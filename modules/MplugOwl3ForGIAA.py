from copy import deepcopy
import inspect
import json
import os
import torch, sys, transformers
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftType
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.tuners.lora import LoraLayer

from PIL import Image
from torchvision import transforms
from einops import rearrange, repeat

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

def manual_scale_and_merge_lora(peft_model, scale_factor: float):
    print(f"Executing FINAL manual LoRA merge with scale factor: {scale_factor}...")
    
    if not hasattr(peft_model, 'get_base_model'):
        raise TypeError("The provided model is not a PeftModel and cannot be merged.")
        
    base_model = peft_model.get_base_model()
    active_adapter = peft_model.active_adapter

    layers_to_replace = []

    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            if active_adapter in module.lora_A:
                base_layer = module.get_base_layer()
                lora_A = module.lora_A[active_adapter].weight
                lora_B = module.lora_B[active_adapter].weight
                scaling = module.scaling[active_adapter]
                
                delta_weight = lora_B @ lora_A * scaling
                scaled_delta_weight = delta_weight * scale_factor
                
                base_layer.weight.data += scaled_delta_weight.to(base_layer.weight.dtype)
                
                parent_name, _, child_name = name.rpartition('.')
                parent_module = peft_model.get_submodule(parent_name)
                layers_to_replace.append((parent_module, child_name, base_layer))

    if not layers_to_replace:
        print("Warning: No LoRA layers were found to merge.")
    else:
        print(f"Found {len(layers_to_replace)} LoRA layers to merge and unload.")

    for parent_module, child_name, base_layer in layers_to_replace:
        setattr(parent_module, child_name, base_layer)

    print("Manual merge and unload completed. The model is now a clean base model.")
    return base_model


class MplugOwl3ForGIAA(nn.Module):
    def __init__(
        self,
        model_path = "models/mPLUG-Owl3-7B",
        preferential_ids = None,
        load_in_8bit = False,
        use_lora = True,
        lora_r = 16,
        lora_alpha = 32,
        lora_dropout = 0.05,
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        processor_image_size = 378,
        processor_cut_enable = True,
    ):
        super().__init__()

        self.grad_flag = False
        self.use_lora = use_lora
        self.processor_image_size = processor_image_size
        self.processor_cut_enable = processor_cut_enable
        
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        
        loading_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if not load_in_8bit else torch.float16,
            "low_cpu_mem_usage": True,
        }

        cuda_available = torch.cuda.is_available()
        if load_in_8bit:
            try:
                import bitsandbytes as bnb  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "bitsandbytes is required for 8-bit loading but could not be imported."
                ) from exc
            if not cuda_available:
                raise RuntimeError("`load_in_8bit=True` requires a CUDA-enabled torch runtime.")
            loading_kwargs["load_in_8bit"] = True
            loading_kwargs["device_map"] = "auto"
            if transformers.utils.is_flash_attn_2_available():
                loading_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                print("flash_attn_2 is unavailable. Falling back to the default attention implementation.")
        elif not cuda_available:
            print("CUDA is unavailable. Falling back to the default device placement.")

        self.LLM = transformers.AutoModel.from_pretrained(
            model_path,
            **loading_kwargs
        )
        
        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.LLM = get_peft_model(self.LLM, lora_config)
            self._freeze_vision_encoder()
            self.LLM.print_trainable_parameters()
        
        with torch.no_grad():
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            self.processor = self.LLM.init_processor(self.tokenizer)
            image_processor_cls = self.processor.image_processor.__class__
            self.processor.image_processor = image_processor_cls(image_size=self.processor_image_size)
            self.processor.image_processor.cut_enable = self.processor_cut_enable
        
        if preferential_ids:
            self.preferential_ids = [id_[0] for id_ in self.tokenizer(preferential_ids)["input_ids"]]
        else:
            self.preferential_ids = [id_[0] for id_ in self.tokenizer(["Excellent", "Good", "Fair", "Poor", "Bad"])["input_ids"]]
    
    def _freeze_vision_encoder(self):
        frozen_param_count = 0
        frozen_module_messages = []

        if hasattr(self.LLM, 'vision_model'):
            for param in self.LLM.vision_model.parameters():
                param.requires_grad = False
                frozen_param_count += param.numel()
            frozen_module_messages.append("vision_model")
            
        if hasattr(self.LLM, 'vision_projector'):
            for param in self.LLM.vision_projector.parameters():
                param.requires_grad = False
                frozen_param_count += param.numel()
            frozen_module_messages.append("vision_projector")
            
        for name, param in self.LLM.named_parameters():
            if any(vision_key in name for vision_key in ['visual', 'vision', 'img']):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_param_count += param.numel()

        if frozen_module_messages or frozen_param_count > 0:
            module_text = ", ".join(frozen_module_messages) if frozen_module_messages else "vision-related modules"
            print(f"Frozen {module_text}; total frozen parameters: {frozen_param_count:,}")
    
    def kl_loss(self, predictions, target_distribution=None, uniform=False, temperature=1.0):
        kl_loss = F.kl_div(
            predictions.log(),
            target_distribution,
            reduction='batchmean'
        )
    
        return kl_loss
    
    def save_lora_weights(self, output_dir):
        if self.use_lora:
            self.LLM.save_pretrained(output_dir)
            print(f"LoRA weights saved to {output_dir}")

    def _merge_current_lora_if_needed(self):
        if isinstance(self.LLM, PeftModel):
            print("Current model already has a LoRA adapter attached. Merging it into the base model first.")
            self.LLM = self.LLM.merge_and_unload()
            self.use_lora = False

    def _load_compatible_peft_config(self, weights_dir):
        config_path = os.path.join(weights_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        peft_type = raw_config.get("peft_type")
        if peft_type is None:
            return None

        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[PeftType(peft_type)]
        valid_fields = set(inspect.signature(config_cls.__init__).parameters.keys()) - {"self"}
        filtered_config = {k: v for k, v in raw_config.items() if k in valid_fields}
        dropped_fields = sorted(set(raw_config.keys()) - set(filtered_config.keys()))
        if dropped_fields:
            print(
                f"Adapter config at {weights_dir} contains unsupported fields for the current peft runtime: "
                f"{', '.join(dropped_fields)}. Ignoring them."
            )
        return config_cls(**filtered_config)

    def _load_peft_model(self, weights_dir, is_trainable=False, device=None):
        compatible_config = self._load_compatible_peft_config(weights_dir)
        load_kwargs = {}
        if device is not None:
            load_kwargs["torch_device"] = str(device)
        return PeftModel.from_pretrained(
            self.LLM,
            weights_dir,
            is_trainable=is_trainable,
            config=compatible_config,
            **load_kwargs
        )
    
    def load_lora_weights(self, weights_dir, trainable=False, device=None):
        self.LLM = self._load_peft_model(weights_dir, is_trainable=trainable, device=device)
        self.use_lora = True
        print(f"LoRA weights loaded from {weights_dir}")
    
    def reinit_lora_weights(
        self, new_lora_r=None, new_lora_alpha=None, 
        new_target_modules=None, new_lora_dropout=None, trainable=True
    ):
        print("Reinitializing LoRA with new configuration...")
        self._merge_current_lora_if_needed()
        
        lora_r = new_lora_r if new_lora_r is not None else 16
        lora_alpha = new_lora_alpha if new_lora_alpha is not None else 32
        target_modules = new_target_modules if new_target_modules is not None else ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_dropout = new_lora_dropout if new_lora_dropout is not None else 0.05
        
        new_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.LLM = get_peft_model(self.LLM, new_lora_config)
        self._freeze_vision_encoder()
        
        if trainable:
            self.LLM.train()
    
        self.use_lora = True
        self.LLM.print_trainable_parameters()
        print(f"LoRA reinitialized with new config")
        print(f"New config: r={lora_r}, alpha={lora_alpha}, targets={target_modules}, dropout={lora_dropout}")
    
    def load_and_reinit_lora_weights(
        self, weights_dir, new_lora_r=None, new_lora_alpha=None,
        new_target_modules=None, new_lora_dropout=None, trainable=True, device=None
    ):
        self._merge_current_lora_if_needed()
        temp_model = self._load_peft_model(weights_dir, is_trainable=False, device=device)

        print("Merging existing LoRA weights...")
        self.LLM = temp_model.merge_and_unload()

        print("Reinitializing LoRA with new configuration...")
        lora_r = new_lora_r if new_lora_r is not None else 16
        lora_alpha = new_lora_alpha if new_lora_alpha is not None else 32
        target_modules = new_target_modules if new_target_modules is not None else ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_dropout = new_lora_dropout if new_lora_dropout is not None else 0.05

        new_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.LLM = get_peft_model(self.LLM, new_lora_config)
        self._freeze_vision_encoder()

        if trainable:
            self.LLM.train()

        self.use_lora = True
        self.LLM.print_trainable_parameters()
        print(f"LoRA weights loaded from {weights_dir} and reinitialized with new config")
        print(f"New config: r={lora_r}, alpha={lora_alpha}, targets={target_modules}, dropout={lora_dropout}")

    def load_sequential_loras(self, first_lora_dir, second_lora_dir, is_trainable=False, device=None):
        temp_model = self._load_peft_model(first_lora_dir, is_trainable=False, device=device)

        print("Merging existing LoRA weights GIAA...")
        self.LLM = temp_model.merge_and_unload()

        print("Merging existing LoRA weights PIAA...")
        self.LLM = self._load_peft_model(second_lora_dir, is_trainable=is_trainable, device=device)
        print(f"LoRA weights loaded GIAA from {first_lora_dir} and PIAA from {second_lora_dir}")

    def load_sequential_loras_control(self, first_lora_dir, second_lora_dir, piaa_scale_factor=1.0, is_trainable=False, device=None):
        print("Loading and merging first LoRA adapter (GIAA)...")
        temp_model = self._load_peft_model(first_lora_dir, is_trainable=False, device=device)
        self.LLM = manual_scale_and_merge_lora(temp_model, piaa_scale_factor)
        print("First LoRA adapter (GIAA) merged.")

        print("\nLoading second LoRA adapter (PIAA)...")
        peft_model_for_piaa = self._load_peft_model(second_lora_dir, is_trainable=is_trainable, device=device)
        self.LLM = manual_scale_and_merge_lora(peft_model_for_piaa, piaa_scale_factor)

        def print_boxed(message):
            border = '#' * (len(message) + 4)
            print(f"\n{border}")
            print(f"# {message} #")
            print(f"{border}\n")

        print_boxed(f"All LoRAs manually merged. GIAA (scale={piaa_scale_factor}), PIAA (scale={piaa_scale_factor}). Final model is ready.")

    
    @torch.no_grad()
    def generate(
        self,
        images: List[Image.Image],
        device: Optional[torch.device] = None,
        msg_input: Optional[List[Dict]] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
            
        batch_size = len(images)
        
        all_inputs = []
        all_prompts = []
        for i in range(batch_size):
            if msg_input is None:
                msg = [
                    {"role": "user", "content": "<|image|>How would you rate the aesthetic quality of this image?"},
                    {"role": "assistant", "content": "The aesthetic quality of the image is"},
                ]
            else:
                msg = deepcopy(msg_input)
            
            prompt_text = self.processor.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt_text)

            inputs = self.processor(
                msg,
                images=[images[i]],
                videos=None,
                cut_enable=self.processor_cut_enable,
            ).to(device)
            all_inputs.append(inputs)
            
        batched_inputs = {}
        for key in all_inputs[0].keys():
            if isinstance(all_inputs[0][key], torch.Tensor):
                batched_inputs[key] = torch.cat([inp[key] for inp in all_inputs], dim=0)
            elif key == "media_offset":
                media_tensors = [inp["media_offset"][0] for inp in all_inputs]
                batched_inputs["media_offset"] = torch.stack(media_tensors, dim=0)

        gen_kwargs = generation_kwargs if generation_kwargs is not None else {}
        default_gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
        }
        default_gen_kwargs.update(gen_kwargs)

        generated_ids = self.LLM.generate(
            **batched_inputs,
            tokenizer=self.tokenizer,
            **default_gen_kwargs
        )
        
        full_generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        generated_descriptions = []
        for i, full_text in enumerate(full_generated_texts):
            clean_text = full_text.split(all_prompts[i])[-1].strip()
            generated_descriptions.append(clean_text)

        del all_inputs, batched_inputs, generated_ids
        torch.cuda.empty_cache()
        
        return generated_descriptions
         
         
    def forward(        
        self,
        images: List[Image.Image] = None,
        videos: List = None,
        labels: List = None,
        device = None,
        msg_input = None,
        **kwargs
    ):
        if device is None:
            device = next(self.parameters()).device
            
        batch_size = len(images)
        
        all_inputs = []
        for i in range(batch_size):
            if msg_input == None:
                msg = [
                    {
                        "role": "user", 
                        "content": f"<|image|>How would you rate the aesthetic quality of this image?" 
                    },
                    {
                        "role": "assistant", 
                        "content": "The aesthetic quality of the image is"},
                ] 
            else:
                msg = deepcopy(msg_input)
            
            with torch.no_grad():
                inputs = self.processor(
                    msg,
                    images=[images[i]],
                    videos=None,
                    cut_enable=self.processor_cut_enable,
                ).to(device)
                all_inputs.append(inputs)
            
        batched_inputs = {}
        for key in all_inputs[0].keys():
            if isinstance(all_inputs[0][key], torch.Tensor):
                batched_inputs[key] = torch.cat([inp[key] for inp in all_inputs], dim=0)
            elif key == "media_offset":
                # media_offset
                media_tensors = [inp["media_offset"][0] for inp in all_inputs]  # tensor
                batched_inputs["media_offset"] = torch.stack(media_tensors, dim=0)  # [batch_size, seq_len]    
            
        outputs = self.LLM(**batched_inputs, output_hidden_states=True)
        outputs_probs_for_distribution = torch.softmax(outputs["logits"][:,-1, self.preferential_ids], -1) # [batch_size, len(self.preferential_ids)]
            
        del all_inputs, batched_inputs, outputs
        torch.cuda.empty_cache()   
            
        if labels is not None: 
            loss_for_batch = self.kl_loss(outputs_probs_for_distribution, labels)
        else:
            loss_for_batch = None
        return outputs_probs_for_distribution, loss_for_batch
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preferential_ids = [
        "Outstanding",
        "Excellent",
        "Superior",
        "Good",
        "Fair",
        "Mediocre",
        "Poor",
        "Bad",
        "Terrible"
    ]
    
    giaa_model = MplugOwl3ForGIAA(preferential_ids=preferential_ids).to(device)
    
    image_folder = "fig"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    batch_images = []
    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        batch_images.append(image)
    
    image_labels = []
    
    with torch.no_grad():
        g, loss = giaa_model(batch_images, device)
        print(g)
        print(loss)

