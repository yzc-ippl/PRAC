import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from tqdm import tqdm
from pathlib import Path

from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA 

from data_loader.data_loader_giaa import get_dataloaders_std
from utils import compute_std, load_config

try:
    _ = torch.compiler.is_compiling
except AttributeError:
    if hasattr(torch, "compiler"):
        torch.compiler.is_compiling = lambda: False
    else:
        class MockCompiler:
            def is_compiling(self): return False
        torch.compiler = MockCompiler()

def PreferSelect_CCM(config, dataset_name="PARA"):
    """Compute the predictive standard deviation for every image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[{dataset_name}] Starting CCM (Confidence/Std Calculation)...")
    
    print("Loading MplugOwl3ForGIAA model...")
    model = MplugOwl3ForGIAA(
        model_path=config['model']['model_path'],
        preferential_ids=config['model']['preferential_ids'],
        load_in_8bit=config['model']['load_in_8bit'],
        use_lora=config['model']['use_lora'],
        lora_r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        lora_target_modules=config['model']['lora_target_modules'],
        processor_image_size=config['data'].get('processor_image_size', config['data']['image_size']),
        processor_cut_enable=config['data'].get('processor_cut_enable', True),
    ).to(device)
    
    if config['model'].get('use_lora_adapter', False):
        adapter_path = config['model']['adapter_path']
        print(f"Loading LoRA adapter from {adapter_path}")
        model.load_lora_weights(adapter_path, device=device)
        
    model.eval()
    
    if dataset_name != "PARA":
        raise ValueError("This release supports the PARA dataset only.")
    print(f"Loading dataloaders for {dataset_name}...")
    train_loader, val_loader = get_dataloaders_std(dataset_name, "train", config)
    test_loader = get_dataloaders_std(dataset_name, "test", config)

    results_all = {}

    # Process all splits with one model so that the prediction cache is consistent.
    def process_loader(loader, desc_text):
        progress_bar = tqdm(loader, desc=f"Processing {desc_text}")
        
        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            labels = groups[1].to(device)
            image_ids_batch = groups[3]
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs, _ = model(images, labels=labels, device=device)
            
            del images, labels
            torch.cuda.empty_cache()
            
            outputs_float = outputs.float()
            
            std_batch = compute_std(config, device, outputs_float)
            
            for i in range(len(image_ids_batch)):
                curr_id = image_ids_batch[i]
                
                if curr_id not in results_all:
                    results_all[curr_id] = []
                
                results_all[curr_id].append(std_batch[i].item())
                
                outputs_list = outputs_float[i].cpu().numpy().tolist()
                results_all[curr_id].append(outputs_list)
        
        print(f"-> Finished {desc_text}. Total records so far: {len(results_all)}")

    process_loader(test_loader, "Test Set")
    process_loader(train_loader, "Train Set")
    process_loader(val_loader, "Validation Set")
    
    output_dir = Path(os.getenv("PRAC_RUN_DIR", "runs/para")) / "preferselect"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_csv = os.path.join(output_dir, f"ccm.csv")
    print(f"Saving results to {output_csv}...")
    
    with open(output_csv, 'w') as f:
        
        for image_id, values in results_all.items():
            std_val = round(values[0], 6)
            distribution = values[1]
            distribution_str = [round(x, 6) for x in distribution]
            
            if dataset_name == "PARA":
                if "-" in str(image_id):
                    parts = str(image_id).split("-", 1)
                    session_id = parts[0]
                    img_sub_id = parts[1]
                else:
                    session_id = "Unknown"
                    img_sub_id = str(image_id)
                
                f.write(f"{session_id},{img_sub_id},{std_val},{distribution_str}\n")
                
            else:
                raise ValueError("This release supports the PARA dataset only.")
                
    print(f"Successfully saved {len(results_all)} records to CSV.")


if __name__ == "__main__":
    target_dataset = os.getenv("PREFERSELECT_DATASET", "PARA")
    config_path = os.getenv("PREFERSELECT_CONFIG_PATH")

    default_config_paths = {
        "PARA": "configs/preferselect_config/para.yaml",
    }

    if config_path is None:
        if target_dataset not in default_config_paths:
            raise ValueError(f"Unsupported dataset name: {target_dataset}")
        config_path = default_config_paths[target_dataset]

    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    PreferSelect_CCM(config, dataset_name=target_dataset)

