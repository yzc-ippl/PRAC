"""
preferSelect_PDM 
:
1. ,
2. (personaoutputs1),
3. 
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import gc
from PIL import Image

from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA
from data_loader.data_loader_para_user_info import get_qualified_users_dataloaders_user_info
from utils import print_parameter_statistics, load_config

try:
    _ = torch.compiler.is_compiling
except AttributeError:
    if hasattr(torch, "compiler"):
        torch.compiler.is_compiling = lambda: False
    else:
        class MockCompiler:
            def is_compiling(self): return False
        torch.compiler = MockCompiler()


def get_completed_users(output_path):
    """Return user IDs whose cached PDM CSV already exists."""
    completed = set()
    if os.path.exists(output_path):
        for f in os.listdir(output_path):
            if f.startswith("pdm_user_") and f.endswith(".csv"):
                user_id = f.replace("pdm_user_", "").replace(".csv", "")
                completed.add(user_id)
    return completed


def _resolve_image_path(img_root_dir, image_id):
    if os.path.isabs(image_id):
        return image_id
    return os.path.join(img_root_dir, image_id)


def _load_and_resize_image(img_root_dir, image_id, image_size):
    if "/" not in str(image_id) and "-" in str(image_id):
        session_id, image_name = str(image_id).split("-", 1)
        img_path = os.path.join(img_root_dir, session_id, image_name)
    else:
        img_path = _resolve_image_path(img_root_dir, image_id)
    image = Image.open(img_path).convert('RGB')
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return image


def collect_unique_image_ids(config, user_dataloaders):
    dataset_cfg = config.get('user_info_dataset', {})
    csv_path = dataset_cfg.get('csv_path')
    image_id_column = dataset_cfg.get('image_id_column', 'image_id')

    print("Collecting unique image ids from user dataloaders...")
    unique_ids = []
    seen_ids = set()
    seen_loader_ids = set()
    for _, (_, user_loader) in tqdm(user_dataloaders.items(), desc="Scanning users"):
        loader_identity = id(user_loader)
        if loader_identity in seen_loader_ids:
            continue
        seen_loader_ids.add(loader_identity)

        for groups in user_loader:
            images_id = groups[3]
            for img_id in images_id:
                if img_id not in seen_ids:
                    seen_ids.add(img_id)
                    unique_ids.append(img_id)

    print(f"Total unique images: {len(unique_ids)}")
    return unique_ids


def compute_generic_predictions(
    model, image_ids, device, cache_path, img_root_dir, image_size
):
    """Compute generic predictions once and cache them for all users."""
    if os.path.exists(cache_path):
        print(f"Loading cached generic predictions from {cache_path}")
        return torch.load(cache_path)

    print(f"Computing generic predictions for {len(image_ids)} unique images...")
    generic_cache = {}

    model.eval()
    batch_size = 8

    for start_idx in tqdm(range(0, len(image_ids), batch_size), desc="Computing generic predictions"):
        end_idx = min(start_idx + batch_size, len(image_ids))
        batch_ids = image_ids[start_idx:end_idx]
        batch_images = []
        valid_batch_ids = []

        for img_id in batch_ids:
            try:
                batch_images.append(
                    _load_and_resize_image(
                        img_root_dir,
                        img_id,
                        image_size,
                    )
                )
                valid_batch_ids.append(img_id)
            except Exception as e:
                print(f"Error loading image {_resolve_image_path(img_root_dir, img_id)}: {e}")

        if not valid_batch_ids:
            continue

        torch.cuda.empty_cache()

        with torch.no_grad():
            outputs, _ = model(batch_images, labels=None, device=device, msg_input=None)

        outputs = outputs.float().cpu()

        for i, img_id in enumerate(valid_batch_ids):
            generic_cache[img_id] = outputs[i]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(generic_cache, cache_path)
    print(f"Generic predictions cached to {cache_path}")

    return generic_cache

def computeKL_optimized(config, user_dataloaders, output_path, cache_path):
    """Compute personalized-versus-generic KL divergence for each user."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    completed_users = get_completed_users(output_path)
    print(f"Already completed: {len(completed_users)} users")

    todo_users = {k: v for k, v in user_dataloaders.items() if k not in completed_users}
    print(f"Remaining users to process: {len(todo_users)}")

    if not todo_users:
        print("All users completed!")
        return

    print("Loading MplugOwl3ForGIAA model (one-time load)...")
    model = MplugOwl3ForGIAA(
        model_path=config['model']['model_path'],
        preferential_ids=config['model']['preferential_ids'],
        load_in_8bit=config['model']['load_in_8bit'],
        use_lora=config['model']['use_lora'],
        lora_r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        lora_target_modules=config['model']['lora_target_modules'],
        processor_image_size=config['data'].get('processor_image_size', config['data'].get('image_size', 378)),
        processor_cut_enable=config['data'].get('processor_cut_enable', True),
    ).to(device)

    if config['model'].get('use_lora_adapter', False):
        print(f"Loading LoRA adapter from {config['model']['adapter_path']}")
        model.load_lora_weights(config['model']['adapter_path'], device=device)

    print_parameter_statistics(model)

    dataset_cfg = config.get('user_info_dataset', {})
    img_root_dir = dataset_cfg.get('img_root_dir')
    image_size = dataset_cfg.get('image_size')

    # ()
    if not os.path.exists(cache_path):
        image_ids = collect_unique_image_ids(config, user_dataloaders)
        generic_cache = compute_generic_predictions(
            model,
            image_ids,
            device,
            cache_path,
            img_root_dir=img_root_dir,
            image_size=image_size,
        )
    else:
        generic_cache = compute_generic_predictions(
            model,
            [],
            device,
            cache_path,
            img_root_dir=img_root_dir,
            image_size=image_size,
        )

    os.makedirs(output_path, exist_ok=True)
    count = 0
    total_users = len(todo_users)

    for user_id, (user_prompt_text, user_loader) in todo_users.items():
        count += 1
        print("\n" + "+"*64)
        print(f"Processing User: {user_id} ({count}/{total_users})")
        print(f"User Info: {user_prompt_text[:100]}...")
        print(f"User Image Count: {len(user_loader.dataset)}")

        KL_results = {}

        # persona
        persona_msg = [
            {
                "role": "user",
                "content": f"<|image|>Imagine that you are the person described as follows: {user_prompt_text} In such case, how would you rate the aesthetic quality of this image?",
            },
            {
                "role": "assistant",
                "content": "The aesthetic quality of the image is"
            },
        ]

        model.eval()
        progress_bar = tqdm(user_loader, desc=f"User {user_id}")

        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            images_id = groups[3]

            torch.cuda.empty_cache()

            with torch.no_grad():
                outputs_persona, _ = model(images, labels=None, device=device, msg_input=persona_msg)

            outputs_persona = outputs_persona.float().cpu()

            for i, img_id in enumerate(images_id):
                outputs_generic = generic_cache[img_id]

                kl_div = F.kl_div(outputs_generic.log(), outputs_persona[i], reduction='batchmean')

                KL_results[img_id] = {
                    'generic': outputs_generic,
                    'persona': outputs_persona[i],
                    'kl': kl_div.item()
                }

        output_file = os.path.join(output_path, f"pdm_user_{user_id}.csv")
        with open(output_file, 'w') as f:
            f.write("session_id,image_id,GIAA_distribution,GIAA+UserPrompt_distribution,KL\n")
            for image_id, values in KL_results.items():
                if "-" in str(image_id):
                    session_id = str(image_id).split("-")[0]
                    img_sub_id = str(image_id).split("-")[1]
                else:
                    session_id = "Unknown"
                    img_sub_id = str(image_id)

                GIAA_distribution = f'"{str(values["generic"].numpy().tolist())}"'
                GIAA_UserPrompt_distribution = f'"{str(values["persona"].numpy().tolist())}"'
                KL_score = values['kl']

                f.write(f"{session_id},{img_sub_id},{GIAA_distribution},{GIAA_UserPrompt_distribution},{KL_score}\n")

        print(f"Saved results for user {user_id}")

        if count % 10 == 0:
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nAll users processed!")


if __name__ == "__main__":
    target_dataset = os.getenv("PREFERSELECT_DATASET", "PARA")

    if target_dataset == "PARA":
        user_info_config_path = "configs/preferselect_config/para_user_info.yaml"
        preferselect_config_path = "configs/preferselect_config/para.yaml"
        run_dir = os.getenv("PRAC_RUN_DIR", "runs/para")
        output_path = os.path.join(run_dir, "preferselect", "pdm")
        cache_path = os.path.join(run_dir, "preferselect", "pdm_cache", "generic_predictions.pt")
        dataset_name = "PARA"
    else:
        raise ValueError("This release supports the PARA dataset only.")

    print(f"Loading user-info configuration from {user_info_config_path}")
    user_info_config = load_config(user_info_config_path)
    print(f"Loading preferselect configuration from {preferselect_config_path}")
    preferselect_config = load_config(preferselect_config_path)
    preferselect_config['dataset_name'] = dataset_name
    preferselect_config['user_info_dataset'] = user_info_config['dataset']

    print("Loading user dataloaders...")
    user_dataloaders = get_qualified_users_dataloaders_user_info(
        user_info_config,
        random_seed=None,
        get_all_image_of_user=True,
        get_all_users=True
    )

    print(f"Total users: {len(user_dataloaders)}")

    computeKL_optimized(preferselect_config, user_dataloaders, output_path, cache_path)

