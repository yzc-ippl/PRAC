import os
#  tokenizers 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
from datetime import datetime
import gc

RUN_DIR = os.getenv("PRAC_RUN_DIR", "runs/para")
log_f = None
original_stdout = None

class Tee:
    def __init__(self, file_stream, terminal_stream):
        self.file = file_stream
        self.terminal = terminal_stream
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()

def start_logging():
    global RUN_DIR, log_f, original_stdout
    RUN_DIR = os.getenv("PRAC_RUN_DIR", "runs/para")
    os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
    log_file = os.path.join(
        RUN_DIR,
        "logs",
        f"prefermerge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    original_stdout = sys.stdout
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(log_f, original_stdout)


def stop_logging():
    global log_f, original_stdout
    if original_stdout is not None:
        sys.stdout = original_stdout
        original_stdout = None
    if log_f is not None:
        log_f.close()
        log_f = None

import csv
import shutil
import math
import torch
import pandas as pd
from safetensors.torch import load_file
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# --- 1.  Qwen  ---
from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA 

from tqdm import tqdm
from data_loader.data_loader_piaa import get_qualified_users_dataloaders_using_image_select
from utils import print_parameter_statistics, print_peft_config, compute_score_and_onehot, compute_cross_entropy, compute_mse_loss, compute_MOS, compute_rank_loss, compute_SRCC_PLCC, load_config
from utils import print_boxed, print_double_boxed
from utils import extract_top_similar_users, find_user_folder

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

MERGE_RESULT_FIELDS = [
    "User",
    "merge_SRCC_before_ft",
    "merge_PLCC_before_ft",
    "merge_SRCC_after_ft",
    "merge_PLCC_after_ft",
    "before_merge_train_srcc",
    "after_merge_train_srcc",
    "whether_merge",
]

# --- Monkey Patch ---
try:
    _ = torch.compiler.is_compiling
except AttributeError:
    if hasattr(torch, "compiler"):
        torch.compiler.is_compiling = lambda: False
    else:
        class MockCompiler:
            def is_compiling(self): return False
        torch.compiler = MockCompiler()

def normalize_lora_key(key):
    return key.replace(".default.", ".")

def resolve_node2_user_scope(merge_config):
    enabled_users = merge_config.get('enabled_users')
    selected_user_ids = None
    if enabled_users is None:
        print("enabled_users not set; keep the original test-user scope")
    elif not isinstance(enabled_users, list):
        print("Warning: enabled_users must be a list. Ignore this option and keep the original test-user scope")
    else:
        normalized_user_ids = []
        seen_user_ids = set()
        for user_id in enabled_users:
            normalized_user_id = str(user_id).strip()
            if normalized_user_id and normalized_user_id not in seen_user_ids:
                normalized_user_ids.append(normalized_user_id)
                seen_user_ids.add(normalized_user_id)
        if normalized_user_ids:
            selected_user_ids = normalized_user_ids
            print(f"enabled_users configured with {len(selected_user_ids)} candidate users")
        else:
            print("enabled_users is empty or invalid after normalization; keep the original test-user scope")

    reference_user_id = merge_config.get('testset_reference_user')
    if reference_user_id is None:
        resolved_reference_user_id = None
        print("testset_reference_user not set; keep per-user test sets")
    else:
        resolved_reference_user_id = str(reference_user_id).strip()
        if resolved_reference_user_id:
            print(f"testset_reference_user configured as {resolved_reference_user_id}")
        else:
            resolved_reference_user_id = None
            print("testset_reference_user is empty after normalization; keep per-user test sets")

    return selected_user_ids, resolved_reference_user_id

def train_progress(config, model, train_loader, test_loader, user_id, count, device, pool_name):
    best_srcc = float('-inf')
    best_plcc = float('-inf')
    early_count = 0
        
    best_model_path = None
    temp_dir = os.path.join(config['output']['lora_pool_dir'], f'{pool_name}/temp_checkpoints_user_{user_id}')
    os.makedirs(temp_dir, exist_ok=True)
    
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    total_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['training']['warmup_steps'], num_training_steps=total_steps)

    for epoch in range(config['training']['epochs']):
        # train
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"       Epoch {epoch+1}/{config['training']['epochs']} Training")

        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            scores_onehot_labels = groups[1].to(device)
            scores_labels = groups[2].to(device)
            
            torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            
            outputs, _ = model(images, labels=None, device=device)
            
            del images
            torch.cuda.empty_cache()

            outputs = outputs.float()

            ce_loss = compute_cross_entropy(outputs, scores_onehot_labels, reduction='mean')
            rank_loss = compute_rank_loss(outputs, scores_onehot_labels, margin=1.0)
            loss = ce_loss + rank_loss * 2
            
            if loss is not None:
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()
                scheduler.step()
        
        print(f"        Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")
        
        plcc, srcc = test_progress(config, test_loader, model, device, f"   After fine-tuning Testing for User {user_id} after epoch {epoch+1}")
        print(f"    For fine-tuning epoch {epoch+1} : PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")

        if srcc > best_srcc:
            if best_model_path and os.path.exists(best_model_path):
                try:
                    shutil.rmtree(best_model_path)
                    print(f"    Removed previous best model: {best_model_path}")
                except Exception as e:
                    print(f"    Warning: Failed to remove previous model: {e}")
            
            best_srcc = srcc
            best_plcc = plcc
            
            current_model_path = os.path.join(temp_dir, f'best_model_epoch_{epoch+1}_srcc_{srcc:.4f}')
            model.save_lora_weights(current_model_path)
            best_model_path = current_model_path
            
            print(f"    New best model saved: SRCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}")
            early_count = 0
            print(f"    Early stopping count reset to {early_count}\n")
        
        elif srcc <= best_srcc:
            early_count += 1
            print(f"    Early stopping count: {early_count}\n")
            if early_count > 30 and best_srcc - srcc > 0.1:
                print_boxed(f"Early stopping at epoch {epoch+1}")
                break
            elif early_count > 60:
                print_boxed(f"Early stopping at epoch {epoch+1}")
                break
        
        elif math.isnan(srcc):
            print_boxed("NaN detected in SRCC, stopping training.")
            break
    
    # ,
    if best_model_path and os.path.exists(best_model_path):
        final_save_path = os.path.join(config['output']['lora_pool_dir'], f'{pool_name}/count_{count}_LoRA_for_user_{user_id}_BestPLCC_{best_plcc:.4f}_BestSRCC_{best_srcc:.4f}')
        shutil.copytree(best_model_path, final_save_path)
        print_boxed(f"Best model saved to: {final_save_path}")
        
        try:
            shutil.rmtree(temp_dir)
            print(f"    Cleaned up temporary directory: {temp_dir}\n\n")
        except Exception as e:
            print(f"    Warning: Failed to clean up temporary directory: {e}")
    
    del optimizer, scheduler
    # model,del
    torch.cuda.empty_cache()
    
    return best_plcc, best_srcc, final_save_path

def train_progress_without_save_LoRA(config, model, train_loader, test_loader, user_id, count, device):
    best_srcc = float('-inf')
    best_plcc = float('-inf')
    early_count = 0
    
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    total_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['training']['warmup_steps'], num_training_steps=total_steps)

    for epoch in range(config['training']['epochs']):
        # train
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"       Epoch {epoch+1}/{config['training']['epochs']} Training")

        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            scores_onehot_labels = groups[1].to(device)
            # scores_labels = groups[2].to(device)
            
            torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            
            outputs, _ = model(images, labels=None, device=device)
            
            del images
            torch.cuda.empty_cache()
            
            outputs = outputs.float()
            
            ce_loss = compute_cross_entropy(outputs, scores_onehot_labels, reduction='mean')
            rank_loss = compute_rank_loss(outputs, scores_onehot_labels, margin=1.0)
            loss = ce_loss + rank_loss * 2
            
            if loss is not None:
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()
                scheduler.step()
        
        print(f"        Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")
        
        plcc, srcc = test_progress(config, test_loader, model, device, f"   After fine-tuning Testing for User {user_id} after epoch {epoch+1}")
        print(f"    For fine-tuning epoch {epoch+1} : PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")

        if srcc >= best_srcc:       
            best_srcc = srcc
            best_plcc = plcc
            print(f"    New best model saved: SRCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}")
            early_count = 0
            print(f"    Early stopping count reset to {early_count}\n")
            
        elif srcc <= best_srcc:
            early_count += 1
            print(f"    Early stopping count: {early_count}\n")
            if early_count > 3 and best_srcc - srcc > 0.08:
                print_boxed(f"Early stopping at epoch {epoch+1}")
                break
            elif early_count > 5:
                print_boxed(f"Early stopping at epoch {epoch+1}")
                break
        
        elif math.isnan(srcc):
            print_boxed("NaN detected in SRCC, stopping training.")
            break
    
    del optimizer, scheduler
    torch.cuda.empty_cache()
    
    return best_plcc, best_srcc
    
def test_progress(config, test_loader, model, device, desc_description):
    score_pred_all = None
    score_gt_all = None
    progress_bar = tqdm(test_loader, desc=desc_description)

    for batch_idx, groups in enumerate(progress_bar):
        model.eval()
        images = groups[0]
        # scores_onehot_labels = groups[1].to(device)
        scores_labels = groups[2].to(device)
    
        torch.cuda.empty_cache()
    
        with torch.no_grad():
            outputs, _ = model(images, labels=None, device=device)
        
        del images
        torch.cuda.empty_cache()

        outputs = outputs.float()

        scores, scores_onehot = compute_score_and_onehot(config, device, outputs)
        
        score_pred_all = scores if score_pred_all is None else torch.cat((score_pred_all, scores))
        score_gt_all = scores_labels if score_gt_all is None else torch.cat((score_gt_all, scores_labels))
    
    plcc, srcc = compute_SRCC_PLCC(score_pred_all, score_gt_all)
    return plcc, srcc

def got_LoRA_for_users(config, user_dataloaders, pool_name):
    save_dir = os.path.join(config['output']['lora_pool_dir'], f"{pool_name}/{pool_name}_index.csv")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # Node1 regenerates the complete pool, so always start with one header.
    with open(save_dir, 'w', encoding='utf-8', newline='') as file:
        file.write("User,basePLCC,baseSRCC,bestPLCC,bestSRCC,LoRAPath\n")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    count = 0
    
    for user_id, (train_loader, test_loader) in user_dataloaders.items():
        all_results_before = {}
        all_results_finetune = {}
        count += 1
        
        print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print_boxed(f"For User: {user_id} count: {count}")

        # load base model
        print("Loading MplugOwl3ForGIAA...")
        giaa_model = MplugOwl3ForGIAA(
            model_path=config['model']['model_path'],
            preferential_ids=config['model']['preferential_ids'],
            load_in_8bit=config['model']['load_in_8bit'],
            use_lora=config['model']['use_lora'],
            lora_r=config['model']['lora_r'],
            lora_alpha=config['model']['lora_alpha'],
            lora_dropout=config['model']['lora_dropout'],
            lora_target_modules=config['model']['lora_target_modules']
        ).to(device)
        
        if config['model']['use_lora_adapter']:
            giaa_model.load_and_reinit_lora_weights(
                config['model']['adapter_path'],
                new_lora_r=4,
                new_lora_alpha=8,
                new_target_modules={'lm_head'},
                new_lora_dropout=0.7,
                trainable=True,
                device=device
            )
            
        print_parameter_statistics(giaa_model)
        print_peft_config(giaa_model)
        
        plcc, srcc = test_progress(config, train_loader, giaa_model, device, f"  Before fine-tuning Testing for User {user_id}")
        base_srcc = srcc
        base_plcc = plcc
        print(f"    Base SRCC: {base_srcc:.4f}, PLCC: {base_plcc:.4f}")
        all_results_before[user_id] = (base_plcc, base_srcc)
        
        print_boxed(f"For User: {user_id} count: {count} start fine-tuning")
        if getattr(test_loader, 'is_shared_test_set', False):
            print(f"    Shared test set size: {getattr(test_loader, 'shared_test_set_size', len(test_loader.dataset))}")
        
        best_plcc, best_srcc, save_path = train_progress(config, giaa_model, train_loader, test_loader, user_id, count, device, pool_name)    
                
        all_results_finetune[user_id] = (best_plcc, best_srcc)
        print(f"before finetune: PLCC {all_results_before[user_id][0]:.4f}, SRCC {all_results_before[user_id][1]:.4f}")
        print(f"after finetune: PLCC {all_results_finetune[user_id][0]:.4f}, SRCC {all_results_finetune[user_id][1]:.4f}")
        
        del giaa_model
        gc.collect()
        torch.cuda.empty_cache()
        
        with open(save_dir, 'a', encoding='utf-8', newline='') as file:
            file.write(f"{user_id},{all_results_before[user_id][0]},{all_results_before[user_id][1]},{all_results_finetune[user_id][0]},{all_results_finetune[user_id][1]},{save_path}\n")
    
    return save_dir

def model_merge(config, user_dataloaders, user_lora_dict, piaa_size, device):
    save_dir = config['output']['piaa_result_dir'] + f"_{random_seed}_mergetest_{piaa_size}.csv"
    
    if not os.path.exists(save_dir):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, "w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=MERGE_RESULT_FIELDS).writeheader()
        user_alreadly_have = []
    else:
        print(f"Warning: {save_dir} already exists, appending results to the file.")
        user_alreadly_have = pd.read_csv(save_dir)['User'].tolist()
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    count = 0
    all_results_before_GIAA = {}
    all_results_finetune_GIAA = {}
    all_results_before_merge = {}
    all_results_finetune_merge = {}
    
    for test_user_id in user_lora_dict.keys():
        if test_user_id in user_alreadly_have:
            print(f"Skipping test user {test_user_id} as results already exist.")
            continue
        
        train_loader = user_dataloaders[test_user_id][0]
        test_loader = user_dataloaders[test_user_id][1]
        count += 1
        print_boxed(f"Test User: {test_user_id}")
    
        all_lora_state_dicts = {}
        
        lora_paths = user_lora_dict[test_user_id]
        lora_paths = list(lora_paths.keys())

        for lora_path in lora_paths:
            if lora_path:
                adapter_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
                adapter_bin = os.path.join(lora_path, "adapter_model.bin")
                if os.path.exists(adapter_safetensors):
                    full_state_dict = load_file(adapter_safetensors)
                elif os.path.exists(adapter_bin):
                    full_state_dict = torch.load(adapter_bin, map_location="cpu")
                else:
                    raise FileNotFoundError(f"No adapter weights found under: {lora_path}")

                lora_state_dict = {
                    key: value.clone()
                    for key, value in full_state_dict.items()
                    if 'lora_' in key
                }
                all_lora_state_dicts[lora_path] = lora_state_dict

                del full_state_dict
                gc.collect()
                torch.cuda.empty_cache()
        
        merged_lora_state_dict = {}
        
        lora_keys = set()
        for lora_state_dict in all_lora_state_dicts.values():
            lora_keys.update(lora_state_dict.keys())
        print(f" {len(lora_keys)} LoRA")
        
        for key in lora_keys:
            merged_param = None
            for lora_path, lora_state_dict in all_lora_state_dicts.items():
                if key in lora_state_dict:
                    weight = user_lora_dict[test_user_id][lora_path]
                    weighted_param = lora_state_dict[key] * weight
                    if merged_param is None:
                        merged_param = weighted_param
                    else:
                        merged_param += weighted_param
            
            if merged_param is not None:
                merged_lora_state_dict[key] = merged_param.to(device)
        
        giaa_model = MplugOwl3ForGIAA(
            model_path=config['model']['model_path'],
            preferential_ids=config['model']['preferential_ids'],
            load_in_8bit=config['model']['load_in_8bit'],
            use_lora=config['model']['use_lora'],
            lora_r=config['model']['lora_r'],
            lora_alpha=config['model']['lora_alpha'],
            lora_dropout=config['model']['lora_dropout'],
            lora_target_modules=config['model']['lora_target_modules']
        ).to(device)
        
        if config['model']['use_lora_adapter']:
            giaa_model.load_and_reinit_lora_weights(
                config['model']['adapter_path'],
                new_lora_r=4,
                new_lora_alpha=8,
                new_target_modules={'lm_head'},
                new_lora_dropout=0.7,
                trainable=True,
                device=device
            )
        # print_parameter_statistics(giaa_model)
        # print_peft_config(giaa_model)
        # for name, param in giaa_model.named_parameters():
        #     if "lora_" in name:
        #         print(f"{name}:\n{param.data}")
        #         print(torch.all(param.data == 0).item())
        print("========== Start merging weights into model ==========")
        updated_count = 0
        with torch.no_grad():
            for name, param in giaa_model.LLM.named_parameters():
                lookup_name = normalize_lora_key(name)
                if lookup_name in merged_lora_state_dict:
                    alpha = 1.0 # 
                    source_weight = merged_lora_state_dict[lookup_name].to(param.device, dtype=param.dtype)
                    fused_weight = (1 - alpha) * param + alpha * source_weight
                    param.copy_(fused_weight)
                    updated_count += 1
        
        print(f"    Merged {updated_count} LoRA parameters.")

        _, before_merge_train_srcc = test_progress(
            config,
            train_loader,
            giaa_model,
            device,
            f"  Before fine-tuning train evaluation for User {test_user_id}",
        )
        
        all_results_before_GIAA[test_user_id] = (0, 0)
        all_results_finetune_GIAA[test_user_id] = (0, 0)
        
        print("========== merge Before fine-tuning ==========")
        plcc, srcc = test_progress(config, test_loader, giaa_model, device, f"  Before fine-tuning Testing for User {test_user_id}")
        base_srcc = srcc
        base_plcc = plcc
        print(f"    merge Base SRCC: {base_srcc:.4f}, PLCC: {base_plcc:.4f}")
        all_results_before_merge[test_user_id] = (base_plcc, base_srcc)
        
        print("========== merge fine-tuning ==========")
        if getattr(test_loader, 'is_shared_test_set', False):
            print(f"    Shared test set size: {getattr(test_loader, 'shared_test_set_size', len(test_loader.dataset))}")
        best_plcc, best_srcc = train_progress_without_save_LoRA(config, giaa_model, train_loader, test_loader, test_user_id, count, device)
        
        if best_srcc < base_srcc:
            best_srcc = base_srcc
            best_plcc = base_plcc
        _, after_merge_train_srcc = test_progress(
            config,
            train_loader,
            giaa_model,
            device,
            f"  After fine-tuning train evaluation for User {test_user_id}",
        )
        print(f"    merge After SRCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}")
        all_results_finetune_merge[test_user_id] = (best_plcc, best_srcc)
        
        with open(save_dir, "a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=MERGE_RESULT_FIELDS).writerow({
                "User": test_user_id,
                "merge_SRCC_before_ft": all_results_before_merge[test_user_id][1],
                "merge_PLCC_before_ft": all_results_before_merge[test_user_id][0],
                "merge_SRCC_after_ft": all_results_finetune_merge[test_user_id][1],
                "merge_PLCC_after_ft": all_results_finetune_merge[test_user_id][0],
                "before_merge_train_srcc": before_merge_train_srcc,
                "after_merge_train_srcc": after_merge_train_srcc,
                "whether_merge": bool(updated_count),
            })
        del giaa_model, merged_lora_state_dict, all_lora_state_dicts, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        
    return all_results_before_GIAA, all_results_finetune_GIAA, all_results_before_merge, all_results_finetune_merge

# --- Main Execution ---
if __name__ == "__main__":
    start_logging()
    whether_having_test_user_final_lora = False
    
    node = os.getenv("PREFERMERGE_NODE", "node1").lower()
    if node not in {"node1", "node2"}:
        raise ValueError("PREFERMERGE_NODE must be node1 or node2")
    # node1,FIM,node2

    dataset = os.getenv("PREFERMERGE_DATASET", "PARA").upper()
    random_seed_list = [int(os.getenv("PREFERMERGE_SEED", "41"))]
    device = os.getenv("PREFERMERGE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    dataset_config_paths = {
        "PARA": {
            "lora_pool": "configs/prefermerge_config/para_lora_pool.yaml",
            "test_users": "configs/prefermerge_config/para_test_users.yaml",
            "merge": "configs/prefermerge_config/para.yaml",
        },
    }
    if dataset not in dataset_config_paths:
        raise ValueError(f"Unsupported dataset: {dataset}. Add its config paths to dataset_config_paths.")
    selected_paths = dataset_config_paths[dataset]
    lora_pool_config = load_config(selected_paths["lora_pool"])
    test_user_config = load_config(selected_paths["test_users"])
    merge_config = load_config(selected_paths["merge"])

    piaa_size = lora_pool_config['dataset']['piaa_size']
    if node == "node2":
        enabled_user_ids, testset_reference_user = resolve_node2_user_scope(merge_config)
    else:
        enabled_user_ids, testset_reference_user = None, None
    
    for random_seed in random_seed_list:
        print(f"random_seed: {random_seed}")
        
        if dataset == "PARA":            
            # stage 1: got test users
            print_double_boxed("stage 1: got test users")
            user_dataloaders_for_test = get_qualified_users_dataloaders_using_image_select(
                test_user_config,
                dataset,
                random_seed=random_seed,
                using_user_prompts=False,
                selected_user_ids=enabled_user_ids,
                testset_reference_user=testset_reference_user,
            )
            print(f"Got {len(user_dataloaders_for_test)} Users for test")

            # stage 2: got users for LoRA-pool
            print_double_boxed("stage 2: got users for LoRA pool")
            if node == "node1":
                user_dataloaders_for_LoRA_pool = get_qualified_users_dataloaders_using_image_select(lora_pool_config, dataset, random_seed=random_seed, using_user_prompts=False)
                diff_users = [k for k in user_dataloaders_for_LoRA_pool.keys() if k not in user_dataloaders_for_test.keys()]
                user_dataloaders_for_LoRA_pool = {k: user_dataloaders_for_LoRA_pool[k] for k in diff_users}
                print_boxed(f"Got {len(user_dataloaders_for_LoRA_pool)} Users for LoRA pool")
            else:
                print_boxed(f"Already successfully got LoRA pool, skip stage 2")
            
            # stage 3: generate LoRA-pool
            # FIM, LoRA-pool  test users  LoRA 
            print_double_boxed("stage 3: generate LoRA-pool")
            if node == "node1":
                save_dir = got_LoRA_for_users(
                    merge_config, user_dataloaders_for_LoRA_pool, "LoRAPool"
                )
                test_index = os.path.join(
                    merge_config['output']['lora_pool_dir'],
                    "TestUsersPool",
                    "TestUsersPool_index.csv",
                )
                os.makedirs(os.path.dirname(test_index), exist_ok=True)
                with open(test_index, "w", newline="", encoding="utf-8") as index_file:
                    writer = csv.DictWriter(index_file, fieldnames=["User", "LoRAPath"])
                    writer.writeheader()
                    for user_id in user_dataloaders_for_test:
                        writer.writerow({"User": user_id, "LoRAPath": ""})
                print_boxed(f"Saved fixed test-user index to {test_index}")
            else:
                save_dir = os.path.join(
                    merge_config['output']['lora_pool_dir'],
                    "LoRAPool",
                    "LoRAPool_index.csv",
                )
                print_boxed("Existing LoRA pool will be reused")
            
            # Load LoRA pool info
            user_lora_dict = {}
            if os.path.exists(save_dir):
                try:
                    with open(save_dir, 'r', encoding='utf-8') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for row in csv_reader:
                            if row['User'] and row['LoRAPath']:
                                user_lora_dict[row['User']] = row['LoRAPath']
                except Exception as e:
                    print(f"Error: {e}")
                
            print_boxed(f"Finally got {len(user_lora_dict)} users in LoRA-pool") 
            
            # stage 4: Select Qualified LoRAs
            if node == "node2":
                print_double_boxed("stage 4: select the qualified LoRA-pool models")
                test_user_final_lora_dict = {}
                select_weight_threshold = float(os.getenv("PRAC_COHORT_BETA", "0.5"))
                select_num = int(os.getenv("PRAC_COHORT_SIZE", "6"))
                selection_file = os.getenv(
                    "PRAC_COHORT_CSV",
                    os.path.join(
                        RUN_DIR,
                        "fim",
                        f"target_users_optimal_results_top{select_num}_{select_weight_threshold}.csv",
                    ),
                )
                similarity_file = os.getenv(
                    "PRAC_FIM_CSV",
                    os.path.join(RUN_DIR, "fim", "user_similarities_fixed.csv"),
                )
                
                test_user_names = list(user_dataloaders_for_test.keys())
                
                for user in test_user_names:
                    print_boxed(f"test user: {user}")
                    select_user_csv = selection_file
                    if not os.path.exists(select_user_csv):
                        print(f"Warning: {select_user_csv} not found, skipping selection logic.")
                        continue

                    select_user = pd.read_csv(select_user_csv)
                    if user in select_user['Target_User_ID'].values:
                        row = select_user[select_user['Target_User_ID'] == user]
                        selected_users_str = row['Selected_Users'].iloc[0]
                        selected_users_list = [u.strip() for u in selected_users_str.split('|')]
                        
                        top_similar = extract_top_similar_users(similarity_file, user, False, top_n=15)
                        filtered_top_similar = top_similar[top_similar['train_user_id'].isin(selected_users_list)]
                        
                        top_n = {}
                        top_n['LoRAUser'] = filtered_top_similar['train_user_id']
                        top_n['weight'] = filtered_top_similar['similarity']
                        top_n['weight_normalized'] = top_n['weight'] / top_n['weight'].sum()
                        
                        lora_paths = []
                        base_lora_dir = os.path.join(merge_config['output']['lora_pool_dir'], f'LoRAPool')
                        for i in range(min(select_num, len(filtered_top_similar))):
                            u_id = list(filtered_top_similar['train_user_id'])[i]
                            path = find_user_folder(base_lora_dir, u_id)
                            # print(f"Find LoRA weight from {path} to merge.")
                            lora_paths.append(path)
                        
                        top_n['LoRAPath'] = lora_paths
                        lora_dict = dict(zip(top_n['LoRAPath'], top_n['weight_normalized']))
                        test_user_final_lora_dict[user] = lora_dict

            if node == "node2":
                # stage 5: Model Merge
                if not whether_having_test_user_final_lora:
                    print_double_boxed("stage 5: Model Merge")
                    all_results_before_giaa, all_results_finetune_giaa, all_results_before_merge, all_results_finetune_merge = model_merge(
                        merge_config, 
                        user_dataloaders_for_test, 
                        test_user_final_lora_dict,
                        piaa_size=piaa_size,
                        device=device
                    )
                print_boxed("Model Merge Done")

    stop_logging()

