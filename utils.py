import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import yaml
from scipy import stats
import torch
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import random
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import glob

def print_boxed(text):
    width = len(text) + 4
    print("+" + "-" * width + "+")
    print("|  " + text + "  |")
    print("+" + "-" * width + "+")
    
def print_double_boxed(text):
    line = "=" * (len(text) + 8)
    print(f"\n{line}\n||  {text}  ||\n{line}\n")

def print_parameter_statistics(model):
    total_params = sum(p.numel() for p in model.parameters())
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    non_trainable_params = total_params - trainable_params
    
    print(f'\nChecking for Lora Parameters:')
    print(f'    Total Parameters: {total_params:,}')
    print(f'    Trainable Parameters: {trainable_params:,}')
    print(f'    Non-trainable Parameters: {non_trainable_params:,}')


def compute_std(config, device, outputs_distribution=None):
    score_distribution = outputs_distribution
    score_mapping = config['data']['score_mapping']
    score_mapping = torch.tensor(score_mapping).to(device)
    mean = (score_distribution @ score_mapping) # [batch_size]
    variance = (score_distribution * score_mapping ** 2).sum(dim=1) - mean ** 2
    std = torch.sqrt(variance)
    return std


def compute_score_and_onehot(config, device, outputs_distribution):
    indices = torch.argmax(outputs_distribution, dim=1)
    
    # one-hot
    onehot_scores = torch.zeros_like(outputs_distribution)
    onehot_scores.scatter_(1, indices.unsqueeze(1), 1).to(device)
    
    # one-hot
    score_mapping = config['data']['score_mapping']

    scores = [score_mapping[idx] for idx in indices]
    scores = torch.tensor(scores, dtype=torch.float32).to(device)
    return scores, onehot_scores


def compute_MOS(config, device, outputs_distribution=None):
    score_distribution = outputs_distribution
    score_mapping = config['data']['score_mapping']
    score_mapping = torch.tensor(score_mapping).to(device)
    MOS_scores = (score_distribution @ score_mapping) # [batch_size]
    return MOS_scores


def compute_cross_entropy(predictions, targets_onehot, reduction='sum', epsilon=1e-12):
    log_probs = F.log_softmax(predictions, dim=1)
    per_sample_loss = -(targets_onehot * log_probs).sum(dim=1)
    
    # reduction
    if reduction == 'mean':
        return per_sample_loss.mean()
    elif reduction == 'sum':
        return per_sample_loss.sum()
    else:
        return per_sample_loss

def compute_mse_loss(predictions, targets, reduction='mean'):
    per_sample_loss = (predictions - targets) ** 2
    
    # reduction
    if reduction == 'mean':
        return per_sample_loss.mean()
    elif reduction == 'sum':
        return per_sample_loss.sum()
    else:
        return per_sample_loss

def compute_rank_loss(predictions, targets_onehot, margin=1.0):
    """
    ranking loss
    """
    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    
    true_scores = torch.argmax(targets_onehot, dim=1).float()
    score_weights = torch.arange(num_classes, dtype=torch.float32, device=predictions.device)
    pred_scores = torch.sum(predictions * score_weights, dim=1)
    
    total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
    valid_pairs = 0
    
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if true_scores[i] != true_scores[j]:
                if true_scores[i] > true_scores[j]:
                    violation = torch.clamp(margin - (pred_scores[i] - pred_scores[j]), min=0.0)
                else:
                    violation = torch.clamp(margin - (pred_scores[j] - pred_scores[i]), min=0.0)
                total_loss = total_loss + violation
                valid_pairs += 1
    
    return total_loss / max(valid_pairs, 1)  # ranking loss

def compute_SRCC_PLCC(y_pred, y_gt):
    pred_np = y_pred.squeeze().cpu().numpy()
    gt_np = y_gt.squeeze().cpu().numpy()
    plcc = stats.pearsonr(pred_np, gt_np)[0]
    srcc = stats.spearmanr(pred_np, gt_np)[0]
    return plcc, srcc


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Keep all generated artifacts in one experiment directory when requested.
    # YAML files use ``runs/para`` as the portable default; replacing that
    # prefix here keeps every stage of the pipeline aligned with PRAC_RUN_DIR.
    run_dir = os.getenv("PRAC_RUN_DIR")
    if not run_dir or run_dir.replace('\\', '/') == 'runs/para':
        return config

    default_prefix = 'runs/para'

    def replace_run_dir(value):
        if isinstance(value, dict):
            return {key: replace_run_dir(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_run_dir(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace('\\', '/')
            if normalized == default_prefix:
                return run_dir
            prefix = default_prefix + '/'
            if normalized.startswith(prefix):
                return os.path.join(run_dir, normalized[len(prefix):])
        return value

    return replace_run_dir(config)

def print_peft_config(model):
    """PEFT"""
    print("=== PEFT Configuration ===")
    
    if hasattr(model.LLM, 'peft_config'):
        for adapter_name, config in model.LLM.peft_config.items():
            print(f"Adapter: {adapter_name}")
            print(f"  Task type: {config.task_type}")
            print(f"  LoRA r: {config.r}")
            print(f"  LoRA alpha: {config.lora_alpha}")
            print(f"  LoRA dropout: {config.lora_dropout}")
            print(f"  Target modules: {config.target_modules}")
            print(f"  Bias: {config.bias}")
    else:
        print("No PEFT config found")

def set_lora_dropout_rate(model, new_dropout_rate):
    """
    LoRAdropout
    """
    print(f"Setting LoRA dropout rate to {new_dropout_rate}")
    modified_count = 0
    
    # LoRAdropout
    for name, module in model.named_modules():
        if hasattr(module, 'lora_dropout'):
            if hasattr(module.lora_dropout, 'p'):
                old_rate = module.lora_dropout.p
                module.lora_dropout.p = new_dropout_rate
                print(f"  Modified {name}: {old_rate} -> {new_dropout_rate}")
                modified_count += 1
            else:
                print(f"  Warning: {name} has lora_dropout but no 'p' attribute")
    
    # PEFT()
    if hasattr(model.LLM, 'peft_config'):
        for adapter_name, config in model.LLM.peft_config.items():
            if hasattr(config, 'lora_dropout'):
                old_dropout = config.lora_dropout
                config.lora_dropout = new_dropout_rate
                print(f"  Modified PEFT config {adapter_name}: {old_dropout} -> {new_dropout_rate}")
    
    print(f"Total modules modified: {modified_count}")
    return modified_count > 0

def extract_top_similar_users(csv_file_path, test_user_id, whether_top=True, top_n=10):
    """Return the most similar training users for one test user."""
    similarity_df = pd.read_csv(csv_file_path, index_col=0)

    test_user_key = f"Test_{test_user_id}"

    if test_user_key not in similarity_df.index:
        raise ValueError(f"User '{test_user_key}' is missing from {csv_file_path}")

    user_similarities = similarity_df.loc[test_user_key]

    result_df = pd.DataFrame({
        'train_user_id': [col.replace('Train_', '') for col in user_similarities.index],
        'similarity': user_similarities.values
    })

    result_df = result_df.sort_values('similarity', ascending=False)

    top_similarities = result_df.head(top_n) if whether_top else result_df

    return top_similarities

def find_user_folder(base_dir, user_id):
    """Find a saved user adapter directory under ``base_dir``."""
    pattern = f"*user_{user_id}*"

    matching_folders = glob.glob(os.path.join(base_dir, pattern))

    if matching_folders:
        return matching_folders[0]
    else:
        return None
