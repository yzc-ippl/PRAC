import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys
import random # 

# --- 1.  Qwen  ---
from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA
from utils import print_parameter_statistics, load_config, print_peft_config
from data_loader.data_loader_piaa import get_user_dataloaders_using_image_select

# --- 2. Monkey Patch ( torch.compiler ) ---
try:
    _ = torch.compiler.is_compiling
except AttributeError:
    if hasattr(torch, "compiler"):
        torch.compiler.is_compiling = lambda: False
    else:
        class MockCompiler:
            def is_compiling(self): return False
        torch.compiler = MockCompiler()

# --- : ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #  CUDA  (,)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

def find_user_folder(base_dir, user_id):
    pattern = f"*user_{user_id}*"
    matching_folders = glob.glob(os.path.join(base_dir, pattern))
    if matching_folders:
        return matching_folders[0]
    else:
        return None

def compute_fisher_information_matrix(shared_model, user_dataloader, batch_size=4, device='cuda'):
    """
     Fisher  (FIM) .
    """
    shared_model.eval() #  eval  ( Dropout )
    shared_model.to(device)

    params = [p for p in shared_model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    fim_diag = torch.zeros(total_params, device=device)

    sample_count = 0

    rating_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    rating_to_idx = {rating: idx for idx, rating in enumerate(rating_levels)}

    #  tqdm ,
    progress_bar = tqdm(user_dataloader, desc='Calculating FIM', leave=False)

    for batch_idx, groups in enumerate(progress_bar):
        images = groups[0]
        labels = groups[2].to(device) # labels: [batch_size]
        
        actual_batch_size = len(images)
        sample_count += actual_batch_size

        shared_model.zero_grad()
        batch_grads = []

        #  (FIM )
        with torch.enable_grad():
            log_probs, _ = shared_model(images, labels=None, device=device)
            #  model  log_probs  logits,
            #  output  logits,: 
            # log_probs = torch.nn.functional.log_softmax(log_probs.float(), dim=-1)
            
            for i in range(actual_batch_size):
                rating_value = labels[i].item()
                
                #  rating 
                if rating_value in rating_to_idx:
                    label_idx = rating_to_idx[rating_value]
                else:
                    closest_rating = min(rating_levels, key=lambda x: abs(x - rating_value))
                    label_idx = rating_to_idx[closest_rating]

                #  log_prob
                sample_log_prob = log_probs[i, label_idx].clamp_min(1e-12).log()

                # : d(log_prob) / d(params)
                grad_params = torch.autograd.grad(
                    sample_log_prob, params,
                    retain_graph=(i < actual_batch_size - 1), 
                    create_graph=False,
                    allow_unused=True
                )

                grad_params = [
                    torch.zeros_like(p) if g is None else g
                    for g, p in zip(grad_params, params)
                ]

                flat_grad = torch.cat([g.flatten() for g in grad_params])
                batch_grads.append(flat_grad)
        
        del images, labels, log_probs

        for grad in batch_grads:
            fim_diag += grad.pow(2).detach()
        
        del batch_grads

    if sample_count > 0:
        fim_diag /= sample_count
    
    return fim_diag

def compute_cosine_similarity(fim1, fim2):
    # fim1, fim2  GPU  CPU ,
    return torch.nn.functional.cosine_similarity(fim1.unsqueeze(0), fim2.unsqueeze(0)).item()

def compute_all_user_similarities(shared_model, train_user_dataloaders, test_user_dataloaders, batch_size=4, device='cuda'):
    train_user_ids = list(train_user_dataloaders.keys())
    test_user_ids = list(test_user_dataloaders.keys())

    # --- :FIM  ---
    # 1.  User ID ( Train  Test,)
    all_unique_users = set(train_user_ids) | set(test_user_ids)
    
    fim_cache = {}
    print(f"Total unique users to compute: {len(all_unique_users)}")

    # 2.  FIM ()
    for user_id in tqdm(all_unique_users, desc="Computing FIMs (Unified)"):
        #  train  dataloader, test 
        dataloader = train_user_dataloaders.get(user_id)
        if dataloader is None:
            dataloader = test_user_dataloaders.get(user_id)
        
        #  FIM
        fim = compute_fisher_information_matrix(shared_model, dataloader, batch_size, device)
        
        #  ( CPU )
        fim_cache[user_id] = fim.cpu()
        
        del fim
        torch.cuda.empty_cache()

    similarity_matrix = np.zeros((len(test_user_ids), len(train_user_ids)))

    print("Calculating Similarity Matrix from Cache...")
    #  CPU ,
    for i, test_uid in enumerate(tqdm(test_user_ids, desc="Sim Matrix Rows")):
        #  GPU  ( fim ,CPU)
        test_fim = fim_cache[test_uid].to(device)

        for j, train_uid in enumerate(train_user_ids):
            train_fim = fim_cache[train_uid].to(device)
            
            sim_score = compute_cosine_similarity(test_fim, train_fim)
            similarity_matrix[i, j] = sim_score
            
            #  del train_fim,,
            
        del test_fim # 
    
    torch.cuda.empty_cache()

    #  API , train_fims  test_fims 
    # ( FIM,)
    final_train_fims = {uid: fim_cache[uid] for uid in train_user_ids}
    final_test_fims = {uid: fim_cache[uid] for uid in test_user_ids}

    return similarity_matrix, final_train_fims, final_test_fims, train_user_ids, test_user_ids


def find_most_similar_users(similarity_matrix, train_user_ids, test_user_ids, top_k=5):
    most_similar_dict = {}

    for i, test_user_id in enumerate(test_user_ids):
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = [(train_user_ids[idx], similarities[idx]) for idx in top_indices]
        most_similar_dict[test_user_id] = top_similarities

    return most_similar_dict

def export_similarity_to_csv(
    similarity_matrix,
    train_user_ids,
    test_user_ids,
    output_file=None,
):
    if output_file is None:
        output_file = os.path.join(
            os.getenv("PRAC_RUN_DIR", "runs/para"),
            "fim",
            "user_similarities_fixed.csv",
        )
    df = pd.DataFrame(
        similarity_matrix,
        index=[f"Test_{uid}" for uid in test_user_ids],
        columns=[f"Train_{uid}" for uid in train_user_ids]
    )
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    df.to_csv(output_file)
    print(f"Similarity matrix saved to {output_file}")

def main(shared_model, train_user_dataloaders, test_user_dataloaders, batch_size=4, device='cuda'):
    similarity_matrix, train_fims, test_fims, train_user_ids, test_user_ids = compute_all_user_similarities(
        shared_model, train_user_dataloaders, test_user_dataloaders, batch_size, device
    )

    most_similar_dict = find_most_similar_users(similarity_matrix, train_user_ids, test_user_ids, top_k=5)
    export_similarity_to_csv(similarity_matrix, train_user_ids, test_user_ids)

    return similarity_matrix, most_similar_dict, train_fims, test_fims

if __name__ == "__main__":
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_config_for_lora_pool = load_config('configs/prefermerge_config/para_lora_pool.yaml')
    config = load_config('configs/prefermerge_config/para.yaml')
    
    run_dir = os.getenv("PRAC_RUN_DIR", "runs/para")
    pool_csv = os.getenv(
        "PRAC_LORA_POOL_INDEX",
        os.path.join(run_dir, "lora_pool", "LoRAPool", "LoRAPool_index.csv"),
    )
    test_csv = os.getenv(
        "PRAC_TEST_USERS_INDEX",
        os.path.join(run_dir, "lora_pool", "TestUsersPool", "TestUsersPool_index.csv"),
    )
    
    train_user_pool = []
    if os.path.exists(pool_csv):
        train_user_pool = list(pd.read_csv(pool_csv)['User'])
        print(f"Train pool size: {len(train_user_pool)}")
    else:
        print("Warning: Train pool CSV not found.")
        
    test_user = []
    if os.path.exists(test_csv):
        test_user = list(pd.read_csv(test_csv)['User'])
        print(f"Test pool size: {len(test_user)}")
    else:
        print("Warning: Test user CSV not found.")

    print("Loading MplugOwl3ForGIAA...")
    # LoRA  seed_everything 
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
        print("Re-initializing LoRA weights (controlled by seed)...")
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
    # print_peft_config(giaa_model)
    
    #  DataLoader
    train_user_dataloaders = {}
    test_user_dataloaders = {}
    
    #  Users
    print("Loading User DataLoaders...")
    # , ()
    all_users_to_load = list(set(train_user_pool + test_user))
    
    for user in tqdm(all_users_to_load, desc="Loading Loaders"):
        #  shuffle  seed 
        train_loader, _ = get_user_dataloaders_using_image_select(
            data_config_for_lora_pool, 'PARA', user, 41
        )
        
        train_user_dataloaders[user] = train_loader
        test_user_dataloaders[user] = train_loader

    print("Start Calculating Similarities...")
    similarity_matrix, most_similar_dict, train_fims, test_fims = main(
        giaa_model, train_user_dataloaders, test_user_dataloaders
    )
    
    print("Done!")

