import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import yaml
from scipy import stats
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA 

from tqdm import tqdm
from data_loader.data_loader_giaa import get_dataloaders
from utils import print_parameter_statistics, compute_MOS, compute_SRCC_PLCC, load_config

def train(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading MplugOwl3ForGIAA model from {config['model']['model_path']}...")
    
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
    
    print_parameter_statistics(model)
    
    print(f"Preparing dataloaders for {dataset}...")
    loaders = get_dataloaders(dataset, "train", config)
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    test_loader = get_dataloaders(dataset, "test", config)
  
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)

    total_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['training']['warmup_steps'], num_training_steps=total_steps)
    
    if config['training'].get('run_zero_shot_test', True):
        print("Starting Zero-shot Evaluation...")
        model.eval()
        MOS_pred_all = None
        MOS_gt_all = None
        progress_bar = tqdm(test_loader, desc=f"Epoch -1 (Zero-shot) Testing")
        
        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            labels = groups[1].to(device)
            MOS_gt = groups[2].to(device)
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs, _ = model(images, videos=None, labels=None, device=device)
                
            del images
            torch.cuda.empty_cache()
            
            MOS_pred = compute_MOS(config, device, outputs.float())
            MOS_pred_all = MOS_pred if MOS_pred_all is None else torch.cat((MOS_pred_all, MOS_pred))
            MOS_gt_all = MOS_gt if MOS_gt_all is None else torch.cat((MOS_gt_all, MOS_gt))
            
        plcc, srcc = compute_SRCC_PLCC(MOS_pred_all, MOS_gt_all)
        print(f"Zero-shot Test PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")
    
    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(config['training']['epochs']):
        # --- Train ---
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Training")
        
        optimizer.zero_grad()
        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            labels = groups[1].to(device)
            MOS_gt = groups[2].to(device)
            
            torch.cuda.empty_cache()

            outputs, loss = model(images, videos=None, labels=labels, device=device)
            
            if loss is not None:
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
            
            if (batch_idx + 1) % config['training']['logging_steps'] == 0:
                avg_loss = total_loss / config['training']['logging_steps']
                progress_bar.set_postfix({'loss': avg_loss})
                total_loss = 0
            
            if (batch_idx + 1) % config['training']['save_steps'] == 0:
                ckpt_path = os.path.join(config['output']['output_dir'], f'checkpoint-epoch-{epoch+1}-{batch_idx+1}')
                print(f"Saving checkpoint to {ckpt_path}")
                model.save_lora_weights(ckpt_path)

        # --- Validation ---
        model.eval()
        total_loss = 0
        MOS_pred_all = None
        MOS_gt_all = None
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Validating")
        
        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            labels = groups[1].to(device)
            MOS_gt = groups[2].to(device)
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs, loss = model(images, videos=None, labels=labels, device=device)
                
            del images
            torch.cuda.empty_cache()
            
            total_loss += loss.item()
            
            #  .float()
            MOS_pred = compute_MOS(config, device, outputs.float())
            MOS_pred_all = MOS_pred if MOS_pred_all is None else torch.cat((MOS_pred_all, MOS_pred))
            MOS_gt_all = MOS_gt if MOS_gt_all is None else torch.cat((MOS_gt_all, MOS_gt))
            
        avg_loss = total_loss / len(val_loader)
        plcc, srcc = compute_SRCC_PLCC(MOS_pred_all, MOS_gt_all)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} Val Loss: {avg_loss:.4f}, PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")
        
        # --- Test ---
        model.eval()
        MOS_pred_all = None
        MOS_gt_all = None
        progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Testing")
        
        for batch_idx, groups in enumerate(progress_bar):
            images = groups[0]
            labels = groups[1].to(device)
            MOS_gt = groups[2].to(device)
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs, _ = model(images, videos=None, labels=None, device=device)
                
            del images
            torch.cuda.empty_cache()
            
            #  .float()
            MOS_pred = compute_MOS(config, device, outputs.float())
            MOS_pred_all = MOS_pred if MOS_pred_all is None else torch.cat((MOS_pred_all, MOS_pred))
            MOS_gt_all = MOS_gt if MOS_gt_all is None else torch.cat((MOS_gt_all, MOS_gt))
            
        plcc, srcc = compute_SRCC_PLCC(MOS_pred_all, MOS_gt_all)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} Test PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")
        
        save_path = os.path.join(config['output']['output_dir'], f'checkpoint-epoch-{epoch+1}-Test-PLCC-{plcc:.4f}-SRCC-{srcc:.4f}')
        print(f"Saving epoch model to {save_path}")
        model.save_lora_weights(save_path)

if __name__ == "__main__":
    # Choose the configuration file based on the dataset you want to train.
    dataset = os.getenv("GIAA_DATASET", "PARA")
    config_path = os.getenv("GIAA_CONFIG_PATH")

    default_config_paths = {
        "PARA": "configs/giaa_train_config/para.yaml",
    }

    if config_path is None:
        if dataset not in default_config_paths:
            raise ValueError("This release supports the PARA dataset only.")
        config_path = default_config_paths[dataset]
    
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    train(config, dataset)

