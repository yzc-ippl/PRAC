import os
#  tokenizers 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from tqdm import tqdm

from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA 

from data_loader.data_loader_giaa import get_dataloaders
from utils import compute_MOS, compute_SRCC_PLCC, load_config, print_parameter_statistics

try:
    _ = torch.compiler.is_compiling
except AttributeError:
    if hasattr(torch, "compiler"):
        torch.compiler.is_compiling = lambda: False
    else:
        class MockCompiler:
            def is_compiling(self): return False
        torch.compiler = MockCompiler()

def eval_model(config, dataset_name="PARA"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[{dataset_name}] Starting Evaluation...")
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
    ).to(device)
    
    #  Adapter 
    if config['model'].get('use_lora_adapter', False):
        adapter_path = config['model']['adapter_path']
        print(f"Loading LoRA adapter from {adapter_path}")
        model.load_lora_weights(adapter_path) 
        
    print_parameter_statistics(model)
    
    testloader = get_dataloaders(dataset_name, "test", config)

    model.eval()
        
    MOS_pred_all = None
    MOS_gt_all = None
    
    progress_bar = tqdm(testloader, desc=f"Testing on {dataset_name}")
    
    for batch_idx, groups in enumerate(progress_bar):
        images = groups[0]
        labels = groups[1].to(device)
        MOS_gt = groups[2].to(device)
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            outputs, loss = model(images, videos=None, labels=labels, device=device)
            
        del images
        torch.cuda.empty_cache()
        
        MOS_pred = compute_MOS(config, device, outputs.float())
        
        MOS_pred_all = MOS_pred if MOS_pred_all is None else torch.cat((MOS_pred_all, MOS_pred))
        MOS_gt_all = MOS_gt if MOS_gt_all is None else torch.cat((MOS_gt_all, MOS_gt))
    
    plcc, srcc = compute_SRCC_PLCC(MOS_pred_all, MOS_gt_all)
    print("="*40)
    print(f"Dataset: {dataset_name}")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print("="*40)

if __name__ == "__main__":
    dataset_name = os.getenv("GIAA_DATASET", "PARA").upper()
    config_path = os.getenv("GIAA_CONFIG_PATH")

    default_config_paths = {
        "PARA": "configs/giaa_eval_config/para.yaml",
    }

    if config_path is None:
        if dataset_name not in default_config_paths:
            raise ValueError("This release supports the PARA dataset only.")
        config_path = default_config_paths[dataset_name]

    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    eval_model(config, dataset_name)

