import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split

class AestheticDataset(Dataset):
    def __init__(self, csv_path, img_root_dir, dataset, image_size=None, std=False):
        """
        Args:
            csv_path (str): CSV
            img_root_dir (str): 
        """
        self.data = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir
        self.dataset = dataset
        self.image_size = image_size
        self.std = std
        
        self.score_cols = [col for col in self.data.columns if col.startswith('aesScore_') 
                          and col != 'aesScore_mean' and col != 'aesScore_std']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset == 'PARA':
            if torch.is_tensor(idx):
                idx = idx.tolist()
                
            session_id = self.data.iloc[idx]['sessionId']
            image_name = self.data.iloc[idx]['imageName']
            image_id = str(session_id) + '-' + image_name
            img_path = os.path.join(self.img_root_dir, str(session_id), image_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                if self.image_size is not None:
                    image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None
                
            label_values = self.data.iloc[idx][self.score_cols].astype(np.float32).values
            label = torch.from_numpy(label_values)
            
            aesScore_mean = torch.tensor(self.data.iloc[idx]['aesScore_mean'], dtype=torch.float32)
            
            if self.std:
                return image, label, aesScore_mean, image_id
            else:
                return image, label, aesScore_mean
        
        else:
            raise ValueError(f"Unsupported GIAA dataset: {self.dataset}")

def custom_collate(batch):
    images = [item[0] for item in batch]
    dist_labels = torch.stack([item[1] for item in batch])
    mean_labels = torch.stack([item[2] for item in batch])
    return images, dist_labels, mean_labels

def custom_collate_std(batch):
    images = [item[0] for item in batch]
    dist_labels = torch.stack([item[1] for item in batch])
    mean_labels = torch.stack([item[2] for item in batch])
    image_ids = [item[3] for item in batch]
    return images, dist_labels, mean_labels, image_ids

def create_dataloaders(train_csv, test_csv, img_root_dir, image_size=384, batch_size=32, val_ratio=0.1, num_workers=4, random_seed=42):
    """
    ,DataLoader
    
    Args:
        train_csv (str): CSV
        test_csv (str): CSV
        img_root_dir (str): 
        batch_size (int): batch
        val_ratio (float): 
        num_workers (int): 
        random_seed (int): 
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = AestheticDataset(
        train_csv, img_root_dir, dataset="PARA", image_size=image_size
    )
    test_dataset = AestheticDataset(
        test_csv, img_root_dir, dataset="PARA", image_size=image_size
    )
    
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=random_seed
    )
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader, test_loader

def print_dataloader_info(train_loader, val_loader, test_loader):
    """
    Print basic information about dataloaders
    """
    # Get a batch sample to check data shape
    images, dist_labels, mean_labels = next(iter(train_loader))
    
    print("\nDataloader initialized:")
    print(f"    Training set: {len(train_loader.dataset)} samples")
    print(f"    Validation set: {len(val_loader.dataset)} samples") 
    print(f"    Test set: {len(test_loader.dataset)} samples")
    print(f"    Batch size: {len(images)}")
    print(f"    Image size: {images[0].size}\n")

def _split_single_dataset(total_dataset, config):
    dataset_size = len(total_dataset)
    test_size = int(dataset_size * config['data']['test_ratio'])
    val_size = int(dataset_size * config['data']['val_ratio'])
    train_size = dataset_size - val_size - test_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"Invalid split sizes: train={train_size}, val={val_size}, test={test_size}. "
            "Please adjust val_ratio/test_ratio."
        )

    return torch.utils.data.random_split(
        total_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['data']['random_seed'])
    )

def get_dataloaders(dataset, mode, config):
    """
    
    
    Args:
        config: 
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    image_size = config['data']['image_size']
    
    if dataset == 'PARA':
        train_dataset = AestheticDataset(
            csv_path=config['data']['giaa_train_data_dir']['PARA_annotation'],
            img_root_dir=config['data']['giaa_train_data_dir']['PARA_image'],
            dataset=dataset,
            image_size=image_size
        )
        
        test_dataset = AestheticDataset(
            csv_path=config['data']['giaa_eval_data_dir']['PARA_annotation'],
            img_root_dir=config['data']['giaa_eval_data_dir']['PARA_image'], 
            dataset=dataset,
            image_size=image_size
        )
        
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * config['data']['val_ratio'])
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config['data']['random_seed'])
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=custom_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=custom_collate
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=custom_collate
        )
        
        print_dataloader_info(train_loader, val_loader, test_loader)
        
        try:
            if mode == 'train':
                return train_loader, val_loader
            elif mode == 'test':    
                return test_loader
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    else:
        raise ValueError(f"Unsupported GIAA dataset: {dataset}")
        
def get_dataloaders_std(dataset, mode, config):
    image_size = config['data']['image_size']
    
    if dataset == 'PARA':
        train_dataset = AestheticDataset(
            csv_path=config['data']['giaa_train_data_dir']['PARA_annotation'],
            img_root_dir=config['data']['giaa_train_data_dir']['PARA_image'],
            dataset=dataset,
            image_size=image_size,
            std=True,
        )
        
        test_dataset = AestheticDataset(
            csv_path=config['data']['giaa_eval_data_dir']['PARA_annotation'],
            img_root_dir=config['data']['giaa_eval_data_dir']['PARA_image'], 
            dataset=dataset,
            image_size=image_size,
            std=True,
        )
        
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * config['data']['val_ratio'])
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config['data']['random_seed'])
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=custom_collate_std
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=custom_collate_std
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=custom_collate_std
        )
        
        try:
            if mode == 'train':
                return train_loader, val_loader
            elif mode == 'test':    
                return test_loader
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    else:
        raise ValueError(f"Unsupported GIAA dataset: {dataset}")


if __name__ == "__main__":
    train_csv = "data/PARA/para_giaa_train_discretized.csv"
    test_csv = "data/PARA/para_giaa_test_discretized.csv"
    img_root_dir = "data/PARA/imgs"
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        test_csv=test_csv,
        img_root_dir=img_root_dir,
        batch_size=32,
        val_ratio=0.1
    )
    
    print("Testing train loader:")
    for images, labels, aesScore_mean in train_loader:
        print(f"Batch size: {len(images)}")
        print(f"Image type: {type(images[0])}")
        print(f"Image size: {images[0].size}")
        print(f"Labels shape: {labels.shape}")
        print(f"Aesthetic score mean: {aesScore_mean.shape}")
        break

