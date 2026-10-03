import os
import yaml
import json
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _resolve_image_path(img_root_dir, image_id):
    if os.path.isabs(image_id):
        return image_id
    return os.path.join(img_root_dir, image_id)


def _get_dataset_name(config):
    return config['dataset'].get('dataset_name', 'PARA')


def _get_user_id_column(config):
    dataset_name = _get_dataset_name(config)
    default_column = 'userId' if dataset_name == 'PARA' else 'user_name'
    return config['dataset'].get('user_id_column', default_column)


def _get_image_id_column(config):
    dataset_name = _get_dataset_name(config)
    default_column = 'imageName' if dataset_name == 'PARA' else 'image_id'
    return config['dataset'].get('image_id_column', default_column)


def _uses_shared_image_pool(config):
    return config['dataset'].get('use_all_images_for_each_user', False)

class PersonalizedAestheticDataset(Dataset):
    def __init__(self, df, config):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing image info and scores
            config (dict): Configuration dictionary
        """
        self.df = df
        self.img_root_dir = config['dataset']['img_root_dir']
        self.image_size = config['dataset']['image_size']
        self.dataset_name = _get_dataset_name(config)
        self.image_id_column = _get_image_id_column(config)
        self.score_column = config['dataset'].get(
            'score_column',
            'aestheticScore' if self.dataset_name == 'PARA' else None
        )
        self.score_dim = config['dataset'].get(
            'score_dim',
            9 if self.dataset_name == 'PARA' else 10
        )
        if self.dataset_name == 'PARA':
            self.score_mapping = {
                1.0: 0, 1.5: 1, 2.0: 2, 2.5: 3,
                3.0: 4, 3.5: 5, 4.0: 6, 4.5: 7, 5.0: 8
            }
        else:
            self.score_mapping = {float(i): i - 1 for i in range(1, self.score_dim + 1)}
    
    def _score_to_onehot(self, score):
        """
        Args:
            score (float): init score
        Returns:
            torch.Tensor: one-hot score
        """
        if score is None or pd.isna(score):
            return torch.zeros(self.score_dim), -1
        score_idx = self.score_mapping[score]
        one_hot = torch.zeros(self.score_dim)
        one_hot[score_idx] = 1
        return one_hot, score_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.dataset_name == 'PARA':
            img_path = os.path.join(self.img_root_dir, row['sessionId'], row['imageName'])
        else:
            image_id = str(row[self.image_id_column])
            img_path = _resolve_image_path(self.img_root_dir, image_id)
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.image_size is not None:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        score_value = None
        if self.score_column and self.score_column in row and not pd.isna(row[self.score_column]):
            try:
                score_value = float(row[self.score_column])
            except Exception:
                score_value = None

        score_onehot, _ = self._score_to_onehot(score_value)
        score = torch.tensor(score_value if score_value is not None else -1.0, dtype=torch.float32)

        return image, score_onehot, score

class PersonalizedAestheticDataset_(Dataset):
    def __init__(self, df, config):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing image info and scores
            config (dict): Configuration dictionary
        """
        self.df = df
        self.img_root_dir = config['dataset']['img_root_dir']
        self.image_size = config['dataset']['image_size']
        self.dataset_name = _get_dataset_name(config)
        self.image_id_column = _get_image_id_column(config)
        self.score_column = config['dataset'].get(
            'score_column',
            'aestheticScore' if self.dataset_name == 'PARA' else None
        )
        self.score_dim = config['dataset'].get(
            'score_dim',
            9 if self.dataset_name == 'PARA' else 10
        )
        if self.dataset_name == 'PARA':
            self.score_mapping = {
                1.0: 0, 1.5: 1, 2.0: 2, 2.5: 3,
                3.0: 4, 3.5: 5, 4.0: 6, 4.5: 7, 5.0: 8
            }
        else:
            self.score_mapping = {float(i): i - 1 for i in range(1, self.score_dim + 1)}
    
    def _score_to_onehot(self, score):
        """
        Args:
            score (float): init score
        Returns:
            torch.Tensor: one-hot score
        """
        if score is None or pd.isna(score):
            return torch.zeros(self.score_dim), -1
        score_idx = self.score_mapping[score]
        one_hot = torch.zeros(self.score_dim)
        one_hot[score_idx] = 1
        return one_hot, score_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.dataset_name == 'PARA':
            img_path = os.path.join(self.img_root_dir, row['sessionId'], row['imageName'])
            img_id = row['sessionId'] + "-" + row['imageName']
        else:
            img_id = str(row[self.image_id_column])
            img_path = _resolve_image_path(self.img_root_dir, img_id)
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.image_size is not None:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        score_value = None
        if self.score_column and self.score_column in row and not pd.isna(row[self.score_column]):
            try:
                score_value = float(row[self.score_column])
            except Exception:
                score_value = None

        score_onehot, _ = self._score_to_onehot(score_value)
        score = torch.tensor(score_value if score_value is not None else -1.0, dtype=torch.float32)

        return image, score_onehot, score, img_id
    
def custom_collate(batch):
    images = [item[0] for item in batch]
    scores_onehot = torch.stack([item[1] for item in batch])
    scores = torch.stack([item[2] for item in batch])
    return images, scores_onehot, scores

def custom_collate_(batch):
    images = [item[0] for item in batch]
    scores_onehot = torch.stack([item[1] for item in batch])
    scores = torch.stack([item[2] for item in batch])
    images_id = [item[3] for item in batch]
    return images, scores_onehot, scores, images_id

def get_user_prompts(path):
    """
    
    """
    with open(path, 'r') as f:
        print("Loading user-info prompts from:", path)
        user_prompts = json.load(f)
        for user_id, prompt in user_prompts.items():
            user_prompts[user_id] = prompt.replace('\n', ' ').strip()
    return user_prompts

def get_all_user_ids(config):
    """
    ID
    """
    try:
        if _uses_shared_image_pool(config):
            print("\nShared image pool mode enabled. User ids will be loaded from user-info JSON.")
            return [], {}

        df = pd.read_csv(config['dataset']['csv_path'])
        user_id_column = _get_user_id_column(config)
        if user_id_column not in df.columns:
            raise ValueError(f"Column '{user_id_column}' not found in {config['dataset']['csv_path']}")

        user_ids = df[user_id_column].astype(str).unique().tolist()
        user_image_counts = df[user_id_column].astype(str).value_counts().to_dict()
        
        print(f"\nTotal number of unique users: {len(user_ids)}")
        print("\nUser statistics:")
        print(f"    Maximum images per user: {max(user_image_counts.values())}")
        print(f"    Minimum images per user: {min(user_image_counts.values())}")
        print(f"    Average images per user: {sum(user_image_counts.values()) / len(user_ids):.2f}")
        
        return user_ids, user_image_counts
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], {}

def get_user_dataloaders(config, user_id, random_seed=None):
    """
    
    """
    df = pd.read_csv(config['dataset']['csv_path'])
    shared_image_pool = _uses_shared_image_pool(config)

    if shared_image_pool:
        user_df = df.reset_index(drop=True)
    else:
        user_id_column = _get_user_id_column(config)
        if user_id_column not in df.columns:
            raise ValueError(f"Column '{user_id_column}' not found in {config['dataset']['csv_path']}")
        user_df = df[df[user_id_column].astype(str) == str(user_id)].reset_index(drop=True)

    random_state = config['dataloader']['random_seed'] if random_seed is None else random_seed

    train_loader = None
    test_loader = None

    if not shared_image_pool:
        train_df, test_df = train_test_split(
            user_df,
            train_size=config['dataloader']['train_size'][config['dataset']['piaa_size']],
            test_size=config['dataloader']['test_size'],
            random_state=random_state,
            shuffle=True
        )

        train_dataset = PersonalizedAestheticDataset(train_df, config)
        test_dataset = PersonalizedAestheticDataset(test_df, config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['dataloader']['batch_size'],
            shuffle=True,
            num_workers=config['dataloader']['num_workers'],
            pin_memory=config['dataloader']['pin_memory'],
            collate_fn=custom_collate
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['dataloader']['batch_size'],
            shuffle=False,
            num_workers=config['dataloader']['num_workers'],
            pin_memory=config['dataloader']['pin_memory'],
            collate_fn=custom_collate
        )

    user_dataset = PersonalizedAestheticDataset_(user_df, config)
    user_loader = DataLoader(
        user_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        collate_fn=custom_collate_
    )

    if shared_image_pool:
        user_loader._shared_image_pool = True
    
    return train_loader, test_loader, user_loader

def get_qualified_users_dataloaders_user_info(config, random_seed=None, get_all_image_of_user=False, get_all_users=False):
    """
    ,user_iduser_prompt
    """
    print(f"\nLoad Datasets for {config['dataset']['piaa_size']} PIAA")
    
    user_prompts = get_user_prompts(config['dataset']['user_info_path'])
    shared_image_pool = _uses_shared_image_pool(config)
    user_ids, user_image_counts = get_all_user_ids(config)
    
    if shared_image_pool:
        qualified_users = sorted(user_prompts.keys())
    else:
        try:
            if config['dataset']['piaa_size'] == "10-shot":
                qualified_users = [
                    user_id for user_id, count in user_image_counts.items() 
                    if count >= config['dataloader']['min_images']['10-shot']
                ]
            elif config['dataset']['piaa_size'] == "100-shot":
                qualified_users = [
                    user_id for user_id, count in user_image_counts.items() 
                    if count >= config['dataloader']['min_images']['100-shot']
                ]
            else:
                raise ValueError(f"Unsupported piaa_size: {config['dataset']['piaa_size']}")
        except Exception as e:
            print(f"Error getting piaa size: {e}")
            return {}
    
    if get_all_users:
        qualified_users = [user_id for user_id in qualified_users if str(user_id) in user_prompts]
        print(f"\nTotal number of qualified users (having user prompts): {len(qualified_users)}")
        
        user_dataloaders = {}
        shared_user_loader = None
        if shared_image_pool:
            _, _, shared_user_loader = get_user_dataloaders(config, None, random_seed)
        for user_id in tqdm(qualified_users):
            try:
                user_prompt = user_prompts[str(user_id)]
                if shared_image_pool:
                    train_loader, test_loader, user_loader = None, None, shared_user_loader
                else:
                    train_loader, test_loader, user_loader = get_user_dataloaders(config, user_id, random_seed)
                if get_all_image_of_user:
                    user_dataloaders[user_id] = (user_prompt, user_loader)
                else:
                    if train_loader is None or test_loader is None:
                        raise ValueError("Train/test dataloaders are unavailable in shared image pool mode.")
                    user_dataloaders[user_id] = (user_prompt, train_loader, test_loader)
            except Exception as e:
                print(f"Error creating dataloaders for user {user_id}: {e}")
                continue
        return user_dataloaders
    
    else: 
        random_seed_for_selection = config['dataloader']['random_seed_for_selection']
        
        # user_prompt,user_size,user_size,user_size
        print(f"\nRandomly choose {config['dataset']['user_size']} users for {config['dataset']['piaa_size']} PIAA")
        random.seed(random_seed_for_selection)
        candidate_size = min(len(qualified_users), config['dataset']['user_size'] + 10)
        selected_qualified_users = random.sample(qualified_users, candidate_size)
        
        # user_prompt
        user_dataloaders = {}
        shared_user_loader = None
        if shared_image_pool:
            _, _, shared_user_loader = get_user_dataloaders(config, None, random_seed)
        for user_id in tqdm(selected_qualified_users):
            try:
                user_prompt = user_prompts[str(user_id)]
                if shared_image_pool:
                    train_loader, test_loader, user_loader = None, None, shared_user_loader
                else:
                    train_loader, test_loader, user_loader = get_user_dataloaders(config, user_id, random_seed)
                if get_all_image_of_user:
                    user_dataloaders[user_id] = (user_prompt, user_loader)
                else:
                    if train_loader is None or test_loader is None:
                        raise ValueError("Train/test dataloaders are unavailable in shared image pool mode.")
                    user_dataloaders[user_id] = (user_prompt, train_loader, test_loader)
                if len(user_dataloaders) == config['dataset']['user_size']:
                    print("\nThe number of selected users has reached the specified value")
                    break
            except Exception as e:
                print(f"Error creating dataloaders for user {user_id}: {e}")
                continue
        
        print(f"\nCreated dataloaders for randomly choosed {len(user_dataloaders)} qualified users")
        return user_dataloaders

