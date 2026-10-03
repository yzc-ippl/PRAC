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


class PersonalizedAestheticDataset(Dataset):
    def __init__(self, df, config, dataset="PARA"):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing image info and scores
            config (dict): Configuration dictionary
        """
        self.df = df
        self.img_root_dir = config['dataset']['img_root_dir']
        self.image_size = config['dataset']['image_size']
        self.dataset = dataset
        if self.dataset != "PARA":
            raise ValueError("This release supports the PARA dataset only.")
        self.score_mapping = {
            1.0: 0, 1.5: 1, 2.0: 2, 2.5: 3,
            3.0: 4, 3.5: 5, 4.0: 6, 4.5: 7, 5.0: 8
        }
        self.num_classes = 9

    def _score_to_onehot(self, score):
        """
        Args:
            score (float): init score
        Returns:
            torch.Tensor: one-hot score
        """
        score_idx = self.score_mapping[score]
        one_hot = torch.zeros(self.num_classes)
        one_hot[score_idx] = 1
        return one_hot, score_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.dataset == "PARA":
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_root_dir, row['sessionId'], row['imageName'])

            try:
                image = Image.open(img_path).convert('RGB')

                if self.image_size is not None:
                    image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None

            score_onehot, _ = self._score_to_onehot(row['aestheticScore'])
            score = torch.tensor(row['aestheticScore'], dtype=torch.float32)

            return image, score_onehot, score

        else:
            raise ValueError("This release supports the PARA dataset only.")


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    images = [item[0] for item in batch]
    scores_onehot = torch.stack([item[1] for item in batch])
    scores = torch.stack([item[2] for item in batch])
    return images, scores_onehot, scores


def get_all_user_ids(config):
    """
    Get all unique user IDs from the dataset.
    """
    try:
        df = pd.read_csv(config['dataset']['csv_path'])
        user_ids = df['userId'].unique().tolist()
        user_image_counts = df['userId'].value_counts().to_dict()

        print(f"\nTotal number of unique users: {len(user_ids)}")
        print("\nUser statistics:")
        print(f"    Maximum images per user: {max(user_image_counts.values())}")
        print(f"    Minimum images per user: {min(user_image_counts.values())}")
        print(f"    Average images per user: {sum(user_image_counts.values()) / len(user_ids):.2f}")

        return user_ids, user_image_counts

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], {}


def get_user_prompts(path="data/user_info_para.json"):
    """Load user prompts from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        user_prompts = json.load(f)
        for user_id, prompt in user_prompts.items():
            user_prompts[user_id] = prompt.replace('\n', ' ').strip()
    return user_prompts


def get_user_dataloaders(config, user_id, random_seed=None):
    """
    Create train/test dataloaders for a specific user.
    """
    df = pd.read_csv(config['dataset']['csv_path'])
    user_df = df[df['userId'] == user_id].reset_index(drop=True)

    random_state = config['dataloader']['random_seed'] if random_seed is None else random_seed

    train_df, test_df = train_test_split(
        user_df,
        train_size=config['dataloader']['train_size'][config['dataset']['piaa_size']],
        test_size=config['dataloader']['test_size'],
        random_state=random_state,
        shuffle=True
    )

    dataset_name = config['dataset'].get('dataset_name', 'PARA')
    train_dataset = PersonalizedAestheticDataset(train_df, config, dataset_name)
    test_dataset = PersonalizedAestheticDataset(test_df, config, dataset_name)

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
    test_loader.is_shared_test_set = False
    test_loader.shared_test_set_size = None

    return train_loader, test_loader


def _load_user_dataframe(config, dataset, user_id):
    if dataset != "PARA":
        raise ValueError("This release supports the PARA dataset only.")
    df = pd.read_csv(config['dataset']['csv_path'])
    user_df = df[df['userId'] == user_id].reset_index(drop=True)
    user_df['image_key'] = user_df['sessionId'] + "-" + user_df['imageName']
    return user_df


def _load_user_image_select_dataframe(config, dataset, user_id):
    if dataset != "PARA":
        raise ValueError("This release supports the PARA dataset only.")
    user_image_select_path = config["image_select"]['user_path']
    image_num = None
    for file in os.listdir(user_image_select_path):
        if not file.startswith("user_") or not file.endswith("_imgs"):
            continue
        body = file[len("user_"):-len("_imgs")]
        if "_" not in body:
            continue
        parsed_user_id, parsed_image_num = body.rsplit("_", 1)
        if parsed_user_id == str(user_id):
            user_image_select_path = os.path.join(user_image_select_path, file)
            image_num = int(parsed_image_num)
            break

    if image_num is None:
        raise FileNotFoundError(f"Cannot find image select folder for user {user_id} under {config['image_select']['user_path']}")

    user_image_select_df = pd.read_csv(
        user_image_select_path + "/" + f"fix_weight_results_user_{user_id}_{image_num}_imgs.csv"
    )
    user_image_select_df['image_key'] = user_image_select_df['session_id'] + "-" + user_image_select_df['image_id']
    return user_image_select_df

def _build_shared_test_image_keys_from_selected_users(
    config,
    dataset,
    selected_user_ids,
    random_seed=None,
    testset_reference_user=None,
):
    if dataset != "PARA":
        raise ValueError("This release supports the PARA dataset only.")

    if not selected_user_ids or len(selected_user_ids) < 2:
        print("Warning: shared test set requires at least 2 enabled users. Fallback to per-user test sets")
        return None

    test_size = int(config['dataloader']['test_size'])
    train_shot_count = int(config['dataloader']['train_size'][config['dataset']['piaa_size']])
    select_image_from = config['dataset'].get('select_image_from', 'top')
    random_state = config['dataloader']['random_seed'] if random_seed is None else random_seed

    common_image_keys = None
    for user_id in selected_user_ids:
        user_df = _load_user_dataframe(config, dataset, user_id)
        user_df['image_key'] = user_df['image_key'].astype(str)
        user_image_keys = set(user_df['image_key'].tolist())
        if common_image_keys is None:
            common_image_keys = user_image_keys
        else:
            common_image_keys &= user_image_keys

        if not common_image_keys:
            print(
                "Warning: enabled_users do not have any shared images. "
                "Fallback to per-user test sets"
            )
            return None

    if len(common_image_keys) < test_size:
        print(
            f"Warning: enabled_users only share {len(common_image_keys)} images, "
            f"which is smaller than test_size={test_size}. Fallback to per-user test sets"
        )
        return None

    reference_user_id = str(testset_reference_user).strip() if testset_reference_user is not None else None
    if not reference_user_id:
        print("Warning: testset_reference_user not set for shared test construction. Fallback to per-user test sets")
        return None

    if reference_user_id not in selected_user_ids:
        print(
            f"Warning: testset_reference_user={reference_user_id} is not in enabled_users. "
            f"Fallback to per-user test sets"
        )
        return None

    try:
        reference_image_select_df = _load_user_image_select_dataframe(config, dataset, reference_user_id)
        reference_ranked_keys = (
            reference_image_select_df.sort_values('weighted_value', ascending=False)['image_key']
            .astype(str)
            .tolist()
        )
        reference_train_keys = set(
            _select_train_image_keys(
                reference_ranked_keys,
                train_shot_count,
                select_image_from,
                random_state
            )
        )
        candidate_common_test_keys = [
            key for key in reference_ranked_keys
            if key in common_image_keys and key not in reference_train_keys
        ]
    except Exception as e:
        print(
            f"Warning: failed to prepare shared test candidates from reference user {reference_user_id}: {e}. "
            f"Fallback to per-user test sets"
        )
        return None

    if len(candidate_common_test_keys) < test_size:
        print(
            f"Warning: enabled_users share {len(common_image_keys)} images, but only "
            f"{len(candidate_common_test_keys)} remain after excluding the reference user's top-{train_shot_count} "
            f"train images. Fallback to per-user test sets"
        )
        return None

    shared_test_image_keys = random.Random(random_state).sample(candidate_common_test_keys, test_size)
    print(
        f"\nUse shared test set from enabled_users common intersection: "
        f"{len(shared_test_image_keys)} images shared by {len(selected_user_ids)} users, "
        f"ranked by reference user {reference_user_id}"
    )
    return shared_test_image_keys

def _select_train_image_keys(ranked_keys, train_shot_count, select_image_from, random_state):
    if len(ranked_keys) < train_shot_count:
        raise ValueError(
            f"Only {len(ranked_keys)} ranked images are available, cannot select {train_shot_count} train images"
        )

    if select_image_from == 'top':
        return ranked_keys[:train_shot_count]
    if select_image_from == 'bottom':
        return ranked_keys[-train_shot_count:]
    if select_image_from == 'random':
        return random.Random(random_state).sample(ranked_keys, train_shot_count)

    raise ValueError(
        f"Unsupported dataset.select_image_from={select_image_from}. Supported values: top, bottom, random"
    )


def _split_user_dataframe_using_image_select(
    config,
    dataset,
    user_id,
    random_seed=None,
    train_split_mode='preferselect',
    shared_test_image_keys=None,
):
    user_df = _load_user_dataframe(config, dataset, user_id)
    user_image_select_df = _load_user_image_select_dataframe(config, dataset, user_id)
    random_state = config['dataloader']['random_seed'] if random_seed is None else random_seed

    if dataset == "PARA":
        train_shot_count = int(config['dataloader']['train_size'][config['dataset']['piaa_size']])
        test_size = int(config['dataloader']['test_size'])
        select_image_from = config['dataset'].get('select_image_from', 'top')
        user_df['image_key'] = user_df['image_key'].astype(str)
        ranked_keys = (
            user_image_select_df.sort_values('weighted_value', ascending=False)['image_key']
            .astype(str)
            .tolist()
        )

        shared_keys = []
        shared_key_set = set()
        if shared_test_image_keys:
            for image_key in shared_test_image_keys:
                normalized_key = str(image_key).strip()
                if normalized_key and normalized_key not in shared_key_set:
                    shared_keys.append(normalized_key)
                    shared_key_set.add(normalized_key)

            user_image_keys = set(user_df['image_key'].tolist())
            missing_shared_keys = [image_key for image_key in shared_keys if image_key not in user_image_keys]
            if missing_shared_keys:
                raise ValueError(
                    f"User {user_id} is missing {len(missing_shared_keys)} shared test images "
                    f"from the enabled_users common intersection"
                )

            test_df = user_df[user_df['image_key'].isin(shared_keys)].copy()
            if len(test_df) != len(shared_keys):
                raise ValueError(
                    f"User {user_id} matched {len(test_df)} shared test images, expected {len(shared_keys)}"
                )
        else:
            train_keys = _select_train_image_keys(ranked_keys, train_shot_count, select_image_from, random_state)
            preferselect_train_df = user_df[user_df['image_key'].isin(train_keys)].copy()
            if len(preferselect_train_df) != train_shot_count:
                raise ValueError(
                    f"User {user_id} only matched {len(preferselect_train_df)} selected images, expected {train_shot_count}"
                )

            remain_df = user_df[~user_df['image_key'].isin(train_keys)]
            if len(remain_df) < test_size:
                raise ValueError(
                    f"User {user_id} only has {len(remain_df)} remaining images, cannot sample {test_size} test images"
                )
            test_df = remain_df.sample(n=test_size, random_state=random_state).copy()
            shared_key_set = set(test_df['image_key'].tolist())

        if train_split_mode == 'preferselect':
            candidate_train_keys = [key for key in ranked_keys if key not in shared_key_set]
            train_keys = _select_train_image_keys(
                candidate_train_keys,
                train_shot_count,
                select_image_from,
                random_state
            )
            train_df = user_df[user_df['image_key'].isin(train_keys)].copy()
            if len(train_df) != train_shot_count:
                raise ValueError(
                    f"User {user_id} only matched {len(train_df)} train images outside the test set, expected {train_shot_count}"
                )
        elif train_split_mode == 'random':
            available_train_df = user_df[~user_df['image_key'].isin(shared_key_set)].copy()
            if len(available_train_df) < train_shot_count:
                raise ValueError(
                    f"User {user_id} only has {len(available_train_df)} images outside the fixed test set, "
                    f"cannot sample {train_shot_count} random train images"
                )
            train_df = available_train_df.sample(n=train_shot_count, random_state=random_state).copy()
        else:
            raise ValueError(
                f"Unsupported train_split_mode={train_split_mode}. Supported values: preferselect, random"
            )
    else:
        raise ValueError("This release supports the PARA dataset only.")

    return train_df, test_df


def get_user_dataloaders_using_image_select(
    config,
    dataset,
    user_id,
    random_seed=None,
    train_split_mode='preferselect',
    shared_test_image_keys=None,
):
    """
    Create train/test dataloaders for a specific user using image selection.
    """
    train_df, test_df = _split_user_dataframe_using_image_select(
        config,
        dataset,
        user_id,
        random_seed=random_seed,
        train_split_mode=train_split_mode,
        shared_test_image_keys=shared_test_image_keys,
    )

    train_dataset = PersonalizedAestheticDataset(train_df, config, dataset)
    test_dataset = PersonalizedAestheticDataset(test_df, config, dataset)

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
    test_loader.is_shared_test_set = shared_test_image_keys is not None
    test_loader.shared_test_set_size = len(test_dataset) if shared_test_image_keys is not None else None

    return train_loader, test_loader


def get_qualified_users_dataloaders_using_image_select(
    config,
    dataset,
    random_seed=None,
    using_user_prompts=False,
    using_image_select=True,
    train_split_mode='preferselect',
    selected_user_ids=None,
    testset_reference_user=None,
):
    """
    Get dataloaders for all qualified users using image selection.
    """
    if dataset != "PARA":
        raise ValueError("This release supports the PARA dataset only.")

    if using_image_select:
        if train_split_mode == 'preferselect':
            print(f"\nLoad Datasets for {config['dataset']['piaa_size']} PIAA, using image select")
        else:
            print(
                f"\nLoad Datasets for {config['dataset']['piaa_size']} PIAA, "
                f"using random train with fixed test from image select"
            )
    else:
        print(f"\nLoad Datasets for {config['dataset']['piaa_size']} PIAA")

    if using_user_prompts:
        user_prompts = get_user_prompts()

    user_image_counts = {}
    for file in os.listdir(config["image_select"]['user_path']):
        if not file.startswith("user_") or not file.endswith("_imgs"):
            continue
        body = file[len("user_"):-len("_imgs")]
        if "_" not in body:
            continue
        user_id, image_count = body.rsplit("_", 1)
        user_image_counts[user_id] = int(image_count)

    try:
        piaa_size = config['dataset']['piaa_size']
        min_images = int(config['dataloader']['min_images'][piaa_size])
        qualified_users = [
            user_id for user_id, count in user_image_counts.items()
            if count >= min_images
        ]
        qualified_users = sorted(qualified_users)
    except Exception as e:
        print(f"Error getting piaa size: {e}")
        return {}

    print(f"\nTotal number of unique users for {dataset}: {len(qualified_users)}")

    fixed_test_users_csv = config['dataset'].get('fixed_test_users_csv')
    if fixed_test_users_csv:
        fixed_user_column = config['dataset'].get(
            'fixed_test_users_column',
            config['dataset'].get('user_id_column', 'user_name')
        )
        fixed_users_df = pd.read_csv(fixed_test_users_csv, encoding='utf-8-sig')
        if fixed_user_column not in fixed_users_df.columns:
            raise ValueError(
                f"Column '{fixed_user_column}' not found in fixed user file: {fixed_test_users_csv}"
            )

        selected_qualified_users = []
        seen_users = set()
        for user_id in fixed_users_df[fixed_user_column].astype(str).tolist():
            user_id = user_id.strip()
            if user_id and user_id not in seen_users:
                selected_qualified_users.append(user_id)
                seen_users.add(user_id)

        missing_users = [user_id for user_id in selected_qualified_users if user_id not in qualified_users]
        if missing_users:
            raise ValueError(
                f"Fixed test users not qualified for {config['dataset']['piaa_size']}: {missing_users}"
            )

        print(
            f"\nUse fixed test users from {fixed_test_users_csv}: "
            f"{len(selected_qualified_users)} users for {config['dataset']['piaa_size']} PIAA"
        )
    else:
        random_seed_for_selection = config['dataloader']['random_seed_for_selection']
        print(
            f"\nRandomly choose {config['dataset']['user_size']} users for "
            f"{config['dataset']['piaa_size']} PIAA, using image select"
        )
        random.seed(random_seed_for_selection)
        user_size = config['dataset'].get('user_size')
        if user_size is None:
            selected_qualified_users = qualified_users
        else:
            user_size = int(user_size)
            if user_size > len(qualified_users):
                raise ValueError(
                    f"user_size={user_size} is larger than qualified users={len(qualified_users)} "
                    f"for {config['dataset']['piaa_size']}"
                )
            selected_qualified_users = random.sample(qualified_users, user_size)

    if selected_user_ids is not None:
        requested_user_ids = []
        requested_user_set = set()
        for user_id in selected_user_ids:
            normalized_user_id = str(user_id).strip()
            if normalized_user_id and normalized_user_id not in requested_user_set:
                requested_user_ids.append(normalized_user_id)
                requested_user_set.add(normalized_user_id)

        filtered_users = [user_id for user_id in selected_qualified_users if user_id in requested_user_set]
        if filtered_users:
            missing_requested_users = [user_id for user_id in requested_user_ids if user_id not in selected_qualified_users]
            if missing_requested_users:
                print(
                    f"Warning: {len(missing_requested_users)} enabled_users are not in the current selected test-user set: "
                    f"{missing_requested_users}"
                )
            print(f"\nApply enabled_users filter: keep {len(filtered_users)} of {len(selected_qualified_users)} selected users")
            selected_qualified_users = filtered_users
        else:
            print("Warning: enabled_users produced no valid users in the current selected set, fallback to the original test users")

    shared_test_image_keys = None
    if using_image_select and testset_reference_user is not None:
        reference_user_id = str(testset_reference_user).strip()
        if reference_user_id and reference_user_id not in qualified_users:
            print(
                f"Warning: testset_reference_user={reference_user_id} is not a qualified user for "
                f"{config['dataset']['piaa_size']}, fallback to per-user test sets"
            )
        elif selected_user_ids is None:
            print(
                "Warning: enabled_users not provided, so a shared cross-user test set cannot be "
                "constructed from a common intersection. Fallback to per-user test sets"
            )
        else:
            shared_test_image_keys = _build_shared_test_image_keys_from_selected_users(
                config,
                dataset,
                selected_qualified_users,
                random_seed=random_seed,
                testset_reference_user=testset_reference_user,
            )

    if using_image_select:
        user_dataloaders = {}
        for user_id in tqdm(selected_qualified_users):
            if using_user_prompts:
                user_prompt = user_prompts[str(user_id)]

            try:
                train_loader, test_loader = get_user_dataloaders_using_image_select(
                    config,
                    dataset,
                    user_id,
                    random_seed,
                    train_split_mode=train_split_mode,
                    shared_test_image_keys=shared_test_image_keys,
                )
            except Exception as e:
                if shared_test_image_keys is None:
                    raise
                print(
                    f"Warning: failed to apply shared test set for user {user_id}: {e}. "
                    f"Fallback to the original per-user test split"
                )
                train_loader, test_loader = get_user_dataloaders_using_image_select(
                    config,
                    dataset,
                    user_id,
                    random_seed,
                    train_split_mode=train_split_mode,
                    shared_test_image_keys=None,
                )

            if using_user_prompts:
                user_dataloaders[user_id] = (user_prompt, train_loader, test_loader)
            else:
                user_dataloaders[user_id] = (train_loader, test_loader)

        if fixed_test_users_csv:
            print(f"\nCreated dataloaders for {len(user_dataloaders)} fixed qualified users, using image select")
        else:
            print(f"\nCreated dataloaders for randomly choosed {len(user_dataloaders)} qualified users, using image select")
    else:
        user_dataloaders = {}
        for user_id in tqdm(selected_qualified_users):
            try:
                train_loader, test_loader = get_user_dataloaders(config, user_id, random_seed)
                user_dataloaders[user_id] = (train_loader, test_loader)
            except Exception as e:
                print(f"Error creating dataloaders for user {user_id}: {e}")
                continue

        if fixed_test_users_csv:
            print(f"\nCreated dataloaders for {len(user_dataloaders)} fixed qualified users")
        else:
            print(f"\nCreated dataloaders for randomly choosed {len(user_dataloaders)} qualified users")
    return user_dataloaders
