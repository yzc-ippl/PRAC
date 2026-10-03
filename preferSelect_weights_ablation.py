import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_ccm_line(line, dataset_name):
    parts = line.rstrip("\n").split(",")
    if dataset_name == "PARA":
        if len(parts) < 4:
            raise ValueError(f"Invalid PARA ccm row: {line[:200]}")
        session_id = parts[0]
        image_id = parts[1]
        std_value = parts[2]
        distribution = ",".join(parts[3:])
        full_image_name = f"{session_id}/{image_id}"
        return full_image_name, float(std_value), distribution
    else:
        raise ValueError("This release supports the PARA dataset only.")


def infer_dataset_name():
    return os.getenv("PREFERSELECT_DATASET", "PARA")


def read_std_file(std_file_path, dataset_name):
    records = []
    with open(std_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            full_image_name, std_value, distribution = _parse_ccm_line(line, dataset_name)
            records.append({
                "full_image_name": full_image_name,
                "Std": std_value,
                "Distribution": distribution,
            })

    std_df = pd.DataFrame(records)

    std_values = std_df["Std"].values
    if len(std_values) > 1:
        min_std = np.min(std_values)
        max_std = np.max(std_values)
        std_df["normalized_std"] = (
            (std_df["Std"] - min_std) / (max_std - min_std) if max_std > min_std else std_df["Std"]
        )
    else:
        std_df["normalized_std"] = std_df["Std"]

    std_dict = dict(zip(std_df["full_image_name"], std_df["normalized_std"]))
    return std_dict, std_df


def process_user_rating_file(user_file, std_dict, dataset_name):
    user_df = pd.read_csv(user_file)

    if dataset_name == "PARA":
        user_df["full_image_name"] = user_df["session_id"].astype(str) + "/" + user_df["image_id"].astype(str)
    else:
        raise ValueError("This release supports the PARA dataset only.")

    kl_values = user_df["KL"].values
    if len(kl_values) > 1:
        min_kl = np.min(kl_values)
        max_kl = np.max(kl_values)
        user_df["normalized_kl"] = (
            (user_df["KL"] - min_kl) / (max_kl - min_kl) if max_kl > min_kl else user_df["KL"]
        )
    else:
        user_df["normalized_kl"] = user_df["KL"]

    user_df["normalized_std"] = user_df["full_image_name"].map(lambda x: std_dict.get(x, np.nan))
    user_df = user_df.dropna(subset=["normalized_std"])

    user_id = os.path.basename(user_file).split(".")[0].replace("pdm_user_", "")
    user_df["user_id"] = user_id
    return user_id, user_df


def calculate_user_weighted_differences(user_df, weight_steps=0.1):
    weights = np.arange(0, 1.01, weight_steps)
    user_weight_results = {}

    for a in weights:
        column_name = f"diff_w{a:.1f}"
        user_df[column_name] = a * user_df["normalized_kl"] + (1 - a) * user_df["normalized_std"]
        user_weight_results[a] = user_df[
            ["user_id", "session_id", "image_id", "full_image_name", "normalized_kl", "normalized_std", column_name]
        ].copy()

    return user_weight_results, user_df


def main(std_file_path, user_files_path, output_dir=None, dataset_name="PARA"):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    std_dict, std_df = read_std_file(std_file_path, dataset_name)
    user_files = sorted(glob(os.path.join(user_files_path, "*.csv")))

    all_users_results = {}
    all_users_weight_results = {}

    for user_file in user_files:
        print(f"process_user: {os.path.basename(user_file)}")
        user_id, user_df = process_user_rating_file(user_file, std_dict, dataset_name)
        user_weight_results, user_final_df = calculate_user_weighted_differences(user_df)

        all_users_results[user_id] = user_final_df
        all_users_weight_results[user_id] = user_weight_results

        if output_dir:
            user_to_save = user_final_df.drop(
                columns=["GIAA_distribution", "GIAA+UserPrompt_distribution", "KL", "full_image_name", "user_id"],
                errors="ignore",
            )
            user_to_save.to_csv(os.path.join(output_dir, f"weights_ablation_results_user_{user_id}.csv"), index=False)
            print(f"user {user_id} results save to {output_dir}")

    return all_users_results, all_users_weight_results


if __name__ == "__main__":
    dataset_name = infer_dataset_name()

    if dataset_name == "PARA":
        base_dir = os.path.join(os.getenv("PRAC_RUN_DIR", "runs/para"), "preferselect")
    else:
        raise ValueError("This release supports the PARA dataset only.")

    ccm_path = str(Path(base_dir) / "ccm.csv")
    pdm_path = str(Path(base_dir) / "pdm")
    output_dir = str(Path(base_dir) / "weights_ablation")

    all_users_results, all_users_weight_results = main(
        ccm_path,
        pdm_path,
        output_dir,
        dataset_name=dataset_name,
    )
