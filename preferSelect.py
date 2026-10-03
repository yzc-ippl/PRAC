import os
from pathlib import Path
import glob

import numpy as np
import pandas as pd


def infer_dataset_name():
    return os.getenv("PREFERSELECT_DATASET", "PARA")


def get_base_dir(dataset_name):
    if dataset_name == "PARA":
        return Path(os.getenv("PRAC_RUN_DIR", "runs/para")) / "preferselect"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def process_user_selection(weighted_folder, output_base_folder, target_weight=0.9):
    """
    
    """
    output_base_folder = Path(output_base_folder)
    output_base_folder.mkdir(parents=True, exist_ok=True)

    target_col = f"diff_w{float(target_weight):.1f}"
    weighted_files = sorted(glob.glob(os.path.join(weighted_folder, "weights_ablation_results_user_*.csv")))

    print(f"Found {len(weighted_files)} user files in {weighted_folder}")
    print(f"Target Weight: {target_weight} (Column: {target_col})")

    for file_path in weighted_files:
        filename = os.path.basename(file_path)
        try:
            user_id = filename.split("weights_ablation_results_user_")[1].split(".")[0]
        except IndexError:
            print(f"Skipping file with unexpected name format: {filename}")
            continue

        print(f"Processing user {user_id}...")

        try:
            df = pd.read_csv(file_path)
            if target_col not in df.columns:
                print(f"Warning: Column '{target_col}' not found for user {user_id}. Skipping.")
                continue

            sorted_df = df.sort_values(by=target_col, ascending=False).reset_index(drop=True)
            image_num = len(sorted_df)

            user_output_folder = output_base_folder / f"user_{user_id}_{image_num}_imgs"
            user_output_folder.mkdir(parents=True, exist_ok=True)

            result_df = sorted_df[["session_id", "image_id", target_col]].copy()
            result_df.rename(columns={target_col: "weighted_value"}, inplace=True)

            output_filename = f"fix_weight_results_user_{user_id}_{image_num}_imgs.csv"
            result_df.to_csv(user_output_folder / output_filename, index=False)

            with open(user_output_folder / "info.txt", "w", encoding="utf-8") as f:
                f.write(f"User ID: {user_id}\n")
                f.write(f"Total Images: {image_num}\n")
                f.write(f"Selected Weight: {target_weight}\n")
                f.write(f"Source Column: {target_col}\n")

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    dataset_name = infer_dataset_name()
    base_dir = get_base_dir(dataset_name)
    weighted_folder = str(base_dir / "weights_ablation")

    weights_list = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]
    for target_weight in weights_list:
        output_base_folder = str(base_dir / f"preferSelect_{target_weight}")
        process_user_selection(weighted_folder, output_base_folder, target_weight)

