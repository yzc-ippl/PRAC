import pandas as pd
import numpy as np
import time
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse

# ==========================================
# 1. 
# ==========================================

def extract_user_id(user_name):
    """ID"""
    if user_name.startswith('Train_'):
        return user_name[6:]
    elif user_name.startswith('Test_'):
        return user_name[5:]
    return user_name

def find_user_in_matrix(similarity_matrix, user_id, prefix_type='test'):
    """Return the matrix label for a test or training user."""
    target_name = f'Test_{user_id}' if prefix_type == 'test' else f'Train_{user_id}'
    if prefix_type == 'test' and target_name in similarity_matrix.index:
        return target_name
    elif prefix_type == 'train' and target_name in similarity_matrix.columns:
        return target_name
    return None

def calculate_objectives(similarity_matrix, target_user, selected_users):
    """: & """
    if not selected_users:
        return 0, 0

    # A: 
    similarity_sum = sum(similarity_matrix.loc[target_user, user] for user in selected_users)

    # B: 
    internal_similarity_sum = 0
    if len(selected_users) > 1:
        for i, user1 in enumerate(selected_users):
            for j, user2 in enumerate(selected_users):
                if i < j:
                    user1_id = extract_user_id(user1)
                    test_user1 = f'Test_{user1_id}'
                    if test_user1 in similarity_matrix.index:
                        internal_similarity_sum += similarity_matrix.loc[test_user1, user2]
                    else:
                        user2_id = extract_user_id(user2)
                        test_user2 = f'Test_{user2_id}'
                        if test_user2 in similarity_matrix.index:
                            internal_similarity_sum += similarity_matrix.loc[test_user2, user1]

    return similarity_sum, internal_similarity_sum

def evaluate_combination(similarity_matrix, target_user, selected_users, weight_beta=0.7):
    """Score relevance against the target and diversity within the cohort."""
    sim_sum, internal_sum = calculate_objectives(similarity_matrix, target_user, selected_users)
    n = len(selected_users)
    pairs = n * (n - 1) / 2
    normalized_sim = sim_sum / n if n > 0 else 0
    normalized_internal = internal_sum / pairs if pairs > 0 else 0
    score = weight_beta * normalized_sim - (1 - weight_beta) * normalized_internal
    return score, sim_sum, internal_sum

# ==========================================
# 2. 
# ==========================================

def greedy_solution(similarity_matrix, target_user, candidate_users, n, weight_beta=0.7):
    """
    :
    : O(n * |candidates|)
    """
    selected = []
    remaining = set(candidate_users)

    for _ in range(n):
        best_candidate = None
        best_score = float('-inf')

        for candidate in remaining:
            test_selected = selected + [candidate]
            score, _, _ = evaluate_combination(similarity_matrix, target_user, test_selected, weight_beta)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

    return selected

def local_search(similarity_matrix, target_user, initial_solution, candidate_users, weight_beta=0.7, max_iterations=100):
    """
    :
    ,
    """
    current_solution = initial_solution[:]
    current_score, _, _ = evaluate_combination(similarity_matrix, target_user, current_solution, weight_beta)

    selected_set = set(current_solution)
    remaining = [c for c in candidate_users if c not in selected_set]

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i, old_user in enumerate(current_solution):
            for new_user in remaining:
                new_solution = current_solution[:]
                new_solution[i] = new_user
                new_score, _, _ = evaluate_combination(similarity_matrix, target_user, new_solution, weight_beta)

                if new_score > current_score:
                    current_solution = new_solution
                    current_score = new_score
                    # remaining
                    remaining.remove(new_user)
                    remaining.append(old_user)
                    improved = True
                    break

            if improved:
                break

    return current_solution, current_score

def multi_start_local_search(similarity_matrix, target_user, candidate_users, n, weight_beta=0.7, num_starts=5):
    """
    :,
    """
    best_solution = None
    best_score = float('-inf')

    # 1: 
    greedy_sol = greedy_solution(similarity_matrix, target_user, candidate_users, n, weight_beta)
    sol, score = local_search(similarity_matrix, target_user, greedy_sol, candidate_users, weight_beta)
    if score > best_score:
        best_score = score
        best_solution = sol

    # 2: ()
    sim_greedy = greedy_solution(similarity_matrix, target_user, candidate_users, n, weight_beta=1.0)
    sol, score = local_search(similarity_matrix, target_user, sim_greedy, candidate_users, weight_beta)
    if score > best_score:
        best_score = score
        best_solution = sol

    # 3-5: 
    np.random.seed(42)  # 
    for start_idx in range(num_starts - 2):
        random_sol = list(np.random.choice(candidate_users, n, replace=False))
        sol, score = local_search(similarity_matrix, target_user, random_sol, candidate_users, weight_beta)
        if score > best_score:
            best_score = score
            best_solution = sol

    return best_solution, best_score

def find_optimal_users(similarity_matrix, target_user_id, candidate_users, n, weight_beta=0.7):
    """
    :
    """
    pid = os.getpid()
    start_time = time.time()

    target_user = find_user_in_matrix(similarity_matrix, target_user_id, 'test')
    if target_user is None:
        print(f"[P{pid}] :  {target_user_id}")
        return target_user_id, [], 0, 0, 0, {}

    best_solution, best_score = multi_start_local_search(
        similarity_matrix, target_user, candidate_users, n, weight_beta, num_starts=5
    )

    final_score, sim_sum, int_sum = evaluate_combination(
        similarity_matrix, target_user, best_solution, weight_beta
    )

    elapsed = time.time() - start_time
    stats = {'search_time': elapsed}

    best_solution_ids = [extract_user_id(u) for u in best_solution]

    print(f"[P{pid}]  {target_user_id} ! : {elapsed:.2f}s, : {final_score:.4f}")

    return target_user_id, best_solution_ids, final_score, sim_sum, int_sum, stats

def worker_wrapper(args):
    """Run one cohort-selection task in a worker process."""
    return find_optimal_users(*args)

# ==========================================
# 3. 
# ==========================================

def load_processed_users(filepath):
    """ID,"""
    processed = set()
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            col_name = 'Target_User_ID'
            if col_name in df.columns:
                processed = set(df[col_name].astype(str).tolist())
        except Exception as e:
            print(f": {e}")
    return processed

def append_result_to_csv(filepath, data):
    """CSV"""
    headers = ['Target_User_ID', 'Selected_Users', 'Score', 'Sim_Sum', 'Avg_Sim',
               'Internal_Sum', 'Avg_Internal', 'Diversity', 'Time_Seconds']

    file_exists = os.path.exists(filepath)

    try:
        with open(filepath, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        print(f": {e}")

# ==========================================
# 4. 
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Select a diverse cohort for each target user.")
    run_dir = os.getenv("PRAC_RUN_DIR", "runs/para")
    default_fim = os.path.join(run_dir, "fim", "user_similarities_fixed.csv")
    default_users = os.path.join(
        run_dir, "lora_pool", "TestUsersPool", "TestUsersPool_index.csv"
    )
    default_output = os.path.join(
        run_dir, "fim", "target_users_optimal_results_top6_0.5.csv"
    )
    parser.add_argument("--csv-file", default=os.getenv("PRAC_FIM_CSV", default_fim))
    parser.add_argument("--target-users-file", default=os.getenv("PRAC_TEST_USERS_INDEX", default_users))
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--beta", "--alpha", dest="beta", type=float, default=0.5,
                        help="Relevance/diversity weight beta from the paper.")
    parser.add_argument("--max-workers", type=int, default=min(12, multiprocessing.cpu_count()))
    parser.add_argument("--output-file", default=os.getenv("PRAC_COHORT_CSV", default_output))
    args = parser.parse_args()

    csv_file = args.csv_file
    df = pd.read_csv(args.target_users_file)
    target_users = df['User'].tolist()

    N = args.top_n
    BETA = args.beta
    MAX_WORKERS = args.max_workers
    output_filename = args.output_file
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    print(f"===  ===")
    print(f": {len(target_users)}")
    print(f"Parameters: N={N}, beta={BETA}, workers={MAX_WORKERS}")
    print(f"Output: {output_filename}")

    # 1. 
    try:
        print(f": {csv_file} ...")
        sim_matrix = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        print(f":  {csv_file}")
        return

    # 2. 
    processed_users = load_processed_users(output_filename)
    print(f":  {len(processed_users)} ")

    todo_users = []
    skipped_count = 0

    for uid in target_users:
        uid_str = str(uid)
        if uid_str in processed_users:
            skipped_count += 1
            continue
        if find_user_in_matrix(sim_matrix, uid_str, 'test'):
            todo_users.append(uid_str)
        else:
            print(f":  {uid_str} ,")

    print(f": {len(todo_users)}  ( {skipped_count} )")

    if not todo_users:
        print(",.")
        return

    # 3.  (target_users)
    all_train = [c for c in sim_matrix.columns if c.startswith('Train_')]
    exclude_set = set([f'Train_{uid}' for uid in target_users])
    candidates = [c for c in all_train if c not in exclude_set]

    print(f": {len(candidates)}")
    print("-" * 60)

    # 4. 
    tasks = [(sim_matrix, uid, candidates, N, BETA) for uid in todo_users]

    # 5. 
    total_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_uid = {executor.submit(worker_wrapper, t): t[1] for t in tasks}

        completed_count = 0
        total_tasks = len(tasks)

        for future in as_completed(future_to_uid):
            try:
                uid, best_combo, score, s_sum, i_sum, stats = future.result()

                avg_s = s_sum / len(best_combo) if best_combo else 0
                pair_count = len(best_combo) * (len(best_combo) - 1) / 2
                avg_i = i_sum / pair_count if pair_count > 0 else 0
                diversity = 1 - avg_i

                row_data = {
                    'Target_User_ID': uid,
                    'Selected_Users': '|'.join(best_combo),
                    'Score': f"{score:.6f}",
                    'Sim_Sum': f"{s_sum:.6f}",
                    'Avg_Sim': f"{avg_s:.6f}",
                    'Internal_Sum': f"{i_sum:.6f}",
                    'Avg_Internal': f"{avg_i:.6f}",
                    'Diversity': f"{diversity:.6f}",
                    'Time_Seconds': f"{stats['search_time']:.2f}"
                }

                append_result_to_csv(output_filename, row_data)

                completed_count += 1
                print(f">>>  [{completed_count}/{total_tasks}]  {uid} .")

            except Exception as e:
                print(f"!!!  {future_to_uid[future]} : {e}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"! : {total_elapsed:.2f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\n..")

