import json
import os
from collections import Counter
from random import Random

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_plausible_seed_trajectories(seed_folders, worker_pool, num_tasks):  # noqa: C901
    # Initialize seed trajectories as all as valid, then filter them down
    seed_trajectories = [list(range(num_tasks)) for _ in seed_folders]

    def get_task_idx_to_seed_trajectory(num_tasks, seed_trajectories):
        r = Random(42)
        task_idx_to_seed_trajectory = {}
        for task_idx in range(num_tasks):
            if task_idx in seed_trajectories[0]:
                task_idx_to_seed_trajectory[task_idx] = 0
            else:
                other_sts = [
                    st_idx
                    for st_idx, st in enumerate(seed_trajectories[1:])
                    if task_idx in st
                ]
                if len(other_sts) > 0:
                    task_idx_to_seed_trajectory[task_idx] = r.choice(other_sts)
        return task_idx_to_seed_trajectory

    def compute_seed_trajectories_metrics(num_tasks, task_idx_to_seed_trajectory):
        all_results = []
        for seed_folder in seed_folders:
            with open(
                os.path.join(seed_folder, "results", "all_results.json"), "r"
            ) as all_results_fp:
                all_results.append(
                    {
                        int(task_idx): score
                        for task_idx, score in json.load(all_results_fp).items()
                    }
                )
        y = []
        y_pred = []
        for task_idx in range(num_tasks):
            if task_idx in task_idx_to_seed_trajectory:
                y_pred.append(1)
                y.append(
                    int(all_results[task_idx_to_seed_trajectory[task_idx]][task_idx])
                )
            else:
                y_pred.append(0)
                y.append(int(max([ar[task_idx] for ar in all_results])))
        return (
            accuracy_score(y, y_pred),
            f1_score(y, y_pred),
            precision_score(y, y_pred),
            recall_score(y, y_pred),
        )

    task_idx_to_seed_trajectory = get_task_idx_to_seed_trajectory(
        num_tasks, seed_trajectories
    )
    all_task_idx_to_seed_trajectory = task_idx_to_seed_trajectory
    acc, f1, p, r = compute_seed_trajectories_metrics(
        num_tasks, task_idx_to_seed_trajectory
    )
    logger.info(f"Starting with {len(task_idx_to_seed_trajectory)} seed trajectories.")
    logger.info(
        f"Seed trajectories are: Accuracy: {acc}, F1: {f1}, Precision: {p}, Recall: {r}"
    )

    # Filter out any trajectories from training data that can be auto-detected as
    # incomplete due to self stopping in an unsupervised way
    def detect_self_stop(st_idx, st):
        def filter_early_stops(task_idx):
            seed_folder_path = seed_folders[st_idx]
            try:
                with open(
                    os.path.join(
                        seed_folder_path, "results", f"render_{task_idx}.html"
                    ),
                    "r",
                ) as result_html_fp:
                    html_content = result_html_fp.read()
            except FileNotFoundError:
                logger.debug(f"File missing: {task_idx}.html")
                return False
            try:
                with open(
                    os.path.join(
                        seed_folder_path, "results", "prompts", f"{task_idx}.json"
                    ),
                    "r",
                ) as prompts_fp:
                    prompts = json.load(prompts_fp)["prompts"]
                    extracted_actions = [
                        p["extracted_action"]
                        for p in prompts
                        if "extracted_action" in p
                    ]
            except FileNotFoundError:
                logger.debug(f"File missing: {task_idx}.json")
                return False
            return (
                "Early stop" not in html_content
                and Counter(extracted_actions).most_common(1)[0][1] < 3
                and "impossible" not in prompts[-1]["response"]
                and "cannot" not in prompts[-1]["response"]
                and "N/A" not in prompts[-1]["extracted_action"]
                and "stop []" not in prompts[-1]["extracted_action"]
                and "stop [No " not in prompts[-1]["extracted_action"]
                and all(
                    "the format was incorrect" not in p.get("current_action", "")
                    for p in prompts
                )
            )

        return list(filter(filter_early_stops, st))

    seed_trajectories = worker_pool.starmap(
        detect_self_stop, enumerate(seed_trajectories)
    )
    task_idx_to_seed_trajectory = get_task_idx_to_seed_trajectory(
        num_tasks, seed_trajectories
    )
    acc, f1, p, r = compute_seed_trajectories_metrics(
        num_tasks, task_idx_to_seed_trajectory
    )
    logger.info(
        f"Filtered down to {len(task_idx_to_seed_trajectory)} higher-quality seed trajectories (self-stop)."
    )
    logger.info(
        f"Seed trajectories are: Accuracy: {acc}, F1: {f1}, Precision: {p}, Recall: {r}"
    )

    return all_task_idx_to_seed_trajectory, task_idx_to_seed_trajectory
