import json
import math
import os
from random import Random


def create_dataset_from_trajectories(seed_folders, seed_trajectories, random_seed=42):
    r = Random(random_seed)
    first_actions = {0: [], 1: []}
    intermediate_actions = {0: [], 1: []}
    last_actions = {0: [], 1: []}
    all_actions = {0: [], 1: []}

    for task_idx, seed_idx in seed_trajectories.items():
        seed_folder_path = seed_folders[seed_idx]

        # Get prompts and responses
        with open(
            os.path.join(seed_folder_path, "results", "prompts", f"{task_idx}.json"),
            "r",
        ) as prompts_fp:
            prompts = json.load(prompts_fp)["prompts"]
            intermediate_rows = []
            for prompt_idx, prompt_object in enumerate(prompts):
                if "extracted_action" not in prompt_object:
                    continue
                row = {
                    "prompt": prompt_object["prompt"],
                    "response": prompt_object["response"],
                    "extracted_action": prompt_object["extracted_action"],
                    "first": prompt_idx == 0,
                    "last": prompt_idx == (len(prompts) - 1),
                }
                if prompt_idx == 0:
                    first_actions[int(seed_idx != 0)].append(row)
                elif prompt_idx == len(prompts) - 1:
                    last_actions[int(seed_idx != 0)].append(row)
                else:
                    intermediate_rows.append(row)
            if len(intermediate_rows) > 0:
                intermediate_actions[int(seed_idx != 0)].append(
                    r.choice(intermediate_rows)
                )

        # Balance first_action, intermediate_action, and last_action examples
        for is_zero_seed_idx, (fa, la, ia) in enumerate(
            zip(
                first_actions.values(),
                intermediate_actions.values(),
                last_actions.values(),
            )
        ):
            minority_class_count = min([len(fa), len(ia), len(la)])
            first_actions[is_zero_seed_idx] = r.sample(fa, minority_class_count)
            intermediate_actions[is_zero_seed_idx] = r.sample(
                ia, min(len(ia), (minority_class_count * 2))
            )
            last_actions[is_zero_seed_idx] = r.sample(la, minority_class_count)
            all_actions[is_zero_seed_idx] = (
                first_actions[is_zero_seed_idx]
                + intermediate_actions[is_zero_seed_idx]
                + last_actions[is_zero_seed_idx]
            )
            r.shuffle(all_actions[is_zero_seed_idx])

    # Create train / validation splits
    num_total_rows = len(all_actions[0]) + len(all_actions[1])
    num_train_rows = math.ceil(num_total_rows * 0.90)
    num_validation_rows = num_total_rows - num_train_rows
    all_rows = all_actions[0] + all_actions[1]
    validation_rows = all_rows[:num_validation_rows]
    train_rows = all_rows[num_validation_rows:]
    r.shuffle(validation_rows)
    r.shuffle(train_rows)

    # Return the rows
    return train_rows, validation_rows
