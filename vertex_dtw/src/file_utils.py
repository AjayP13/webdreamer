import json
from glob import glob

from tqdm.auto import tqdm


def load_jsons_from_dir(results_dir, score_threshold=0.0):
    task_trajectories = {}

    # load json files and extract task id from filename
    json_files = glob(f"{results_dir}/results/prompts/*.json")
    json_files = {fn.split("/")[-1].split(".")[0]: fn for fn in json_files}

    # load trajectories for all tasks in directory
    for task_id, fn in tqdm(
        json_files.items(), total=len(json_files), desc="Loading trajectories"
    ):
        with open(fn, "r") as f:
            data = json.load(f)

        # get score
        score = data["score"]

        # add to results
        if score >= score_threshold:
            task_trajectories[task_id] = data

    return task_trajectories


def load_model_results(results_dir, reference_task_ids=None, score_threshold=0.0):
    models = glob(f"{results_dir}/*")
    models_names = [m.split("/")[-1] for m in models]
    results = {}

    for model_name, model_dir in tqdm(
        zip(models_names, models),
        total=len(models_names),
        desc="Loading model trajectories",
    ):
        results[model_name] = load_jsons_from_dir(
            model_dir, score_threshold=score_threshold
        )
        # Remove tasks that are not in the reference solutions
        if reference_task_ids is not None:
            for task_id in list(results[model_name].keys()):
                if task_id not in reference_task_ids:
                    del results[model_name][task_id]

    return results


def load_task_mappings(task_mappings_file):
    with open(task_mappings_file, "r") as f:
        task_mappings = json.load(f)
    return task_mappings


def merge_references(results):
    merged_reference = {task: [ref] for task, ref in results[0].items()}
    for res in results:
        for task in res.keys():
            if task not in merged_reference:
                merged_reference[task] = []
            merged_reference[task].append(res[task])
    return merged_reference
