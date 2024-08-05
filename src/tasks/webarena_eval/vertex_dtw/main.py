import argparse
import os
import warnings

# suppress warnings for optional dependencies from symbolicai
warnings.filterwarnings("ignore")

# DTW implementation
import numpy as np
from symai import GlobalSymbolPrimitive
from symai.functional import EngineRepository
from tqdm import tqdm

from .src.capabilities import group_by_capability
from .src.file_utils import (
    load_jsons_from_dir,
    load_model_results,
    load_task_mappings,
    merge_references,
)
from .src.symbolicai import measure
from .src.vertex_dtw import vertex_dtw

# Setup SymboliAI framework
GlobalSymbolPrimitive("measure", measure)
EngineRepository.register_from_plugin(
    "embedding",
    plugin="ExtensityAI/embeddings",
    kwargs={"model": "all-mpnet-base-v2"},
    allow_engine_override=True,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute VERTEX scores from DTW")
    parser.add_argument(
        "--results", type=str, required=True, help="Directory containing model results"
    )
    parser.add_argument(
        "--references",
        type=str,
        required=True,
        help="Directory containing reference trajectories",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Directory containing baseline trajectories",
    )
    parser.add_argument(
        "--capabilities",
        type=str,
        required=True,
        help="File containing task to capability mappings",
    )
    parser.add_argument(
        "--trivial_tasks",
        type=str,
        required=True,
        help="File containing trivial task ids",
    )
    parser.add_argument(
        "--latex", action="store_true", help="Print results in latex format"
    )
    return parser.parse_args()


def compute_vertex_score(
    results, references, baseline, capabilities, trivial_tasks, trial_weights
):
    # load baseline, reference trajectories and model results
    baseline = load_jsons_from_dir(baseline, score_threshold=0.0)
    references = load_model_results(
        os.path.dirname(references[0]),
        score_threshold=1.0,
        filter=[os.path.basename(p) for p in references],
    )
    references = merge_references(list(references.values()))
    results = load_model_results(
        os.path.dirname(results[0]),
        reference_task_ids=references.keys(),
        score_threshold=0.0,
        filter=[os.path.basename(p) for p in results],
    )

    # load task mappings
    task_to_capability = load_task_mappings(capabilities)
    trivial_task_ids = load_task_mappings(trivial_tasks)

    # compute vertex_dtw for each model
    vertex_scores = {model: {} for model in results.keys()}

    for task_id, ref in tqdm(
        references.items(), desc="VERTEX_DTW", total=len(references)
    ):
        for model in vertex_scores.keys():
            if task_id not in results[model]:
                vertex_scores[model][task_id] = 0
                continue
            test_score = vertex_dtw(ref, baseline[task_id], results[model][task_id])
            vertex_scores[model][task_id] = test_score

    # Filter trial weights to just those we have references for
    supported_task_ids = sorted(int(task_id) for task_id in references)
    trial_weights = [
        [w[task_id] for task_id in supported_task_ids] for w in trial_weights
    ]

    # weight by capability and remove trivial tasks
    vertex_scores_capabilities = group_by_capability(
        vertex_scores, task_to_capability, trivial_task_ids, filter_trivial=False
    )
    vertex_scores_notrivial = group_by_capability(
        vertex_scores, task_to_capability, trivial_task_ids, filter_trivial=True
    )

    # accumulate results
    return {
        model: [
            np.average(
                list(
                    map(
                        lambda x: x[1],
                        sorted(vertex_scores[model].items(), key=lambda x: x[0]),
                    )
                ),
                weights=w,
            )
            for w in trial_weights
        ]
        for model in vertex_scores.keys()
    }
