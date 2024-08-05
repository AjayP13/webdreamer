import numpy as np


def group_by_capability(vertex_scores, task_id_to_capability_id, trivial_task_ids,  filter_trivial=False):
    vertex_scores_by_capability = {model: {c: -1 for c in task_id_to_capability_id.values()} for model in vertex_scores.keys()}
    for model in sorted(vertex_scores.keys()):
        for task_id, score in vertex_scores[model].items():
            if filter_trivial and int(task_id) in trivial_task_ids:
                continue
            old = vertex_scores_by_capability[model][task_id_to_capability_id[task_id]]
            if score > old:
                vertex_scores_by_capability[model][task_id_to_capability_id[task_id]] = score
    result = {}
    for model in vertex_scores_by_capability.keys():
        result[model] = np.mean([x for x in vertex_scores_by_capability[model].values() if x != -1])
    return result
 