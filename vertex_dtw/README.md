# VERTEX_DTW
Implementation of VERTEX_DTW from the paper `Large Language Models Can Self-Improve At Web Agent Tasks` which extends the VERTEX score proposed in `SymbolicAI: A framework for logic-based approaches combining generative models and solvers`.
VERTEX_DTW enables comparison of trajectories with different length by adding an alignment step.
This implementation uses `Dynamic Time Warping` (DTW) to facilitate the alignment.



# Usage

```shell
python main.py \
    --results       <results-dir> \
    --references    <references-dir> \
    --baseline      <random-baseline-dir> \
    --capabilities  <capability_mapping.json> \
    --trivial_tasks <trivial_task_ids.json> \
    [--latex] # print the results table as latex code
```

By default, results, references and baselines are expected to be organized in the following directory structure:

```
results
 - model1
 -- results/prompts
 --- 0.json
 --- 1.json
 --- ...
 --- N.json
```

# References

SymbolicAI: A framework for logic-based approaches combining generative models and solvers

```bibtex
@misc{dinu2024symbolicai,
      title={SymbolicAI: A framework for logic-based approaches combining generative models and solvers}, 
      author={Marius-Constantin Dinu and Claudiu Leoveanu-Condrei and Markus Holzleitner and Werner Zellinger and Sepp Hochreiter},
      year={2024},
      eprint={2402.00854},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Dynamic Time Warping

```bibtex
@inproceedings{
    Berndt:94dtw,
    title={Using Dynamic Time Warping to Find Patterns in Time Series},
    author={Donald J. Berndt and James Clifford},
    booktitle={KDD Workshop},
    year={1994},
    url={https://api.semanticscholar.org/CorpusID:929893}
}
```