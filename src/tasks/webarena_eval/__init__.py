"""``webarena_eval`` implements everything related to WebArena evaluation."""

import glob
import itertools
import json
import os
import pathlib
import sys
from collections import OrderedDict, defaultdict
from pprint import pprint
from shutil import rmtree
from subprocess import check_output
from time import sleep

import click
import multiprocess
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from ._datadreamer_support import _patch_to_support_datadreamer_in_webarena
from ._patch_webarena import _get_all_results, _patch_to_support_webarena
from ._warnings_silencer import ignore_webarena_warnings


@click.argument("provider", type=click.Choice(["openai", "huggingface", "trivial"]))
@click.argument("model", type=str)
@click.option(
    "--adapter",
    "-a",
    type=str,
    required=False,
    default=None,
    help="The PEFT adapter to apply.",
)
@click.option(
    "--top-p", "-p", type=float, required=False, default=0.0, help="The top_p to use."
)
@click.option(
    "--dd", is_flag=True, show_default=True, default=False, help="Disable DataDreamer."
)
@click.option(
    "--tests",
    "-t",
    type=str,
    required=False,
    default="all",
    help="The range of tests to run.",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    required=False,
    default=1,
    help="The number of workers to run the tests in parallel over.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    required=False,
    default=1,
    help="The batch size to use when calling an LLM.",
)
@click.option(
    "--continuous-batching-interval",
    "-c",
    type=float,
    required=False,
    default=1.0,
    help="The time in seconds to wait to accumulate inputs into a batch before calling an LLM.",
)
@click.option(
    "--continuous-batching-interval",
    "-c",
    type=float,
    required=False,
    default=1.0,
    help="The time in seconds to wait to accumulate inputs into a batch before calling an LLM.",
)
@click.option(
    "--resume",
    is_flag=True,
    show_default=True,
    default=False,
    help="Resume from a previous run.",
)
def webarena_eval(  # noqa: C901
    ctx,
    provider,
    model,
    adapter,
    top_p,
    dd,
    tests,
    workers,
    batch_size,
    continuous_batching_interval,
    resume,
):
    """This command runs the WebArena evluation."""
    # Setup environment for WebArena
    if adapter is not None:
        adapter = os.path.normpath(os.path.abspath(os.path.join("../", adapter)))
    os.chdir("../webarena_benchmark/")
    sys.argv = sys.argv[0:1]

    # Setup DataDreamer
    if not dd:
        is_trivial = provider == "trivial"
        if is_trivial:
            # Pretend like we are running with "openai" and "gpt-4" when running the
            # trivial baseline
            provider, model = "openai", "gpt-4"
        _patch_to_support_datadreamer_in_webarena(
            provider, adapter, batch_size, continuous_batching_interval, is_trivial
        )
        provider = "datadreamer"

    # Patch to support WebArena
    _patch_to_support_webarena(dd)

    with ignore_webarena_warnings():
        # Clean out old tests and generate new tests
        logger.info("Generating WebArena tests...")
        assert os.system("git clean -f -x t config_files/ 1>/dev/null") == 0
        exec(open("scripts/generate_test_data.py").read(), {"__name__": "__main__"})

        # Clean out old credentials and get new auth credentials
        logger.info("Generating WebArena authentication credentials...")
        rmtree("./.auth", ignore_errors=True)
        os.makedirs("./.auth", exist_ok=True)
        exec(open("browser_env/auto_login.py").read(), {"__name__": "__main__"})

        # Run tests
        pathlib.Path("_results.zip").unlink(missing_ok=True)
        pathlib.Path("_results_light.zip").unlink(missing_ok=True)
        if not resume:
            rmtree("/dev/shm/.playwright", ignore_errors=True)
            rmtree("./results", ignore_errors=True)
            rmtree("./log_files", ignore_errors=True)
        os.makedirs("/dev/shm/.playwright", exist_ok=True)
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./results/traces", exist_ok=True)
        if not dd:
            os.makedirs("./results/prompts", exist_ok=True)

        tests = tests.replace(" ", "").lower()
        total_num_tests = int(
            check_output(
                "ls config_files/ | grep json | grep -v test | wc -l", shell=True
            )
            .decode("utf8")
            .strip()
        )
        if tests == "all":
            tests = "0-" + str(total_num_tests)
        test_ranges = tests.split(",")
        needs_to_run = True
        while needs_to_run:
            # Figure out what tests to run
            tests_to_run = []
            for test_range in test_ranges:
                test_start_idx = test_range.split("-")[0]
                test_end_idx = (
                    test_range.split("-")[1]
                    if len(test_range.split("-")) > 1
                    else str(int(test_range.split("-")[0]) + 1)
                )
                for test_idx in range(int(test_start_idx), int(test_end_idx)):
                    tests_to_run.append(test_idx)

            # Define a helper to run a single test
            def run_test(test_idx):
                logger.info(
                    (
                        f"Running WebArena test #{test_idx}"
                        f" with {provider} ({model})..."
                    )
                )
                if not dd:
                    from ._datadreamer_support import call_llm_history

                    call_llm_history.clear()
                sys.argv = [
                    "run.py",
                    "--instruction_path",
                    "agent/prompts/jsons/p_cot_id_actree_2s.json",
                    "--test_start_idx",
                    str(test_idx),
                    "--test_end_idx",
                    str(test_idx + 1),
                    "--provider",
                    provider,
                    "--model",
                    model,
                    "--top_p",
                    str(top_p),
                    "--result_dir",
                    "./results/",
                ]
                try:
                    exec(open("run.py").read(), {"__name__": "__main__"})
                except (ZeroDivisionError, json.JSONDecodeError) as e:
                    logger.debug(f"Test #{test_idx} potentially failed: {str(e)}.")

            # Run each test in multiprocessing Pool to parallelize
            try:
                multiprocess.set_start_method("fork")
            except RuntimeError:
                pass
            pool = multiprocess.Pool(processes=workers, maxtasksperchild=1)
            pool.map(run_test, tests_to_run, chunksize=1)
            pool.close()
            pool.join()

            # Check for failed runs
            logger.info("Checking for failed runs...")
            with open("./results/log_files.txt", "w+") as logs_fp:
                for log_file_name in sorted(glob.glob("log_files/*.log")):
                    logs_fp.write(log_file_name + "\n")
            needs_to_run = (
                os.system(
                    "python3 scripts/check_error_runs.py"
                    " ./results --delete_errors --tolerance 0"
                )
                != 0
            ) or (len(_get_all_results()) < len(tests_to_run))
            if needs_to_run:
                logger.info("Found failed results...retrying.")
                sleep(30)
            else:
                logger.info("No failed results found.")

        # Zip up results
        logger.info("Zipping up results...")
        pathlib.Path("./results/all_results.json.flock").unlink(missing_ok=True)
        assert (
            os.system("zip -q -r _results.zip ./config_files ./results ./log_files")
            == 0
        )
        rmtree("./results/traces", ignore_errors=True)
        assert (
            os.system(
                "zip -q -r _results_light.zip ./config_files ./results ./log_files"
            )
            == 0
        )
        os.rename("_results.zip", "./results/_results.zip")
        os.rename("_results_light.zip", "./results/_results_light.zip")
        logger.info(f"Done: {os.path.join(os.getcwd(), './results/_results.zip')}")
        rmtree("/dev/shm/.playwright", ignore_errors=True)
        os.makedirs("/dev/shm/.playwright", exist_ok=True)


@click.argument("results_path", type=str)
@click.option("--to", "-t", type=str, required=True, help="The Google Drive folder ID.")
def webarena_eval_gupload(  # noqa: C901
    ctx, results_path, to
):
    """This command uploads WebArena evaluation results to Google Drive."""
    # Setup environment for WebArena eval uploading
    os.chdir("../")

    logger.info(f"Uploading: {results_path}")
    from ...project import reporter

    reporter.upload(results_path, tgt=to, service="drive", type="dataset")
    logger.info(f"Done uploading: {results_path}")


@click.argument("parent_folder_path", type=str)
@click.argument("seed_folders", type=str)
@click.option(
    "--threshold",
    "-t",
    type=float,
    required=False,
    default=0.6,
    help="The threshold at which to detect capability paraphrases.",
)
@click.option(
    "--filter_trivial",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether or not to filter trivial tasks.",
)
def webarena_eval_capabilities(  # noqa: C901
    ctx, parent_folder_path, seed_folders, threshold, filter_trivial
):
    """This command trains models as WebArena agents."""
    # Setup environment for WebArena training
    os.chdir("../")

    # Normalize arguments
    parent_folder_path = os.path.normpath(os.path.abspath(parent_folder_path))
    seed_folders = [
        os.path.normpath(os.path.join(parent_folder_path, sf.strip()))
        for sf in seed_folders.split(",")
    ]

    # Initialize a worker pool
    worker_pool = multiprocess.Pool(processes=20)

    # Get total number of tasks
    num_tasks = int(
        check_output(
            f"ls {seed_folders[0]}/config_files/ | grep json | grep -v test | wc -l",
            shell=True,
        )
        .decode("utf8")
        .strip()
    )
    logger.info(f"Detected {num_tasks} tasks.")

    # Get seed model names
    seed_model_names = {}
    for seed_folder in seed_folders:
        seed_model_names[seed_folder] = os.path.basename(seed_folder)

    # Get capability for each task type
    def capability_template_for_capability_idx(task_idx):
        with open(
            os.path.join(seed_folders[0], "config_files", f"{task_idx}.json"), "r"
        ) as config_file_fp:
            config_file = json.load(config_file_fp)
            intent_template_id = config_file["intent_template_id"]
            intent_template = config_file["intent_template"]
            sites = set(config_file["sites"])
            return (intent_template_id, {"sites": sites, "template": intent_template})

    capability_idx_to_template = dict(
        worker_pool.starmap(
            capability_template_for_capability_idx, zip(range(num_tasks))
        )
    )

    logger.info("=============================")
    print("Capabilities:", file=sys.stderr)
    pprint(
        {k: v["template"] for k, v in capability_idx_to_template.items()},
        stream=sys.stderr,
    )
    logger.info("=============================\n\n")

    # Get capability for each task type
    def capability_for_task_idx(task_idx):
        with open(
            os.path.join(seed_folders[0], "config_files", f"{task_idx}.json"), "r"
        ) as config_file_fp:
            intent_template_id = json.load(config_file_fp)["intent_template_id"]
            return (task_idx, intent_template_id)

    task_idx_to_capability = dict(
        worker_pool.starmap(capability_for_task_idx, zip(range(num_tasks)))
    )

    # Get capability groups
    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    capability_paraphrases = defaultdict(set)
    for sites in map(lambda v: v["sites"], capability_idx_to_template.values()):
        templates = OrderedDict(
            map(
                lambda kv: (kv[0], kv[1]["template"]),
                filter(
                    lambda kv: kv[1]["sites"] == sites,
                    capability_idx_to_template.items(),
                ),
            )
        )
        paraphrases = util.paraphrase_mining(
            model, list(templates.values()), top_k=(len(templates) ** 2)
        )
        for capability_idx in templates.keys():
            if capability_idx not in capability_paraphrases:
                capability_paraphrases[capability_idx] = set([capability_idx])
        for score, i, j in paraphrases:
            i = list(templates.keys())[i]
            j = list(templates.keys())[j]
            if score >= threshold:
                capability_paraphrases[i].add(j)
                capability_paraphrases[j].add(i)

    def capability_idx_to_group(capability_idx):
        return sorted(capability_paraphrases[capability_idx])[0]

    logger.info("=============================")
    print(f"Capability Groups ({threshold}):", file=sys.stderr)
    pprint(capability_paraphrases, stream=sys.stderr)
    logger.info("=============================\n\n")

    logger.info("=============================")
    print(f"Capability Group Paraphrases ({threshold}):", file=sys.stderr)
    print(
        json.dumps(
            {
                k: [capability_idx_to_template[idx]["template"] for idx in v]
                for k, v in capability_paraphrases.items()
                if k == sorted(v)[0]
            },
            indent=4,
        ),
        file=sys.stderr,
    )
    logger.info("=============================\n\n")

    # Get results for each model
    def score_for_task_idx(seed_folder):
        with open(
            os.path.join(seed_folder, "results", "all_results.json"), "r"
        ) as all_results_fp:
            return (
                seed_model_names[seed_folder],
                set([int(k) for k, v in json.load(all_results_fp).items() if v == 1.0]),
            )

    results_for_model = dict(worker_pool.starmap(score_for_task_idx, zip(seed_folders)))

    # Filter trivial results
    trivial_task_idxs = [22, 24, 101, 115, 166, 218, 219, 183, 168, 201, 191, 253, 225, 247, 234, 235, 301, 302, 313, 382, 368, 376, 491, 723, 726, 772, 790, 783, 789, 792, 793, 794, 795, 796, 797, 798, 791, 8]  # fmt: off
    trivial_results_for_model = defaultdict(set)
    for model_name in results_for_model:
        for task_idx in results_for_model[model_name].copy():
            if task_idx in trivial_task_idxs:
                if filter_trivial:
                    results_for_model[model_name].remove(task_idx)
                trivial_results_for_model[model_name].add(task_idx)

    # Get capabilities for each model
    capabilities_for_model = {}
    for model_name in results_for_model:
        capabilities_for_model[model_name] = set(
            [
                capability_idx_to_group(task_idx_to_capability[task_idx])
                for task_idx in results_for_model[model_name]
            ]
        )

    # Print evaluated capabilities
    for model_name in capabilities_for_model:
        logger.info(
            f"Model name: {model_name}   Task Count: {len(results_for_model[model_name])}   Capabilities Count: {len(capabilities_for_model[model_name])} Trivial %: {len(trivial_results_for_model[model_name]) / len(results_for_model[model_name].union(trivial_results_for_model[model_name]))}"
        )

    for model_a, model_b in itertools.permutations(seed_model_names.values(), 2):
        a_cap = capabilities_for_model[model_a]
        b_cap = capabilities_for_model[model_b]
        enables = list(a_cap.difference(b_cap))
        loses = list(b_cap.difference(a_cap))
        logger.info("=============================")
        logger.info(
            f"Model name: {model_a} vs. {model_b} enables {len(enables)} capabilities: {enables}  and loses {len(loses)} capabilities: {loses}"
        )
        print("\n\nEnables:", file=sys.stderr)
        for capability_group_idx in enables:
            task_idxs = [
                str(task_idx)
                for task_idx in results_for_model[model_a]
                if capability_idx_to_group(task_idx_to_capability[task_idx])
                == capability_group_idx
            ]
            print(
                capability_idx_to_template[task_idx_to_capability[int(task_idxs[0])]][
                    "template"
                ]
                + f" ({', '.join(task_idxs)})",
                file=sys.stderr,
            )
        print("\n\nDisables:", file=sys.stderr)
        for capability_group_idx in loses:
            task_idxs = [
                str(task_idx)
                for task_idx in results_for_model[model_b]
                if capability_idx_to_group(task_idx_to_capability[task_idx])
                == capability_group_idx
            ]
            print(
                capability_idx_to_template[task_idx_to_capability[int(task_idxs[0])]][
                    "template"
                ]
                + f" ({', '.join(task_idxs)})",
                file=sys.stderr,
            )
        logger.info("=============================\n\n")


__all__ = ["webarena_eval", "webarena_eval_gupload", "webarena_eval_capabilities"]
