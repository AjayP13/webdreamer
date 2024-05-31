import json
import logging
import os
from functools import partial
from shutil import copytree, rmtree
from subprocess import check_output

import click
import multiprocess
from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers, OpenAI
from datadreamer.steps import DataSource, Prompt, concat, zipped
from datadreamer.utils.fs_utils import safe_fn
from loguru import logger

from ...project import get_torch_cpu_device, get_torch_devices
from ._augmentation_helpers import (
    GenerateSyntheticObjectivesAndURLs,
    GenerateSyntheticObservationsAndActions,
    convert_to_finetuning_format,
    extract_structured_information_from_prompt,
)
from ._dataset_from_trajectories import create_dataset_from_trajectories
from ._seed_trajectories import get_plausible_seed_trajectories


@click.argument("parent_folder_path", type=str)
@click.argument("seed_folders", type=str)
@click.argument("output_folder_path", type=str)
@click.option(
    "--synth",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether or not to generate synthetic `(observation, previous_action) -> next_action` augmented examples.",
)
@click.option(
    "--synth_only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether or not to only use synthetic data for fine-tuning.",
)
@click.option(
    "--synth_provider",
    type=click.Choice(["openai", "huggingface"]),
    required=False,
    default=None,
    help="The provider of the LLM to generate synthetic data with.",
)
@click.option(
    "--synth_model",
    type=str,
    required=False,
    default=None,
    help="The model name of the LLM to generate synthetic data with.",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "huggingface"]),
    required=True,
    help="The provider of the LLM to fine-tune.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help="The model name of the LLM to fine-tune.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    required=False,
    default=16,
    help="The training batch size.",
)
def webarena_train(  # noqa: C901
    ctx,
    parent_folder_path,
    seed_folders,
    output_folder_path,
    synth,
    synth_only,
    synth_provider,
    synth_model,
    provider,
    model,
    batch_size,
):
    from datadreamer.trainers import TrainHFFineTune, TrainOpenAIFineTune
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    """This command trains models as WebArena agents."""
    # Setup environment for WebArena training
    os.chdir("../")

    # Normalize arguments
    parent_folder_path = os.path.normpath(os.path.abspath(parent_folder_path))
    seed_folders = [
        os.path.normpath(os.path.join(parent_folder_path, sf.strip()))
        for sf in seed_folders.split(",")
    ]
    output_folder_path = os.path.normpath(os.path.abspath(output_folder_path))
    if synth and (synth_provider is None and synth_model is None):
        synth_provider = provider
        synth_model = model
    elif synth and (synth_provider is None or synth_model is None):
        raise RuntimeError("You need to provide both synth_provider and synth_model.")
    assert not synth_only or synth, "You must pass --synth to use --synth_only."

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
    seed_model_names = []
    for seed_folder in seed_folders:
        with open(
            os.path.join(seed_folder, "results", "config.json"), "r"
        ) as seed_folder_config_fp:
            seed_model_names.append(json.load(seed_folder_config_fp)["model"])

    # Get seed trajectories
    all_seed_trajectories, filtered_seed_trajectories = get_plausible_seed_trajectories(
        seed_folders, worker_pool, num_tasks
    )

    # Create a dataset from seed trajectories
    (
        train_rows_from_all_seed_trajectories,
        val_rows_from_all_seed_trajectories,
    ) = create_dataset_from_trajectories(seed_folders, all_seed_trajectories)
    (
        train_rows_from_filtered_seed_trajectories,
        val_rows_from_filtered_seed_trajectories,
    ) = create_dataset_from_trajectories(seed_folders, filtered_seed_trajectories)

    # Create synthetic train and validation datasets and train
    with DataDreamer(output_folder_path, log_date=True):
        # Load the rows from the plausible trajectories as a dataset
        train_rows = DataSource(
            "Load train rows",
            [
                {k: v for k, v in row.items() if k in ["prompt", "response"]}
                for row in train_rows_from_filtered_seed_trajectories
            ],
        )
        val_rows = DataSource(
            "Load validation rows",
            [
                {k: v for k, v in row.items() if k in ["prompt", "response"]}
                for row in val_rows_from_filtered_seed_trajectories
            ],
        )

        # Get devices available
        all_devices = (
            get_torch_devices()
            if len(get_torch_devices()) > 0
            else get_torch_cpu_device()
        )
        two_devices = all_devices[:2]

        # Create a quantization_config if needed
        if provider == "huggingface" or synth_provider == "huggingface":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None

        # Synthetically augment the train rows
        if synth:
            if synth_provider == "huggingface":
                synth_llm = HFTransformers(
                    synth_model,
                    device=two_devices,
                    dtype="bfloat16",
                    quantization_config=quantization_config,
                )
            elif provider == "openai":
                synth_llm = OpenAI(synth_model)
            all_seed_rows = DataSource(
                "Load all seed rows for synthetic augmentation",
                train_rows_from_all_seed_trajectories,
            )
            filtered_seed_rows = DataSource(
                "Load seed rows for synthetic augmentation",
                train_rows_from_filtered_seed_trajectories,
            )
            structured_all_seed_trajectories = all_seed_rows.map(
                function=extract_structured_information_from_prompt,
                name="Extract Structured Information from All Plausible Seed Trajectories",
                remove_columns=["prompt"],
                lazy=False,
            )
            structured_filtered_seed_trajectories = filtered_seed_rows.map(
                function=extract_structured_information_from_prompt,
                name="Extract Structured Information from Plausible Seed Trajectories",
                remove_columns=["prompt"],
                lazy=False,
            )
            synthetic_objective_and_url = GenerateSyntheticObjectivesAndURLs(
                "Generate synthetic objectives and URLs",
                args={
                    "llm": synth_llm,
                    "structured_plausible_seed_trajectories": structured_all_seed_trajectories,
                    "k": 4,
                    "n": len(train_rows.output) + len(val_rows.output),
                },
            )
            synthetic_structured_train_rows = GenerateSyntheticObservationsAndActions(
                "Generate synthetic observations and actions",
                inputs={
                    "objective": synthetic_objective_and_url.output["objective"],
                    "url": synthetic_objective_and_url.output["url"],
                    "steps_to_solve": synthetic_objective_and_url.output[
                        "steps_to_solve"
                    ],
                    "selected_step": synthetic_objective_and_url.output[
                        "selected_step"
                    ],
                    "first": synthetic_objective_and_url.output["first"],
                    "last": synthetic_objective_and_url.output["last"],
                },
                args={
                    "llm": synth_llm,
                    "structured_plausible_seed_trajectories": structured_filtered_seed_trajectories,
                },
            )
            synthetic_train_rows = synthetic_structured_train_rows.map(
                function=partial(
                    convert_to_finetuning_format, filtered_seed_rows.output[0]["prompt"]
                ),
                name="Create Synthetic Train Rows in Fine-Tuning Format",
                remove_columns=["objective", "url", "observation", "previous_action"],
                lazy=False,
            ).shuffle()
            # Final Train Rows
            if synth_only:
                final_train_rows = synthetic_train_rows.take(
                    n=len(train_rows.output),
                    name="Load Final Fully Synthetic Train Rows",
                )
                final_val_rows = synthetic_train_rows.skip(
                    n=len(train_rows.output), name="Skip to get Synthetic Val Rows"
                ).take(n=len(val_rows.output), name="Take Synthetic Train Rows")
            else:
                final_train_rows = concat(
                    train_rows,
                    synthetic_train_rows.take(
                        n=len(train_rows.output), name="Take Synthetic Train Rows"
                    ),
                    name="Combine Train Rows with Synthetic Train Rows",
                ).shuffle(
                    name="Load Final Synthetically-Augmented Train Rows", lazy=False
                )
                final_val_rows = val_rows

            # Unload the LLM used for synthetic data generation
            synth_llm.unload_model()
        else:
            # Create the final train and validation rows
            final_train_rows = train_rows
            final_val_rows = val_rows

        # Determine training technique
        all_seed_same = all([sorted(seed_folders)[0] in sf for sf in seed_folders])
        self_augmented = synth_model == model
        self_improved = seed_model_names == [model] or all_seed_same
        training_technique = (
            (
                (
                    "self-synthetic"
                    if self_augmented and self_improved
                    else "distilled-synthetic"
                )
                if synth_only
                else (
                    "self-augmented"
                    if self_augmented and self_improved
                    else "distilled-augmented"
                )
            )
            if synth
            else ("self-improved" if self_improved else "distilled-self-improved")
        )
        if training_technique.startswith("self-") and len(seed_folders) > 1:
            training_technique += f"-round-{len(seed_folders)}"
        training_id = f" ({training_technique} -- {safe_fn(model)})"

        # Fine-tune a model
        BATCH_SIZE = batch_size
        if provider == "huggingface":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            trainer = TrainHFFineTune(
                "Perform HF Fine-Tune for WebArena" + training_id,
                model_name=model,
                peft_config=LoraConfig(),
                device=two_devices,
                dtype="bfloat16",
                quantization_config=quantization_config,
                log_level=logging.DEBUG,
            )
            trainer.train(
                train_input=final_train_rows.output["prompt"],
                train_output=final_train_rows.output["response"],
                validation_input=final_val_rows.output["prompt"],
                validation_output=final_val_rows.output["response"],
                epochs=500,
                batch_size=1,
                gradient_accumulation_steps=(BATCH_SIZE // len(two_devices)),
                gradient_checkpointing=True,
                learning_rate=1e-5,
            )
            trainer.unload_model()
            base_llm = HFTransformers(
                model,
                device=all_devices,
                dtype="bfloat16",
                quantization_config=quantization_config,
                cache_folder_path="/tmp/eval_cache/",
            )
            finetuned_llm = HFTransformers(
                model,
                adapter_name=trainer.model_path,
                device=all_devices,
                dtype="bfloat16",
                quantization_config=quantization_config,
                cache_folder_path="/tmp/eval_cache/",
            )
        elif provider == "openai":
            trainer = TrainOpenAIFineTune(
                "Perform OpenAI Fine-Tune for WebArena" + training_id, model_name=model
            )
            trainer.train(
                train_input=final_train_rows.output["prompt"],
                train_output=final_train_rows.output["response"],
                validation_input=final_val_rows.output["prompt"],
                validation_output=final_val_rows.output["response"],
                epochs=8,
                batch_size=32,
            )
            trainer.unload_model()
            base_llm = OpenAI(model)
            finetuned_llm = OpenAI(trainer.model)

        # Evaluate Base LLM on eval set
        base_generations = Prompt(
            "Evaluate the Base LLM on the Validation Set" + training_id,
            inputs={"prompts": final_val_rows.output["prompt"]},
            args={
                "llm": base_llm,
                "batch_size": 1,
                "max_new_tokens": 384,
                "temperature": 0.9,
            },
            outputs={"generations": "base_generation"},
        ).select_columns(
            ["base_generation"], name="Select the Base Generations Column" + training_id
        )
        base_llm.unload_model()

        # Evaluate Trained LLM on eval set
        finetuned_generations = Prompt(
            "Evaluate the Fine-Tuned LLM on the Validation Set" + training_id,
            inputs={"prompts": final_val_rows.output["prompt"]},
            args={
                "llm": finetuned_llm,
                "batch_size": 1,
                "max_new_tokens": 384,
                "temperature": 0.9,
            },
            outputs={"generations": "finetuned_generation"},
        ).select_columns(
            ["finetuned_generation"],
            name="Select the Fine-Tuned Generations Column" + training_id,
        )
        finetuned_llm.unload_model()

        # Rename validation row columns
        final_val_rows.rename_columns(
            {"response": "expected_response"},
            name="Rename Validation Set Columns" + training_id,
        )

        # Combine all of the columns
        evaluated_dataset = zipped(
            final_val_rows,
            base_generations,
            finetuned_generations,
            name="Build Final Evaluated Dataset" + training_id,
        )

        # Create final output
        final_output_folder_path = os.path.join(
            output_folder_path, "_trained_models", training_technique, safe_fn(model)
        )
        rmtree(final_output_folder_path, ignore_errors=True)
        os.makedirs(final_output_folder_path, exist_ok=True)

        # Save synthetic data
        if synth:
            with open(
                os.path.join(final_output_folder_path, "synthetic_dataset.json"), "w+"
            ) as synthetic_dataset_fp:
                json.dump(
                    synthetic_train_rows.export_to_list(),
                    synthetic_dataset_fp,
                    indent=4,
                )

        # Save training information
        with open(
            os.path.join(final_output_folder_path, "training_info.json"), "w+"
        ) as training_info_fp:
            json.dump(
                dict(
                    parent_folder_path=parent_folder_path,
                    seed_folders=seed_folders,
                    output_folder_path=output_folder_path,
                    synth=synth,
                    synth_only=synth_only,
                    synth_provider=synth_provider,
                    synth_model=synth_model,
                    provider=provider,
                    model=model,
                    batch_size=batch_size,
                ),
                training_info_fp,
                indent=4,
            )

        # Save model
        if provider == "huggingface":
            copytree(
                trainer.model_path, os.path.join(final_output_folder_path, "model")
            )
        elif provider == "openai":
            with open(
                os.path.join(final_output_folder_path, "model_name.json"), "w+"
            ) as model_name_fp:
                json.dump({"model_name": trainer.model}, model_name_fp, indent=4)

        # Save evaluation predictions
        with open(
            os.path.join(final_output_folder_path, "evaluation_predictions.json"), "w+"
        ) as evaluation_predictions_fp:
            json.dump(
                evaluated_dataset.export_to_list(), evaluation_predictions_fp, indent=4
            )

        # List all outputs
        print("\n=======================")
        print("Outputs:")
        print("=======================")
        for output in os.listdir(final_output_folder_path):
            norm_path = os.path.normpath(
                os.path.abspath(os.path.join(final_output_folder_path, output))
            )
            print(f"\n* {norm_path}")


__all__ = ["webarena_train"]
