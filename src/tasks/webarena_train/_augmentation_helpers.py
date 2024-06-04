import itertools
import json
import logging
import re
from functools import partial
from random import Random
from types import SimpleNamespace

from datadreamer.datasets import OutputDatasetColumn
from datadreamer.embedders import SentenceTransformersEmbedder
from datadreamer.retrievers import EmbeddingRetriever
from datadreamer.steps import Step
from datasets import Dataset


def flatten_list(nested_list):
    return list(itertools.chain(*nested_list))


def intersperse(lst, item):
    # From: https://stackoverflow.com/a/5921708
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def split_by(val, delimiter):
    delimiter = "\n" + delimiter
    if isinstance(val, str):
        val = [val]
    prev_length = len(val)
    val = flatten_list(
        [intersperse(("\n" + v).split(delimiter), delimiter) for v in val]
    )
    assert len(val) > prev_length, f"Delimiter `{delimiter}` was not found."
    val = [v.strip() for v in val if len(v.strip()) > 0]
    return val


def get_value_for_key(val, key):
    assert key in val, f"Key `{key}` was not found."
    idx = len(val) - 1 - val[::-1].index(key)
    return val[idx + 1]


def extract_structured_information_from_prompt(row):
    prompt = row["prompt"]
    prompt = split_by(
        prompt, "The actions you can perform fall into several categories:"
    )
    prompt = split_by(prompt, "Homepage:")
    prompt = split_by(
        prompt, "To be successful, it is very important to follow the following rules:"
    )
    prompt = split_by(prompt, "Here are a few examples:")
    prompt = split_by(prompt, "Observation\n:")
    prompt = split_by(prompt, "OBSERVATION:")
    prompt = split_by(prompt, "URL:")
    prompt = split_by(prompt, "OBJECTIVE:")
    prompt = split_by(prompt, "PREVIOUS ACTION:")
    prompt = split_by(prompt, "Action:")
    return {
        "instruction": get_value_for_key(
            prompt, "The actions you can perform fall into several categories:"
        ),
        "observation": get_value_for_key(prompt, "OBSERVATION:"),
        "url": get_value_for_key(prompt, "URL:"),
        "objective": get_value_for_key(prompt, "OBJECTIVE:"),
        "previous_action": get_value_for_key(prompt, "PREVIOUS ACTION:"),
    }


def create_objective_and_url_generation_prompts(
    structured_plausible_seed_trajectories, results, k, n
):
    prompts = []
    for _ in range(n):
        r = Random(hash(json.dumps(structured_plausible_seed_trajectories + results)))
        results_k = min(k // 2, len(results))
        orig_k = k - results_k
        examples = r.sample(structured_plausible_seed_trajectories, orig_k) + r.sample(
            results, results_k
        )
        r.shuffle(examples)
        current_page = r.choice(
            ["first page", "intermediate page", "intermediate page", "final page"]
        )
        examples_str = "\n\n".join(
            f"OBJECTIVE: {e['objective']}\nURL: {e['url']}" for e in examples
        )
        prompts.append(
            {
                "prompt": f"Here are a few example objectives (tasks) a user might be asked to perform on a webpage. Closely following these example objectives, generate a potential objective a user might want to perform on another American website that is similar to the examples. (in terms of reasoning required, requiring navigating to multiple pages or taking multiple steps to solve, etc.) The new objective should not be on a website that is the same or is similar to any of the example objective's websites/domains, it should be a completely different website. Ensure the objective has a definitive, objective answer, and not a subjective answer. Return just the objective and a domain name (no path in the URL, just the hostname) of the website (in the same OBJECTIVE:/URL: format) and nothing else.\n\n{examples_str}",
                "first": current_page == "first page",
                "last": current_page == "final page",
            }
        )
    return prompts


def extract_structured_synthetic_objective_and_url(response):
    response = response.split("\n\n")[0]
    response = split_by(response, "OBJECTIVE:")
    response = split_by(response, "URL:")
    return {
        "objective": get_value_for_key(response, "OBJECTIVE:").split("\n")[0].strip(),
        "url": get_value_for_key(response, "URL:").split("\n")[0].strip(),
    }


class GenerateSyntheticObjectivesAndURLs(Step):
    def setup(self):
        self.register_arg("llm", required=True)
        self.register_arg("structured_plausible_seed_trajectories", required=True)
        self.register_arg("k", required=True, help="How many examples to include.")
        self.register_arg("n", required=True, help="Amount to generate.")
        self.register_output("objective")
        self.register_output("url")
        self.register_output("steps_to_solve")
        self.register_output("selected_step")
        self.register_output("first")
        self.register_output("last")

    def run(self):
        r = Random(42)
        llm = self.args["llm"]
        structured_plausible_seed_trajectories = self.args[
            "structured_plausible_seed_trajectories"
        ].export_to_list()
        k = self.args["k"]
        n = self.args["n"]
        results = []
        for i in range(n):
            if len(results) > 0:
                indexed_texts = OutputDatasetColumn(
                    self,
                    Dataset.from_list([{"objective": r["objective"]} for r in results]),
                )
                indexed_texts._step = SimpleNamespace(
                    fingerprint=str(self.fingerprint) + str(len(results))
                )
                prev_results = EmbeddingRetriever(
                    texts=indexed_texts,
                    embedder=SentenceTransformersEmbedder(
                        "sentence-transformers/all-distilroberta-v1", device=0
                    ),
                )
                prev_results.get_logger = partial(
                    prev_results.get_logger, log_level=logging.CRITICAL
                )
            self.progress = i / n
            prompts = create_objective_and_url_generation_prompts(
                structured_plausible_seed_trajectories=structured_plausible_seed_trajectories,
                results=results,
                k=k,
                n=1,
            )
            for attempt in range(9999999):
                response = llm.run(
                    [prompts[0]["prompt"]],
                    batch_size=1,
                    max_new_tokens=200,
                    top_p=1.0,
                    force=attempt > 0,
                    log_level=logging.CRITICAL,
                )[0]
                try:
                    structured = extract_structured_synthetic_objective_and_url(
                        response
                    )
                    if len(results) > 0:
                        closest_previous_result = prev_results.run(
                            queries=[structured["objective"]], k=1
                        )[0]
                        assert closest_previous_result["scores"][0] < 0.70
                    break
                except AssertionError:
                    pass
            for attempt in range(9999999):
                try:
                    steps_to_solve = llm.run(
                        [
                            f"OBJECTIVE: {structured['objective']}\nURL: {structured['url']}\n\nHere is an objective a user can perform on the webpage. The user may need to perform multiple actions / steps (clicking, typing, scrolling, storing/remembering information, or recalling stored information) in order to solve the objective. Assuming the user is starting with a web browser that is already loaded with the website, output the required / necessary steps the user must take on the page to solve the objective, one step per line. Each step MUST involve either clicking, scrolling, typing, or stopping (when the objective is complete). DO NOT output steps that don't involve one of these actions. If a step does not involve clicking, scrolling, typing, or stopping, such as remembering/recalling/calculating information, combine it instead with the next step in the sequence that does. Return nothing else other than the necessary steps, no bullets and no numbered lists."
                        ],
                        batch_size=1,
                        max_new_tokens=1000,
                        top_p=1.0,
                        force=attempt > 0,
                        log_level=logging.CRITICAL,
                    )[0]
                    if attempt < 2:
                        assert "scroll" not in steps_to_solve.split("\n")[0].lower()
                    break
                except AssertionError:
                    pass
            steps_to_solve = [
                line.strip()
                for line in steps_to_solve.split("\n")
                if len(line.strip()) > 0
            ]
            if "scroll" in steps_to_solve[0]:
                steps_to_solve = steps_to_solve[1:]
            steps_to_solve = [
                f"{line_idx+1}. " + re.sub(r"^\d+\.\s*", "", line)
                for line_idx, line in enumerate(steps_to_solve)
            ]
            steps_to_solve_str = "\n".join(steps_to_solve)
            num_of_intermediate_steps = max(1, len(steps_to_solve[1:-1]))
            range_of_steps = list(range(len(steps_to_solve)))
            steps_to_samples = (
                ([range_of_steps[0]] * num_of_intermediate_steps)
                + range_of_steps[1:-1] * 2
                + ([range_of_steps[-1]] * num_of_intermediate_steps)
            )
            selected_step = r.choice(steps_to_samples)
            page_description = (
                f"what page a user would be on after they perform Step #{selected_step}"
                if selected_step > 0
                else "the homepage"
            )
            structured["url"] = (
                llm.run(
                    [
                        f"OBJECTIVE: {structured['objective']}\nWEBSITE: {structured['url']}\nSTEPS:\n{steps_to_solve_str}\n\nHere is an objective a user can perform on a website starting from the homepage and some steps a user may take to solve the objective. Output a realistic and valid URL (don't use placeholders like '123', 'example', 'acme', etc.) for {page_description}. Return just the URL and nothing else."
                    ],
                    batch_size=1,
                    max_new_tokens=1000,
                    top_p=1.0,
                    log_level=logging.CRITICAL,
                )[0]
                .split("\n")[0]
                .strip()
            )
            if len(results) > 0:
                del prev_results
            results.append(
                {
                    **structured,
                    "steps_to_solve": steps_to_solve,
                    "selected_step": selected_step,
                    "first": prompts[0]["first"],
                    "last": prompts[0]["last"],
                }
            )
            del structured
        return results

    @property
    def version(self) -> float:
        return 24.0


def replace_past_tense_in_action(action):
    return (
        action.replace("clicked [", "click [")
        .replace("typed [", "type [")
        .replace("hovered [", "hover [")
        .replace("pressed [", "press [")
        .replace("scrolled [", "scroll [")
        .replace("stopped [", "stop [")
    )


def extract_structured_synthetic_observation_and_actions(response, attempt):
    assert response.count("WEBPAGE:") == 1, "ESOA1"
    assert response.count("PREVIOUS ACTION:") == 1, "ESOA2"
    assert response.count("NEXT ACTION:") == 1, "ESOA3"
    response = split_by(response, "WEBPAGE:")
    response = split_by(response, "PREVIOUS ACTION:")
    response = split_by(response, "NEXT ACTION:")
    next_action = replace_past_tense_in_action(
        get_value_for_key(response, "NEXT ACTION:").strip()
    )
    assert next_action.count("```") == 2, "ESOA4"
    assert (
        "```click" in next_action
        or "```type" in next_action
        or "```hover" in next_action
        or "```scroll" in next_action
        or "```stop" in next_action
    ), f"ESOA5: {next_action}"
    assert next_action.startswith("Let's think step-by-step."), "ESOA6"
    assert re.search(r"(click|hover|type)\s*\[[A-Za-z]+", next_action) is None, "ESOA7"
    return {
        "observation": get_value_for_key(response, "WEBPAGE:"),
        "previous_action": replace_past_tense_in_action(
            get_value_for_key(response, "PREVIOUS ACTION:")
        )
        .split("(")[0]
        .strip(),
        "response": next_action,
    }


def extract_structured_synthetic_observation(response, next_action):
    assert response.count("WEBPAGE:") == 1, "ESO1"
    assert response.count("PREVIOUS ACTION:") == 0, "ESO2"
    assert response.count("NEXT ACTION:") == 0, "ESO3"
    assert response.count("OBJECTIVE:") == 0, "ESO4"
    assert response.count("URL:") == 0, "ESO5"
    if re.search(r"```(click|hover|type)\s*\[([0-9]+)", next_action) is not None:
        assert (
            "["
            + re.search(r"```(click|hover|type)\s*\[([0-9]+)", next_action).groups()[1]
            + "] "
        ) in response, "ESO6"
    response = split_by(response, "WEBPAGE:")
    return {
        "observation": re.split(r"\n\n[^\[]", get_value_for_key(response, "WEBPAGE:"))[
            0
        ].strip()
    }


class GenerateSyntheticObservationsAndActions(Step):
    def setup(self):
        self.register_input("objective", required=True)
        self.register_input("url", required=True)
        self.register_input("steps_to_solve", required=True)
        self.register_input("selected_step", required=True)
        self.register_input("first", required=True)
        self.register_input("last", required=True)
        self.register_arg("llm", required=True)
        self.register_arg("structured_plausible_seed_trajectories", required=True)
        self.register_output("objective")
        self.register_output("url")
        self.register_output("observation")
        self.register_output("previous_action")
        self.register_output("response")

    def run(self):
        llm = self.args["llm"]
        structured_plausible_seed_trajectories = self.args[
            "structured_plausible_seed_trajectories"
        ].export_to_list()
        results = []
        for i, (
            objective,
            url,
            steps_to_solve,
            selected_step,
            first,
            last,
        ) in enumerate(
            zip(
                self.inputs["objective"],
                self.inputs["url"],
                self.inputs["steps_to_solve"],
                self.inputs["selected_step"],
                self.inputs["first"],
                self.inputs["last"],
            )
        ):
            self.progress = i / len(self.inputs["objective"])

            # Prepare
            steps_to_solve_str = "\n".join(steps_to_solve)

            # Generate previous_action, next_action, and observation (that will be thrown away)
            for attempt in range(9999999):
                num_examples = 2
                random_seed = hash(str((i, 0, attempt)))
                r2 = Random(random_seed)
                for _ in range(num_examples):
                    try:
                        examples = r2.sample(
                            [
                                e
                                for e in structured_plausible_seed_trajectories
                                if e["first"] == first
                                and e["last"] == last
                                and "Let's think step-by-step." in e["response"]
                            ],
                            k=num_examples,
                        )
                        break
                    except ValueError:
                        num_examples -= 1
                examples_str = "\n\n".join(
                    [
                        f"Example {str(e_idx+1)}:\n\nOBJECTIVE: {e['objective']}\nURL: {e['url']}\nWEBPAGE:\n{e['observation']}\nPREVIOUS ACTION: {e['previous_action']}\nNEXT ACTION: {e['response'].strip()}"
                        for e_idx, e in enumerate(examples)
                    ]
                )
                prompt = (
                    f"Here are {num_examples} example objectives a user might be asked to perform on a URL / webpage (provided in accessibility tree format). The goal is to perform a series of incremental actions that can complete the objective. The previous action that was taken and the next action a user should take towards completing the objective along with a \"Let's think step-by-step.\" explanation is also provided for the {num_examples} examples. All actions possible for the user are:\n\n{examples[0]['instruction']}\n\nThe action should always be placed inside ``````. For example, \"In summary, the next action I will perform is ```click [1234]```\"."
                    f"\n\n{examples_str}\n\n"
                    f"Following the structure of these {num_examples} examples closely, for the objective and URL below, generate a realistic full-length webpage accessibility tree, realistic previous action, and realistic next action that a user needs to perform on the webpage in order to complete Step #{selected_step+1} of the OVERALL PLAN towards the objective. Provide the actions and webpage in the same format (WEBPAGE:/PREVIOUS ACTION:/NEXT ACTION:). Ensure the next action is Step #{selected_step+1}, the next action begins with \"Let's think step-by-step.\" and ends with \"In summary, the next action I will perform is ```...```\", and the [id] for any actions is an ID number not a string. Do not mention or reference the OVERALL PLAN or Step #{selected_step+1} directly in the output. Return nothing else other than the two actions and the webpage.\n\n"
                    f"OBJECTIVE: {objective}\n"
                    f"URL: {url}\n"
                    f"OVERALL PLAN:\n{steps_to_solve_str}\n"
                    f"CURRENT STEP: {selected_step+1}"
                )
                response = llm.run(
                    [prompt],
                    batch_size=1,
                    max_new_tokens=3000,
                    top_p=1.0,
                    seed=random_seed,
                    log_level=logging.CRITICAL,
                )[0]
                response = response.replace("press_enter_after=", "")
                response = response.replace("direction=", "")
                try:
                    structured_actions = (
                        extract_structured_synthetic_observation_and_actions(
                            response, attempt
                        )
                    )
                    del structured_actions["observation"]
                    break
                except AssertionError as e:
                    print("ASSERTIONERROR1", e, response)
                    pass

            # Generate the observation (that will be kept)
            for attempt in range(9999999):
                random_seed = hash(str((i, 1, attempt)))
                prompt = (
                    f"Here are {num_examples} example objectives a user might be asked to perform on a URL / webpage (provided in accessibility tree format). The goal is to perform a series of incremental actions that can complete the objective. The previous action that was taken and the next action a user should take towards completing the objective along with a \"Let's think step-by-step.\" explanation is also provided for the {num_examples} examples. All actions possible for the user are:\n\n{examples[0]['instruction']}\n\nThe action should always be placed inside ``````. For example, \"In summary, the next action I will perform is ```click [1234]```\"."
                    f"\n\n{examples_str}\n\n"
                    f"Following the structure of these {num_examples} examples closely, for the objective, URL, previous action, and next action below, generate a realistic full-length webpage accessibility tree (don't use placeholders like '123', 'example', 'acme', etc.). Ensure the page is in English and is structured such that performing the next action described would realistically complete or make incremental progress towards completing the objective. Provide the webpage in the same format (WEBPAGE:) and return nothing else other than the webpage.\n\n"
                    f"OBJECTIVE: {objective}\n"
                    f"URL: {url}\n"
                    f"PREVIOUS ACTION: {structured_actions['previous_action']}\n"
                    f"NEXT ACTION: {structured_actions['response']}"
                )
                response = llm.run(
                    [prompt],
                    batch_size=1,
                    max_new_tokens=3000,
                    top_p=1.0,
                    seed=random_seed,
                )[0]

                try:
                    structured_observation = extract_structured_synthetic_observation(
                        response=response, next_action=structured_actions["response"]
                    )
                    break
                except AssertionError as e:
                    print("ASSERTIONERROR2", e)
                    pass

            # Build the results
            results.append(
                {
                    "objective": objective,
                    "url": url,
                    **structured_observation,
                    **structured_actions,
                }
            )
            del structured_actions
            del structured_observation

        return results

    @property
    def version(self) -> float:
        return 8.0


def convert_to_finetuning_format(seed_prompt, row):
    seed_structured_information = extract_structured_information_from_prompt(
        {"prompt": seed_prompt}
    )
    prompt = seed_prompt
    assert prompt.count(seed_structured_information["observation"]) == 1
    prompt = prompt.replace(
        seed_structured_information["observation"], row["observation"]
    )
    assert prompt.count(seed_structured_information["observation"]) == 0

    assert prompt.count(seed_structured_information["url"]) == 1
    prompt = prompt.replace(seed_structured_information["url"], row["url"].strip("<>"))
    assert prompt.count(seed_structured_information["url"]) == 0

    assert prompt.count(seed_structured_information["objective"]) == 1
    prompt = prompt.replace(seed_structured_information["objective"], row["objective"])
    assert prompt.count(seed_structured_information["objective"]) == 0

    assert prompt.count(seed_structured_information["previous_action"]) == 1
    prompt = prompt.replace(
        seed_structured_information["previous_action"], row["previous_action"]
    )
    assert prompt.count(seed_structured_information["previous_action"]) == 0

    row["prompt"] = prompt
    return row
