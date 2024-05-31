import atexit
import multiprocessing
import os
from functools import partial

from transformers import AutoTokenizer

from ._llm_server import call_server, launch_server
from ._warnings_silencer import ignore_webarena_warnings

call_llm_history = []


def _patch_to_support_datadreamer_in_webarena(  # noqa: C901
    orig_provider, adapter, batch_size, continuous_batching_interval, is_trivial
):
    with ignore_webarena_warnings():
        # Patch to support the older OpenAI version
        from ._patch_webarena import _patch_openai_to_support_older_version

        _patch_openai_to_support_older_version()

        # Run datadreamer server
        try:
            multiprocessing.set_start_method("fork")
        except RuntimeError:
            pass
        port = 1395
        os.system(
            f"lsof -i tcp:{port}"
            " | awk 'NR!=1 {print $2}' | xargs -I{} kill -9 {} 2>/dev/null"
        )
        process = multiprocessing.Process(
            target=launch_server,
            args=(
                port,
                orig_provider,
                adapter,
                batch_size,
                continuous_batching_interval,
                is_trivial,
            ),
        )
        process.start()
        parent_pid = os.getpid()
        atexit.register(
            partial(
                lambda pid: process.terminate() if pid == os.getpid() else None,
                parent_pid,
            )
        )

        # Patch LLM config
        from llms import lm_config

        _construct_llm_config = lm_config.construct_llm_config

        def new_construct_llm_config(args, *other_args, **kwargs):
            args.provider = orig_provider
            result = _construct_llm_config(args, *other_args, **kwargs)
            args.provider = "datadreamer"
            return result

        lm_config.construct_llm_config = new_construct_llm_config

        # Patch tokenizers
        from llms import tokenizers

        class Tokenizer(tokenizers.Tokenizer):
            def __init__(self, provider, model_name, *args, **kwargs):
                provider = orig_provider
                if provider == "openai":
                    super().__init__(provider, model_name, *args, **kwargs)
                else:
                    while True:
                        # To handle multiple processes concurrently trying to load
                        # a tokenizer. Hugging Face tokenizers is not multi-process
                        # safe
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                            break
                        except FileExistsError:
                            pass
                    self.tokenizer.add_special_tokens = False
                    self.tokenizer.add_bos_token = False
                    self.tokenizer.add_eos_token = False

        tokenizers.Tokenizer = Tokenizer

        # Patch call_llm
        import llms
        from llms import utils

        def new_call_llm(lm_config, prompt):
            global call_llm_history
            response = call_server(port, "call_llm", lm_config, prompt)
            call_llm_history[-1] = {
                **call_llm_history[-1],
                "prompt": prompt,
                "response": response,
            }
            return response

        utils.call_llm = new_call_llm
        llms.call_llm = utils.call_llm

        # Patch prompt constructors
        from agent.prompts import PromptConstructor
        from llms.lm_config import LMConfig

        _get_lm_api_input = PromptConstructor.get_lm_api_input

        def new_get_lm_api_input(self, intro, examples, current):
            orig_lm_config = self.lm_config
            orig_lm_config_dict = orig_lm_config.__dict__
            orig_lm_config_dict["provider"] = "openai"
            orig_lm_config_dict["mode"] = "completion"
            self.lm_config = LMConfig(**orig_lm_config_dict)
            result = _get_lm_api_input(self, intro, examples, current)
            self.lm_config = orig_lm_config
            return result

        PromptConstructor.get_lm_api_input = new_get_lm_api_input

        _extract_action = PromptConstructor.extract_action

        def new_extract_action(self, response):
            from browser_env import ActionParsingError

            try:
                action = _extract_action(self, response)
            except ActionParsingError:
                action_splitter = self.instruction["meta_data"]["action_splitter"]
                if action_splitter == "```":
                    # Some open source LMs only do single backticks, possibly due to
                    # tokenization issues, so we'll support that.
                    self.instruction["meta_data"]["action_splitter"] = "`"
                action = _extract_action(self, response)
                self.instruction["meta_data"]["action_splitter"] = action_splitter

            call_llm_history[-1]["extracted_action"] = action
            return action

        PromptConstructor.extract_action = new_extract_action

        # Patch generate_from_openai_chat_completion to support
        # `llm_fuzzy_match` or `llm_ua_match`
        from llms.providers import openai_utils

        def new_generate_from_openai_chat_completion(
            messages,
            model,
            temperature,
            max_tokens,
            top_p,
            context_length,
            stop_token=None,
        ):
            assert "You are a helpful assistant" in messages[0]["content"]
            assert (
                "Help a teacher to grade the answer of a student given a question"
                in messages[1]["content"]
            ) or (
                "The task described above is inherently unachievable"
                in messages[1]["content"]
            )
            return call_server(
                port,
                "call_eval_llm",
                LMConfig(
                    provider="openai",
                    model=model,
                    mode="chat",
                    gen_config={
                        "fuzzy_or_ua": True,
                        "system_prompt": messages[0]["content"],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop_token": stop_token,
                        "context_length": context_length,
                    },
                ),
                messages[1]["content"],
            )

        openai_utils.generate_from_openai_chat_completion = (
            new_generate_from_openai_chat_completion
        )

        # Patch construct_agent
        import agent

        _construct_agent = agent.construct_agent

        def new_construct_agent(*args, **kwargs):
            global call_llm_history
            new_agent = _construct_agent(*args, **kwargs)
            _next_action = new_agent.next_action

            def new_next_action(trajectory, intent, meta_data):
                if len(call_llm_history) > 0:
                    call_llm_history[-1]["current_action"] = meta_data[
                        "action_history"
                    ][-1]
                call_llm_history.append(
                    {"previous_action": meta_data["action_history"][-1]}
                )
                return _next_action(trajectory, intent, meta_data)

            new_agent.next_action = new_next_action
            return new_agent

        agent.construct_agent = new_construct_agent
