# Run a single server in a background process that serves requests from all workers
# against an LLM with DataDreamer and returns responses.

import atexit
import datetime
import multiprocessing
import os
import subprocess
import sys
from collections import defaultdict
from functools import partial
from queue import Empty
from time import sleep
from uuid import uuid4

import zerorpc
from expiringdict import ExpiringDict
from loguru import logger


def _launch_server(  # noqa: C901
    port, orig_provider, adapter, batch_size, continuous_batching_interval, is_trivial
):
    import gevent.queue
    import gevent.time
    from gevent import Greenlet

    prompt_queues = defaultdict(lambda: gevent.queue.Queue(maxsize=0))
    result_dicts = defaultdict(
        lambda: ExpiringDict(max_len=sys.maxsize, max_age_seconds=500)
    )

    # Define a continuous batcher to run in the background
    def continuous_batching_runner(prompt_queue, results):
        from datadreamer.llms import HFTransformers, OpenAI
        from datadreamer.utils.fingerprint_utils import stable_fingerprint

        llms = {}

        while True:
            current_batch = []
            for _ in range(batch_size):
                try:
                    current_batch.append(
                        prompt_queue.get(
                            block=True, timeout=continuous_batching_interval
                        )
                    )
                except Empty:
                    pass
            if len(current_batch) == 0:
                continue
            lm_config = current_batch[0][1]
            is_fuzzy_or_ua = "fuzzy_or_ua" in lm_config["gen_config"]
            keys = [row[0] for row in current_batch]
            prompts = [row[2] for row in current_batch]
            cache_folder_path = "../datadreamer_cache/"
            if stable_fingerprint(lm_config) not in llms:
                if is_trivial and not is_fuzzy_or_ua:
                    llm = None
                elif orig_provider == "openai" or is_fuzzy_or_ua:
                    llm = OpenAI(
                        lm_config["model"],
                        cache_folder_path=cache_folder_path,
                        system_prompt=lm_config["gen_config"]["system_prompt"],
                    )
                    llms[stable_fingerprint(lm_config)] = llm
                elif orig_provider == "huggingface":
                    from transformers import BitsAndBytesConfig

                    llm = HFTransformers(
                        lm_config["model"],
                        adapter_name=adapter,
                        device_map="auto",
                        cache_folder_path=cache_folder_path,
                        dtype="bfloat16",
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        ),
                        trust_remote_code=True,
                    )
                llms[stable_fingerprint(lm_config)] = llm
            else:
                llm = llms[stable_fingerprint(lm_config)]
            try:
                if is_trivial and not is_fuzzy_or_ua:
                    # For the "trivial" baseline, always return `stop [N/A]`
                    responses = ["```stop [N/A]```"] * len(prompts)
                else:
                    responses = llm.run(
                        prompts,
                        max_new_tokens=lm_config["gen_config"].get(
                            "max_tokens",
                            lm_config["gen_config"].get("max_new_tokens", None),
                        ),
                        temperature=lm_config["gen_config"]["temperature"],
                        top_p=lm_config["gen_config"]["top_p"],
                        batch_size=batch_size,
                        stop=lm_config["gen_config"].get(
                            "stop_token",
                            lm_config["gen_config"].get("stop_sequences", None),
                        ),
                        n=1,
                        repetition_penalty=None,
                        logit_bias=None,
                        adaptive_batch_size=True,
                        force=(
                            lm_config["gen_config"]["temperature"] > 0.0
                            and lm_config["gen_config"]["top_p"] > 0.0
                        ),
                    )
            except RuntimeError as e:
                print(
                    f"[{str(datetime.datetime.now())}] LLM server crashed, automatically retrying...: {str(e)}"
                )
                sys.exit(1)
                raise e
            for key, response in zip(keys, responses):
                results[key] = response

    # Define a server interface
    class DataDreamerRPC(object):
        def _call_llm(self, func_name, lm_config, prompt):
            key = uuid4().hex
            lm_config = lm_config
            prompt_queues[func_name].put((key, lm_config, prompt))
            while key not in result_dicts[func_name]:
                gevent.time.sleep(0.3)
            return result_dicts[func_name].pop(key)

        def call_llm(self, lm_config, prompt):
            return self._call_llm("call_llm", lm_config, prompt)

        def call_eval_llm(self, lm_config, prompt):
            return self._call_llm("call_eval_llm", lm_config, prompt)

    # Redirect output to /dev/null
    sys.stderr = open(os.devnull, "w")

    # Run continous batchers
    def catch_exception(*args):
        sys.exit(1)

    Greenlet.spawn(
        partial(
            continuous_batching_runner,
            prompt_queues["call_llm"],
            result_dicts["call_llm"],
        )
    ).link_exception(catch_exception)
    Greenlet.spawn(
        partial(
            continuous_batching_runner,
            prompt_queues["call_eval_llm"],
            result_dicts["call_eval_llm"],
        )
    ).link_exception(catch_exception)

    # Run server
    s = zerorpc.Server(DataDreamerRPC(), heartbeat=None)
    s.bind(f"tcp://0.0.0.0:{port}")
    s.run()


def launch_server(
    port, orig_provider, adapter, batch_size, continuous_batching_interval, is_trivial
):
    while True:
        process = multiprocessing.Process(
            target=_launch_server,
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
        print(f"[{str(datetime.datetime.now())}] Booted LLM server.")
        parent_pid = os.getpid()
        atexit.register(
            partial(
                lambda pid, p: p.terminate()
                if (pid == os.getpid() and p.is_alive())
                else None,
                parent_pid,
                process,
            )
        )
        process.join()
        sleep(1)
        print(f"[{str(datetime.datetime.now())}] Rebooting LLM server...")

        # Kill any clients that were connected to the server before it died
        os.system(
            "ps -aux | grep zerorpc.Client"
            " | awk 'NR!=1 {print $2}' | xargs -I{} kill -9 {} 2>/dev/null"
        )


def call_server(port, func_name, lm_config, prompt):
    lm_config = {
        "model": lm_config.model,
        "mode": lm_config.mode,
        "gen_config": lm_config.gen_config,
    }
    lm_config["gen_config"]["system_prompt"] = lm_config["gen_config"].get(
        "system_prompt", None
    )
    returncode = None
    while returncode != 0:
        call_process = subprocess.Popen(
            [
                "python3",
                "-c",
                (
                    f"import zerorpc; "
                    f"client = zerorpc.Client(heartbeat=None, timeout=1200); "
                    f"client.connect('tcp://127.0.0.1:{port}'); "
                    f"print(client.{func_name}({repr(lm_config)}, {repr(prompt)}))"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = call_process.communicate()
        stdout = stdout.decode("utf8")
        stderr = stderr.decode("utf8")
        returncode = call_process.returncode
        if returncode != 0:
            logger.error(
                f"Error reaching LLM server, automatically retrying...: {stderr}"
            )
            sleep(1)
    return stdout
