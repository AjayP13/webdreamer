#!/bin/bash

# Make requirements.txt from webarena_benchmark compatible
(
    cd webarena_benchmark || exit 1
    git checkout requirements.txt
    NEW_WEBARENA_REQUIREMENTS=$(grep -v -E 'transformers|openai' requirements.txt)
    echo "$NEW_WEBARENA_REQUIREMENTS" >requirements.txt
)

# Install symbolicai for webarena-eval-capabilities
if [ "$PROJECT_SAFE_TASK" == "webarena_eval_capabilities" ]; then
    pip3 install symbolicai==0.6.2
fi
