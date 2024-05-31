#!/bin/bash

# Make requirements.txt from webarena_benchmark compatible
(
    cd webarena_benchmark || exit 1
    git checkout requirements.txt
    NEW_WEBARENA_REQUIREMENTS=$(grep -v -E 'transformers|openai' requirements.txt)
    echo "$NEW_WEBARENA_REQUIREMENTS" > requirements.txt
)
