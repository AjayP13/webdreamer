#!/bin/bash

# Restore requirements.txt from webarena_benchmark
(
    cd webarena_benchmark || exit 1
    git checkout requirements.txt
)

# Install browsers for WebArena automation via playwright
playwright install | (grep -v BEWARE || true)