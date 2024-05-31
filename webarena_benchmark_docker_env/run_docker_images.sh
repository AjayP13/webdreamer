#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")"/../ || exit

# Setup logs directory
export LOGS_DIR="$(pwd)/webarena_benchmark_docker_env/docker_images/logs"
rm -rf "$LOGS_DIR"
mkdir -p "$LOGS_DIR"

# Run two jobs
(
    cd webarena_benchmark/environment_docker/webarena-homepage || exit 1
    git checkout templates/index.html
    perl -pi -e "s|<your-server-hostname>|http://<your-server-hostname>|g" templates/index.html
)
qsub -N bench_1 -l mem=4G -pe parallel-onenode 12 -l h=nlpgrid19 webarena_benchmark_docker_env/run_docker_images_helper.sh "1 3 4 5"
qsub -N bench_2 -l mem=4G -pe parallel-onenode 4 -l h=nlpgrid20 webarena_benchmark_docker_env/run_docker_images_helper.sh "2"
