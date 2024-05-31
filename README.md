# WebDreamer

This repository implements the [Large Language Models Can Self-Improve At Web Agent Tasks](https://arxiv.org/abs/2405.20309) paper.

This project evaluates & trains LLMs as web agents on the [WebArena benchmark](https://github.com/web-arena-x/webarena) with synthetic data using [DataDreamer](https://github.com/datadreamer-dev/DataDreamer) and evaluates using the VERTEX score from [SymbolicAI](https://github.com/ExtensityAI/symbolicai). 🤖🌐

## Setup and Install

The ideal version of Python for this project is **Python 3.10**. The project can be cloned and setup with:
```bash
git clone --recurse-submodules git@github.com:AjayP13/webdreamer.git
cd webdreamer/
git config --local core.hooksPath ./.githooks/
./.githooks/post-checkout
```

### Environment Setup

Before running, you will want to edit the `project.env` file and fill in all the environment variables with the needed values.


## Running

To run the project you can simply do the following command to see the list of options of tasks that can be run:

```
./run.sh --help
```

The `./run.sh` file will automatically setup a virtual environment and setup project dependencies on each run. After the first time you run this, all project dependencies will be setup. To skip checking / installing dependencies to make future runs faster, see the `PROJECT_SKIP_INSTALL_REQS` environment variable in `project.env`.


## Formatting and Linting

You can automatically format and lint the code with:
```
./format.sh
```

Additionally, a pre-commit hook will automatically format & enforce Python style through linting when committing.
