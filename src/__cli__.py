import click

from . import project
from .project.builtin_tasks import register_builtin_tasks
from .tasks.webarena_eval import (
    webarena_eval,
    webarena_eval_capabilities,
    webarena_eval_gupload,
)
from .tasks.webarena_train import webarena_train


# Register main
@click.group()
@click.pass_context
@click.option(
    "--local_rank", type=int, required=False, hidden=True
)  # DeepSpeed passes this when used
def _main(*args, **kwargs):
    # Run init
    project.init()


# Register built-in tasks
register_builtin_tasks(_main)

# Register tasks
_main.command()(click.pass_context(webarena_eval))
_main.command()(click.pass_context(webarena_eval_gupload))
_main.command()(click.pass_context(webarena_eval_capabilities))
_main.command()(click.pass_context(webarena_train))
