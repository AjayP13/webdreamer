import contextlib
import warnings


@contextlib.contextmanager
def ignore_webarena_warnings():
    """WebArena throws some warnings we can silence to make output logs cleaner."""
    from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=BeartypeDecorHintPep585DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message='Field "model_id" has conflict.*',
            module="pydantic._internal._fields",
        )
        yield None
