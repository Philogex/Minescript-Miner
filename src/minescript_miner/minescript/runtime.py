from typing import Callable, TypeVar

import minescript as m


Result = TypeVar("Result")


def query(function: Callable[..., Result], *args, **kwargs) -> Result:
    with m.script_loop:
        return function(*args, **kwargs)
