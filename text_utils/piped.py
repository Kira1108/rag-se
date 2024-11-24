from functools import reduce, partial
from typing import Callable
from .common import *
from .html import *

def composed(*funcs) -> Callable:
    
    """Function Chain Composition"""
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)


process_web_content:Callable[[str], str] = composed(
    html_to_text,
    partial(remove_non_chinese, min_len = 20), 
    partial(remove_shot_lines, min_len = 35),
)

