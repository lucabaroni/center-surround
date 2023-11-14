from .demo_generator import render_image

from functools import partial
from nnfabrik.builder import resolve_fn


resolve_generator = partial(resolve_fn, default_base="generators")