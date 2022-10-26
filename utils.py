"""
Provide utilities for the Stable Diffusion Wrapper

Classes:
    LazyLoad: lazy load torch models
"""


import gc
from typing import Callable


class LazyLoad:
    """
    Lazy load torch models.

    Instance attributes:
        loader (Callable): function which loads the model
        args (tuple): args to pass to `loader`
        kwargs (dict): kwargs to pass to `loader`
    NOTE: These attributes are deleted when `load` is called.

    Methods:
        load: load the model, and replace this class' attributes
            with the model's
    """
    def __init__(self, func: Callable, *args, **kwargs):
        self.loader = func
        self.args = args
        self.kwargs = kwargs
        self._model = None

    def load(self):
        """Load the model if not already loaded."""
        if self._model is None:
            self._model = self.loader(*self.args, **self.kwargs)
            for attribute_name in dir(self._model):
                if attribute_name in ["__dict__", "__weakref__", "__class__"]:
                    continue
                attribute = getattr(self._model, attribute_name)
                setattr(self, attribute_name, attribute)
            setattr(self, "__class__", getattr(self._model, "__class__"))
            setattr(self, "load", lambda: None)
            delattr(self, "loader")
            delattr(self, "args")
            delattr(self, "kwargs")
            delattr(self, "_model")
            gc.collect()
