"""Decider's native FP16 runtime for Apple Silicon."""
__version__ = "0.1.0"


def __getattr__(name):
    if name in {"Decider", "Native"}:
        from .native import Decider
        return Decider
    raise AttributeError(name)


__all__ = ["Decider", "Native", "__version__"]
