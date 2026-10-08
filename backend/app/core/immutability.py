"""Small helpers for immutable, JSON-compatible contract values."""


class FrozenDict(dict):
    """A JSON-serializable mapping that rejects ordinary mutation."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


def deep_freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(deep_freeze(item) for item in value)
    return value
