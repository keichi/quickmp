from ._quickmp import *
from ._version import __version__

__all__ = [
    "initialize",
    "finalize",
    "get_device_count",
    "use_device",
    "get_current_device",
    "get_stream_count",
    "sliding_dot_product",
    "compute_mean_std",
    "selfjoin",
    "abjoin",
    "__version__",
]
