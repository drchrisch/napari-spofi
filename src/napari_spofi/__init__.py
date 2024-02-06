__version__ = "0.0.1a2"
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import SpotFinderWidget

__all__ = (
    "SpotFinderWidget",
)
