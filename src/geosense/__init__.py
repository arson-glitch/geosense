"""GeoSense - where was this photo taken? (IE University, AI - Machine Learning & Analytics)"""
from .data import load_classes, load_manifest

try:
    from .data import GeoSenseDataset  # noqa: F401  (needs torch)
except ImportError:
    pass

__version__ = "0.1.0"
