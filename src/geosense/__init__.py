"""GeoSense - where was this photo taken? (IE University, AI - Machine Learning & Analytics)"""
from .data import crop_ui, load_classes, load_continents, load_splits

try:
    from .data import GeoSenseDataset  # noqa: F401  (needs torch)
except ImportError:
    pass

__version__ = "0.2.0"
