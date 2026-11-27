"""
pypvroof - A Python package for PV roof mapping and analysis
"""

__version__ = "0.2.0"

from .main import MetadataExtraction
from .dsm import DSMService, DSMPatch, FetchResult, IGNLidarService, IGNCorrelService, LocalRasterService
from .theilsen import TheilSenParams
from .bulk import characterize_roofs

__all__ = [
    'MetadataExtraction',
    'characterize_roofs',
    'TheilSenParams',
    'DSMService',
    'DSMPatch',
    'FetchResult',
    'IGNLidarService',
    'IGNCorrelService',
    'LocalRasterService',
]
