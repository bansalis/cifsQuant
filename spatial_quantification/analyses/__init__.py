"""Analysis modules for spatial quantification."""

from .population_dynamics import PopulationDynamics
from .distance_analysis import DistanceAnalysis
from .infiltration_analysis import InfiltrationAnalysis
from .infiltration_analysis_optimized import InfiltrationAnalysisOptimized
from .neighborhoods import NeighborhoodAnalysis
from .neighborhoods_optimized import NeighborhoodAnalysisOptimized
from .advanced import AdvancedAnalysis
from .tumor_microenvironment_analysis import TumorMicroenvironmentAnalysis

# SpatialCells-based analyses (improved region detection)
try:
    from .per_tumor_analysis_spatialcells import PerTumorAnalysisSpatialCells
    from .infiltration_analysis_spatialcells import InfiltrationAnalysisSpatialCells
    HAS_SPATIALCELLS_ANALYSES = True
except ImportError:
    HAS_SPATIALCELLS_ANALYSES = False
    PerTumorAnalysisSpatialCells = None
    InfiltrationAnalysisSpatialCells = None

__all__ = [
    'PopulationDynamics',
    'DistanceAnalysis',
    'InfiltrationAnalysis',
    'InfiltrationAnalysisOptimized',
    'NeighborhoodAnalysis',
    'NeighborhoodAnalysisOptimized',
    'AdvancedAnalysis',
    'TumorMicroenvironmentAnalysis',
    'PerTumorAnalysisSpatialCells',
    'InfiltrationAnalysisSpatialCells'
]
