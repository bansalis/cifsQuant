"""Analysis modules for spatial quantification."""

from .population_dynamics import PopulationDynamics
from .distance_analysis import DistanceAnalysis
from .infiltration_analysis import InfiltrationAnalysis
from .infiltration_analysis_optimized import InfiltrationAnalysisOptimized
from .neighborhoods import NeighborhoodAnalysis
from .neighborhoods_optimized import NeighborhoodAnalysisOptimized
from .advanced import AdvancedAnalysis
from .tumor_microenvironment_analysis import TumorMicroenvironmentAnalysis
from .spatial_permutation_testing import SpatialPermutationTesting

# Coexpression and overlap analyses
try:
    from .coexpression_analysis_comprehensive import CoexpressionAnalysisComprehensive
    from .spatial_overlap_analysis import SpatialOverlapAnalysis
    HAS_COMPREHENSIVE_ANALYSES = True
except ImportError:
    HAS_COMPREHENSIVE_ANALYSES = False
    CoexpressionAnalysisComprehensive = None
    SpatialOverlapAnalysis = None

# SpatialCells-based analyses (improved region detection)
try:
    from .per_tumor_analysis_spatialcells import PerTumorAnalysisSpatialCells
    from .infiltration_analysis_spatialcells import InfiltrationAnalysisSpatialCells
    from .marker_region_analysis_spatialcells import MarkerRegionAnalysisSpatialCells
    from .kpnt_correlation_analysis import KPNTCorrelationAnalysis
    HAS_SPATIALCELLS_ANALYSES = True
except ImportError:
    HAS_SPATIALCELLS_ANALYSES = False
    PerTumorAnalysisSpatialCells = None
    InfiltrationAnalysisSpatialCells = None
    MarkerRegionAnalysisSpatialCells = None
    KPNTCorrelationAnalysis = None

__all__ = [
    'PopulationDynamics',
    'DistanceAnalysis',
    'InfiltrationAnalysis',
    'InfiltrationAnalysisOptimized',
    'NeighborhoodAnalysis',
    'NeighborhoodAnalysisOptimized',
    'AdvancedAnalysis',
    'TumorMicroenvironmentAnalysis',
    'SpatialPermutationTesting',
    'PerTumorAnalysisSpatialCells',
    'InfiltrationAnalysisSpatialCells',
    'MarkerRegionAnalysisSpatialCells',
    'KPNTCorrelationAnalysis',
    'CoexpressionAnalysisComprehensive',
    'SpatialOverlapAnalysis'
]
