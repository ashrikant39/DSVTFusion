from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, PillarVFE3D
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE, DynamicPillarVFE_3d,\
    DynamicPillarWithBoxVFE, DynamicPillarWithClassFeatsVFE, DynamicPillarWithFeatureSeg,\
    DynamicPillarWithClassSeg, DynamicPillarWithFullBoxSeg
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'PillarVFE3D': PillarVFE3D,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynPillarVFE3D': DynamicPillarVFE_3d,
    'DynamicPillarWithBoxVFE':DynamicPillarWithBoxVFE,
    'DynamicPillarWithClassFeatsVFE': DynamicPillarWithClassFeatsVFE,
    'DynamicPillarWithFeatureSeg': DynamicPillarWithFeatureSeg,
    'DynamicPillarWithClassSeg': DynamicPillarWithClassSeg,
    'DynamicPillarWithFullBoxSeg': DynamicPillarWithFullBoxSeg
}
