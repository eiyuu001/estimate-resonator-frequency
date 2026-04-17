from typing import Type
from bare_shift_boundary_estimator import (
    BareShiftBoundaryEstimator,
    ConfigBareShiftBoundaryEstimator,
    HighFrequencyStrengthBareShiftBoundaryEstimator,
)


estimators: dict[str, Type[BareShiftBoundaryEstimator]] = {
    'config': ConfigBareShiftBoundaryEstimator,
    'high_frequency_strength': HighFrequencyStrengthBareShiftBoundaryEstimator,
}


def create_bare_shift_boundary_estimator(
    conf, image_dir_prefix=None
) -> BareShiftBoundaryEstimator:
    cls = estimators[conf['bare_shift_boundary_estimator']['type']]
    args = conf['bare_shift_boundary_estimator']['args']
    return cls(image_dir_prefix=image_dir_prefix, **args)
