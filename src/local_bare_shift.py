from collections import Counter
from operator import attrgetter
from typing import Sequence

from bare_shift import BareShiftBoundary
from estimate_resonator_frequency import Resonance, Peak


def sort_by_count_x_desc(peaks: Sequence[Peak]):
    return sorted(
        Counter(map(attrgetter('x'), peaks)).items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )


def estimate_local_bare_shift_boundary(ys: Sequence[float], resonance: Resonance):
    x_fixed = next(
        x for x, _ in sort_by_count_x_desc(resonance.peaks) if x >= resonance.x
    )

    y_fixed = next(
        peak.y
        for peak in sorted(resonance.peaks, key=attrgetter('y'), reverse=True)
        if peak.x == x_fixed
    )

    if y_fixed + 1 < len(ys):
        return BareShiftBoundary(
            low_power=ys[y_fixed],
            high_power_min=ys[y_fixed + 1],
            high_power_max=ys[-1],
        )
    else:
        return BareShiftBoundary(
            low_power=ys[-1],
            high_power_min=None,
            high_power_max=None,
        )
