from dataclasses import dataclass
from typing import Callable, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt


def find_first_left(
    arr: Sequence,
    start: int,
    predicate: Callable[[float], bool],
) -> Optional[int]:
    for i in range(start, -1, -1):
        if predicate(arr[i]):
            return i
    return None


@dataclass(frozen=True)
class Correlation:
    coefs_abs: Sequence[float]
    coefs_rel: Sequence[float]

    @classmethod
    def from_zs(cls, zs: Sequence[Sequence[float]], idx_base: int):
        coefs_abs = np.corrcoef(zs)[idx_base]
        coefs_rel = [np.corrcoef(zs)[i][i + 1] for i in range(len(zs) - 1)]
        return cls(coefs_abs, coefs_rel)


@dataclass(frozen=True)
class CorrelationBasedMinimumUsablePowerEstimator:
    coef_min: float

    def estimate_idx(
        self,
        zs: Sequence[Sequence[float]],
        idx_base: int,
        *,
        artifact_prefix: str | None = None,
    ) -> int:
        correlation = Correlation.from_zs(zs, idx_base)

        if artifact_prefix is not None:
            self.plot(artifact_prefix, correlation)

        correlated_rightmost = find_first_left(
            correlation.coefs_rel, idx_base, lambda x: x >= self.coef_min
        )
        if correlated_rightmost is None:
            return idx_base
        first_below_threshold = find_first_left(
            correlation.coefs_rel, correlated_rightmost, lambda x: x < self.coef_min
        )
        if first_below_threshold is None:
            return 0

        return first_below_threshold + 1

    def plot(
        self,
        artifact_prefix: str,
        correlation: Correlation,
    ):
        plt.clf()
        plt.plot(correlation.coefs_abs, label='abs')
        plt.plot(correlation.coefs_rel, label='rel')
        plt.grid()
        plt.legend()
        plt.savefig(artifact_prefix + '0_corrcoefs.png')
