from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq


@dataclass(frozen=True)
class BareShiftBoundaryEstimator(ABC):
    image_dir_prefix: str | None

    @abstractmethod
    def estimate_bare_shift_boundary(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        zs: Sequence[Sequence[float]],
    ) -> tuple[float, float | None, float | None]:
        pass


@dataclass(frozen=True)
class ConfigBareShiftBoundaryEstimator(BareShiftBoundaryEstimator):
    low_power: float
    high_power_min: float
    high_power_max: float

    def estimate_bare_shift_boundary(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        zs: Sequence[Sequence[float]],
    ):
        return self.low_power, self.high_power_min, self.high_power_max


@dataclass(frozen=True)
class HighFrequencyStrengthBareShiftBoundaryEstimator(BareShiftBoundaryEstimator):
    strength_limit: float

    def estimate_bare_shift_boundary(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        zs: Sequence[Sequence[float]],
    ):
        high_freq = np.asarray(
            [self.compute_high_frequency_strength(trace) for trace in zs],
            dtype=np.float64,
        )

        bare_shift_boundary = self.compute_first_local_minimum_index(high_freq)

        self.plot_fft(xs, zs)
        self.plot_high_frequency_strength(high_freq)

        if bare_shift_boundary + 1 < len(ys):
            return ys[bare_shift_boundary], ys[bare_shift_boundary + 1], ys[-1]
        else:
            return ys[bare_shift_boundary], None, None

    def compute_first_local_minimum_index(
        self,
        high_frequency_strength: npt.NDArray[np.float64],
    ) -> int:
        arr = np.concatenate(([float('inf')], high_frequency_strength, [float('inf')]))

        diffs = np.diff(arr)
        is_local_min = np.logical_and((diffs < 0)[:-1], (diffs > 0)[1:])

        is_strength_less_than_limit = high_frequency_strength < self.strength_limit
        candidates = np.logical_and(is_local_min, is_strength_less_than_limit)

        if np.any(candidates):
            result = np.argmax(candidates)
            return int(result)
        else:
            return len(high_frequency_strength) - 1

    def plot_high_frequency_strength(self, high_freq: npt.NDArray[np.float64]):
        if self.image_dir_prefix is None:
            return

        plt.clf()
        plt.plot(high_freq)
        plt.grid()
        plt.savefig(self.image_dir_prefix + '1_high_frequency_strength.png')

    def plot_fft(
        self,
        xs: Sequence[float],
        zs: Sequence[Sequence[float]],
    ):
        if self.image_dir_prefix is None:
            return

        plt.clf()

        N = len(zs[0])
        xs_fft = fftfreq(N, xs[1] - xs[0])[: N // 2]

        for i, trace in enumerate(zs):
            if i % 2 == 1:
                continue

            zs_fft = fft(trace)
            plt.plot(xs_fft[1:], 2.0 / N * np.abs(zs_fft[1 : N // 2]), label=f'y={i}')

        plt.grid()
        plt.legend()
        plt.savefig(self.image_dir_prefix + '0_fft.png')

    @staticmethod
    def compute_high_frequency_strength(trace: Sequence[float]):
        N = len(trace)

        trace_fft = fft(trace)
        trace_fft = np.abs(trace_fft[9 : N // 2])
        return np.mean(trace_fft)
