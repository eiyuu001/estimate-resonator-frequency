from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


class LowPowerEstimator(ABC):
    @abstractmethod
    def estimate_low_power(
        self,
        ys: Sequence[float],
        zs: Sequence[Sequence[float]],
    ) -> float:
        pass


@dataclass(frozen=True)
class ConfigLowPowerEstimator(LowPowerEstimator):
    low_power: float

    def estimate_low_power(
        self,
        ys: Sequence[float],
        zs: Sequence[Sequence[float]],
    ):
        return self.low_power
