from abc import ABC, abstractmethod
from typing import Tuple, Mapping, Any
import numpy as np

class PolicyInterface(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
    @abstractmethod
    def infer(self, obs: Mapping[str, Any], prev_chunk_left_over: Any | None, inference_delay: int) -> Tuple[np.ndarray, np.ndarray]:
        pass