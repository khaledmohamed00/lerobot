# ---------------------------------------------------------------------
# Robot interface
# ---------------------------------------------------------------------
# use ABC class abstractmethod if needed
from abc import ABC, abstractmethod
from typing import Dict
from threading import Lock

import numpy as np


class RobotInterface(ABC):
    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def step(self, action: np.ndarray):
        pass

    def reset(self):
        pass

class RobotWrapper:
    """
    Thread-safe wrapper around a robot interface.
    Uses a Lock to ensure that observation retrieval and action sending
    are thread-safe operations.
    Args:
        robot: The robot instance to wrap.
    """

    def __init__(self, robot: RobotInterface):
        self.robot = robot
        # Keep real Lock; avoid the "lock=True" bug.
        self.lock = Lock()

    def get_observation(self) -> Dict[str, np.ndarray]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: np.ndarray):
        with self.lock:
            return self.robot.step(action)
