# ---------------------------------------------------------------------
# Robot interface
# ---------------------------------------------------------------------
# use ABC class abstractmethod if needed
from abc import ABC, abstractmethod
from typing import Dict
from threading import Lock

import torch

class RobotInterface(ABC):
    @abstractmethod
    def get_observation(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def send_action(self, action: torch.Tensor):
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

    def get_observation(self) -> Dict[str, torch.Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: torch.Tensor):
        with self.lock:
            return self.robot.send_action(action)

    def observation_features(self):
        with self.lock:
            return getattr(self.robot, "observation_features", [])

    def action_features(self):
        with self.lock:
            return getattr(self.robot, "action_features", [])