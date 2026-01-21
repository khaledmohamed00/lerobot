from typing import Dict
import time

import numpy as np

from .robot_interface import RobotInterface

class RobotDummy(RobotInterface):
    """
    A dummy robot interface for testing purposes.
    Implements the RobotInterface methods with no-op or fixed behavior.
    """
    def __init__(self, fps: float, dummy_data_path: str = ""):
        self.fps = fps
        self.data = np.load(dummy_data_path) if dummy_data_path else None
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Return a fixed dummy observation.
        """
        data_loaded = {k: self.data[k] for k in self.data.files} if self.data is not None else {}
        return data_loaded

    def send_action(self, action: np.ndarray):
        """
        Dummy method to simulate sending an action to the robot.
        Does nothing.
        """
        # logger fps
        sleep_time = 1.0 / float(self.fps)
        time.sleep(sleep_time)
