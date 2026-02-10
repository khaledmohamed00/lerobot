import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time
#from rcs.utils import SimpleFrameRate
from .robot_interface import RobotInterface

import gymnasium as gym
import numpy as np


@dataclass
class LatestObs:
    obs: Dict[str, Any]
    info: Dict[str, Any]
    t: float
    seq: int

class LatestBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._data: Optional[LatestObs] = None
        self._seq = 0

    def put(self, obs: Dict[str, Any], info: Dict[str, Any], t: float):
        with self._lock:
            self._seq += 1
            self._data = LatestObs(obs=obs, info=info, t=t, seq=self._seq)
            self._event.set()

    def get_latest(self, wait: bool = True, timeout: Optional[float] = None) -> LatestObs:
        if wait:
            ok = self._event.wait(timeout)
            if not ok:
                raise TimeoutError("No observation available")
        with self._lock:
            assert self._data is not None
            return self._data

class RCSRobot(RobotInterface):
    def __init__(self, env: gym.Env):
        self.env = env
        self.latest = LatestBuffer()
        #self.rate_limiter = SimpleFrameRate(30, "vla inference")

    def reset(self):
        obs, info = self.env.reset()
        self.latest.put(obs, info or {}, time.time())
        return obs, info

    def step(self, vec8: np.ndarray):
        single_action = {"joints": vec8[:7], "gripper": vec8[7]}
        obs, _, _, truncated, info = self.env.step(single_action)
        self.latest.put(obs, info or {}, time.time())

    def get_observation(self, timeout: float = 1.0):
        return self.latest.get_latest(wait=True, timeout=timeout)['obs']