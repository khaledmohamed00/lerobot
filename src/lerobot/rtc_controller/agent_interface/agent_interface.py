from typing import Any
import io
import base64
import time
from typing import Union
from abc import ABC, abstractmethod
from typing import Tuple, Mapping, Any

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

from agents.client import RemoteAgent
from agents.policies import Obs
#from ..lerobot_policy import LeRobotPolicy_interface

# interface for PolicyClient to convert observations
class PolicyClient(RemoteAgent):
    def __init__(self, host: str,
                 port: int,
                 model: str,
                 on_same_machine: bool,
                 instruction: str ):  
        super().__init__(host, port, model, on_same_machine)

    def get_obs(self, obs: dict,
                prev_chunk_left_over: Any = None,
                inference_delay: Any = None,
                reset=False) -> Obs:
        # observations preprocessing
        # obs keys    <list of keys>
        pass    

class PolicyAgent:
    def __init__(self, policy: Union[PolicyClient, Any] ):
        self.policy = policy

    def act(self, obs: dict,
            prev_chunk_left_over: Any = None,
            inference_delay: Any = None,
        ) -> Any:
        if isinstance(self.policy, PolicyClient):
            obs_converted = self.policy.get_obs(obs,
                                                prev_chunk_left_over=prev_chunk_left_over,
                                                inference_delay=inference_delay)
            action = self.policy.act(obs_converted)

            return action
        elif isinstance(self.policy, Any):
            # if numpy arrays, convert to torch tensors
            for k in obs:
                if isinstance(obs[k], np.ndarray):
                    obs[k] = torch.from_numpy(obs[k]).unsqueeze(0)
            prev_chunk_left_over = (torch.from_numpy(prev_chunk_left_over).unsqueeze(0)
                                    if prev_chunk_left_over is not None else None)

            action = self.policy.infer(obs,
                                      prev_chunk_left_over=prev_chunk_left_over,
                                      inference_delay=inference_delay,
                                      )
            return action
        else:
            raise ValueError("Unsupported policy type")