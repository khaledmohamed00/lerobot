from typing import Any
import io
import base64
import time
from typing import Union

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from agents.client import RemoteAgent
from agents.policies import Obs

from .lerobot_policy import LeRobotPolicy

class PolicyClient(RemoteAgent):
    def __init__(self, host: str= "airtower.utn-mi.de",
                 port: int= 20997,
                 model: str = "lerobot_pi",
                 on_same_machine: bool = False,
                 instructions: str = ""):  
        super().__init__(host, port, model, on_same_machine)
    
    def get_obs(self, obs: dict,
                prev_chunk_left_over: Any = None, inference_delay: Any = None,
                reset=False) -> Obs:
        # observations preprocessing
        # obs keys    <list of keys>
        #['observation.images.image', # torch.Size([1, 3, 256, 256]) 
        # 'observation.images.image2', # torch.Size([1, 3, 256, 256])
        # 'observation.state', # torch.Size([1, 8])
        # 'action', # torch.Size([1, 50, 7])
        # 'timestamp', # torch.Size([1])
        # 'frame_index', # torch.Size([1])
        # 'episode_index', # torch.Size([1])
        # 'index', # torch.Size([1])
        # 'task_index', # torch.Size([1])
        # 'action_is_pad', # torch.Size([1, 50])  # bool
        # 'task'] # ["string"]

        # side = obs["frames"]["side"]["rgb"]["data"]
        # wrist = obs["frames"]["wrist"]["rgb"]["data"]
        # gripper = obs["gripper"]
        # joints = obs["joints"]

        side = obs["observation.images.image"].transpose(1, 2, 0).astype(np.uint8)  # HWC
        wrist = obs["observation.images.image2"].transpose(1, 2, 0).astype(np.uint8)  # HWC
        state = obs["observation.state"]  # 8-dim
        joints = state[:7]  # first 7 values are joint positions
        gripper = state[7]  # the last value is gripper open/close value
        prev_chunk_left_over = prev_chunk_left_over.cpu().numpy() if prev_chunk_left_over is not None else None
        inference_delay = inference_delay if inference_delay is not None else 0
        if self.on_same_machine:
            return Obs(cameras=dict(rgb_side=side, rgb_wrist=wrist),
                       gripper=gripper, info=dict(joints=joints,
                                                   prev_chunk_left_over=prev_chunk_left_over,
                                                   inference_delay=inference_delay))
        else:
            # encode to jpeg to reduce the size
            # with jpeg encoding 70 - 80 Hz transfer speed, without 17 fps
            side_bytes = io.BytesIO()
            Image.fromarray(
                side
            ).save(side_bytes, format="JPEG", quality=80)

            wrist_bytes = io.BytesIO()
            Image.fromarray(
                wrist
            ).save(wrist_bytes, format="JPEG", quality=80)

            return Obs(cameras=dict(rgb_side=base64.urlsafe_b64encode(side_bytes.getvalue()).decode("utf-8"), rgb_wrist=base64.urlsafe_b64encode(wrist_bytes.getvalue()).decode("utf-8")),
                    gripper=gripper, info=dict(joints=joints,
                                                prev_chunk_left_over=prev_chunk_left_over,
                                                inference_delay=inference_delay))
    

class PolicyAgent:
    def __init__(self, policy: Union[PolicyClient, LeRobotPolicy] ):
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
        elif isinstance(self.policy, LeRobotPolicy):
            action = self.policy.infer(obs,
                                      prev_chunk_left_over=prev_chunk_left_over,
                                      inference_delay=inference_delay,
                                      )
            return action
        else:
            raise ValueError("Unsupported policy type")