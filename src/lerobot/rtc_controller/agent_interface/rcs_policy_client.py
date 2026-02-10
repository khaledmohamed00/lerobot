import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from agents.client import RemoteAgent
from agents.policies import Obs
from PIL import Image
from .agent_interface import PolicyClient

class RCSPolicyClient(PolicyClient):
    def __init__(self, host: str= "airtower.utn-mi.de",
                 port: int= 20997,
                 model: str = "lerobot_pi",
                 on_same_machine: bool = False,
                 ):  
        super().__init__(host, port, model, on_same_machine)

    def get_obs(self, obs: dict[str, np.ndarray],
                prev_chunk_left_over: np.ndarray = None,
                inference_delay: int = None,
                reset=False) -> Obs:

        side = obs["frames"]["side"]["rgb"]["data"]
        wrist = obs["frames"]["wrist"]["rgb"]["data"]
        joints = obs["joints"]  
        gripper = obs["gripper"]
        
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