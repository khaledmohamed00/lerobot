from typing import Dict
import time
import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch

from .robot_interface import RobotInterface
# ---------------------------------------------------------------------
# Simulation robot backed by a dataset
# ---------------------------------------------------------------------
class Robot_simulation(RobotInterface):
    def __init__(self, dataset_repo_id: str = "HuggingFaceVLA/libero", batch_size: int = 1, fps: float = 30.0):
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps

        ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
        # If you want correct delta timestamps, uncomment this once cfg.policy is available
        # delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        self.dataset = LeRobotDataset(
            dataset_repo_id,
            # delta_timestamps=delta_timestamps,
        )
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.loader_iter = iter(self.data_loader)

        # NOTE: These are placeholders. If your pipeline expects real names, set them properly.
        self.observation_features = []
        self.action_features = []
        print(f"[SIM] Dataset simulation initialized | repo_id={dataset_repo_id}")

    def get_observation(self) -> Dict[str, np.ndarray]:
        try:
            obs = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.data_loader)
            obs = next(self.loader_iter)
        # go over the dictionay transfer if torch tensor to numpy and squeeze the batch dimension
        for k in obs:
            if isinstance(obs[k], torch.Tensor):
                obs[k] = obs[k].squeeze(0).detach().cpu().numpy()
        return obs

    def send_action(self, action: np.ndarray):
        # logger fps
        sleep_time = 1.0 / float(self.fps)
        time.sleep(sleep_time)


