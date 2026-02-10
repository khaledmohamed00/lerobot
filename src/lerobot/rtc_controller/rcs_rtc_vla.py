import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from time import sleep
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time

import json_numpy
import numpy as np
from omegaconf import OmegaConf
from rcs._core.common import RobotPlatform
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.creators import SimEnvCreator
from rcs.envs.utils import (
    default_mujoco_cameraset_cfg,
    default_sim_gripper_cfg,
    default_sim_robot_cfg,
)
from rcs.utils import SimpleFrameRate
from rcs_fr3.creators import RCSFR3EnvCreator
from rcs_fr3.desk import FCI, ContextManager, Desk
from rcs_fr3.utils import default_fr3_hw_gripper_cfg, default_fr3_hw_robot_cfg
from rcs_realsense.utils import default_realsense

from rcs_toolbox.real_grasp_collector_async import PickUpDemo
from rcs_toolbox.utils import load_creds_fr3_desk

from .rcs_policy_client import RCSPolicyClient, PolicyClient
from .rcs_robot import RCSRobot

from .rtc_controller import RTCController
json_numpy.patch()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ROBOT_INSTANCE = RobotPlatform.HARDWARE
INSTRUCTION = "pick the green box"


# 24 29 36 38 39 48
default_cfg = OmegaConf.create(
    {
        "host": "airtower.utn-mi.de",
        "port": 20997,
        "model": "openpi",
        "robot_ip": "192.168.102.1",
        "debug": True,
        # "description": "openpi_utn_wrist",
        "n_experiment": 1
        ,
        "description": "openpi_2025-09-14_20-20-04_pi0_rcs_real_sim_async_sim ckpt 29999",
    }
)
cli_conf = OmegaConf.from_cli()
cfg = OmegaConf.merge(cli_conf, default_cfg)
DEBUG = cfg.debug


# class RTCController:
#     def __init__(self, agent: PolicyClient,
#                  robot: RCSRobot,
#                  instruction: str,
#         ):
        
#         self.agent = agent
#         self.robot = robot
#         self.instruction = instruction
        
#     def _get_actions_loop(self):
#         pass
#     def _actor_loop(self):
#         pass
#     def start(self):
#         # start two threads, one for getting actions from the agent and one for executing them on the robot
#         pass

def intialize_resource_manager():
    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:
        user, pw = load_creds_fr3_desk()
        resource_manager = FCI(
            Desk(cfg.robot_ip, user, pw), unlock=False, lock_when_done=False, guiding_mode_when_done=True
        )
    else:
        resource_manager = ContextManager()
    return resource_manager

def hardware_env_creator():
    env_c = RCSFR3EnvCreator()(
        ip=cfg.robot_ip,
        robot_cfg=default_fr3_hw_robot_cfg(async_control=True),
        control_mode=ControlMode.CARTESIAN_TQuat,
        gripper_cfg=default_fr3_hw_gripper_cfg(async_control=True),
        # urdf_path="/home/gamal/RobotControlStack/robot-control-stack/assets/scenes/fr3_empty_world/robot.urdf",
    )
    np.random.seed(cfg.n_experiment)
    c = PickUpDemo(env_c)
    # c.pickup_from_neutral()
    p = c.get_random_pose()
    c.place_cube(p)
    sleep(1)
    env_c.unwrapped.robot.stop_control_thread()
    env_c.unwrapped.robot.move_home()

    camera_dict = {
        "wrist": "230422272017",
        "side": "243122074917",
    }

    camera_set = HardwareCameraSet([default_realsense(camera_dict)]) if camera_dict is not None else None
    robot_cfg = default_fr3_hw_robot_cfg(async_control=True) # async
    robot_cfg.speed_factor = 0.2
    env = RCSFR3EnvCreator()(
        ip=cfg.robot_ip,
        camera_set=camera_set,
        robot_cfg=robot_cfg,
        control_mode=ControlMode.JOINTS, # openpi
        # control_mode=ControlMode.CARTESIAN_TRPY, # openvla / octo
        collision_guard=None,
        gripper_cfg=default_fr3_hw_gripper_cfg(async_control=True), # async
        # max_relative_movement=(0.1, np.deg2rad(15)),#45 # openvla / octo  # Not openpi
        # relative_to=RelativeTo.LAST_STEP, # openvla / octo # Not openpi
    )
    return env, camera_set

def sim_env_creator():
    env = SimEnvCreator()(
        control_mode=ControlMode.CARTESIAN_TRPY,
        robot_cfg=default_sim_robot_cfg(scene="fr3_empty_world"),
        collision_guard=False,
        gripper_cfg=default_sim_gripper_cfg(),
        cameras=default_mujoco_cameraset_cfg(),
        max_relative_movement=(0.5, np.deg2rad(45)),
        relative_to=RelativeTo.LAST_STEP,
    )
    env.get_wrapper_attr("sim").open_gui()
    return env, None

def main():

    resource_manager = intialize_resource_manager()

    with resource_manager:

        if ROBOT_INSTANCE == RobotPlatform.HARDWARE:
            env, camera_set = hardware_env_creator()

        else:
            env = sim_env_creator()

        video_path = Path(f"videos_experiment_{cfg.description.replace(':', '_')}")
        video_path.mkdir(parents=True, exist_ok=True)
        timestamp = f"experiment_{cfg.description}_{cfg.n_experiment}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if camera_set is not None:  
            camera_set.record_video(video_path, timestamp)
        
        robot_control = RCSRobot(env)
        agent = RCSPolicyClient(host=cfg.host, port=cfg.port, model=cfg.model, on_same_machine=False)
        controller = RTCController(agent, robot_control, instruction=INSTRUCTION)
        #input("robot is about to be controlled by AI, press enter to start")
        with env:
            controller.start()


if __name__ == "__main__":
    main()
# current venv:
# /home/gamal/RobotControlStack/rcs2/.venv311