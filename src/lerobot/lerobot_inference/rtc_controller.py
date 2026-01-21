import logging
import math
import time
import traceback
from threading import Event, Thread
from dataclasses import dataclass

import numpy as np

from .robot_interface import RobotWrapper  
from .robot import Robot_simulation  
from .lerobot_policy import LeRobotPolicy
from .agent_interface import PolicyAgent
from .agent_interface import PolicyClient
from .client_agent_policy import SimPolicyClient
from .action_queue import ActionQueue
from .latency_tracker import LatencyTracker
# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(threadName)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Dedicated flow logger (turn to DEBUG when needed)
rtc_logger = logging.getLogger("rtc.flow")
rtc_logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = True
    execution_horizon: int = 10

@dataclass(frozen=False)
class SimpleDemoConfig:
    # Fields that affect behavior in THIS script
    fps: int = 10
    duration: float = 120.0
    inference_delay: int = 4
    rtc: RTCConfig = RTCConfig()
    horizon: int = 50
    action_queue_size_to_get_new_actions: int = 36

    def __post_init__(self):
        self.action_queue_size_to_get_new_actions = self.horizon - (self.rtc.execution_horizon + self.inference_delay)


def log_rtc_step(tag: str, **kwargs):
    # single-line, grep-friendly RTC logs
    parts = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    rtc_logger.info(f"[RTC] {tag:<10s} | {parts}")

class RTCController:
    def __init__(self, cfg: SimpleDemoConfig, policy_agent: PolicyAgent, robot: RobotWrapper):
        self.cfg = cfg
        self.policy_agent = policy_agent
        self.robot = robot
        self.queue = ActionQueue(self.cfg.rtc) # Thread-safe action queue
        self.latency_tracker = LatencyTracker()  # Track latency of action chunks

        self.shutdown = Event()
        self.exc: Exception | None = None

        H, D, E = self.cfg.horizon, self.cfg.inference_delay, self.cfg.rtc.execution_horizon
        if E > (H - D):
            raise ValueError(
                f"execution_horizon must be <= horizon - inference_delay "
                f"({E} <= {H - D})"
            )

        self.trigger_qsize = self.cfg.action_queue_size_to_get_new_actions
        self.time_per_tick = 1.0 / float(self.cfg.fps)

        log_rtc_step(
            "BOOT",
            horizon=H,
            exec_horizon=E,
            delay=D,
            trigger=self.trigger_qsize,
            fps=self.cfg.fps,
        )

    def start(self):
        self.t_get = Thread(target=self._get_actions_loop, name="GetActions", daemon=True)
        self.t_act = Thread(target=self._actor_loop, name="Actor", daemon=True)
        self.t_get.start()
        self.t_act.start()

    def stop(self):
        self.shutdown.set()
        self.t_get.join()
        self.t_act.join()
        if self.exc:
            raise self.exc

    def _fail(self, e: Exception):
        self.exc = e
        logger.error("Worker crashed: %s", e)
        logger.error(traceback.format_exc())
        self.shutdown.set()

    def _get_actions_loop(self):
        try:
            while not self.shutdown.is_set():
                qsize = self.queue.qsize()

                if qsize > self.trigger_qsize:
                    time.sleep(0.01)
                    continue

                t0 = time.perf_counter()
                action_index = self.queue.get_action_index()
                prev_left = self.queue.get_left_over()
                prev_len = int(prev_left.shape[0]) if prev_left is not None else 0

                log_rtc_step(
                    "REQ_INFER",
                    q=qsize,
                    action_idx=action_index,
                    prev_leftover=prev_len,
                )

                inference_latency = self.latency_tracker.max()  # seconds
                inference_delay = math.ceil(inference_latency / self.time_per_tick)
                obs = self.robot.get_observation()
                out = self.policy_agent.act(
                    obs, prev_chunk_left_over=prev_left, inference_delay=inference_delay
                )
                if isinstance(self.policy_agent.policy, PolicyClient):
                    post = out.action  # numpy [T, action_dim]
                    orig = out.original_action  # numpy [T, action_dim]
                elif isinstance(self.policy_agent.policy, LeRobotPolicy):
                    post, orig = out  # both torch [T, action_dim]
                new_latency = time.perf_counter() - t0
                new_delay = math.ceil(new_latency / self.time_per_tick)
                self.latency_tracker.add(new_latency)
                
                log_rtc_step(
                    "INFER_DONE",
                    latency=f"{new_latency:.4f}s",
                    orig_len=int(orig.shape[0]),
                    post_len=int(post.shape[0]),
                    drop=inference_delay,
                )

                if self.cfg.action_queue_size_to_get_new_actions < self.cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[act] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                self.queue.merge(orig, post, new_delay, action_index)

                log_rtc_step(
                    "MERGE",
                    insert_at=action_index,
                    q_after=self.queue.qsize(),
                )

        except Exception as e:
            self._fail(e)

    def _actor_loop(self):
        try:
            interval = 1.0 / float(self.cfg.fps)
            next_tick = time.perf_counter()
            exec_count = 0

            while not self.shutdown.is_set():
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick += interval

                action = self.queue.get()
                if action is None:
                    continue

                self.robot.send_action(action)
                exec_count += 1

                log_rtc_step(
                    "EXEC",
                    exec_count=exec_count,
                    action_idx=self.queue.get_action_index(),
                    act_max=f"{np.abs(action).max():.4f}",
                    q=self.queue.qsize(),
                )

        except Exception as e:
            self._fail(e)

# @parser.wrap()
# def demo_cli(cfg: RTCDemoConfig):
def demo_cli():
    cfg = SimpleDemoConfig()
    robot = RobotWrapper(Robot_simulation(fps=cfg.fps))

    port = 20997
    local = False
    is_agent = True
    model = "lerobot_pi"
    if local == True:
    # test local connection
        host = "localhost"
        on_same_machine = True
        if is_agent == True:
            policy_client = SimPolicyClient(host=host, port=port, model=model, on_same_machine=on_same_machine)
            obs = robot.get_observation()
            obs_converted = policy_client.get_obs(obs)
            instruction = "pick the green box"
            policy_client.reset(obs=obs_converted, instruction=instruction)
        else:
            from .lerobot_policy import RTCDemoConfig, LeRobotPolicy
            from lerobot.configs.policies import PreTrainedConfig
            default_checkpoint_path = "/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999"
            policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=default_checkpoint_path)
            lerobot_cfg = RTCDemoConfig(policy=policy_cfg)
            lerobot_cfg.policy.pretrained_path = default_checkpoint_path
            print("Initializing LeRobotPolicy with path:", default_checkpoint_path)
            print("default_checkpoint_path:", default_checkpoint_path)
            policy_client = LeRobotPolicy(cfg=lerobot_cfg)
            print("LeRobotPolicy initialized")
            # pass
    else:
    # test remote connection
        #host = "airtower.utn-mi.de"
        host = "localhost"

        on_same_machine = False
        policy_client = SimPolicyClient(host=host, port=port, model=model, on_same_machine=on_same_machine)
        obs = robot.get_observation()
        obs_converted = policy_client.get_obs(obs)
        instruction = "pick the green box"
        policy_client.reset(obs=obs_converted, instruction=instruction)
    
    policy_agent = PolicyAgent(policy=policy_client)
    ctrl = RTCController(cfg, policy_agent, robot)
    ctrl.start()

    start = time.time()
    last = 0.0

    while (time.time() - start) < cfg.duration and not ctrl.shutdown.is_set():
        time.sleep(0.2)
        elapsed = time.time() - start
        if elapsed - last >= 5.0:
            last = elapsed
            logger.info(
                "[MAIN] Heartbeat | elapsed=%.1fs | q=%d | idx=%d",
                elapsed,
                ctrl.queue.qsize(),
                ctrl.queue.get_action_index(),
            )

    ctrl.stop()

if __name__ == "__main__":
    demo_cli()