
import logging
import math
import time
import traceback
from threading import Event, Thread
from typing import Optional

# Local, backend-agnostic modules
from .action_queue import ActionQueue          # NumPy version you already have
from .latency_tracker import LatencyTracker    # simple float tracker


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(threadName)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

rtc_logger = logging.getLogger("rtc.flow")
rtc_logger.setLevel(logging.INFO)


def log_rtc_step(tag: str, **kwargs):
    parts = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    rtc_logger.info(f"[RTC] {tag:<10s} | {parts}")


# ---------------------------------------------------------------------
# RTC Controller
# ---------------------------------------------------------------------
class RTCController:
    """
    Backend-agnostic RTC controller.
    Depends only on:
      - ActionQueue interface
      - LatencyTracker interface
      - policy.get_actions()
      - robot.get_observation(), robot.send_action()
    """

    def __init__(self, cfg, policy, robot):
        self.cfg = cfg
        self.policy = policy
        self.robot = robot

        self.queue = ActionQueue(cfg.rtc)
        self.latency_tracker = LatencyTracker()

        self.shutdown = Event()
        self.exc: Optional[Exception] = None

        H = cfg.horizon
        D = cfg.inference_delay
        E = cfg.rtc.execution_horizon

        if E > (H - D):
            raise ValueError(
                f"execution_horizon must be <= horizon - inference_delay "
                f"({E} <= {H - D})"
            )

        self.trigger_qsize = cfg.action_queue_size_to_get_new_actions
        self.time_per_tick = 1.0 / float(cfg.fps)

        log_rtc_step(
            "BOOT",
            horizon=H,
            exec_horizon=E,
            delay=D,
            trigger=self.trigger_qsize,
            fps=cfg.fps,
        )

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Producer thread
    # --------------------------------------------------
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

                # Estimated delay (for policy conditioning only)
                est_latency = self.latency_tracker.max()
                est_delay = (
                    math.ceil(est_latency / self.time_per_tick)
                    if est_latency > 0
                    else 0
                )

                obs = self.robot.get_observation()
                post, orig = self.policy.get_actions(
                    obs,
                    prev_chunk_left_over=prev_left,
                    inference_delay=est_delay,
                )

                # Measure real delay
                new_latency = time.perf_counter() - t0
                new_delay = math.ceil(new_latency / self.time_per_tick)
                self.latency_tracker.add(new_latency)

                log_rtc_step(
                    "INFER_DONE",
                    latency=f"{new_latency:.4f}s",
                    orig_len=int(orig.shape[0]),
                    post_len=int(post.shape[0]),
                    drop=new_delay,
                    est_delay=est_delay,
                )

                if self.trigger_qsize < self.cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions too small: "
                        "should be >= execution_horizon + inference_delay"
                    )

                # IMPORTANT: merge with REAL delay
                self.queue.merge(orig.detach().cpu().numpy(), post.detach().cpu().numpy(), new_delay, action_index)

                log_rtc_step(
                    "MERGE",
                    insert_at=action_index,
                    q_after=self.queue.qsize(),
                )

        except Exception as e:
            self._fail(e)

    # --------------------------------------------------
    # Consumer thread
    # --------------------------------------------------
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

                # torch OR numpy compatible
                action_out = action.cpu() if hasattr(action, "cpu") else action
                self.robot.send_action(action_out)
                exec_count += 1

                # logging-safe max
                if hasattr(action_out, "abs"):
                    v = action_out.abs().max()
                    v = v.item() if hasattr(v, "item") else float(v)
                else:
                    v = float(abs(action_out).max())

                log_rtc_step(
                    "EXEC",
                    exec_count=exec_count,
                    action_idx=self.queue.get_action_index(),
                    act_max=f"{v:.4f}",
                    q=self.queue.qsize(),
                )

        except Exception as e:
            self._fail(e)



def demo():
    from .lerobot_policy import RTCDemoConfig  # noqa: E402
    from .robot_interface import RobotWrapper  # noqa: E402
    from .robot import Robot_simulation  # noqa: E402
    from .lerobot_policy import LeRobotPolicy
    from lerobot.configs.policies import PreTrainedConfig

    policy_path = "/home/epez82ox/repos/pi0_droid_pytorch_29999"
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=policy_path)
    cfg = RTCDemoConfig(policy=policy_cfg)
    cfg.policy.pretrained_path = policy_path
    policy = LeRobotPolicy(cfg=cfg)
    robot = RobotWrapper(Robot_simulation(fps=cfg.fps))
    ctrl = RTCController(cfg, policy, robot)
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
    demo()