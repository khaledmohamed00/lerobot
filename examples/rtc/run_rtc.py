import logging
import math
import sys
import time
import traceback
from threading import Event, Thread
from typing import Dict
import os
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.utils import init_logging


# File location
current_file = Path(__file__).resolve()
# repo_root = parent of parent
repo_root = current_file.parents[2]
# Path you actually want to import from
rtc_examples_path = repo_root / "examples" / "rtc"
# Add once
if rtc_examples_path not in map(Path, sys.path):
    sys.path.append(str(rtc_examples_path))

from rtc_inference import RTCDemoConfig  # noqa: E402
from robot_interface import RobotWrapper  # noqa: E402
from robot import Robot_simulation  # noqa: E402
from rtc_inference import PI0_INFERENCE

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


def log_rtc_step(tag: str, **kwargs):
    # single-line, grep-friendly RTC logs
    parts = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    rtc_logger.info(f"[RTC] {tag:<10s} | {parts}")


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


# ---------------------------------------------------------------------
# Thread: get action chunks from policy
# ---------------------------------------------------------------------
def get_actions(
    policy: PI0_INFERENCE,
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    try:
        logger.info("[GET_ACTIONS] Thread started")

        latency_tracker = LatencyTracker()
        fps = cfg.fps
        time_per_tick = 1.0 / float(fps)

        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        # print get_actions_threshold for debugging in logger
        logger.info(f"[GET_ACTIONS] Action queue size threshold to get new actions: {get_actions_threshold}")
        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        log_rtc_step(
            "BOOT",
            rtc_enabled=cfg.rtc.enabled,
            horizon=cfg.rtc.execution_horizon,
            fps=fps,
            q_threshold=get_actions_threshold,
        )

        while not shutdown_event.is_set():
            qsize = action_queue.qsize()
            if qsize <= get_actions_threshold:
                # --- book-keeping
                t0 = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                prev_leftover_len = int(prev_actions.shape[0]) if prev_actions is not None else 0

                # --- delay estimation from observed inference latency
                inference_latency_est = latency_tracker.max()
                inference_delay = int(math.ceil(inference_latency_est / time_per_tick)) if inference_latency_est > 0 else 0

                log_rtc_step(
                    "REQ_INFER",
                    q=qsize,
                    action_idx=action_index_before_inference,
                    prev_leftover=prev_leftover_len,
                    latency_est=f"{inference_latency_est:.4f}s",
                    infer_delay=inference_delay,
                )

                # --- get obs and preprocess
                obs = robot.get_observation()
                rtc_logger.debug(f"[SIM] Observation fetched | keys={list(obs.keys())}")

                postprocessed_actions, original_actions = policy.get_actions(
                    obs,
                    prev_chunk_left_over=prev_actions,
                )
                # --- latency update
                new_latency = time.perf_counter() - t0
                new_delay = int(math.ceil(new_latency / time_per_tick))
                latency_tracker.add(new_latency)

                log_rtc_step(
                    "INFER_DONE",
                    latency=f"{new_latency:.4f}s",
                    new_delay=new_delay,
                    orig_len=int(original_actions.shape[0]),
                    post_len=int(postprocessed_actions.shape[0]),
                )

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions too small. "
                        "Should be >= execution_horizon + inference_delay."
                    )

                # --- merge into queue
                action_queue.merge(
                    original_actions,
                    postprocessed_actions,
                    new_delay,
                    action_index_before_inference,
                )

                log_rtc_step(
                    "MERGE",
                    insert_at=action_index_before_inference,
                    delay=new_delay,
                    q_after=action_queue.qsize(),
                )
            else:
                time.sleep(0.05)

        logger.info("[GET_ACTIONS] Thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


# ---------------------------------------------------------------------
# Thread: execute actions from queue
# ---------------------------------------------------------------------
def actor_control(
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    try:
        logger.info("[ACTOR] Thread started")

        action_count = 0
        action_interval = 1.0 / float(cfg.fps)

        while not shutdown_event.is_set():
            tick_t0 = time.perf_counter()

            action = action_queue.get()
            if action is not None:
                action_cpu = action.cpu()
                robot.send_action(action_cpu)
                sleep_time = 1.0 / float(cfg.fps)
                rtc_logger.debug(
                    f"[SIM] Action received | shape={tuple(action.shape)} | max={action.abs().max().item():.4f} | sleep={sleep_time:.4f}"
                )
                action_count += 1

                # Log “truth”: what got executed
                # get_action_index() is global index AFTER popping; still useful as a running counter.
                log_rtc_step(
                    "EXEC",
                    exec_count=action_count,
                    action_idx=action_queue.get_action_index(),
                    act_max=f"{action_cpu.abs().max().item():.4f}",
                    act_mean=f"{action_cpu.mean().item():.4f}",
                    q=action_queue.qsize(),
                )

            dt = time.perf_counter() - tick_t0
            time.sleep(max(0.0, (action_interval - dt) - 0.001))

        logger.info(f"[ACTOR] Thread shutting down | total_executed={action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    init_logging()
    logger.info(f"[MAIN] Using device: {cfg.device}")
    logger.info(f"[MAIN] RTC enabled: {cfg.rtc.enabled}")
    logger.info(f"[MAIN] FPS: {cfg.fps}")
    logger.info(f"[MAIN] Duration: {cfg.duration}s")
    logger.info(f"[MAIN] Execution horizon: {cfg.rtc.execution_horizon} steps")
    logger.info(f"[MAIN] Inference delay: {cfg.inference_delay} steps")
    # graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # policy
    policy = PI0_INFERENCE(cfg=cfg)
    assert policy.policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    logger.info(f"[MAIN] Policy ready | name={policy.policy.name}")
    robot = Robot_simulation(fps=cfg.fps)
    logger.info("[MAIN] Robot simulation initialized")
    robot_wrapper = RobotWrapper(robot)
    logger.info("[MAIN] Robot simulation wrapper ready")
    # action queue
    action_queue = ActionQueue(cfg.rtc)

    # threads
    t_get = Thread(
        target=get_actions,
        args=(policy, robot_wrapper, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    t_act = Thread(
        target=actor_control,
        args=(robot_wrapper, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )

    t_get.start()
    logger.info("[MAIN] Started get actions thread")
    t_act.start()
    logger.info("[MAIN] Started actor thread")

    logger.info(f"[MAIN] Running demo for {cfg.duration} seconds...")
    start_time = time.time()
    last_heartbeat = 0.0

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(0.2)
        elapsed = time.time() - start_time
        if elapsed - last_heartbeat >= 5.0:
            last_heartbeat = elapsed
            logger.info(
                "[MAIN] Heartbeat | elapsed=%.1fs | qsize=%d | action_index=%d",
                elapsed,
                action_queue.qsize(),
                action_queue.get_action_index(),
            )

    logger.info("[MAIN] Duration reached or shutdown requested -> stopping")
    shutdown_event.set()

    if t_get.is_alive():
        logger.info("[MAIN] Joining get actions thread")
        t_get.join()
    if t_act.is_alive():
        logger.info("[MAIN] Joining actor thread")
        t_act.join()

    logger.info("[MAIN] Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")

