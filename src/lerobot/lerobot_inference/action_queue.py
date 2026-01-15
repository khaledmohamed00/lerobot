import logging
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)



class ActionQueue:
    """
    Thread-safe queue for managing action chunks in real-time control.

    This is a NumPy-only replacement for the LeRobot/Torch ActionQueue.
    It stores:
      - queue: processed actions to execute (T, A)
      - original_queue: original actions for leftover computation (T, A)

    API mirrors your original class.
    """

    def __init__(self, enable_rtc: bool = True):
        self.queue: Optional[np.ndarray] = None
        self.original_queue: Optional[np.ndarray] = None
        self.lock = Lock()
        self.last_index = 0
        self.enable_rtc = enable_rtc

    def get(self) -> Optional[np.ndarray]:
        """Return next action (A,) or None. Returns a copy to prevent mutation."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        """Number of remaining unconsumed actions."""
        with self.lock:
            if self.queue is None:
                return 0
            return int(len(self.queue) - self.last_index)

    def empty(self) -> bool:
        """True if queue is empty."""
        return self.qsize() <= 0

    def get_action_index(self) -> int:
        """Index of next action to be consumed."""
        with self.lock:
            return int(self.last_index)

    def get_left_over(self) -> Optional[np.ndarray]:
        """Return remaining original actions (remaining_steps, action_dim) or None."""
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].copy()

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: Optional[int] = 0,
    ) -> None:
        """
        Merge new actions into the queue.

        RTC enabled: replace queue and drop first `real_delay`.
        RTC disabled: append and maintain continuity.
        """
        # Basic shape sanity (optional but helpful)
        if original_actions.ndim != 2 or processed_actions.ndim != 2:
            raise ValueError("original_actions and processed_actions must be 2D arrays (T, A)")
        if original_actions.shape != processed_actions.shape:
            raise ValueError("original_actions and processed_actions must have the same shape")
        if real_delay < 0:
            raise ValueError("real_delay must be >= 0")

        with self.lock:
            self._check_delays(real_delay, action_index_before_inference)

            if self.enable_rtc:
                self._replace_actions_queue(original_actions, processed_actions, real_delay)
            else:
                self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int) -> None:
        """
        Replace queue with new actions (RTC mode), dropping first `real_delay`.
        Prevent empty-queue edge case by clamping delay to [0, T].
        """
        T = int(processed_actions.shape[0])

        # Clamp delay to avoid slicing past end; empty queue is allowed but often undesirable.
        rd = max(0, min(int(real_delay), T))

        self.original_queue = original_actions[rd:].copy()
        self.queue = processed_actions[rd:].copy()
        self.last_index = 0

        logger.debug("original_actions shape: %s", self.original_queue.shape)
        logger.debug("processed_actions shape: %s", self.queue.shape)
        logger.debug("real_delay: %d (clamped to %d)", real_delay, rd)

    def _append_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray) -> None:
        """
        Append new actions to the queue (non-RTC mode).
        Drops already-consumed prefix, then resets last_index.
        """
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            self.last_index = 0
            return

        # Concatenate
        self.original_queue = np.concatenate([self.original_queue, original_actions], axis=0)
        self.queue = np.concatenate([self.queue, processed_actions], axis=0)

        # Drop consumed prefix
        li = int(self.last_index)
        if li > 0:
            self.original_queue = self.original_queue[li:].copy()
            self.queue = self.queue[li:].copy()

        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: Optional[int] = None) -> None:
        """
        Compare actions consumed during inference (index diff) vs real_delay.
        This mirrors your original warning logic.
        """
        if action_index_before_inference is None:
            return

        indexes_diff = int(self.last_index - int(action_index_before_inference))
        if indexes_diff != int(real_delay):
            logger.warning(
                "[ACTION_QUEUE] Indexes diff != real_delay | indexes_diff=%d real_delay=%d",
                indexes_diff,
                int(real_delay),
            )

