import gc
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

@dataclass
class RTCEvalConfig(HubMixin):
    """Configuration for RTC evaluation."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Dataset configuration
    #dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=20,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            debug=True,
            debug_maxlen=1000,
        )
    )

    # Device configuration
    device: str | None = field(
        default=None,
        metadata={"help": "Device to run on (cuda, cpu, mps, auto)"},
    )

    # Output configuration
    output_dir: str = field(
        default="rtc_debug_output",
        metadata={"help": "Directory to save debug visualizations"},
    )

    # Seed configuration
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )

    inference_delay: int = field(
        default=4,
        metadata={"help": "Inference delay for RTC"},
    )

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")


    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class pi0_inference:
    def __init__(self, 
                 cfg: RTCEvalConfig | None = None,
                 ):

        self.cfg = cfg
        self.pretrained_policy_path = self.cfg.policy.pretrained_path if cfg else None
        self.rtc_enabled = self.cfg.rtc.enabled if cfg else True
        self.inference_delay = self.cfg.inference_delay if cfg else 4
        self.execution_horizon = self.cfg.rtc.execution_horizon if cfg else 20
        self.device = self.cfg.device if cfg else "auto"
        self.prev_chunk_left_over = None
        if self.cfg.rtc.prefix_attention_schedule.lower() == "exp":
            self.attention_schedule = RTCAttentionSchedule.EXP
        elif self.cfg.rtc.prefix_attention_schedule.lower() == "linear":
            self.attention_schedule = RTCAttentionSchedule.LINEAR
        else:
            raise ValueError(f"Unknown attention schedule: {self.cfg.rtc.prefix_attention_schedule}")

        # Auto-detect device if not specified
        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.cfg.device = self.device
        self.cfg.policy.device = self.device

        self.policy = self._init_policy("Policy")
        # Turn on RTC
        self.policy.config.rtc_config = self.cfg.rtc
        self.policy.init_rtc_processor()

        # Create preprocessor/postprocessor
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg.policy,
            pretrained_path=self.cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

    def _init_policy(self, name: str):
        """Initialize a single policy instance with specified RTC configuration.

        Args:
            name: Name identifier for logging purposes

        Returns:
            Configured policy instance with optional torch.compile applied
        """
        logging.info(f"Initializing {name}...")

        # Load policy from pretrained
        policy_class = get_policy_class(self.cfg.policy.type)
        self.cfg.policy.compile_model = self.cfg.use_torch_compile
        policy = policy_class.from_pretrained(self.cfg.policy.pretrained_path, config=self.cfg.policy)
        policy = policy.to(self.device)
        policy.eval()

        policy.init_rtc_processor()

        logging.info(f"  RTC enabled: {self.cfg.rtc.enabled}")
        logging.info(f"  RTC debug: {self.cfg.rtc.debug}")
        logging.info(f"  Policy config: {self.cfg.policy}")

        logging.info(f"✓ {name} initialized successfully")
        
        return policy

    def get_actions(self, obs):
        """Run inference with RTC enabled policy.

        Args:
            obs: Observations for the policy

        Returns:
            Actions predicted by the policy
        """
        with torch.no_grad():
            if self.rtc_enabled:
                actions = self.policy.predict_action_chunk(
                    obs,
                    inference_delay=self.inference_delay,
                    prev_chunk_left_over=self.prev_chunk_left_over,
                )
                self.prev_chunk_left_over = actions[:,self.execution_horizon:,:]
            else:
                actions = self.policy.predict_action_chunk(obs)
        return actions
    
    def run_inference(self):
        """Run inference with RTC enabled policy.

        Args:
            obs: Observations for the policy

        Returns:
            Actions predicted by the policy
        """
        dataset_repo_id = "HuggingFaceVLA/libero"
        ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
        # Calculate delta_timestamps from policy's delta_indices
        delta_timestamps = resolve_delta_timestamps(self.cfg.policy, ds_meta)

        # Create dataset with calculated delta_timestamps
        self.dataset = LeRobotDataset(
           dataset_repo_id,
            delta_timestamps=delta_timestamps,
        )
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        loader_iter = iter(data_loader)

        for idx in range(10):  # Run for 10 batches
            try:
                obs = next(loader_iter)
            except StopIteration:
                break

            preprocessed_obs = self.preprocessor(obs)

            actions = self.get_actions(preprocessed_obs)
            print("Predicted actions shape:", actions.shape)

@parser.wrap()
def main(cfg: RTCEvalConfig):
    pi0_inf = pi0_inference(cfg=cfg)
    pi0_inf.run_inference()

if __name__ == "__main__":
    # pretrained_policy_path = "/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999"
    #--policy.path="/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999" \

    main()
