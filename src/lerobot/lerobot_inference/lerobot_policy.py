import logging
from dataclasses import dataclass, field
from typing import Tuple, Mapping, Any
from abc import ABC, abstractmethod

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.hub import HubMixin

from .Policy_interface import PolicyInterface


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 300.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = field(
        default=36,
        metadata={"help": "Number of actions to execute before requesting new actions"})

    horizon : int = field(default=50, metadata={"help": "Action Chunk size"})
    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})
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
        elif self.policy and self.policy.pretrained_path:
            pass
        else:
            raise ValueError("Policy path is required")
        self.action_queue_size_to_get_new_actions: int = self.horizon - (self.rtc.execution_horizon + self.inference_delay)

        # Validate that robot configuration is provided
        # if self.robot is None:
        #     raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]




class LeRobotPolicy(PolicyInterface):
    def __init__(self, 
                 cfg: RTCDemoConfig | None = None,
                 ):

        self.cfg = cfg
        self.pretrained_policy_path = self.cfg.policy.pretrained_path if cfg else None
        self.rtc_enabled = self.cfg.rtc.enabled if cfg else True
        self.inference_delay = self.cfg.inference_delay if cfg else 4
        self.execution_horizon = self.cfg.rtc.execution_horizon if cfg else 20
        self.device = self.cfg.device if cfg else "auto"
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
        # Turn on RTC
        policy.config.rtc_config = self.cfg.rtc
        policy.init_rtc_processor()
        # Apply torch.compile if enabled
        if self.cfg.use_torch_compile:
            policy = self._apply_torch_compile(policy)
        logging.info(f"  RTC enabled: {self.rtc_enabled}")
        logging.info(f"  RTC debug: {self.cfg.rtc.debug}")
        logging.info(f"  Policy config: {self.cfg.policy}")

        logging.info(f"✓ {name} initialized successfully")

        return policy

    def _apply_torch_compile(self, policy):
        """Apply torch.compile to the policy's predict_action_chunk method.

        Args:
            policy: Policy instance to compile

        Returns:
            Policy with compiled predict_action_chunk method
        """

        # PI models handle their own compilation
        if policy.type == "pi05" or policy.type == "pi0":
            return policy

        try:
            # Check if torch.compile is available (PyTorch 2.0+)
            if not hasattr(torch, "compile"):
                logging.warning(
                    f"torch.compile is not available. Requires PyTorch 2.0+. "
                    f"Current version: {torch.__version__}. Skipping compilation."
                )
                return policy

            logging.info("Applying torch.compile to predict_action_chunk...")
            logging.info(f"  Backend: {self.cfg.torch_compile_backend}")
            logging.info(f"  Mode: {self.cfg.torch_compile_mode}")
            logging.info(f"  Disable CUDA graphs: {self.cfg.torch_compile_disable_cudagraphs}")

            # Compile the predict_action_chunk method
            # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
            compile_kwargs = {
                "backend": self.cfg.torch_compile_backend,
                "mode": self.cfg.torch_compile_mode,
            }

            # Disable CUDA graphs if requested (prevents tensor aliasing issues)
            if self.cfg.torch_compile_disable_cudagraphs:
                compile_kwargs["options"] = {"triton.cudagraphs": False}

            original_method = policy.predict_action_chunk
            compiled_method = torch.compile(original_method, **compile_kwargs)
            policy.predict_action_chunk = compiled_method
            logging.info("✓ Successfully compiled predict_action_chunk")

        except Exception as e:
            logging.error(f"Failed to apply torch.compile: {e}")
            logging.warning("Continuing without torch.compile")
        return policy

    def infer(self, obs: dict[str, torch.Tensor], prev_chunk_left_over: torch.Tensor | None, inference_delay: int=4) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with RTC enabled policy.

        Args:
            obs: Observations for the policy
            prev_chunk_left_over: Leftover actions from previous chunk
            inference_delay: Inference delay in steps

        Returns:
            Actions predicted by the policy
        """

        with torch.no_grad():
            preprocessed_obs = self.preprocessor(obs)
            if self.rtc_enabled:
                original_actions = self.policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_chunk_left_over,
                )
            else:
                original_actions = self.policy.predict_action_chunk(preprocessed_obs)
            actions = self.postprocessor(original_actions)
        actions = actions.squeeze(0).detach().cpu().numpy()
        original_actions = original_actions.squeeze(0).detach().cpu().numpy()
        return actions, original_actions

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

            actions = self.infer(obs)
            print("Predicted actions shape:", actions.shape)

@parser.wrap()
def main(cfg: RTCDemoConfig):
    pi_inf = LeRobotPolicy(cfg=cfg)
    pi_inf.run_inference()

if __name__ == "__main__":
    # pretrained_policy_path = "/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999"
    #--policy.path="/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999" \

    main()
