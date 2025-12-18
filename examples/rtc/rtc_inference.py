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


class pi0_inference:
    def __init__(self, pretrained_policy_path: str,
                 rtc_enabled: bool =  True,
                 execution_horizon: int = 10,
                 inference_delay: int = 4,
                 max_guidance_weight: float = 10.0,
                 rtc_debug: bool = False,
                 device: str | None = "auto",
                 attention_schedule: str = "exp",
                 use_torch_compile: bool = False,
                 ):

        self.pretrained_policy_path = pretrained_policy_path
        self.rtc_enabled = rtc_enabled
        self.inference_delay = inference_delay
        self.execution_horizon = execution_horizon
        self.device = device
        self.prev_chunk_left_over = None

        if attention_schedule.lower() == "exp":
            self.attention_schedule = RTCAttentionSchedule.EXP
        elif attention_schedule.lower() == "linear":
            self.attention_schedule = RTCAttentionSchedule.LINEAR
        else:
            raise ValueError(f"Unknown attention schedule: {attention_schedule}")
        
        # Auto-detect device if not specified
        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        policy_cfg = PreTrainedConfig.from_pretrained(self.pretrained_policy_path)
        policy_cfg.pretrained_path = self.pretrained_policy_path
        self.cfg = RTCEvalConfig(
            policy=policy_cfg,
            use_torch_compile=use_torch_compile,
            #dataset=DatasetConfig(),
            rtc=RTCConfig(
                enabled=self.rtc_enabled,
                execution_horizon=self.execution_horizon,
                max_guidance_weight=max_guidance_weight,
                prefix_attention_schedule=self.attention_schedule,
                debug=rtc_debug,
            ),
            device=self.device,
        )
        
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

        config = PreTrainedConfig.from_pretrained(self.cfg.policy.pretrained_path)

        if self.cfg.policy.type == "pi05" or self.cfg.policy.type == "pi0":
            config.compile_model = self.cfg.use_torch_compile

        policy = policy_class.from_pretrained(self.cfg.policy.pretrained_path, config=config)
        policy = policy.to(self.device)
        policy.eval()

        # Configure RTC
        rtc_config = RTCConfig(
            enabled=self.cfg.rtc.enabled,
            execution_horizon=self.cfg.rtc.execution_horizon,
            max_guidance_weight=self.cfg.rtc.max_guidance_weight,
            prefix_attention_schedule=self.cfg.rtc.prefix_attention_schedule,
            debug=self.cfg.rtc.debug,
            debug_maxlen=self.cfg.rtc.debug_maxlen,
        )
        policy.config.rtc_config = rtc_config
        policy.init_rtc_processor()

        logging.info(f"  RTC enabled: {self.cfg.rtc.enabled}")
        logging.info(f"  RTC debug: {self.cfg.rtc.debug}")
        logging.info(f"  Policy config: {config}")

        # Apply torch.compile to predict_action_chunk method if enabled
        if self.cfg.use_torch_compile:
            policy = self._apply_torch_compile(policy, name)

        logging.info(f"✓ {name} initialized successfully")

        # Apply torch.compile to predict_action_chunk method if enabled
        if self.cfg.use_torch_compile:
            policy = self._apply_torch_compile(policy, name)

        logging.info(f"✓ {name} initialized successfully")
        return policy

    def _apply_torch_compile(self, policy, policy_name: str):
        """Apply torch.compile to the policy's predict_action_chunk method.

        Args:
            policy: Policy instance to compile
            policy_name: Name for logging purposes

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
                    f"  [{policy_name}] torch.compile is not available. Requires PyTorch 2.0+. "
                    f"Current version: {torch.__version__}. Skipping compilation."
                )
                return policy

            logging.info(f"  [{policy_name}] Applying torch.compile to predict_action_chunk...")
            logging.info(f"    Backend: {self.cfg.torch_compile_backend}")
            logging.info(f"    Mode: {self.cfg.torch_compile_mode}")
            logging.info(f"    Disable CUDA graphs: {self.cfg.torch_compile_disable_cudagraphs}")
            logging.info("    Note: Debug tracker excluded from compilation via @torch._dynamo.disable")

            # Compile the predict_action_chunk method
            # - Debug tracker is excluded from compilation via @torch._dynamo.disable
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
            logging.info(f"  ✓ [{policy_name}] Successfully compiled predict_action_chunk")

        except Exception as e:
            logging.error(f"  [{policy_name}] Failed to apply torch.compile: {e}")
            logging.warning(f"  [{policy_name}] Continuing without torch.compile")

        return policy

    def _destroy_policy(self, policy, policy_name: str):
        """Explicitly destroy a policy and free all associated memory.

        This method performs aggressive cleanup to ensure maximum memory is freed,
        which is critical for large models (e.g., VLAs with billions of parameters).

        Args:
            policy: Policy instance to destroy
            policy_name: Name for logging purposes
        """
        logging.info(f"  Destroying {policy_name} and freeing memory...")

        try:
            # Step 1: Move policy to CPU to free GPU/MPS memory
            policy.cpu()

            # Step 2: Delete the policy object
            del policy

            # Step 3: Force garbage collection to reclaim memory immediately
            gc.collect()

            # Step 4: Clear device-specific caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            logging.info(f"  ✓ {policy_name} destroyed and memory freed")

        except Exception as e:
            logging.warning(f"  Warning: Error during {policy_name} cleanup: {e}")


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

            actions = pi0_inf.get_actions(preprocessed_obs)
            print("Predicted actions shape:", actions.shape)

if __name__ == "__main__":
    pretrained_policy_path = "/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999"
    pi0_inf = pi0_inference(
        pretrained_policy_path=pretrained_policy_path,
        rtc_enabled=True,
        execution_horizon=10,
        inference_delay=4,
        max_guidance_weight=10.0,
        rtc_debug=True,
        device="auto",
        attention_schedule="exp",
        use_torch_compile=False,
    )
    
    pi0_inf.run_inference()
