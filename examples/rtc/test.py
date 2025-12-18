# from lerobot.policies.pi0.modeling_pi0 import PI0Policy
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained_policy_path = "/home/gamal/pi0_droid_pytorch_29999"

# policy = PI0Policy.from_pretrained(pretrained_policy_path).to(device)

# # load pytorch model weights using pytorch
# #policy.load_state_dict(torch.load(f"{pretrained_policy_path}/model.safetensors", map_location=device))
# # Alternatively, you can load the entire model directly

# # model = torch.load(f"{pretrained_policy_path}/model.safetensors",
# #                    map_location=device,
# #                    weights_only=False)
# print()

from lerobot.policies.pi0 import PI0Policy, PI0Config
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.action_queue import ActionQueue

# Load Pi0 with RTC enabled
policy_cfg = PI0Config()

# Enable RTC
policy_cfg.rtc_config = RTCConfig(
    enabled=True,
    execution_horizon=10,  # How many steps to blend with previous chunk
    max_guidance_weight=10.0,  # How strongly to enforce consistency
    prefix_attention_schedule=RTCAttentionSchedule.EXP,  # Exponential blend
)
pretrained_policy_path = "/home/gamal/pi0_droid_pytorch_29999"

# Load the policy
policy = PI0Policy.from_pretrained(pretrained_policy_path, config=policy_cfg)

# Now use predict_action_chunk with RTC parameters
inference_delay = 4  # How many steps of inference latency, this values should be calculated based on the inference latency of the policy

# Initialize the action queue
action_queue = ActionQueue(policy_cfg.rtc_config)

# # Start in a separate thread with the following function
# def get_actions():
#   while True:
#     if should_get_actions:

#       prev_actions = action_queue.get_left_over()
#       obs = get_robot_observations(robot)

#       # Generate actions WITH RTC
#       actions = policy.predict_action_chunk(
#           obs,
#           inference_delay=inference_delay,
#           prev_chunk_left_over=prev_actions,
#       )

#       action_queue.merge(
#           actions, actions, inference_delay
#       )

# for step in range(num_steps):
#     action = action_queue.get()

#     # Execute the first N actions
#     execute_actions(action)