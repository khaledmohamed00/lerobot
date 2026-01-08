from lerobot.lerobot_inference.rtc_controller import RTCDemoConfig, LeRobotPolicy
from lerobot.configs.policies import PreTrainedConfig

if __name__ == "__main__":
    policy_path = "/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999"
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=policy_path)
    cfg = RTCDemoConfig(policy=policy_cfg)
    cfg.policy.pretrained_path = policy_path
    policy = LeRobotPolicy(cfg=cfg)
    print("Policy initialized with config:", cfg)
# from lerobot.lerobot_inference import main
# if __name__ == "__main__":
#     main.demo_cli()