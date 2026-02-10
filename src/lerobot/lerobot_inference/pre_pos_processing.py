import torch
import json
# import library responsible for loading safetensor model
from safetensors.torch import load_file as load_safetensors
STATE_SPACE = 8
# load checkpoint torch safetensor file
def load_checkpoint_safetensor(checkpoint_path: str, device: str = 'cpu') -> dict:
    """
    Load a model checkpoint from a safetensor file.

    Args:
        checkpoint_path (str): Path to the safetensor checkpoint file.
        device (str): Device to map the loaded tensors to. Default is 'cpu'.

    Returns:
        dict: A dictionary containing the model state.
    """
    # Load the safetensor file
    state_dict = load_safetensors(checkpoint_path, device=device)
    return state_dict

# load json config file
def load_json_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def compare_states(state1: dict, state2: dict) -> None:
    """
    Compare two state dictionaries and print the keys that have equal values.

    Args:
        state1 (dict): First state dictionary.
        state2 (dict): Second state dictionary.
    """
    for key in state1.keys():
        if key in state2 and torch.equal(state1[key], state2[key]):
            print(f"Key '{key}' has equal values in both states.")
        else:
            print(f"Key '{key}' differs between states or is missing in the second state.")

def overwrite_state_from_json(state: dict, json_config: dict) -> dict:
    """
    Overwrite the state dictionary values with those from the JSON config if keys match.

    Args:
        state (dict): The original state dictionary.
        json_config (dict): The JSON configuration dictionary.
    Returns:
        dict: The updated state dictionary.
    """
    # example 
    #json_config['norm_stats']['state']['mean'] = state['observation.state.mean']
    #json_config['norm_stats']['actions']['mean'] = state['action.mean']
    # state keys to overwrite are:
    # mean
    # std
    # q01
    # q99
    # Overwrite the state dictionary values with mean and std from the JSON config
    keys = ['mean', 'std', 'q01', 'q99']
    for key in keys:
        state[f'observation.state.{key}'] = torch.tensor(json_config['norm_stats']['state'][key])[:STATE_SPACE]
        state[f'action.{key}'] = torch.tensor(json_config['norm_stats']['actions'][key])[:STATE_SPACE]
    return state

def check_after_overwrite(state: dict, json_config: dict) -> None:
    """
    Check the state dictionary after overwriting with the JSON config.
    Check if the values match those in the JSON config.

    Args:
        state (dict): The state dictionary.
        json_config (dict): The JSON configuration dictionary.
    """
    # torch.equal cannot compare tensors with different shapes
    keys = ['mean', 'std', 'q01', 'q99']
    for key in keys:
        state_value = state[f'observation.state.{key}']
        json_value = torch.tensor(json_config['norm_stats']['state'][key])
        if torch.equal(state_value, json_value):
            print(f"observation.state.{key} matches the JSON config.")
        else:
            print(f"observation.state.{key} does not match the JSON config.")

        state_value = state[f'action.{key}']
        json_value = torch.tensor(json_config['norm_stats']['actions'][key])
        if torch.equal(state_value, json_value):
            print(f"action.{key} matches the JSON config.")
        else:
            print(f"action.{key} does not match the JSON config.")

def save_checkpoint_safetensor(state: dict, save_path: str) -> None:
    """
    Save a model state dictionary to a safetensor file.

    Args:
        state (dict): The model state dictionary.
        save_path (str): Path to save the safetensor file.
    """
    from safetensors.torch import save_file as save_safetensors
    save_safetensors(state, save_path)

if __name__ == "__main__":
    # Example usage
    root_dir = "/home/gamal/pi0_fintuned/pi0_droid_pytorch_29999/"
    preprocessing_checkpoint_path = f"{root_dir}/backup_pre_post/policy_preprocessor_step_5_normalizer_processor.safetensors"
    postprocessing_checkpoint_path = f"{root_dir}/backup_pre_post/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    config_path = f"{root_dir}/norm_stats.json"

    preprocessing_state = load_checkpoint_safetensor(preprocessing_checkpoint_path, device='cpu')
    postprocessing_state = load_checkpoint_safetensor(postprocessing_checkpoint_path, device='cpu')
    config = load_json_config(config_path)

    # compare_states(preprocessing_state, postprocessing_state)
    # collect all key that contain "state" or "action" in their name
    preprocessing_keys = [key for key in preprocessing_state.keys() if "state" in key or "action" in key]
    postprocessing_keys = [key for key in postprocessing_state.keys() if "state" in key or "action" in key]
    #print("Preprocessing State Keys:", preprocessing_keys)
    #print("Postprocessing State Keys:", postprocessing_keys)
    # compare preprocessing and postprocessing data if they equal in values
    # print("Config:", config)
    updated_preprocessing_state = overwrite_state_from_json(preprocessing_state, config)
    updated_postprocessing_state = overwrite_state_from_json(postprocessing_state, config)
    # compare_states(updated_preprocessing_state, updated_postprocessing_state)
    check_after_overwrite(updated_preprocessing_state, config)
    check_after_overwrite(updated_postprocessing_state, config)
    # save updated states
    save_checkpoint_safetensor(updated_preprocessing_state, f"{root_dir}/policy_preprocessor_step_5_normalizer_processor.safetensors")
    save_checkpoint_safetensor(updated_postprocessing_state, f"{root_dir}/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    print("Done.")