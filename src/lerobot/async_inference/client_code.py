import time
import pickle  # nosec
import grpc
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import TimedObservation
ENV_DT = 1.0 / 30.0  # just for channel options; doesn't affect timings

def load_img(path: str, hw=(224, 224)) -> np.ndarray:
    return np.array(Image.open(path).resize((hw[1], hw[0])))

def roundtrip_once(stub, payload, step: int) -> float:
    obs = TimedObservation(
        timestamp=time.time(),
        observation=payload,
        timestep=step,
    )

    t0 = time.perf_counter()

    # Send observation
    obs_bytes = pickle.dumps(obs)  # includes numpy arrays, same approach
    obs_iter = send_bytes_in_chunks(obs_bytes, services_pb2.Observation, silent=True)
    stub.SendObservations(obs_iter)

    # Get actions (poll until non-empty)
    while True:
        msg = stub.GetActions(services_pb2.Empty())
        if len(msg.data) > 0:
            _actions = pickle.loads(msg.data)  # list[TimedAction]
            break

    t1 = time.perf_counter()
    return t1 - t0

def benchmark(image_folder_path, image_size=(224, 224, 3), host="localhost", runs=1000, port=8000):

    side_image_path = f"{image_folder_path}/side_observer_30.png"
    wrist_image_path = f"{image_folder_path}/side_right_30.png"
    side_image = load_img(side_image_path, image_size)
    wrist_image = load_img(wrist_image_path, image_size)

    payload = {
        "images": {
            "side_image": side_image,
            "wrist_image": wrist_image,
        },
        "instruction": "do something",
    }

    channel = grpc.insecure_channel(
        host,
        grpc_channel_options(initial_backoff=f"{ENV_DT:.4f}s"),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    # handshake
    stub.Ready(services_pb2.Empty())

    times = []
    for i in tqdm(range(runs)):
        dt = roundtrip_once(stub, payload, i)
        times.append(dt)

    channel.close()
    avg_time = sum(times) / runs
    max_time = max(times)
    min_time = min(times)
    std_dev = np.std(np.array(times))
    results = {
        
        "avg": avg_time,
        "max": max_time,
        "min": min_time,
        "std": std_dev,
        "times": times,
    }
    return results


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Client agent for benchmarking")
    parser.add_argument("--host", type=str, default="multihead.utn-mi.de", help="Host of the server")
    parser.add_argument("--port", type=int, default=8000, help="Port of the server")
    parser.add_argument("--on_same_machine", action="store_true", help="Whether the client is running on the same machine as the server")
    parser.add_argument("--runs", type=int, default=1000, help="Number of runs for the benchmark")
    parser.add_argument("--imgs_folder_path", type=str, default="/home/gamal/vlagent_benchmark/imgs", help="Path to the folder containing the images for benchmarking")
    parser.add_argument("--output_folder_path", type=str, default="/home/gamal/vlagent_benchmark/outputs", help="Path to the folder where benchmark results will be saved")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    on_same_machine = args.on_same_machine
    port = args.port
    #image_size = (224, 224)
    image_size = (720, 1280)

    # Build payload once (fixed images like your OpenPI test)
    if on_same_machine == True:
        host = f"0.0.0.0:{port}"
    else:
        #SERVER = f"airtower.utn-mi.de:{port}"
        host = args.host + f":{port}"

    runs = 1000
    model = "lerobot"
    image_folder_path = "/home/gamal/vlagent_benchmark/imgs"
    output_folder_path = f"/home/gamal/vlagent_benchmark/outputs/{model}"
    for image_size in [(224, 224), (720, 1280)]:
        results = benchmark(image_folder_path, image_size=image_size, host=host, runs=runs, port=port)
        print(model, "benchmark results:")
        print(f"runs: {runs} ")
        print("size:", image_size)
        print("on same machine" , on_same_machine)
        print("avg:", results["avg"])
        print("min:", results["min"])
        print("max:", results["max"])
        print("std:", results["std"])
        results["model"] = model
        results["on_same_machine"] = on_same_machine
        results["image_size"] = image_size
        results["runs"] = runs
        os.makedirs(output_folder_path, exist_ok=True)
        json_path = f"{output_folder_path}/benchmark_results_{model}_{'local' if on_same_machine else 'remote'}_{image_size[0]}x{image_size[1]}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Benchmark results saved to {json_path}")
        time.sleep(5)  # to avoid overloading the server

# Example usage:
# For local testing (client and server on the same machine):
# python lerobot/src/lerobot/async_inference/client_code.py --on_same_machine --runs 1000 --port 8000 --imgs_folder_path /home/gamal/vlagent_benchmark/imgs --output_folder_path /home/gamal/vlagent_benchmark/outputs
# For remote testing (client and server on different machines):
# python lerobot/src/lerobot/async_inference/client_code.py --host multihead.utn-mi.de --port 8000 --runs 1000 --imgs_folder_path /home/gamal/vlagent_benchmark/imgs --output_folder_path /home/gamal/vlagent_benchmark/outputs

# Server command (run on the server machine):
# python -m lerobot.async_inference.policy_server_dummy --host=0.0.0.0 --port=8000 --fps=30 --inference_latency=0.0 --obs_queue_timeout=1 --dummy_actions=true --dummy_action_dim=7 --dummy_actions_per_chunk=50