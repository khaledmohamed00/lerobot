import time
import pickle  # nosec
import grpc
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import TimedObservation
local = True
if local:
    SERVER = "127.0.0.1:8050"
else:
    SERVER = "airtower.utn-mi.de:8000"
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

def main():
    # Build payload once (fixed images like your OpenPI test)
    image_size = (224, 224)
    #image_size = (720, 1280)
    side_image = load_img("/home/epez82ox/repos/imgs/side_observer_30.png", image_size)
    wrist_image = load_img("/home/epez82ox/repos/imgs/side_right_30.png", image_size)

    payload = {
        "images": {
            "side_image": side_image,
            "wrist_image": wrist_image,
        },
        "instruction": "do something",
    }

    channel = grpc.insecure_channel(
        SERVER,
        grpc_channel_options(initial_backoff=f"{ENV_DT:.4f}s"),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    # handshake
    stub.Ready(services_pb2.Empty())

    times = []
    for i in tqdm(range(1000)):
        dt = roundtrip_once(stub, payload, i)
        times.append(dt)

    channel.close()
    print("Results over 1000 runs:")
    print("size:", image_size)
    print("on same machine:", local)
    print("avg:", sum(times) / len(times), "min:", min(times), "max:", max(times))
    print("standard deviation:", np.std(np.array(times)))
    results_dict = {
        "average_time": sum(times)/len(times),
        "max_time": max(times),
        "min_time": min(times),
        "std_dev": np.std(np.array(times)),
        "times": times,
    }
    import json
    import os
    model  = "lerobot"
    dir_path = "/home/epez82ox/repos/time_benchmarks/time_benchmarks"
    os.makedirs(dir_path, exist_ok=True)
    json_path = f"{dir_path}/benchmark_results_{model}_{'local' if local else 'remote'}_{image_size[0]}x{image_size[1]}.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Benchmark results saved to {json_path}")
if __name__ == "__main__":
    main()
