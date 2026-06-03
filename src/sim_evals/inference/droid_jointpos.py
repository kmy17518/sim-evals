import os
from collections import deque

import tyro
import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools

from .abstract_client import InferenceClient

class Client(InferenceClient):
    def __init__(self, 
                remote_host:str = "localhost", 
                remote_port:int = 8000,
                open_loop_horizon:int = 8,
                history_frames: int = None,
                 ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.client = websocket_client_policy.WebsocketClientPolicy(
            remote_host, remote_port
        )

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

        # Observation-history window. The piwan policies (A* MEM history; B* real-frame Wan
        # conditioning video) train on the last `history_frames` CONSECUTIVE frames ending at the
        # current step. `infer` is called every sim step (even when not re-querying), so we keep a
        # per-step rolling buffer here and send the last N frames at query time -> the server feeds
        # the exact stride-1 window the model trained on. Configurable via PIWAN_HISTORY_FRAMES
        # (default 6 = the current A*/B* config); set 1 to send single frames (legacy behavior).
        if history_frames is None:
            history_frames = int(os.environ.get("PIWAN_HISTORY_FRAMES", 6))
        self.history_frames = int(history_frames)
        self._base_history = deque(maxlen=max(1, self.history_frames))
        self._wrist_history = deque(maxlen=max(1, self.history_frames))

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        # new episode -> drop the previous episode's frame history (no cross-episode leakage)
        self._base_history.clear()
        self._wrist_history.clear()

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        curr_obs = self._extract_observation(obs)
        # resize once; push EVERY step so the rolling window is stride-1 consecutive frames
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        self._base_history.append(base_img)
        self._wrist_history.append(wrist_img)

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_1_left": base_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            if self.history_frames > 1:
                # last N consecutive frames (earliest..current); the server trims/earliest-pads
                # to the model's history_frames. uint8 [T,224,224,3] keeps the payload small.
                request_data["observation/exterior_image_1_left_history"] = np.stack(
                    list(self._base_history), axis=0
                )
                request_data["observation/wrist_image_left_history"] = np.stack(
                    list(self._wrist_history), axis=0
                )
            self.pred_action_chunk = self.client.infer(request_data)["actions"]

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        both = np.concatenate([base_img, wrist_img], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

if __name__ == "__main__":
    import torch
    args = tyro.cli(Args)
    client = Client(args)
    fake_obs = {
        "splat": {
            "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),

        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction) # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
