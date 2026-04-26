import os
import cv2
import imageio
import numpy as np
import torch
from collections import defaultdict
from typing import Optional

class Recorder:
    def __init__(self, save_dir: str, vis_dim=(640, 480)):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.vis_dim = vis_dim
        self.combined_frames = []
        self.tensors = defaultdict(list)

    def add(self, camera_obses: dict[str, torch.Tensor], color: Optional[tuple] = None):
        frames = []
        for camera, obs in camera_obses.items():
            assert obs.dim() == 3 and obs.size(0) == 3
            assert obs.dtype == torch.uint8

            tensor = obs.cpu()
            self.tensors[camera].append(tensor)
            frame = tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
            frame = cv2.resize(frame, self.vis_dim)
            frames.append(frame)

        if frames:
            stacked = np.hstack(frames)
            if color is not None:
                stacked[:10, :, :] = color  # Optional visual cue
            self.combined_frames.append(cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

    def add_numpy(self, obs: dict, cameras: list[str], color: Optional[tuple] = None):
        frames = []
        for cam in cameras:
            image = obs[cam]
            assert len(image.shape) == 3 and image.shape[-1] == 3
            image = cv2.resize(image, self.vis_dim)
            frames.append(image)

        if frames:
            stacked = np.hstack(frames)
            if color is not None:
                stacked[:10, :, :] = color
            self.combined_frames.append(cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

    #def save(self, name: str, fps: int = 10):
    #    if not self.combined_frames:
    #        print("No frames to save.")
    #        return None
    #
    #    mp4_path = os.path.join(self.save_dir, f"{name}.mp4")
    #
    #    try:
    #        writer = imageio.get_writer(
    #            mp4_path,
    #            fps=fps,
    #            codec='mpeg4',
    #            format='ffmpeg',  # ensure ffmpeg backend
    #            quality=8,  # or bitrate='5000k'
    #        )
    #
    #        for frame in self.combined_frames:
    #            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #            writer.append_data(frame_rgb)
    #
    #        writer.close()
    #        print(f"Saved MP4 to {mp4_path}")
    #        self.combined_frames.clear()
    #        return mp4_path
    #
    #    except Exception as e:
    #        print(f"Failed to save MP4 using libopenh264: {e}")
    #        return None

    def save(self, name: str, fps: int = 10):
        if not self.combined_frames:
            print("No frames to save.")
            return None

        path = os.path.join(self.save_dir, f"{name}.mp4")
        H, W, _ = self.combined_frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))

        for frame in self.combined_frames:
            out.write(frame)
        out.release()
        print(f"Saved video to {path}")
        self.combined_frames.clear()
        return path


    def save_images(self, name: str):
        if not self.combined_frames:
            print("No frames to save.")
            return None

        path = os.path.join(self.save_dir, f"{name}.jpg")
        image = np.concatenate(self.combined_frames, axis=0)
        cv2.imwrite(path, image)
        print(f"Saved image to {path}")
        self.combined_frames.clear()
        return path

