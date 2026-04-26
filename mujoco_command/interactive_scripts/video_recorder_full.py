# Copyright 2026 Hency
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# envs/utils/obs_stream_recorder.py
import os
import time
import json
import shutil
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# def _as_uint8_bgr(img: np.ndarray) -> np.ndarray:
#     """
#     你 obs 里一般是 uint8 HxWx3。
#     如果你是 RGB，这里可以改成 cv2.cvtColor(img, cv2.COLOR_RGB2BGR)。
#     默认按 BGR 写（OpenCV VideoWriter 的常规用法）。
#     """
#     if img.dtype != np.uint8:
#         img = np.clip(img, 0, 255).astype(np.uint8)
#     if img.ndim == 2:
#         img = np.repeat(img[:, :, None], 3, axis=2)
#     if img.shape[2] == 1:
#         img = np.repeat(img, 3, axis=2)
#     return img

def _as_uint8_bgr(img: np.ndarray, in_color: str = "bgr") -> np.ndarray:
    """
    将输入图像统一转成 uint8 + BGR（三通道），用于 OpenCV VideoWriter.
    in_color: "bgr" or "rgb"
    """
    if img is None:
        return None

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # 关键：如果输入是 RGB，写之前转 BGR
    if in_color.lower() == "rgb":
        img = img[:, :, ::-1]

    return img



class _StreamWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        out_path: Path,
        fps: float,
        size: Tuple[int, int],
        q: "queue.Queue[Tuple[float, np.ndarray]]",
        stop_event: threading.Event,
        drop_counter: Dict[str, int],
        resize: bool = True,
        fourcc: str = "mp4v",
        in_color: str = "bgr",
    ):
        super().__init__(daemon=True)
        self.stream_name = name
        self.out_path = out_path
        self.fps = float(fps)
        self.w, self.h = int(size[0]), int(size[1])
        self.q = q
        self.stop_event = stop_event
        self.drop_counter = drop_counter
        self.resize = resize
        self.fourcc = fourcc

        self.writer = None
        self.ts_list = []  # 每帧的时间戳（epoch seconds）
        self.in_color = in_color

    def _open_writer(self):
        _ensure_dir(self.out_path.parent)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, (self.w, self.h), True)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.out_path}")

    def run(self):
        self._open_writer()

        while (not self.stop_event.is_set()) or (not self.q.empty()):
            try:
                ts, frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            # frame = _as_uint8_bgr(frame)
            frame = _as_uint8_bgr(frame, in_color=self.in_color)

            if self.resize and (frame.shape[1] != self.w or frame.shape[0] != self.h):
                frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)

            self.writer.write(frame)
            self.ts_list.append(float(ts))
            self.q.task_done()

        if self.writer is not None:
            self.writer.release()


class ObsStreamRecorder:
    """
    只负责把 obs 里的图像流持续写成 mp4（实时写盘，不依赖 action）。
    每个 stream 一个 mp4 + 一个 timestamps list（保存到 meta.json / meta.pkl 都行）。
    """

    def __init__(
        self,
        data_folder: str,
        stream_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
        queue_size: int = 256,
        drop_if_full: bool = True,
        resize: bool = True,
    ):
        """
        stream_cfg:
          {
            "tactile_image": {"fps": 24, "size": (480,120)},
            "external_img":  {"fps": 24, "size": (320,240)},
            "wrist_image":   {"fps": 24, "size": (320,240)},
          }
        """
        self.data_folder = Path(data_folder)
        _ensure_dir(self.data_folder)

        if stream_cfg is None:
            # 你可以按需改 key 名
            stream_cfg = {
                "tactile_image": {"fps": 24, "size": (480, 120)},
                "external_img":  {"fps": 24, "size": (320, 240)},
                "wrist_image":   {"fps": 24, "size": (320, 240)},
            }
        self.stream_cfg = stream_cfg

        self.queue_size = int(queue_size)
        self.drop_if_full = bool(drop_if_full)
        self.resize = bool(resize)

        self._episode_dir: Optional[Path] = None
        self._stop_event = threading.Event()
        self._workers: Dict[str, _StreamWorker] = {}
        self._queues: Dict[str, "queue.Queue[Tuple[float, np.ndarray]]"] = {}
        self._drop_counter: Dict[str, int] = {}
        self._started = False

        # 如果你也想存“每次 record_obs 的总时间戳”，保留这个
        self._global_ts = []

    def start_episode(self, idx: int):
        if self._started:
            raise RuntimeError("ObsStreamRecorder already started. Call end_episode() first.")

        self._episode_dir = self.data_folder / f"demo{idx:05d}_obsstream"
        self._episode_dir.mkdir(parents=True, exist_ok=False)

        self._stop_event.clear()
        self._workers.clear()
        self._queues.clear()
        self._drop_counter = {k: 0 for k in self.stream_cfg.keys()}
        self._global_ts = []
        self._started = True

        # 立即为每个 stream 启 worker
        for k, cfg in self.stream_cfg.items():
            fps = cfg.get("fps", 24)
            size = cfg.get("size", (320, 240))
            in_color = cfg.get("in_color", "bgr")
            out_path = self._episode_dir / f"{k}.mp4"

            q = queue.Queue(maxsize=self.queue_size)
            self._queues[k] = q

            worker = _StreamWorker(
                name=k,
                out_path=out_path,
                fps=fps,
                size=size,
                q=q,
                stop_event=self._stop_event,
                drop_counter=self._drop_counter,
                resize=self.resize,
                fourcc=cfg.get("fourcc", "mp4v"),
                in_color=in_color,
            )
            self._workers[k] = worker
            worker.start()

    def record_obs(self, obs: Dict[str, Any], timestamp: Optional[float] = None):
        """
        在你的控制循环里每一步都调一次：
            self.obs_stream_recorder.record_obs(obs, time.time())
        非阻塞：队列满了就丢帧（可配置）。
        """
        if not self._started:
            return
        if timestamp is None:
            timestamp = time.time()

        self._global_ts.append(float(timestamp))

        for k in self.stream_cfg.keys():
            frame = obs.get(k, None)
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                q = self._queues[k]
                item = (float(timestamp), frame.copy())
                if self.drop_if_full:
                    if q.full():
                        # 丢掉最旧的一帧，保证“最新画面”更跟手
                        try:
                            q.get_nowait()
                            q.task_done()
                        except queue.Empty:
                            pass
                        self._drop_counter[k] += 1
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        self._drop_counter[k] += 1
                else:
                    q.put(item)  # 可能阻塞，不推荐用于实时控制

    def end_episode(self, save: bool = True):
        if not self._started:
            return

        ep_dir = self._episode_dir
        assert ep_dir is not None

        # 停 worker 并 flush 队列
        self._stop_event.set()
        for k, q in self._queues.items():
            q.join()
        for w in self._workers.values():
            w.join(timeout=2.0)

        # 写 meta
        meta = {
            "created_time": time.time(),
            "streams": {},
            "global_ts_len": len(self._global_ts),
            "drop_counter": dict(self._drop_counter),
            "stream_cfg": self.stream_cfg,
        }
        for k, w in self._workers.items():
            meta["streams"][k] = {
                "mp4": f"{k}.mp4",
                "frame_count": len(w.ts_list),
                "timestamps": w.ts_list[:],  # 如果太大你也可以只存起止/间隔
            }

        with open(ep_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if not save:
            shutil.rmtree(ep_dir, ignore_errors=True)

        # reset state
        self._episode_dir = None
        self._started = False
        self._workers.clear()
        self._queues.clear()
        self._global_ts = []
