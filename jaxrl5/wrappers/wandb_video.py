from typing import Optional

import gymnasium as gym
import numpy as np
import wandb


class WANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        pixel_hw: int = 84,
        render_kwargs={},
        max_videos: Optional[int] = None,
    ):
        super().__init__(env)

        self._name = name
        self._pixel_hw = pixel_hw
        self._render_kwargs = render_kwargs
        self._max_videos = max_videos
        self._video = []

    def _add_frame(self, obs):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if isinstance(obs, dict) and "pixels" in obs:
            if obs["pixels"].ndim == 4:
                self._video.append(obs["pixels"][..., -1])
            else:
                self._video.append(obs["pixels"])
        else:
            self._video.append(
                self.env.render(
                    height=self._pixel_hw,
                    width=self._pixel_hw,
                    mode="rgb_array",
                    **self._render_kwargs,
                )
            )

    def reset(self, **kwargs):
        self._video.clear()
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        self._add_frame(obs)
        return (obs, info) if isinstance(result, tuple) else obs

    def step(self, action: np.ndarray):

        result = self.env.step(action)
        if not isinstance(result, tuple):
            raise TypeError("Expected tuple return from env.step")

        if len(result) == 4:
            obs, reward, done, info = result
            cost = None
            terminated, truncated = done, False
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            cost = None
            done = bool(terminated or truncated)
        elif len(result) == 6:
            obs, reward, cost, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            raise ValueError("Unsupported number of return values from env.step")

        self._add_frame(obs)

        if done and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            video = np.moveaxis(np.stack(self._video), -1, 1)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            video = wandb.Video(video, fps=20, format="mp4")
            wandb.log({self._name: video}, commit=False)

        if len(result) == 4:
            return obs, reward, done, info
        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, cost, terminated, truncated, info
