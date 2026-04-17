"""自适应帧采样器：基于 FFT 频域分析 + 瞬时突变触发的动态采样率控制。

将显著度时序信号做短时傅里叶变换（STFT），高频能量占比驱动采样率——
高频高采样、低频低采样。同时保留瞬时阈值作为即时触发机制，解决 FFT 窗口滞后。

三层控制：
  1. 稳态层：STFT 高频能量比 → 目标 fps（EMA 平滑）
  2. 反馈层：detector 回传 change_score 超阈值 → spike boost
  3. 预检测层：producer 对跳过帧做轻量 absdiff → 超阈值强制采样
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import torch


@dataclass
class AdaptiveFrameSamplerConfig:
    """自适应帧采样器的运行时配置。"""

    enabled: bool = True
    stft_window_size: int = 32
    stft_hop_size: int = 4
    highfreq_cutoff_ratio: float = 0.3
    fps_min: float = 4.0
    fps_max: float = 15.0
    gamma: float = 0.7
    spike_threshold: float = 0.4
    spike_boost_seconds: float = 2.0
    smoothing_alpha: float = 0.3
    precheck_diff_threshold: float = 15.0
    precheck_resize_width: int = 160


class AdaptiveFrameSampler:
    """根据显著度时序的频域特征动态调节视频帧采样率。

    三层控制：
    - 稳态层：STFT → 高频能量比 → 目标 fps（EMA 平滑）
    - 反馈层：detector 回传 change_score 超阈值 → spike boost
    - 预检测层：producer 对跳过帧做 absdiff → 超阈值强制采样
    """

    def __init__(self, config: AdaptiveFrameSamplerConfig, source_fps: float):
        self.config = config
        self._source_fps = max(source_fps, 1.0)
        self._score_buffer: Deque[float] = deque(maxlen=config.stft_window_size)
        self._current_target_fps: float = config.fps_max
        self._last_spike_time: float = -1e9
        self._last_sampled_time: float = -1e9
        self._frames_since_last_fft: int = 0
        self._spectral_activity_ratio: float = 0.0
        self._lock = threading.Lock()
        self._precheck_ref_gray: Optional[np.ndarray] = None
        self._precheck_force_count: int = 0

    def update(self, change_score: float, smoothed_score: float, timestamp: float) -> None:
        """detector 线程调用：反馈显著度信号，更新 FFT 和 spike 状态。"""
        self._score_buffer.append(smoothed_score)
        self._check_spike(change_score, timestamp)
        self._frames_since_last_fft += 1
        if self._frames_since_last_fft >= self.config.stft_hop_size:
            self._recompute_fft()
            self._frames_since_last_fft = 0

    def should_sample(self, frame_index: int, timestamp: float) -> bool:
        """producer 线程调用：根据当前 effective fps 决定是否处理该帧。"""
        with self._lock:
            eff_fps = self._effective_fps_locked(timestamp)
        interval = 1.0 / max(eff_fps, 0.1)
        if (timestamp - self._last_sampled_time) >= interval:
            self._last_sampled_time = timestamp
            return True
        return False

    def precheck_skipped_frame(self, frame_bgr: np.ndarray) -> bool:
        """producer 对跳过帧做轻量帧间差分，超阈值返回 True 强制采样。

        仅用 resize + grayscale + absdiff，约 0.1ms/帧。
        """
        import cv2

        w = self.config.precheck_resize_width
        h = int(frame_bgr.shape[0] * w / max(frame_bgr.shape[1], 1))
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._precheck_ref_gray is None:
            self._precheck_ref_gray = gray
            return False

        diff = cv2.absdiff(gray, self._precheck_ref_gray)
        mean_diff = float(diff.mean())

        if mean_diff >= self.config.precheck_diff_threshold:
            self._precheck_ref_gray = gray
            self._precheck_force_count += 1
            return True
        return False

    def mark_sampled(self, frame_bgr: np.ndarray, timestamp: float) -> None:
        """producer 采样帧后调用，更新预检测参考帧和时间戳。"""
        import cv2

        self._last_sampled_time = timestamp
        w = self.config.precheck_resize_width
        h = int(frame_bgr.shape[0] * w / max(frame_bgr.shape[1], 1))
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        self._precheck_ref_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    @property
    def effective_fps(self) -> float:
        with self._lock:
            return self._effective_fps_locked()

    @property
    def spectral_activity_ratio(self) -> float:
        return self._spectral_activity_ratio

    @property
    def current_target_fps(self) -> float:
        with self._lock:
            return self._current_target_fps

    @property
    def precheck_force_count(self) -> int:
        return self._precheck_force_count

    def _effective_fps_locked(self, timestamp: float = 0.0) -> float:
        if (timestamp - self._last_spike_time) < self.config.spike_boost_seconds:
            return self.config.fps_max
        return self._current_target_fps

    def _recompute_fft(self) -> None:
        if len(self._score_buffer) < self.config.stft_window_size:
            return

        signal = torch.tensor(list(self._score_buffer), dtype=torch.float32)
        signal = signal - signal.mean()
        window = torch.hann_window(len(signal))
        spectrum = torch.fft.rfft(signal * window)
        power = spectrum.abs().pow(2)

        cutoff_bin = max(1, int(len(power) * self.config.highfreq_cutoff_ratio))
        total_energy = power[1:].sum().item()
        high_energy = power[cutoff_bin:].sum().item()
        r_high = high_energy / (total_energy + 1e-8)

        raw_fps = self.config.fps_min + (
            self.config.fps_max - self.config.fps_min
        ) * (r_high ** self.config.gamma)

        alpha = self.config.smoothing_alpha
        with self._lock:
            self._current_target_fps = (
                alpha * raw_fps + (1 - alpha) * self._current_target_fps
            )
        self._spectral_activity_ratio = r_high

    def _check_spike(self, change_score: float, timestamp: float) -> None:
        if change_score >= self.config.spike_threshold:
            with self._lock:
                self._last_spike_time = timestamp
