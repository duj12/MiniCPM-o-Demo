#!/usr/bin/env python3
"""回放声卡混音：TTS 优先，打断只丢 TTS，软暂停留住缓冲。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator_replay import (  # noqa: E402
    SR, TTS_SR, AudioMix, VirtualPlayer, _pcm_i16_to_float,
    playback_reported_active,
)


def test_tts_ducks_source_and_stop_keeps_source() -> None:
    mix = AudioMix(sample_rate=8)
    mix.push_source(np.ones(8, dtype=np.float32))
    mix.push_tts(np.full(4, 0.5, dtype=np.float32))
    out = np.zeros(4, dtype=np.float32)
    mix.fill(out)
    assert np.allclose(out, 0.5)
    assert mix.src.n == 4
    mix.clear_tts()
    out2 = np.zeros(4, dtype=np.float32)
    mix.fill(out2)
    assert np.allclose(out2, 1.0)
    assert mix.pending_s() == 0.0


def test_hold_keeps_the_unplayed_tail() -> None:
    mix = AudioMix(sample_rate=8)
    mix.push_tts(np.ones(8, dtype=np.float32))
    out = np.zeros(3, dtype=np.float32)
    mix.fill(out)
    assert np.allclose(out, 1.0)
    assert mix.tts_played == 3
    mix.hold(True)
    assert mix.pending_s() == 0.0
    held = np.ones(4, dtype=np.float32)
    mix.fill(held)
    assert np.all(held == 0.0)
    assert mix.tts.n == 5
    mix.hold(False)
    rest = np.zeros(5, dtype=np.float32)
    mix.fill(rest)
    assert np.allclose(rest, 1.0)
    assert mix.tts_played == 8


def test_fade_reaches_silence() -> None:
    mix = AudioMix(sample_rate=8)
    mix.push_tts(np.ones(4, dtype=np.float32))
    mix.fade_to(0.0, 4)
    out = np.zeros(4, dtype=np.float32)
    mix.fill(out)
    assert out[0] > out[-1] > 0.0
    assert abs(mix.gain) < 1e-6


def test_player_pause_freezes_then_resume_keeps_the_tail() -> None:
    class Clock:
        def __init__(self) -> None:
            self.t = 1000.0

        def now(self) -> float:
            return self.t

    clock = Clock()
    player = VirtualPlayer(clock)
    player.begin_response("a", "", 200)
    player.schedule(np.zeros(TTS_SR, dtype=np.int16))
    clock.t = 1000.7
    played = player.mark_paused()
    assert abs(played / SR - 0.5) < 0.02
    clock.t = 1005.0
    assert player.remaining_s() == 0.0
    assert player.played_samples() == played
    at = player.resume()
    assert abs(at - 1005.2) < 1e-6
    assert abs(player.remaining_s() - 0.7) < 0.05
    assert player.resume() == 0.0


def test_paused_player_still_counts_as_playback() -> None:
    assert playback_reported_active(0.0, True) is True
    assert playback_reported_active(0.0, False) is False
    assert playback_reported_active(1.2, False) is True


def test_empty_fill_is_silence() -> None:
    mix = AudioMix(sample_rate=8)
    out = np.ones(6, dtype=np.float32)
    mix.fill(out)
    assert np.all(out == 0.0)


def test_resample_16k_to_24k() -> None:
    pcm = np.arange(160, dtype=np.int16)
    out = _pcm_i16_to_float(pcm, 16000, 24000)
    assert out.shape == (240,)
    assert out.dtype == np.float32


if __name__ == "__main__":
    test_tts_ducks_source_and_stop_keeps_source()
    test_hold_keeps_the_unplayed_tail()
    test_fade_reaches_silence()
    test_player_pause_freezes_then_resume_keeps_the_tail()
    test_paused_player_still_counts_as_playback()
    test_empty_fill_is_silence()
    test_resample_16k_to_24k()
    print("ok")
