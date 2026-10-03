from fractions import Fraction

import pytest
import torch
from comfy_api.latest import InputImpl, Types

from cardamon_nodes.video import CardamonNodesJoinShotVideos

SAMPLE_RATE = 32000


def shot(frames, first_value, audio_samples=None, fps=24, width=8):
    # Frame i holds first_value + i, and each audio sample holds its shot's first_value.
    images = (
        (first_value + torch.arange(frames, dtype=torch.float32))
        .view(-1, 1, 1, 1)
        .expand(frames, 4, width, 3)
        .clone()
    )
    audio = None
    if audio_samples is not None:
        audio = {
            "waveform": torch.full((1, 2, audio_samples), float(first_value)),
            "sample_rate": SAMPLE_RATE,
        }
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
    )


def join(videos, trim_frames=22):
    return (
        CardamonNodesJoinShotVideos.execute(videos, [trim_frames])
        .args[0]
        .get_components()
    )


def frame_values(components):
    return components.images[:, 0, 0, 0].tolist()


def test_cuts_the_end_of_all_but_the_last_video():
    joined = join([shot(124, 0), shot(124, 1000), shot(73, 2000)])
    values = frame_values(joined)
    assert len(values) == 102 + 102 + 73
    assert values[101] == 101 and values[102] == 1000
    assert values[203] == 1101 and values[204] == 2000
    assert values[-1] == 2072


def test_audio_is_cut_at_the_kept_frames_on_the_joined_timeline():
    # decoded H3 audio runs a little longer than its video (207 latents * 800 samples for 124 frames)
    joined = join([shot(124, 1, 165600), shot(124, 2, 165600), shot(124, 3, 165600)])
    waveform = joined.audio["waveform"][0, 0]
    boundaries = [round(frames / 24 * SAMPLE_RATE) for frames in (102, 204, 328)]
    assert waveform.shape[-1] == boundaries[-1]
    assert waveform[boundaries[0] - 1] == 1 and waveform[boundaries[0]] == 2
    assert waveform[boundaries[1] - 1] == 2 and waveform[boundaries[1]] == 3


def test_short_audio_is_padded_with_silence():
    joined = join([shot(124, 1, 1000), shot(73, 2, 1000)])
    waveform = joined.audio["waveform"][0, 0]
    assert waveform.shape[-1] == round((102 + 73) / 24 * SAMPLE_RATE)
    assert waveform[999] == 1 and waveform[1000] == 0


def test_videos_without_audio():
    assert join([shot(124, 0), shot(124, 1000)]).audio is None


def test_single_video_is_not_trimmed():
    assert len(frame_values(join([shot(124, 0)]))) == 124


def test_rejects_mismatched_videos():
    with pytest.raises(ValueError, match="fps"):
        join([shot(124, 0), shot(124, 0, fps=30)])
    with pytest.raises(ValueError, match="16x4"):
        join([shot(124, 0), shot(124, 0, width=16)])
    with pytest.raises(ValueError, match="audio"):
        join([shot(124, 0, 1000), shot(124, 0)])
    with pytest.raises(ValueError, match="too few"):
        join([shot(22, 0), shot(124, 0)])
