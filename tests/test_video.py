from fractions import Fraction

import av
import pytest
import torch
from comfy_api.latest import InputImpl, Types

from cardamon_nodes.video import CardamonNodesJoinShotVideos

SAMPLE_RATE = 32000
# Grey levels: each shot has its own, and the frames that should be cut off have MARKER.
MARKER = 0.95


def shot(
    frames, level, audio_samples=None, fps=24, size=64, marked_tail=0, audio_value=None
):
    images = torch.full((frames, size, size, 3), level)
    if marked_tail:
        images[-marked_tail:] = MARKER
    audio = None
    if audio_samples is not None:
        value = level if audio_value is None else audio_value
        audio = {
            "waveform": torch.full((1, 2, audio_samples), float(value)),
            "sample_rate": SAMPLE_RATE,
        }
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
    )


def join(videos, trim_frames=22, codec="auto"):
    return CardamonNodesJoinShotVideos.execute(videos, [trim_frames], [codec]).args[0]


def frame_levels(video):
    return video.get_components().images[:, 32, 32, 0].tolist()


def test_cuts_the_end_of_all_but_the_last_video():
    joined = join(
        [
            shot(124, 0.2, marked_tail=22),
            shot(124, 0.5, marked_tail=22),
            shot(73, 0.8),
        ]
    )
    levels = frame_levels(joined)
    assert len(levels) == 102 + 102 + 73
    assert all(abs(v - 0.2) < 0.03 for v in levels[:102])
    assert all(abs(v - 0.5) < 0.03 for v in levels[102:204])
    assert all(abs(v - 0.8) < 0.03 for v in levels[204:])


def test_holds_compressed_parts_not_frames():
    joined = join([shot(124, 0.2), shot(124, 0.5)])
    assert isinstance(joined, InputImpl.VideoFromList)
    assert all(isinstance(part, InputImpl.VideoFromFile) for part in joined.videos)


def test_audio_is_cut_at_the_kept_frames_on_the_joined_timeline():
    # decoded H3 audio runs a little longer than its video (207 latents * 800 samples for 124 frames)
    joined = join(
        [
            shot(124, 0.2, 165600, audio_value=1),
            shot(124, 0.5, 165600, audio_value=2),
            shot(124, 0.8, 165600, audio_value=3),
        ]
    )
    waveform = joined.complete_audio["waveform"][0, 0]
    boundaries = [round(frames / 24 * SAMPLE_RATE) for frames in (102, 204, 328)]
    assert waveform.shape[-1] == boundaries[-1]
    assert waveform[boundaries[0] - 1] == 1 and waveform[boundaries[0]] == 2
    assert waveform[boundaries[1] - 1] == 2 and waveform[boundaries[1]] == 3


def test_short_audio_is_padded_with_silence():
    joined = join([shot(124, 0.2, 1000, audio_value=1), shot(73, 0.5, 1000, audio_value=2)])
    waveform = joined.complete_audio["waveform"][0, 0]
    assert waveform.shape[-1] == round((102 + 73) / 24 * SAMPLE_RATE)
    assert waveform[999] == 1 and waveform[1000] == 0


def test_videos_without_audio():
    assert join([shot(124, 0.2), shot(124, 0.5)]).complete_audio is None


def test_single_video_is_not_trimmed():
    assert len(frame_levels(join([shot(124, 0.2)]))) == 124


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_saved_file_has_all_frames_and_audio_of_the_same_length(tmp_path, codec):
    joined = join(
        [shot(124, 0.2, 165600), shot(124, 0.5, 165600), shot(73, 0.8, 97600)],
        codec=codec,
    )
    path = tmp_path / "joined.mp4"
    # as Save Video calls it with "auto" encoding settings
    joined.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec(codec))
    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video_stream))
        assert video_stream.codec_context.name in ({"h264"} if codec == "h264" else {"av1", "libdav1d", "libaom-av1"})
    with av.open(str(path)) as container:
        audio_samples = sum(frame.samples for frame in container.decode(container.streams.audio[0]))
        audio_rate = container.streams.audio[0].rate
    assert frames == 102 + 102 + 73
    # AAC works in 1024-sample frames, so allow up to one frame of difference
    assert abs(audio_samples / audio_rate - frames / 24) < 1024 / audio_rate


def test_rejects_mismatched_videos():
    with pytest.raises(ValueError, match="fps"):
        join([shot(124, 0.2), shot(124, 0.2, fps=30)])
    with pytest.raises(ValueError, match="32x32"):
        join([shot(124, 0.2), shot(124, 0.2, size=32)])
    with pytest.raises(ValueError, match="audio"):
        join([shot(124, 0.2, 1000), shot(124, 0.2)])
    with pytest.raises(ValueError, match="too few"):
        join([shot(22, 0.2), shot(124, 0.2)])
