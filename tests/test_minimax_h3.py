import pytest
import torch

import comfy.nested_tensor
from cardamon_nodes.minimax_h3 import (
    AUDIO_FRAME_OFFSET_KEY,
    FRAME_RESCALE,
    CardamonMiniMaxH3AddLatentGuide,
    CardamonMiniMaxH3ExtractLatent,
    frames_for_tokens,
    is_empty_latent,
    tokens_for_frames,
    valid_clip_frames,
)


def av_latent(frames, height=64, width=96):
    tokens = tokens_for_frames(frames)
    audio_t = round(frames / 24 * 40)
    # Encode each position in the values so slices can be checked.
    video = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1, 1).expand(1, 24, tokens, height // 16, width // 16).clone()
    audio = torch.arange(audio_t, dtype=torch.float32).view(1, 1, 1, audio_t).expand(1, 32, 2, audio_t).clone()
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def extract(latent, start_frame, length=0):
    return CardamonMiniMaxH3ExtractLatent.execute(latent, start_frame, length).args


def add_guide(target, guide, frame_idx=0, use_video=True, use_audio=True, positive=None):
    positive = positive if positive is not None else [[torch.zeros(1, 1, 8), {}]]
    return CardamonMiniMaxH3AddLatentGuide.execute(positive, target, guide, frame_idx, use_video, use_audio).args[0]


@pytest.mark.parametrize("frames", [1, 5, 22, 39, 124, 362])
def test_token_frame_round_trip(frames):
    assert frames_for_tokens(tokens_for_frames(frames)) == frames


@pytest.mark.parametrize("frames, expected", [(1, 1), (4, 1), (5, 5), (21, 5), (22, 22), (40, 39)])
def test_valid_clip_frames(frames, expected):
    assert valid_clip_frames(frames) == expected


def test_extract_last_frames_snaps_to_chunk_and_runs_to_end():
    latent, start, length = extract(av_latent(124), -22)
    # 124 - 22 = 102 = 6 * 17, already a chunk boundary
    assert (start, length) == (102, 22)
    video, audio = latent["samples"].tensors
    assert video.shape[2] == tokens_for_frames(22)
    assert video[0, 0, :, 0, 0].tolist() == [30, 31, 32, 33, 34, 35, 36]
    assert audio[0, 0, 0, 0] == 170  # 102 * 5/3
    assert audio.shape[-1] == round(124 / 24 * 40) - 170
    assert latent[AUDIO_FRAME_OFFSET_KEY] == pytest.approx(0.0)


def test_extract_snaps_start_down_and_length_to_valid_clip():
    latent, start, length = extract(av_latent(124), 40, 30)
    assert (start, length) == (34, 22)
    video = latent["samples"].tensors[0]
    assert video[0, 0, :, 0, 0].tolist() == [10, 11, 12, 13, 14, 15, 16]


def test_extract_single_frame():
    latent, start, length = extract(av_latent(124), 17, 1)
    assert (start, length) == (17, 1)
    assert latent["samples"].tensors[0][0, 0, :, 0, 0].tolist() == [5]


def test_extract_audio_offset_when_chunk_is_between_audio_latents():
    latent, start, _ = extract(av_latent(124), 17)
    audio = latent["samples"].tensors[1]
    # frame 17 is at audio position 28.33: the slice starts at audio latent 29
    assert audio[0, 0, 0, 0] == 29
    assert (start + latent[AUDIO_FRAME_OFFSET_KEY]) * FRAME_RESCALE == pytest.approx(29)


def test_extract_from_a_section_keeps_audio_aligned_with_the_source():
    section, _, _ = extract(av_latent(124), 17)
    nested, start, _ = extract(section, 17)
    # frame 17 of the section is frame 34 of the source, at audio position 56.67
    assert nested["samples"].tensors[1][0, 0, 0, 0] == 57
    assert (17 + start + nested[AUDIO_FRAME_OFFSET_KEY]) * FRAME_RESCALE == pytest.approx(57)


def test_extract_rejects_start_outside_video():
    with pytest.raises(ValueError):
        extract(av_latent(124), 124)
    with pytest.raises(ValueError):
        extract(av_latent(124), -125)


def test_add_guide_anchors_video_and_audio_at_matching_times():
    section, start, _ = extract(av_latent(124), 17, 22)
    positive = add_guide(av_latent(124), section, frame_idx=5)
    video_kf, audio_kf = positive[0][1]["minimax_keyframes"]
    assert video_kf["resolved_frame_index"] == 5
    assert video_kf["latent"].shape[2] == tokens_for_frames(22)
    # audio keeps its offset from the video it came with
    offset = section[AUDIO_FRAME_OFFSET_KEY]
    assert audio_kf["resolved_frame_index"] == pytest.approx(5 + offset)


def test_add_guide_keeps_existing_keyframes():
    existing = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 6)}
    section, _, _ = extract(av_latent(124), -22)
    positive = add_guide(av_latent(124), section, frame_idx=-22,
                         positive=[[torch.zeros(1, 1, 8), {"minimax_keyframes": [existing]}]])
    keyframes = positive[0][1]["minimax_keyframes"]
    assert keyframes[0] is existing
    assert keyframes[1]["resolved_frame_index"] == 102


def test_add_guide_crops_audio_to_remaining_duration():
    section, _, _ = extract(av_latent(124), 0, 39)
    positive = add_guide(av_latent(124), section, frame_idx=102, use_video=False)
    (audio_kf,) = positive[0][1]["minimax_keyframes"]
    assert audio_kf["audio_latent"].shape[-1] == round(124 / 24 * 40) - 170


def test_add_guide_rejects_mismatched_size_and_overflow():
    section, _, _ = extract(av_latent(124, width=128), -22)
    with pytest.raises(ValueError, match="128x64"):
        add_guide(av_latent(124), section)
    section, _, _ = extract(av_latent(124), 0, 39)
    with pytest.raises(ValueError, match="does not fit"):
        add_guide(av_latent(124), section, frame_idx=100)


def test_guides_build_a_valid_model_layout():
    from comfy.ldm.minimax.model import PackedLayout

    target = av_latent(124)
    video, audio = target["samples"].tensors
    section, _, _ = extract(av_latent(124), 17)  # fractional audio offset
    keyframes = add_guide(target, section)[0][1]["minimax_keyframes"]
    layout = PackedLayout(8, video.shape[2], video.shape[3], video.shape[4], audio.shape[-1], keyframes=keyframes)
    cond_video = keyframes[0]["latent"]
    cond_rows = cond_video.shape[2] * cond_video.shape[3] * cond_video.shape[4] // 4
    audio_rows = keyframes[1]["audio_latent"].shape[-1] * 2
    assert layout.seq_len == 8 + cond_rows + audio_rows + audio.shape[-1] * 2 + video.shape[2] * video.shape[3] * video.shape[4] // 4


@pytest.mark.parametrize("empty", [{"samples": torch.zeros(0)}, None], ids=["empty", "none"])
def test_empty_guide_passes_through_extract_and_adds_no_guide(empty):
    section, start, length = extract(empty, -22)
    assert is_empty_latent(section) and (start, length) == (0, 0)
    positive = [[torch.zeros(1, 1, 8), {}]]
    assert add_guide(av_latent(124), section, positive=positive) is positive
