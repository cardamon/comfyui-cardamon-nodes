"""MiniMax H3 latent guides: cut a section out of a generated AV latent and anchor it in a new video.

Continuing a video: extract the last frames of a generated latent, then anchor them at
frame 0 of the next generation, so it starts where the previous one ended.

H3 latent facts this module relies on (see ComfyUI's comfy/ldm/minimax/ and
comfy_extras/nodes_minimax_h3.py):
- An AV latent is a NestedTensor (video [B,24,T,H/16,W/16], audio [B,32,2,T_audio]).
- The video VAE encodes independent 17-frame chunks into 5 tokens each covering
  (1, 4, 4, 4, 4) pixel frames, and a video has 17k+5 frames (5k+2 tokens).
  A slice that starts on a chunk boundary is therefore identical to VAE-encoding
  those frames on their own, and guide clips must be 1 or 17k+5 frames long.
- Video runs at 24 fps and audio latents at 40 per second, so one pixel frame
  spans FRAME_RESCALE = 5/3 audio latents. The model places guides on a shared time
  axis and accepts fractional frame indexes.
"""

import math

import comfy.nested_tensor
import node_helpers
from comfy_api.latest import io

from .latent_files import empty_latent, is_empty_latent

# Copied from comfy.ldm.minimax.model so this package still loads on ComfyUI builds without H3.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
CHUNK_FRAMES = sum(FRAME_PER_TOKEN)  # 17 pixel frames per 5 latent tokens
CHUNK_TOKENS = len(FRAME_PER_TOKEN)

# Offset in pixel frames of a latent's audio relative to its video, set when a slice's audio
# cannot start exactly on its first video frame.
AUDIO_FRAME_OFFSET_KEY = "cardamon_audio_frame_offset"


def frames_for_tokens(tokens):
    return sum(FRAME_PER_TOKEN[k % CHUNK_TOKENS] for k in range(tokens))


def tokens_for_frames(frames):
    """Token count of a valid clip length (1 or 17k+5 frames)."""
    return 1 if frames == 1 else (frames - 5) // CHUNK_FRAMES * CHUNK_TOKENS + 2


def valid_clip_frames(frames):
    """Largest valid guide clip length (1 or 17k+5 frames) that fits in `frames`."""
    if frames < 5:
        return 1
    return frames - (frames - 5) % CHUNK_FRAMES


def split_av_latent(latent, node_name):
    samples = latent["samples"]
    if (
        not getattr(samples, "is_nested", False)
        or len(samples.tensors) != 2
        or samples.tensors[0].ndim != 5
        or samples.tensors[0].shape[1] != 24
    ):
        raise ValueError(f"{node_name} expects a MiniMax H3 AV latent")
    return samples.tensors[0], samples.tensors[1]


class CardamonMiniMaxH3ExtractLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonMiniMaxH3ExtractLatent",
            display_name="Extract MiniMax H3 Latent Section",
            category="cardamon/minimax_h3",
            description="Cut a section, typically the last frames, out of a MiniMax H3 AV latent, to use as a latent guide. "
            "Sections start on the VAE's 17-frame chunk boundaries and are 1 or 17k+5 frames long.",
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input(
                    "start_frame",
                    default=-22,
                    min=-9999,
                    max=9999,
                    tooltip="First frame of the section, snapped down to a multiple of 17. "
                    "Negative values count from the end of the video.",
                ),
                io.Int.Input(
                    "length",
                    default=0,
                    min=0,
                    max=9999,
                    tooltip="Frames to extract, snapped down to 1 or 17k+5 (5, 22, 39...). 0 extracts up to the end of the video.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(
                    display_name="start_frame",
                    tooltip="Snapped start frame in the source video.",
                ),
                io.Int.Output(
                    display_name="length", tooltip="Snapped section length in frames."
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, start_frame, length) -> io.NodeOutput:
        if is_empty_latent(latent):
            # Pass "no guide" through, e.g. for the first shot in a list.
            return io.NodeOutput(empty_latent(), 0, 0)
        video, audio = split_av_latent(latent, "CardamonMiniMaxH3ExtractLatent")
        total_frames = frames_for_tokens(video.shape[2])

        resolved = start_frame if start_frame >= 0 else total_frames + start_frame
        if not 0 <= resolved < total_frames:
            raise ValueError(
                f"start_frame {start_frame} is outside the video's {total_frames} frames"
            )
        chunk = resolved // CHUNK_FRAMES
        start = chunk * CHUNK_FRAMES

        available = total_frames - start
        frames = valid_clip_frames(min(length, available) if length > 0 else available)
        token_start = chunk * CHUNK_TOKENS
        video = video[
            :, :, token_start : token_start + tokens_for_frames(frames)
        ].clone()

        # Audio latents rarely line up with a chunk boundary: start on the next one and record the offset.
        offset = latent.get(AUDIO_FRAME_OFFSET_KEY, 0.0)
        audio_start = math.ceil(FRAME_RESCALE * (start - offset) - 1e-6)
        audio_end = min(
            audio.shape[-1], round(FRAME_RESCALE * (start + frames - offset))
        )
        audio = audio[..., audio_start:audio_end].clone()

        out = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            AUDIO_FRAME_OFFSET_KEY: audio_start / FRAME_RESCALE + offset - start,
        }
        return io.NodeOutput(out, start, frames)


class CardamonMiniMaxH3AddLatentGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonMiniMaxH3AddLatentGuide",
            display_name="Add Latent Guide for MiniMax H3",
            category="cardamon/minimax_h3",
            description="Like Add Guide for MiniMax H3, but anchors an already encoded frame, clip and/or audio "
            "(e.g. from Extract MiniMax H3 Latent Section). Chain several nodes to anchor several guides.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input(
                    "latent", tooltip="The MiniMax H3 AV latent that will be sampled."
                ),
                io.Latent.Input(
                    "guide",
                    tooltip="MiniMax H3 AV latent to anchor. Its width and height must match the target latent. "
                    "Clips are cropped down to 1 or 17k+5 frames. Only the first batch item is used. "
                    "An empty latent (from Load Latent List) adds no guide.",
                ),
                io.Int.Input(
                    "frame_idx",
                    default=0,
                    min=-9999,
                    max=9999,
                    tooltip="Frame to anchor the guide's first frame at. Negative values count from the end of the video.",
                ),
                io.Boolean.Input("use_video", default=True),
                io.Boolean.Input(
                    "use_audio",
                    default=True,
                    tooltip="Anchor the guide's audio as well, cropped to the video's remaining duration.",
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(
        cls, positive, latent, guide, frame_idx, use_video, use_audio
    ) -> io.NodeOutput:
        if is_empty_latent(guide):
            return io.NodeOutput(positive)
        video, audio = split_av_latent(latent, "CardamonMiniMaxH3AddLatentGuide")
        guide_video, guide_audio = split_av_latent(
            guide, "CardamonMiniMaxH3AddLatentGuide (guide input)"
        )
        if not use_video and not use_audio:
            raise ValueError("enable use_video and/or use_audio")
        frame_count = frames_for_tokens(video.shape[2])
        resolved = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if not 0 <= resolved < frame_count:
            raise ValueError(
                f"frame_idx {frame_idx} is outside the video's {frame_count} frames"
            )

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        existing = len(keyframes)

        if use_video:
            if guide_video.shape[3:] != video.shape[3:]:
                raise ValueError(
                    f"the guide is {guide_video.shape[4] * 16}x{guide_video.shape[3] * 16} "
                    f"but the target video is {video.shape[4] * 16}x{video.shape[3] * 16}"
                )
            guide_frames = valid_clip_frames(frames_for_tokens(guide_video.shape[2]))
            if resolved + guide_frames > frame_count:
                raise ValueError(
                    f"a {guide_frames} frame guide at frame_idx {frame_idx} does not fit in the video's {frame_count} frames"
                )
            keyframes.append(
                {
                    "resolved_frame_index": resolved,
                    "latent": guide_video[:1, :, : tokens_for_frames(guide_frames)],
                }
            )

        if use_audio and guide_audio.shape[-1] > 0:
            audio_frame = resolved + guide.get(AUDIO_FRAME_OFFSET_KEY, 0.0)
            max_rt = math.floor(audio.shape[-1] - FRAME_RESCALE * audio_frame)
            if max_rt < 1:
                raise ValueError(
                    f"frame_idx {frame_idx} is past the end of the video's audio track"
                )
            keyframes.append(
                {
                    "resolved_frame_index": audio_frame,
                    "audio_latent": guide_audio[:1, ..., :max_rt],
                }
            )

        if len(keyframes) == existing:
            raise ValueError("the guide has no audio to anchor; enable use_video")
        positive = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
        return io.NodeOutput(positive)


NODES = [CardamonMiniMaxH3ExtractLatent, CardamonMiniMaxH3AddLatentGuide]
