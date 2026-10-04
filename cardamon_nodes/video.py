"""Join shot videos into one video."""

import torch
from comfy_api.latest import AudioInput, Input, InputImpl, Types, io


def kept_frames(components, trim_frames):
    """Frames kept from each video: all but `trim_frames` off the end of every video but the last."""
    frame_rate = components[0].frame_rate
    size = components[0].images.shape[1:]
    for i, c in enumerate(components):
        if c.frame_rate != frame_rate:
            raise ValueError(
                f"video {i + 1} is {c.frame_rate} fps, video 1 is {frame_rate} fps"
            )
        if c.images.shape[1:] != size:
            raise ValueError(
                f"video {i + 1} is {c.images.shape[2]}x{c.images.shape[1]}, video 1 is {size[1]}x{size[0]}"
            )
    last = len(components) - 1
    kept = [
        c.images.shape[0] - (trim_frames if i < last else 0)
        for i, c in enumerate(components)
    ]
    for i, frames in enumerate(kept):
        if frames < 1:
            raise ValueError(
                f"video {i + 1} has {components[i].images.shape[0]} frames, too few to cut {trim_frames}"
            )
    return kept


def join_audio(components, kept, frame_rate):
    """The videos' audio joined into one track, each cut where its kept frames end.

    Cut points are sample positions on the joined timeline, rounded once, so audio that runs
    slightly longer or shorter than its video can't drift out of sync over many shots.
    """
    audios = [c.audio for c in components]
    if all(a is None for a in audios):
        return None
    if any(a is None for a in audios):
        raise ValueError("some videos have audio and some don't")
    sample_rate = audios[0]["sample_rate"]
    channels = audios[0]["waveform"].shape[1]
    for i, a in enumerate(audios):
        if a["sample_rate"] != sample_rate or a["waveform"].shape[1] != channels:
            raise ValueError(f"video {i + 1}'s audio format differs from video 1's")

    parts = []
    start_frame = 0
    for audio, frames in zip(audios, kept):
        start = round(start_frame / frame_rate * sample_rate)
        end = round((start_frame + frames) / frame_rate * sample_rate)
        waveform = audio["waveform"][..., : end - start]
        if waveform.shape[-1] < end - start:
            waveform = torch.nn.functional.pad(
                waveform, (0, end - start - waveform.shape[-1])
            )
        parts.append(waveform)
        start_frame += frames
    return AudioInput(
        {"waveform": torch.cat(parts, dim=-1), "sample_rate": sample_rate}
    )


def join_shot_videos(videos, trim_frames, codec):
    """Join the videos into one, without holding all kept frames in memory a second time.

    Each shot's kept frames are a view of its frames, encoded on its own into a compressed part
    (VideoFromList does this one part at a time). The joined audio is added once for the whole
    video, rather than per part, so per-part audio encoding can't shift it.
    """
    components = [v.get_components() for v in videos]
    kept = kept_frames(components, trim_frames)
    frame_rate = components[0].frame_rate
    first = videos[0]
    color_space = first.get_color_space()
    parts: list[Input.Video] = [
        InputImpl.VideoFromComponents(
            Types.VideoComponents(images=c.images[:frames], frame_rate=frame_rate),
            bit_depth=first.get_bit_depth(),
            color_space="sRGB" if color_space == "auto" else color_space,
        )
        for c, frames in zip(components, kept)
    ]
    return InputImpl.VideoFromList(
        parts,
        complete_audio=join_audio(components, kept, frame_rate),
        codec=Types.VideoCodec(codec),
    )


class CardamonNodesJoinShotVideos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonNodesJoinShotVideos",
            display_name="Join Shot Videos",
            category="Cardamon Nodes/video",
            description="Join a list of shot videos into one video, cutting frames off the end of every shot but the last, "
            "e.g. the frames the next shot continues from. Audio is cut at the same points.",
            inputs=[
                io.Video.Input(
                    "videos",
                    tooltip="Shot videos in order, e.g. a list from Create Video after an accumulating End Loop.",
                ),
                io.Int.Input(
                    "trim_frames",
                    default=22,
                    min=0,
                    max=9999,
                    tooltip="Frames to cut off the end of every video except the last.",
                ),
                io.Combo.Input(
                    "codec",
                    options=Types.VideoCodec.as_input(),
                    default="auto",
                    tooltip="Codec the shots are compressed with while joining (auto: H.264). Use the codec Save Video "
                    "uses, so it can copy the result instead of encoding it a second time.",
                ),
            ],
            outputs=[io.Video.Output()],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, videos, trim_frames, codec=None) -> io.NodeOutput:
        return io.NodeOutput(
            join_shot_videos(videos, trim_frames[0], codec[0] if codec else "auto")
        )


NODES = [CardamonNodesJoinShotVideos]
