# comfyui-cardamon-nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

Clone this repo into your ComfyUI `custom_nodes` directory and restart ComfyUI:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/cardamon/comfyui-cardamon-nodes.git
# or using ssh: git clone git@github.com:cardamon/comfyui-cardamon-nodes.git
```

## Nodes

### MiniMax H3

- **Extract MiniMax H3 Latent Section**: cuts a section, typically the last frames, out of a
  MiniMax H3 AV latent (video and audio). Sections start on the VAE's 17-frame chunk boundaries and
  are 1 or 17k+5 frames long, which makes a section identical to VAE-encoding those frames on
  their own. Outputs the snapped start frame and length.
- **Add Latent Guide for MiniMax H3**: like the built-in *Add Guide for MiniMax H3*, but anchors an
  already encoded frame, clip and/or audio at a frame of the video being generated. The guide must
  have the same width and height as the target latent. An empty guide (e.g. on a loop's first
  iteration) adds no guide, and *Extract* passes it through unchanged.

To continue a generated video, extract its last frames (e.g. `start_frame` -22) and anchor them
at `frame_idx` 0 of the next generation. The next video then starts with those frames, so skip
the first `length` frames of its decoded output when joining the two videos.

### Prompts

- **Shot Prompts**: builds one prompt per shot from a template, for generating a longer video as a
  series of shots. Every `{{ SHOT }}` in the template is replaced by the shot prompt, and an empty
  template uses the shot prompts as they are. Connect a text node (e.g. *Text (Multiline)*) to
  each shot input; a new input appears when the last one is connected, and empty shot prompts are
  skipped. The output is a list, so the nodes it connects to run once per prompt.

### Latent files

- **Save Latent (Output Dir)**: saves a latent to the output directory. Unlike the built-in
  *Save Latent*, it handles nested latents such as MiniMax H3 AV latents.
- **Load Latent (Output Dir)**: loads a `.latent` file straight from the output directory, so it
  doesn't have to be moved to the input directory first. It also reads files saved by the built-in
  *Save Latent*. Refresh the node definitions (press R) to list newly saved files.

## Generating shots in a loop

ComfyUI's built-in *Start Loop* and *End Loop* nodes run the shots one after another in a single
run, each continuing from the previous one:

- *Shot Prompts* → *Start Loop* in `List` mode. Use `list_item` as the shot's prompt.
- *Add Latent Guide for MiniMax H3*: connect `current_iteration_value` to `guide` and anchor it at
  `frame_idx` 0. On the first shot there is no value yet, and no guide is added.
- Sample the shot, then *Extract MiniMax H3 Latent Section* (e.g. `start_frame` -22) →
  `next_iteration_value` of *End Loop*.
- *Save Latent (Output Dir)* on the sampled latent, connected to one of End Loop's `termination`
  inputs, saves each shot as soon as it has been generated.
- Connect the sampled latent to `output_value` and enable `accumulate` to get all shots as a list.

## Tips

- **Trim after the VAE decode, not before.** The H3 video decoder decodes each 17-frame chunk
  together with the start of the next one. A latent trimmed before decoding loses that context, so
  the frames just before the cut decode differently than in the full video.
- **For the smoothest audio, start sections at a multiple of 51 frames.** Audio latents run at 40
  per second and video at 24 fps, so they only line up every 51 frames (every third 17-frame
  chunk). At other start frames, the guide's audio begins up to about 25 ms after its first video
  frame. For example, the last 22 frames of a 124-frame video start at frame 102 (2 × 51).

## Development

Nodes use the ComfyUI V3 node API (`comfy_api.latest`). Each node lives in its own
module under `cardamon_nodes/` and is registered by adding its class to `NODES` in
`cardamon_nodes/__init__.py`. Node IDs start with `Cardamon`, and categories start with `cardamon/`.

Tests import ComfyUI's modules from a local ComfyUI checkout:

```sh
COMFYUI_PATH=/path/to/ComfyUI pytest
```
