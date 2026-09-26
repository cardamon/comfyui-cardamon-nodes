# comfyui-cardamon-nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

Clone this repo into your ComfyUI `custom_nodes` directory and restart ComfyUI:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/cardamon/comfyui-cardamon-nodes.git
```

## Nodes

### MiniMax H3

- **Extract MiniMax H3 Latent Section**: cuts a section, typically the last frames, out of a
  MiniMax H3 AV latent (video and audio). Sections start on the VAE's 17-frame chunk boundaries and
  are 1 or 17k+5 frames long, which makes a section identical to VAE-encoding those frames on
  their own. Outputs the snapped start frame and length.
- **Add Latent Guide for MiniMax H3**: like the built-in *Add Guide for MiniMax H3*, but anchors an
  already encoded frame, clip and/or audio at a frame of the video being generated. The guide must
  have the same width and height as the target latent.

To continue a generated video, extract its last frames (e.g. `start_frame` -22) and anchor them
at `frame_idx` 0 of the next generation. The next video then starts with those frames, so skip
the first `length` frames of its decoded output when joining the two videos.

### Latent files

- **Save Latent (Output Dir)**: saves a latent to the output directory. Unlike the built-in
  *Save Latent*, it handles nested latents such as MiniMax H3 AV latents.
- **Load Latent (Output Dir)**: loads a `.latent` file straight from the output directory, so it
  doesn't have to be moved to the input directory first. It also reads files saved by the built-in
  *Save Latent*. Refresh the node definitions (press R) to list newly saved files.

## Development

Nodes use the ComfyUI V3 node API (`comfy_api.latest`). Each node lives in its own
module under `cardamon_nodes/` and is registered by adding its class to `NODES` in
`cardamon_nodes/__init__.py`. Node IDs start with `Cardamon`, and categories start with `cardamon/`.

Tests import ComfyUI's modules from a local ComfyUI checkout:

```sh
COMFYUI_PATH=/path/to/ComfyUI pytest
```
