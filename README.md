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

### Video

- **Join Shot Videos**: joins a list of shot videos, such as *Create Video*'s output after an
  accumulating *End Loop*, into one video. It cuts `trim_frames` frames off the end of every shot
  but the last: the frames the next shot continues from. Each shot's audio is cut at the same
  point on the joined timeline, so it stays in sync however many shots there are.

### LoRAs

- **LoRA Stack**: applies several LoRAs to a model, and to a text encoder if `clip` is connected.
  *Add LoRA* opens a browser over the `loras` model directory: click a folder to open it, click
  a LoRA to add it, or use *Add all* to add every LoRA in the current folder (not its subfolders).
  Each LoRA takes one row on the node, with an enable toggle, its name, its strength and a button
  to remove it. A disabled LoRA is not loaded at all, and keeps its strength for when it's enabled
  again. The checkbox next to *Add LoRA* disables all LoRAs when all are enabled, and otherwise
  enables all; it shows a dash when only some are enabled. The strength applies to both the model
  and the text encoder.

### Latent files

- **Save Latent (Output Dir)**: saves a latent to the output directory. Unlike the built-in
  *Save Latent*, it handles nested latents such as MiniMax H3 AV latents.
- **Load Latent (Output Dir)**: loads a `.latent` file straight from the output directory, so it
  doesn't have to be moved to the input directory first. It also reads files saved by the built-in
  *Save Latent*. Refresh the node definitions (press R) to list newly saved files.

## Generating shots in a loop

ComfyUI's built-in *Start Loop* and *End Loop* nodes run the shots one after another in a single
run, each continuing from the previous one. The example workflow
[`example_workflows/minimax-h3-seamless-multishot.json`](example_workflows/minimax-h3-seamless-multishot.json)
is set up like this:

1. **Prompts:** *Shot Prompts* builds one prompt per shot from a template, and its list goes to
   *Start Loop* in `List` mode. Everything between *Start Loop* and *End Loop* runs once per shot.
2. **Guide from the previous shot:** *MiniMax H3 Image to Video* gets `list_item` as its prompt.
   *Add Latent Guide for MiniMax H3* anchors `current_iteration_value` at `frame_idx` 0. On the
   first shot there is nothing to continue from yet, so no guide is added.
3. **Sample and pass on:** after sampling, *Extract MiniMax H3 Latent Section* (`start_frame` -22)
   takes the shot's last frames. They go to End Loop's `next_iteration_value` and become the next
   shot's guide.
4. **Save as you go (optional):** *Save Latent (Output Dir)* saves each extracted section as soon
   as its shot is done. It's connected to one of End Loop's `termination` inputs, which makes it
   run on every iteration. The loop doesn't need it, because End Loop passes the section on
   directly.
5. **Decode after the loop:** the sampled latent goes to End Loop's `output_value` with
   `accumulate` on, so End Loop outputs every shot as a list. *VAE Decode*, *VAE Decode Audio*,
   *Create Video* and *Save Video* come after End Loop and run once per shot.

Output nodes such as *Save Video* can't be inside the loop, because nothing can connect them to
End Loop. If they are, ComfyUI refuses the run with "Loop body is not closed".

The example workflow saves every shot as its own video, and each shot after the first starts
with the 22 frames it continues from. To get one seamless video instead, connect *Create Video*
to *Join Shot Videos* with `trim_frames` 22, and that to *Save Video*.

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
