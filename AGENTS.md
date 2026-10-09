# comfyui-cardamon-nodes

A ComfyUI custom node pack, installed by cloning it into `ComfyUI/custom_nodes/`. The README
describes every node for users; this file is for working on the code.

## Layout and conventions

- `__init__.py` (repo root) is ComfyUI's entry point: it sets `WEB_DIRECTORY = "./web"` and imports
  `comfy_entrypoint` only inside `if __package__:`, because pytest also imports this file standalone.
- `cardamon_nodes/<area>.py` holds the nodes, one module per area (images, latent_files, loras,
  minimax_h3, prompts, utilities, video). Each module ends with a `NODES` list, which
  `cardamon_nodes/__init__.py` collects.
- Use the V3 node API (`comfy_api.latest`: `io.ComfyNode`, `define_schema`, `execute`), as most
  built-in ComfyUI nodes do now.
- Node IDs start with `CardamonNodes`, and categories with `Cardamon Nodes/`.
  `tests/test_registration.py` enforces both.
- `web/*.js` are frontend extensions, one per node that needs custom UI.
- Use f-strings, and match the existing formatting (Black style).
- Update the README's node descriptions together with the code.

## ComfyUI behaviour worth knowing

- **Lists:** an `is_output_list` output makes downstream nodes run once per item, but every item
  runs through a node before anything downstream starts. So list items can't feed each other in
  one run; use the built-in **Start Loop / End Loop** for that. Lists that meet at one node pair up
  by position, not as a cross product.
- **Loops:** in Start Loop's `List` mode, the carried value (`current_iteration_value`) is `None` on
  the first iteration, so nodes that take it must accept `None`. Output nodes inside a loop must be
  connected to one of End Loop's `termination` inputs, or the run is refused with "Loop body is not
  closed". A node with no output, like Save Video, can't be inside a loop at all.
- **Autogrow inputs** turn widget inputs into sockets (`force_input`). Their values arrive as a dict,
  so sort them by the number at the end of the name. In API prompts, the inputs are named
  `<group>.<prefix><n>`, e.g. `shots.shot_0`.
- **No dynamic outputs:** every link is checked against the outputs declared in the schema. Declare
  a maximum and let the frontend show only the ones in use, removing slots only from the end so
  the remaining indexes stay the same (see Load Images (Paths)).
- **`fingerprint_inputs`** only receives widget values, not linked values. With `is_input_list`,
  values arrive as lists, but `validate_inputs` may still get plain values, so handle both.
- **Partial runs:** `app.queuePrompt(0, 1, [String(node.id)])` runs only that node and what it
  depends on, and only works for output nodes (`is_output_node=True`).
- **Paths:** `folder_paths.get_annotated_filepath` refuses any path outside its base directory.
- **Video:** `InputImpl.VideoFromList` encodes its parts lossily (H.264 CRF 18 or AV1 CRF 24) as soon
  as it's constructed. Save Video only copies them without re-encoding when the codec matches and
  no CRF is set. `get_frame_count()` on file-backed videos is an estimate.
- **H3 noise masks:** a `noise_mask` on an H3 latent is a nested pair, video `[B,1,T,H,W]` and audio
  `[B,1,2,T]`, where 0 keeps a token and 1 generates it (ComfyUI commit ff6c8a8).

## Frontend (web/)

- **Assume Nodes 2.0 mode** (`Comfy.VueNodes.Enabled`). Features don't need to work in the classic
  canvas mode. Its color picker gives `#RRGGBBAA`; the canvas mode's picker has no alpha.
- **Imports:** `../../scripts/app.js` and `api.js`; set things up in `registerExtension({ nodeCreated })`.
- **State:** keep node state as JSON in a widget the UI edits and hides (`widget.hidden = true`,
  `options.hidden = true`, `computeSize = () => [0, -4]`), and draw the UI with `node.addDOMWidget`.
- **Workflow loading:** the saved values are set after `nodeCreated`, so re-render in `onConfigure`.
  Check `app.configuringGraph` to avoid acting while a workflow loads.
- **DOM widget height includes its margin,** 10 px above and below by default. Set `margin`
  explicitly and include `2 * margin` in `getMinHeight` / `getMaxHeight`, or the content gets
  squeezed or overflows. After changes, call `node.setSize([w, node.computeSize()[1]])`.
- **Confirmation dialog:** `app.extensionManager.dialog.confirm({ title, message, type: "delete" })`
  resolves to true or false.

## MiniMax H3

The H3 code in ComfyUI is in `comfy/ldm/minimax/{model,vae,audio_vae}.py`,
`comfy_extras/nodes_minimax_h3.py` and `comfy/model_base.py` (`MiniMaxH3`).

- **Latents:** an AV latent is a `NestedTensor`: video `[B,24,T,H/16,W/16]` plus audio `[B,32,2,T_audio]`.
  Video is 24 fps; audio latents run at 40 per second, so one frame is 5/3 audio latents.
- **Chunks:** the video VAE encodes independent 17-frame chunks into 5 tokens each, covering
  (1, 4, 4, 4, 4) frames. Valid lengths are 1 frame or 17k+5 frames (5k+2 tokens). A slice starting on
  a chunk boundary (a frame index divisible by 17) is identical to VAE-encoding those frames on
  their own. Only frames at multiples of 17 exist as a single token.
- **Audio alignment:** chunk boundaries line up with whole audio latents only every 51 frames.
  `cardamon_audio_frame_offset` in a latent dict records the offset of an extracted section's
  audio. Shot lengths of 51k+22 frames make the last-22-frame sections start exactly on one.
- **Decoding:** the decoder works in 7-token windows (a chunk plus 2 tokens of look-ahead) and
  blends 5 frames between windows. Decoding a latent trimmed before decode changes its last ~15
  frames, so trim after decoding, or join the latents and decode once.
- **Guides:** `minimax_keyframes` in conditioning are `{resolved_frame_index, latent, audio_latent}`.
  The model accepts fractional frame indexes. Keyframes compete with reference videos (`refs`) on the
  reference model; a section prepended to the latent with a noise mask doesn't.
- **Prompts:** see the prompt guides on Hugging Face (MiniMaxAI/MiniMax-H3, `docs/VIDEO_PROMPT_WRITING_GUIDE_*_en.md`).
  They have three sections: `integrated_multimodal_description`, `overall_soundscape` and
  `non_diegetic_music`.

## Testing

- **Python:** use the conda env `py312` (`conda run -n py312 …`). It has torch and all of ComfyUI's
  requirements. Run the tests with `COMFYUI_PATH=<ComfyUI checkout> pytest`, which uses the real
  `comfy_api`.
- **ComfyUI clone:** use a throwaway one, so the real install isn't touched:
  `git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git`, then symlink this repo to
  `custom_nodes/comfyui-cardamon-nodes`.
  - `python main.py --cpu --quick-test-for-ci` checks that the nodes load.
  - `python main.py --cpu --port 8199` plus `POST /prompt` and `GET /history/<id>` runs workflows
    through the API.
- **UI tests:** Playwright in its own venv (not py312), driving `/usr/bin/chromium` headless against
  the clone with Nodes 2.0 mode switched on. Check behaviour with screenshots and
  `page.evaluate` on `app.graph`.
- **Model weights:** none are available locally. Check structural behaviour with randomly
  initialised models (e.g. an SD VAE built from `AutoencoderKL` with the default `ddconfig`, or
  parts of the H3 VAE). Real-model results come from the user's own ComfyUI install.
