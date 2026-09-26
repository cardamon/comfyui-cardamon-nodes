# comfyui-cardamon-nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

Clone this repo into your ComfyUI `custom_nodes` directory and restart ComfyUI:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/cardamon/comfyui-cardamon-nodes.git
```

## Development

Nodes use the ComfyUI V3 node API (`comfy_api.latest`). Each node lives in its own
module under `cardamon_nodes/` and is registered by adding its class to `NODES` in
`cardamon_nodes/__init__.py`. Node IDs start with `Cardamon`, and categories start with `cardamon/`.

Tests import ComfyUI's modules from a local ComfyUI checkout:

```sh
COMFYUI_PATH=/path/to/ComfyUI pytest
```
