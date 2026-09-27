"""ComfyUI entry point. ComfyUI imports this package from custom_nodes/ and calls comfy_entrypoint()."""

# Frontend extensions (e.g. the LoRA Stack's UI)
WEB_DIRECTORY = "./web"

# pytest also imports this file, but as a standalone module where the relative import would fail.
if __package__:
    from .cardamon_nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
