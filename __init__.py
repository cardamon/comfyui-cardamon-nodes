"""ComfyUI entry point. ComfyUI imports this package from custom_nodes/ and calls comfy_entrypoint()."""

# pytest also imports this file, but as a standalone module where the relative import would fail.
if __package__:
    from .cardamon_nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
