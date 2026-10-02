from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from . import images, latent_files, loras, minimax_h3, prompts, utilities, video

# Every node class this package provides. Add new nodes here.
NODES: list[type[io.ComfyNode]] = [
    *images.NODES,
    *latent_files.NODES,
    *loras.NODES,
    *minimax_h3.NODES,
    *prompts.NODES,
    *utilities.NODES,
    *video.NODES,
]


class CardamonExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> CardamonExtension:
    return CardamonExtension()
