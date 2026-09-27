from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from . import latent_files, loras, minimax_h3, prompts, video

# Every node class this package provides. Add new nodes here.
NODES: list[type[io.ComfyNode]] = [
    *latent_files.NODES,
    *loras.NODES,
    *minimax_h3.NODES,
    *prompts.NODES,
    *video.NODES,
]


class CardamonExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> CardamonExtension:
    return CardamonExtension()
