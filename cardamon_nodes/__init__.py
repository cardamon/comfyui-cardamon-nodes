from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from . import latent_files, minimax_h3, prompts

# Every node class this package provides. Add new nodes here.
NODES: list[type[io.ComfyNode]] = [
    *latent_files.NODES,
    *minimax_h3.NODES,
    *prompts.NODES,
]


class CardamonExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> CardamonExtension:
    return CardamonExtension()
