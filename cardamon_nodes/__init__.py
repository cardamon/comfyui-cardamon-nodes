from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# Every node class this package provides. Add new nodes here.
NODES: list[type[io.ComfyNode]] = []


class CardamonExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> CardamonExtension:
    return CardamonExtension()
