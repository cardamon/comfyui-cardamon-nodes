import asyncio

import pytest

from cardamon_nodes import NODES, CardamonNodesExtension, comfy_entrypoint


def test_entrypoint_returns_extension_with_all_nodes():
    extension = asyncio.run(comfy_entrypoint())
    assert isinstance(extension, CardamonNodesExtension)
    assert asyncio.run(extension.get_node_list()) == NODES


def test_node_ids_are_unique():
    node_ids = [node.GET_SCHEMA().node_id for node in NODES]
    assert len(node_ids) == len(set(node_ids))


@pytest.mark.parametrize("node", NODES, ids=lambda n: n.__name__)
def test_node_schema_is_namespaced(node):
    # Prefixes keep our nodes from clashing with built-in or other custom nodes.
    schema = node.GET_SCHEMA()
    assert schema.node_id.startswith("CardamonNodes")
    assert schema.category.split("/")[0] == "Cardamon Nodes"
