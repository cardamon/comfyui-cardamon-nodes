// Load Latent (Output Dir) (cardamon_nodes/latent_files.py): when a VAE is connected, selecting a
// latent, or connecting the VAE, runs just this node and the nodes its VAE depends on (a partial
// run), which decodes the latent's last frame and shows it on the node.

import { app } from "../../scripts/app.js";

const NODE_CLASS = "CardamonNodesLoadLatent";

function vaeConnected(node) {
  const input = node.inputs?.find((i) => i.name === "vae");
  return input?.link != null;
}

function preview(node) {
  if (!vaeConnected(node)) return;
  // queueNodeIds limits the run to this (output) node and what it depends on.
  app.queuePrompt(0, 1, [String(node.id)]);
}

app.registerExtension({
  name: "cardamon.LoadLatentPreview",

  nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;
    const widget = node.widgets?.find((w) => w.name === "latent");
    if (!widget) return;

    const callback = widget.callback;
    widget.callback = function (...args) {
      const result = callback?.apply(this, args);
      preview(node);
      return result;
    };

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, link, input) {
      const result = onConnectionsChange?.apply(this, arguments);
      // Connecting a VAE while the graph is being loaded shouldn't start a run.
      if (connected && input?.name === "vae" && !app.configuringGraph) preview(this);
      return result;
    };
  },
});
