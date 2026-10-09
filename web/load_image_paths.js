// UI for Load Images (Paths) (cardamon_nodes/images.py): one path field per image plus an empty
// one at the end. Typing in the empty field adds an image output and a new empty field below it.
// A Clear all button (with confirmation) removes every path and output.
// The paths live in the node's hidden "paths" widget as a JSON list. The node declares
// MAX_IMAGE_PATHS outputs; only the first paths.length are shown, so output indexes never shift.

import { app } from "../../scripts/app.js";

const NODE_CLASS = "CardamonNodesLoadImagePaths";
const MAX_PATHS = 50;
const ROW_HEIGHT = 26;
const ROW_GAP = 2;
const MARGIN = 2;

const CSS = `
.cardamon-paths { display: flex; flex-direction: column; gap: ${ROW_GAP}px; font-size: 12px; }
.cardamon-paths-header { display: flex; justify-content: flex-end; height: ${ROW_HEIGHT}px; align-items: center; }
.cardamon-paths-clear { height: 22px; padding: 0 8px; cursor: pointer; font-size: 12px;
  background: var(--comfy-input-bg, #222); color: var(--descrip-text, #999);
  border: 1px solid var(--border-color, #444); border-radius: 4px; }
.cardamon-paths-clear:hover:not(:disabled) { color: var(--error-text, #f66); }
.cardamon-paths-clear:disabled { opacity: 0.4; cursor: default; }
.cardamon-paths-input { height: ${ROW_HEIGHT}px; box-sizing: border-box; padding: 0 6px; width: 100%;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #444); border-radius: 4px; font-size: 12px; }
.cardamon-paths-input::placeholder { color: var(--descrip-text, #888); }
`;

function addStyle() {
  if (document.getElementById("cardamon-paths-style")) return;
  const style = document.createElement("style");
  style.id = "cardamon-paths-style";
  style.textContent = CSS;
  document.head.append(style);
}

function pathsWidget(node) {
  return node.widgets?.find((w) => w.name === "paths");
}

function readPaths(node) {
  try {
    const paths = JSON.parse(pathsWidget(node)?.value || "[]");
    return Array.isArray(paths) ? paths.map(String) : [];
  } catch {
    return [];
  }
}

function writePaths(node, paths) {
  pathsWidget(node).value = JSON.stringify(paths);
  app.graph?.setDirtyCanvas(true, true);
}

function fileName(path) {
  const trimmed = path.trim().replace(/^["']|["']$/g, "");
  return trimmed.split(/[\\/]/).filter(Boolean).pop() || "";
}

// Show exactly one output per path, labelled with its file name.
function syncOutputs(node, paths) {
  node.outputs ??= [];
  while (node.outputs.length > paths.length) node.removeOutput(node.outputs.length - 1);
  while (node.outputs.length < paths.length) node.addOutput(`image_${node.outputs.length}`, "IMAGE");
  paths.forEach((path, i) => {
    node.outputs[i].label = fileName(path) || `image ${i + 1}`;
  });
}

function contentHeight(fields) {
  const rows = fields + 1; // header + fields
  return rows * ROW_HEIGHT + (rows - 1) * ROW_GAP + 2 * MARGIN;
}

function fitNode(node) {
  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph?.setDirtyCanvas(true, true);
}

function render(node) {
  const container = node.cardamonPaths;
  if (!container) return;
  const paths = readPaths(node);
  syncOutputs(node, paths);

  const header = document.createElement("div");
  header.className = "cardamon-paths-header";
  const clear = document.createElement("button");
  clear.className = "cardamon-paths-clear";
  clear.textContent = "Clear all";
  clear.title = "Remove all paths and image outputs";
  clear.disabled = paths.length === 0;
  clear.addEventListener("click", () => clearAll(node));
  header.append(clear);
  container.replaceChildren(header);

  const addField = (value, index) => {
    const input = document.createElement("input");
    input.className = "cardamon-paths-input";
    input.type = "text";
    input.value = value;
    input.placeholder = index === 0 ? "Path to an image" : "Path to another image";
    input.spellcheck = false;
    input.title = value; // long paths don't fit
    input.addEventListener("input", () => {
      input.title = input.value;
      const current = readPaths(node);
      if (index < current.length) {
        current[index] = input.value;
      } else if (input.value !== "") {
        // Typing in the empty last field: it becomes a path, with an output and a new empty field.
        if (current.length >= MAX_PATHS) return;
        current.push(input.value);
        if (current.length < MAX_PATHS) {
          container.append(addField("", current.length));
          node.cardamonFields = current.length + 1;
        }
        clear.disabled = false;
      } else {
        return;
      }
      writePaths(node, current);
      syncOutputs(node, current);
      fitNode(node);
    });
    return input;
  };
  paths.forEach((path, i) => container.append(addField(path, i)));
  if (paths.length < MAX_PATHS) container.append(addField("", paths.length));
  node.cardamonFields = Math.min(paths.length + 1, MAX_PATHS);
  fitNode(node);
}

async function clearAll(node) {
  const message = "Remove all image paths, and the outputs and links that belong to them?";
  const dialog = app.extensionManager?.dialog;
  const confirmed = dialog?.confirm
    ? await dialog.confirm({ title: "Clear all paths", message, type: "delete" })
    : window.confirm(message);
  if (!confirmed) return;
  writePaths(node, []);
  render(node);
}

app.registerExtension({
  name: "cardamon.LoadImagePaths",

  nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;
    addStyle();
    const hidden = pathsWidget(node);
    if (hidden) {
      hidden.hidden = true;
      hidden.options = { ...hidden.options, hidden: true };
      hidden.computeSize = () => [0, -4];
    }

    const container = document.createElement("div");
    container.className = "cardamon-paths";
    node.cardamonPaths = container;
    node.addDOMWidget("path_fields", "cardamon_path_fields", container, {
      serialize: false,
      margin: MARGIN,
      getMinHeight: () => contentHeight(node.cardamonFields ?? 1),
      getMaxHeight: () => contentHeight(node.cardamonFields ?? 1),
    });

    // Loading a workflow sets the paths (and the saved outputs) after the node is created.
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      render(this);
      return result;
    };
    render(node);
  },
});
