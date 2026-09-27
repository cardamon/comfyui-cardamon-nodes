// UI for the LoRA Stack node (cardamon_nodes/loras.py): an Add LoRA button that opens a browser
// over the loras directory, and one compact row per LoRA with an enable toggle, its name, strength
// and a remove button. The stack itself lives in the node's hidden "loras" widget as JSON:
// [{"name", "strength", "enabled"}]; entries without "enabled" are enabled.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "CardamonLoraStack";
const ROW_HEIGHT = 24;
const ROW_GAP = 2;

const CSS = `
.cardamon-lora-rows { display: flex; flex-direction: column; gap: ${ROW_GAP}px; font-size: 12px; }
.cardamon-lora-empty { color: var(--descrip-text, #999); padding: 2px 4px; }
.cardamon-lora-row { display: flex; align-items: center; gap: 4px; height: ${ROW_HEIGHT}px; }
.cardamon-lora-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--input-text, #ddd); padding-left: 4px; }
.cardamon-lora-row.off .cardamon-lora-name, .cardamon-lora-row.off .cardamon-lora-strength { opacity: 0.4; }
.cardamon-lora-toggle { margin: 0 0 0 2px; width: 14px; height: 14px; flex: none; cursor: pointer;
  accent-color: var(--p-primary-color, #4a9eff); }
.cardamon-lora-strength { width: 56px; height: 20px; box-sizing: border-box; padding: 0 4px; text-align: right;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #444); border-radius: 4px; }
.cardamon-lora-remove { width: 20px; height: 20px; padding: 0; line-height: 18px; cursor: pointer;
  background: none; color: var(--descrip-text, #999); border: 1px solid transparent; border-radius: 4px; }
.cardamon-lora-remove:hover { color: var(--error-text, #f66); border-color: var(--border-color, #444); }

.cardamon-lora-overlay { position: fixed; inset: 0; z-index: 10000; background: rgba(0, 0, 0, 0.5);
  display: flex; align-items: center; justify-content: center; }
.cardamon-lora-dialog { width: min(560px, 92vw); max-height: 80vh; display: flex; flex-direction: column;
  background: var(--comfy-menu-bg, #353535); color: var(--fg-color, #fff);
  border: 1px solid var(--border-color, #555); border-radius: 8px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  font-size: 14px; }
.cardamon-lora-header { display: flex; align-items: center; gap: 8px; padding: 10px 12px;
  border-bottom: 1px solid var(--border-color, #555); }
.cardamon-lora-crumbs { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cardamon-lora-crumb { cursor: pointer; }
.cardamon-lora-crumb:hover { text-decoration: underline; }
.cardamon-lora-button { padding: 4px 10px; cursor: pointer; white-space: nowrap;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #555); border-radius: 4px; }
.cardamon-lora-button:disabled { opacity: 0.4; cursor: default; }
.cardamon-lora-list { overflow-y: auto; padding: 4px 0; }
.cardamon-lora-item { display: flex; gap: 8px; padding: 5px 14px; cursor: pointer; }
.cardamon-lora-item:hover { background: var(--comfy-input-bg, #222); }
.cardamon-lora-item.added { color: var(--descrip-text, #999); }
.cardamon-lora-icon { width: 1.2em; text-align: center; flex: none; }
.cardamon-lora-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cardamon-lora-message { padding: 12px 14px; color: var(--descrip-text, #999); }
`;

function addStyle() {
  if (document.getElementById("cardamon-lora-stack-style")) return;
  const style = document.createElement("style");
  style.id = "cardamon-lora-stack-style";
  style.textContent = CSS;
  document.head.append(style);
}

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// Paths from /models/loras use the server's separator; keep them as-is, since they are the names
// the node loads, and split on either separator for display.
function pathParts(name) {
  return name.split(/[\\/]/);
}

function displayName(name) {
  return pathParts(name).pop().replace(/\.safetensors$/i, "");
}

// ---- stack state -------------------------------------------------------------------------------

function stackWidget(node) {
  return node.widgets?.find((w) => w.name === "loras");
}

function readStack(node) {
  try {
    const stack = JSON.parse(stackWidget(node)?.value || "[]");
    return Array.isArray(stack) ? stack : [];
  } catch {
    return [];
  }
}

function writeStack(node, stack) {
  stackWidget(node).value = JSON.stringify(stack);
  renderRows(node);
  app.graph?.setDirtyCanvas(true, true);
}

function addLoras(node, names) {
  const stack = readStack(node);
  const present = new Set(stack.map((entry) => entry.name));
  for (const name of names) {
    if (!present.has(name)) stack.push({ name, strength: 1.0, enabled: true });
  }
  writeStack(node, stack);
}

// ---- rows on the node --------------------------------------------------------------------------

function rowsHeight(node) {
  const count = Math.max(readStack(node).length, 1);
  return count * ROW_HEIGHT + (count - 1) * ROW_GAP + 4;
}

function renderRows(node) {
  const container = node.cardamonLoraRows;
  if (!container) return;
  const stack = readStack(node);
  container.replaceChildren();
  if (stack.length === 0) {
    container.append(element("div", "cardamon-lora-empty", "No LoRAs added"));
  }
  stack.forEach((entry, index) => {
    const row = element("div", "cardamon-lora-row");
    const enabled = entry.enabled !== false;
    row.classList.toggle("off", !enabled);

    const toggle = element("input", "cardamon-lora-toggle");
    toggle.type = "checkbox";
    toggle.checked = enabled;
    toggle.title = enabled ? "Enabled: click to disable (the LoRA is not loaded at all)" : "Disabled: click to enable";
    toggle.addEventListener("change", () => {
      const current = readStack(node);
      current[index].enabled = toggle.checked;
      writeStack(node, current);
    });

    const name = element("span", "cardamon-lora-name", displayName(entry.name));
    name.title = entry.name;

    const strength = element("input", "cardamon-lora-strength");
    strength.type = "number";
    strength.step = "0.05";
    strength.value = String(entry.strength);
    strength.title = "Strength";
    strength.addEventListener("change", () => {
      const value = parseFloat(strength.value);
      const current = readStack(node);
      if (Number.isNaN(value)) {
        strength.value = String(current[index].strength);
        return;
      }
      current[index].strength = value;
      writeStack(node, current);
    });

    const remove = element("button", "cardamon-lora-remove", "✕");
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      const current = readStack(node);
      current.splice(index, 1);
      writeStack(node, current);
    });

    row.append(toggle, name, strength, remove);
    container.append(row);
  });
  // Grow or shrink the node to fit its rows.
  node.setSize([node.size[0], node.computeSize()[1]]);
}

// ---- file browser popup ------------------------------------------------------------------------

function buildTree(names) {
  const root = { dirs: new Map(), files: [] };
  for (const name of names) {
    const parts = pathParts(name);
    let dir = root;
    for (const part of parts.slice(0, -1)) {
      if (!dir.dirs.has(part)) dir.dirs.set(part, { dirs: new Map(), files: [] });
      dir = dir.dirs.get(part);
    }
    dir.files.push(name);
  }
  return root;
}

const byName = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });

async function openBrowser(node) {
  const overlay = element("div", "cardamon-lora-overlay");
  const dialog = element("div", "cardamon-lora-dialog");
  const header = element("div", "cardamon-lora-header");
  const crumbs = element("div", "cardamon-lora-crumbs");
  const addAll = element("button", "cardamon-lora-button");
  const close = element("button", "cardamon-lora-button", "Close");
  const list = element("div", "cardamon-lora-list");
  header.append(crumbs, addAll, close);
  dialog.append(header, list);
  overlay.append(dialog);

  const dismiss = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey, true);
  };
  const onKey = (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      dismiss();
    }
  };
  close.addEventListener("click", dismiss);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) dismiss();
  });
  document.addEventListener("keydown", onKey, true);
  document.body.append(overlay);

  list.append(element("div", "cardamon-lora-message", "Loading…"));
  let tree;
  try {
    const response = await api.fetchApi("/models/loras");
    tree = buildTree(await response.json());
  } catch (error) {
    list.replaceChildren(element("div", "cardamon-lora-message", `Could not list LoRAs: ${error}`));
    return;
  }

  const path = [];
  const render = () => {
    let dir = tree;
    for (const part of path) dir = dir.dirs.get(part);
    const added = new Set(readStack(node).map((entry) => entry.name));

    crumbs.replaceChildren();
    ["loras", ...path].forEach((part, depth) => {
      if (depth > 0) crumbs.append(" / ");
      const crumb = element("span", "cardamon-lora-crumb", part);
      crumb.addEventListener("click", () => {
        path.length = depth;
        render();
      });
      crumbs.append(crumb);
    });

    const files = [...dir.files].sort((a, b) => byName(displayName(a), displayName(b)));
    const toAdd = files.filter((name) => !added.has(name));
    addAll.textContent = `Add all (${toAdd.length})`;
    addAll.title = "Add every LoRA in this folder, without its subfolders";
    addAll.disabled = toAdd.length === 0;
    addAll.onclick = () => {
      addLoras(node, toAdd);
      render();
    };

    list.replaceChildren();
    const item = (icon, label, className, onClick) => {
      const row = element("div", `cardamon-lora-item ${className}`);
      row.append(element("span", "cardamon-lora-icon", icon), element("span", "cardamon-lora-label", label));
      row.addEventListener("click", onClick);
      list.append(row);
      return row;
    };
    if (path.length > 0) {
      item("↩", "..", "", () => {
        path.pop();
        render();
      });
    }
    for (const name of [...dir.dirs.keys()].sort(byName)) {
      item("📁", name, "", () => {
        path.push(name);
        render();
      });
    }
    for (const name of files) {
      const isAdded = added.has(name);
      const row = item(isAdded ? "✓" : "", displayName(name), isAdded ? "added" : "", () => {
        addLoras(node, [name]);
        render();
      });
      row.title = isAdded ? `${name} (already in the stack)` : name;
    }
    if (dir.dirs.size === 0 && files.length === 0) {
      list.append(element("div", "cardamon-lora-message", "No LoRAs here"));
    }
  };
  render();
}

// ---- node setup --------------------------------------------------------------------------------

app.registerExtension({
  name: "cardamon.LoraStack",

  nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;
    addStyle();

    const hidden = stackWidget(node);
    if (hidden) {
      hidden.hidden = true;
      hidden.options = { ...hidden.options, hidden: true };
      hidden.computeSize = () => [0, -4];
    }

    const button = node.addWidget("button", "add_lora", null, () => openBrowser(node), { serialize: false });
    button.label = "Add LoRA";

    const container = element("div", "cardamon-lora-rows");
    node.cardamonLoraRows = container;
    node.addDOMWidget("lora_rows", "cardamon_lora_rows", container, {
      serialize: false,
      getMinHeight: () => rowsHeight(node),
      getMaxHeight: () => rowsHeight(node),
    });

    // Loading a workflow sets the hidden widget's value after the node is created.
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      renderRows(this);
      return result;
    };
    renderRows(node);
  },
});
