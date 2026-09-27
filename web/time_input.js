// Shows the Time node's value in seconds as you type (cardamon_nodes/utilities.py).
// parseTime follows the same rules as parse_time in Python; keep the two in sync.

import { app } from "../../scripts/app.js";

const NODE_CLASS = "CardamonTime";
const SECONDS_FIELD = /^\d+(?:[.,]\d*)?$/;
const WHOLE_FIELD = /^\d+$/;

export function parseTime(text) {
  let value = text.trim();
  let sign = 1;
  if (value.startsWith("-")) {
    sign = -1;
    value = value.slice(1).trim();
  }
  const fields = value.split(":");
  const last = fields[fields.length - 1];
  if (fields.length > 3 || !SECONDS_FIELD.test(last) || !fields.slice(0, -1).every((f) => WHOLE_FIELD.test(f))) {
    return null;
  }
  const whole = fields.slice(0, -1).map((f) => parseInt(f, 10));
  const seconds = parseFloat(last.replace(",", "."));
  if (fields.length > 1 && (seconds >= 60 || whole.slice(1).some((v) => v >= 60))) {
    return null;
  }
  let total = 0;
  for (const v of whole) total = total * 60 + v;
  return sign * (whole.length ? total * 60 + seconds : seconds);
}

function preview(text) {
  if (typeof text !== "string") return { text: "", error: false };
  const seconds = parseTime(text);
  if (seconds === null) return { text: "Invalid time, expected hh:mm:ss.fff", error: true };
  return { text: `= ${seconds} s`, error: false };
}

app.registerExtension({
  name: "cardamon.Time",

  nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;
    const widget = node.widgets?.find((w) => w.name === "time");
    if (!widget) return;

    const label = document.createElement("div");
    label.style.cssText = "font-size: 12px; line-height: 18px; padding: 0 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;";
    // The widget's height includes its margin above and below the element.
    node.addDOMWidget("seconds_preview", "cardamon_time_preview", label, {
      serialize: false,
      margin: 2,
      getMinHeight: () => 22,
      getMaxHeight: () => 22,
    });
    // Make room for the preview line.
    node.setSize([node.size[0], node.computeSize()[1]]);

    const update = () => {
      const { text, error } = preview(widget.value);
      label.textContent = text;
      label.style.color = error ? "var(--error-text, #f66)" : "var(--descrip-text, #999)";
      app.graph?.setDirtyCanvas(true, true);
    };

    const callback = widget.callback;
    widget.callback = function (...args) {
      const result = callback?.apply(this, args);
      update();
      return result;
    };
    // Loading a workflow sets the value after the node is created.
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      update();
      return result;
    };
    update();
  },
});
