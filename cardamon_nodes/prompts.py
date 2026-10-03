"""Prompt helpers for generating a video as a series of shots."""

import re

from comfy_api.latest import io

SHOT_PLACEHOLDER = re.compile(r"\{\{\s*SHOT\s*\}\}")
MAX_SHOTS = 50


def shot_index(name):
    # Autogrow names end in the input's number: "shot_0", "shot_1", ... "shot_10"
    return int(name.rsplit("_", 1)[-1])


def build_shot_prompts(template, shots):
    """One prompt per non-empty shot, with {{ SHOT }} in the template replaced by the shot prompt."""
    if not template.strip():
        template = "{{ SHOT }}"
    elif not SHOT_PLACEHOLDER.search(template):
        raise ValueError("the template has no {{ SHOT }} placeholder")
    prompts = [
        SHOT_PLACEHOLDER.sub(lambda _: shot.strip(), template)
        for shot in shots
        if shot and shot.strip()
    ]
    if not prompts:
        raise ValueError("all shot prompts are empty")
    return prompts


class CardamonNodesShotPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonNodesShotPrompts",
            display_name="Shot Prompts",
            category="Cardamon Nodes/prompt",
            description="Build one prompt per shot from a template. Outputs a list, so connected nodes run once per prompt. "
            "A new shot input appears as soon as the last one is connected. Empty shot prompts are skipped.",
            inputs=[
                io.String.Input(
                    "template",
                    multiline=True,
                    default="",
                    tooltip="Prompt with a {{ SHOT }} placeholder, replaced by each shot prompt. Leave empty to use the shot prompts as they are.",
                ),
                io.Autogrow.Input(
                    "shots",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input("shot", multiline=True),
                        prefix="shot_",
                        min=1,
                        max=MAX_SHOTS,
                    ),
                ),
            ],
            outputs=[io.String.Output(display_name="prompts", is_output_list=True)],
        )

    @classmethod
    def execute(cls, template, shots) -> io.NodeOutput:
        ordered = [shots[name] for name in sorted(shots, key=shot_index)]
        return io.NodeOutput(build_shot_prompts(template, ordered))


NODES = [CardamonNodesShotPrompts]
