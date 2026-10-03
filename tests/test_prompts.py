import pytest

from cardamon_nodes.prompts import CardamonNodesShotPrompts, build_shot_prompts

TEMPLATE = """integrated_multimodal_description: [Shot 1] {{ SHOT }}

overall_soundscape: Quiet indoor tone."""


def run(template, **shots):
    return CardamonNodesShotPrompts.execute(template, shots).args[0]


def test_one_prompt_per_shot_in_input_order():
    prompts = run(TEMPLATE, shot_0="A man enters.", shot_1="He sits down.")
    assert prompts == [
        TEMPLATE.replace("{{ SHOT }}", "A man enters."),
        TEMPLATE.replace("{{ SHOT }}", "He sits down."),
    ]


def test_orders_by_input_number_not_name():
    shots = {f"shot_{i}": str(i) for i in (10, 2, 0, 1)}
    assert CardamonNodesShotPrompts.execute("", shots).args[0] == ["0", "1", "2", "10"]


def test_empty_template_uses_shots_as_is():
    assert run("", shot_0="  A man enters.\n") == ["A man enters."]
    assert run("  \n", shot_0="A man enters.") == ["A man enters."]


def test_empty_shots_are_skipped():
    assert run(
        TEMPLATE, shot_0="", shot_1="  \n", shot_2="He sits down.", shot_3=None
    ) == [TEMPLATE.replace("{{ SHOT }}", "He sits down.")]


def test_placeholder_spacing_and_repeats():
    assert build_shot_prompts("{{SHOT}} / {{  SHOT  }}", ["x"]) == ["x / x"]


def test_shot_text_is_inserted_literally():
    assert build_shot_prompts("a {{ SHOT }} b", [r"C:\new \1 {{ SHOT }}"]) == [
        r"a C:\new \1 {{ SHOT }} b"
    ]


def test_template_without_placeholder_is_rejected():
    with pytest.raises(ValueError, match="placeholder"):
        run("no placeholder here", shot_0="x")


def test_all_shots_empty_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        run(TEMPLATE, shot_0=" ", shot_1="")
