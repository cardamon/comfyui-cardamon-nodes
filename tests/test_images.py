import pytest
import torch

from cardamon_nodes.images import CardamonImageStitch, parse_color

WHITE = "#ffffff"


def solid(value, height=4, width=6, channels=3, batch=1):
    return torch.full((batch, height, width, channels), value / 10)


def stitch(*images, direction="right", match=False, wrap_after=0, spacing=0, color=WHITE):
    inputs = {f"image_{i}": image for i, image in enumerate(images)}
    return CardamonImageStitch.execute(inputs, direction, match, wrap_after, spacing, color).args[0]


def layout(out):
    # The grey level of each pixel as a digit, row by row (white spacing is 10).
    return ["".join(str(round(v)) if round(v) < 10 else "." for v in (out[0, y, :, 0] * 10).tolist()) for y in range(out.shape[1])]


def test_parse_color():
    assert parse_color("#ff8000") == (1.0, 128 / 255, 0.0, 1.0)
    assert parse_color("#00000080") == (0.0, 0.0, 0.0, 128 / 255)
    with pytest.raises(ValueError):
        parse_color("white")


def test_right_wraps_into_rows_aligned_left():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1), wrap_after=2)
    assert layout(out) == ["12", "3."]


def test_left_runs_from_the_right_and_aligns_rows_right():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1), direction="left", wrap_after=2)
    assert layout(out) == ["21", ".3"]


def test_down_wraps_into_columns_aligned_top():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1), direction="down", wrap_after=2)
    assert layout(out) == ["13", "2."]


def test_up_runs_from_the_bottom_and_aligns_columns_bottom():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1), direction="up", wrap_after=2)
    assert layout(out) == ["2.", "13"]


def test_no_wrap_is_one_line():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1))
    assert layout(out) == ["123"]


def test_spacing_between_images_and_lines():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1), wrap_after=2, spacing=1)
    assert layout(out) == ["1.2", "...", "3.."]


def test_smaller_images_are_centered_across_their_line():
    out = stitch(solid(1, 3, 1), solid(2, 1, 1))
    assert layout(out) == ["1.", "12", "1."]


def test_match_image_size_resizes_to_image_1():
    out = stitch(solid(1, 4, 6), solid(2, 8, 2), match=True)
    assert out.shape == (1, 4, 12, 3)
    assert layout(out)[0] == "111111222222"


def test_transparent_spacing_gives_rgba():
    out = stitch(solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1), wrap_after=2, spacing=1, color="#ff000040")
    assert out.shape[-1] == 4
    assert out[0, 0, 0].tolist() == pytest.approx([0.1, 0.1, 0.1, 1.0])  # image keeps full alpha
    assert out[0, 0, 1].tolist() == pytest.approx([1.0, 0.0, 0.0, 64 / 255])  # spacing
    assert out[0, 2, 2].tolist() == pytest.approx([1.0, 0.0, 0.0, 64 / 255])  # empty end of last row


def test_opaque_spacing_keeps_rgb_unless_an_image_has_alpha():
    assert stitch(solid(1), solid(2), color="#000000").shape[-1] == 3
    rgba = solid(2, channels=4)
    rgba[..., 3] = 0.5
    out = stitch(solid(1), rgba)
    assert out.shape[-1] == 4
    assert out[0, 0, 0, 3] == 1.0  # the RGB image gets full alpha
    assert out[0, 0, 6, 3] == 0.5  # the RGBA image keeps its own


def test_batches_repeat_their_last_image():
    a = torch.cat([solid(1, 1, 1), solid(2, 1, 1), solid(3, 1, 1)])
    out = stitch(a, solid(5, 1, 1))
    assert out.shape[0] == 3
    assert [layout(out[i : i + 1])[0] for i in range(3)] == ["15", "25", "35"]


def test_inputs_are_ordered_by_number():
    images = {f"image_{i}": solid(i % 10, 1, 1) for i in (10, 2, 0, 1)}
    out = CardamonImageStitch.execute(images, "right", False, 0, 0, WHITE).args[0]
    assert layout(out) == ["0120"]


def test_single_image_is_returned_as_is():
    image = solid(1)
    assert torch.equal(stitch(image), image)
