import json
from pathlib import Path

import pytest

from cardamon_nodes.utilities import CardamonNodesTime, parse_time

# Shared with the browser check of web/time_input.js, which must give the same results.
CASES = json.loads((Path(__file__).parent / "time_cases.json").read_text())


@pytest.mark.parametrize("text, seconds", CASES["valid"])
def test_valid_times(text, seconds):
    assert parse_time(text) == pytest.approx(seconds, abs=1e-12)
    assert CardamonNodesTime.execute(text).args[0] == parse_time(text)


@pytest.mark.parametrize("text", CASES["invalid"])
def test_invalid_times(text):
    with pytest.raises(ValueError):
        parse_time(text)
    assert CardamonNodesTime.validate_inputs(text) is not True


def test_linked_input_is_not_validated_early():
    assert CardamonNodesTime.validate_inputs() is True
