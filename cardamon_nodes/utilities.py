"""Small utility nodes."""

import re

from comfy_api.latest import io

SECONDS_FIELD = re.compile(r"\d+(?:[.,]\d*)?")
WHOLE_FIELD = re.compile(r"\d+")


def parse_time(text):
    """Seconds in "hh:mm:ss.fff", "mm:ss.fff" or "ss.fff", with any number of decimals.

    The leading field may exceed 59 ("90:00" is 90 minutes), a leading "-" makes the time negative,
    and "," works as the decimal separator. web/time_input.js implements the same rules for the
    preview on the node, so keep the two in sync.
    """
    value = text.strip()
    sign = 1
    if value.startswith("-"):
        sign = -1
        value = value[1:].strip()
    fields = value.split(":")
    if len(fields) > 3 or not SECONDS_FIELD.fullmatch(fields[-1]):
        raise ValueError(f"invalid time {text!r}, expected hh:mm:ss.fff")
    if not all(WHOLE_FIELD.fullmatch(f) for f in fields[:-1]):
        raise ValueError(f"invalid time {text!r}, expected hh:mm:ss.fff")
    whole = [int(f) for f in fields[:-1]]
    seconds = float(fields[-1].replace(",", "."))
    # Fields after the leading one are minutes or seconds of the field before them.
    if len(fields) > 1 and (seconds >= 60 or any(v >= 60 for v in whole[1:])):
        raise ValueError(f"invalid time {text!r}: minutes and seconds after the first field must be below 60")
    total = 0
    for v in whole:
        total = total * 60 + v
    return sign * (total * 60 + seconds) if whole else sign * seconds


class CardamonTime(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonTime",
            display_name="Time",
            category="cardamon/utils",
            description="A length or offset in time, entered as hh:mm:ss.fff, mm:ss.fff or seconds, output as seconds. "
            "The node shows the seconds as you type.",
            inputs=[
                io.String.Input(
                    "time",
                    default="00:00:10.000",
                    tooltip="hh:mm:ss.fff, mm:ss.fff or ss.fff, with any number of decimals. A leading - makes it negative.",
                ),
            ],
            outputs=[io.Float.Output(display_name="seconds")],
        )

    @classmethod
    def execute(cls, time) -> io.NodeOutput:
        return io.NodeOutput(parse_time(time))

    @classmethod
    def validate_inputs(cls, time=None):
        # time is None when it comes from a link; that value is checked when the node runs.
        if time is None:
            return True
        try:
            parse_time(time)
        except ValueError as e:
            return str(e)
        return True


NODES = [CardamonTime]
