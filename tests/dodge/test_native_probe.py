from __future__ import annotations

from dodge.native.probe import (
    PROBE_PREFIX,
    input_probe_cartridge,
    parse_probe_output,
    semantics_probe_cartridge,
)


def test_semantics_probe_cartridge_is_seeded_and_covers_compatibility_primitives() -> (
    None
):
    source = semantics_probe_cartridge(42)

    assert "srand(42)" in source
    assert "rnd(10)" in source
    assert f'"{PROBE_PREFIX}|rng_first|"' in source
    assert f'"{PROBE_PREFIX}|rng_limit|"' in source
    assert f'"{PROBE_PREFIX}|rng|"' not in source
    assert f'"{PROBE_PREFIX}|numeric|"' not in source
    assert "del(values,2)" in source
    assert f'"{PROBE_PREFIX}|list_len|"' in source
    assert f'"{PROBE_PREFIX}|list_1|"' in source
    assert f'"{PROBE_PREFIX}|list_2|"' in source
    assert f'"{PROBE_PREFIX}|list_3|"' in source
    assert "camera(3,4)" in source
    assert "pget(0,0)" in source


def test_input_probe_cartridge_covers_pressed_and_held_buttons() -> None:
    source = input_probe_cartridge()

    assert "btn(0)" in source
    assert "btnp(0)" in source
    assert "__probe_bool(btn(0))" in source
    assert "input_frame" in source
    assert "input_btnp3" in source
    assert "isdead=true" in source


def test_parse_probe_output_groups_records_by_probe_name() -> None:
    result = parse_probe_output(
        "noise\n__dodge_probe__|rng|0.5|2\n__dodge_probe__|input|1|true\n"
    )

    assert result == {"rng": ["0.5|2"], "input": ["1|true"]}
