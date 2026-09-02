from __future__ import annotations

from pathlib import Path

from dodge.native.compat import (
    PicoCompat,
    PicoFixed,
    PicoInput,
    PicoRng,
    pico_ceil,
    pico_floor,
    pico_mid,
    pico_mod,
    pico_round,
)
from dodge.native.compatibility import build_compatibility_report


def test_v89_pico_fixed_matches_pemsa_probe_boundaries() -> None:
    assert pico_floor(-1.2) == -2
    assert pico_ceil(-1.2) == -1
    assert pico_mid(0, 9, 4) == 4
    assert pico_mod(7, 4) == 3
    assert pico_round(4.49) == 4
    assert pico_round(4.5) == 5
    assert PicoFixed.from_float(-1.2).raw == -78_643


def test_v89_rng_matches_pemsa_glibc_rand_sequence_and_probe_values() -> None:
    rng = PicoRng(42)

    assert [rng.rand_int() for _ in range(3)] == [
        71_876_166,
        708_592_740,
        1_483_128_881,
    ]

    rng.seed(42)
    assert round(rng.rnd().to_double(), 4) == 0.0335
    assert round(rng.rnd(10).to_double(), 4) == 3.2996


def test_v89_rng_checkpoint_restores_the_exact_stream() -> None:
    rng = PicoRng(7)
    rng.rnd()
    checkpoint = rng.checkpoint()
    expected = [rng.rand_int() for _ in range(4)]

    rng.restore(checkpoint)

    assert [rng.rand_int() for _ in range(4)] == expected


def test_v89_input_edges_and_stat_values_are_explicit() -> None:
    input_state = PicoInput()
    input_state.advance(32, mouse_x=64, mouse_y=63, mouse_button=1)

    assert input_state.btn(5) is True
    assert input_state.btnp(5) is True
    assert input_state.btn(0) is False
    assert input_state.stat(32) == 64
    assert input_state.stat(33) == 63
    assert input_state.stat(34) == 1
    assert input_state.stat(99) is None

    input_state.advance(1)

    assert input_state.btn(0) is True
    assert input_state.btnp(0) is True
    assert input_state.btn(5) is False
    assert input_state.btnp(5) is False


def test_v89_compat_persistent_slots_are_fixed_point_and_checkpointable() -> None:
    compat = PicoCompat.from_seed(42)
    compat.dset(12, 1.5)
    checkpoint = compat.persistent.checkpoint()
    compat.dset(12, 2)

    compat.persistent.restore(checkpoint)

    assert compat.dget(12) == PicoFixed.from_float(1.5)


def test_v89_compatibility_report_keeps_probe_values_separate_and_provenanced() -> None:
    observed = {
        "draw": ["5"],
        "input": {
            "input_btn0": ["0", "1", "0", "0", "0"],
            "input_btn1": ["0", "0", "0", "1", "0"],
            "input_btn2": ["0", "0", "0", "1", "0"],
            "input_btn3": ["0", "0", "0", "0", "1"],
            "input_btnp0": ["0", "1", "0", "0", "0"],
            "input_btnp1": ["0", "0", "0", "1", "0"],
            "input_btnp2": ["0", "0", "0", "1", "0"],
            "input_btnp3": ["0", "0", "0", "0", "1"],
            "input_frame": ["1", "2", "3", "4", "5"],
        },
        "list_1": ["1"],
        "list_2": ["3"],
        "list_3": ["4"],
        "list_len": ["3"],
        "numeric_ceil": ["-1"],
        "numeric_floor": ["-2"],
        "numeric_mid": ["4"],
        "numeric_mod": ["3"],
        "rng_first": ["0.0334"],
        "rng_limit": ["3.2996"],
    }
    source = Path("src/dodge/game/dodge.p8")
    report = build_compatibility_report(
        seed=42,
        observed=observed,  # type: ignore[arg-type]
        source=source,
        pemsa=Path("src/dodge/runtime/pemsa"),
    )

    assert report["status"] == "accepted"
    assert report["seed"] == 42
    assert report["provenance"]["source"]["sha256"]
    assert report["provenance"]["pemsa"]["sha256"]
    assert all("|" not in str(record["observed"]) for record in report["records"])
    assert {record["name"] for record in report["records"]} >= {
        "rng_first",
        "rng_limit",
        "numeric_floor",
        "input_btnp3",
    }
