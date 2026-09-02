from __future__ import annotations

import json
from pathlib import Path

from dodge.control import CARTRIDGE_PATH, PEMSA_PATH, ControlRuntimeError
from dodge.native.compat import (
    PicoCompat,
    PicoFixed,
    PicoInput,
    pico_ceil,
    pico_floor,
    pico_mid,
    pico_mod,
)
from dodge.native.manifest import file_identity, manifest_for_path
from dodge.native.probe import run_input_probe, run_semantics_probe

COMPATIBILITY_SCHEMA_VERSION = 1


def run_compatibility_report(
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    pemsa: Path = PEMSA_PATH,
    runner=None,
) -> dict[str, object]:
    observed = run_semantics_probe(seed=seed, runner=runner)
    observed["input"] = run_input_probe(runner=runner)
    return build_compatibility_report(
        seed=seed,
        observed=observed,
        source=source,
        pemsa=pemsa,
    )


def build_compatibility_report(
    *,
    seed: int,
    observed: dict[str, list[str]],
    source: Path = CARTRIDGE_PATH,
    pemsa: Path = PEMSA_PATH,
) -> dict[str, object]:
    try:
        source_manifest = manifest_for_path(source)
        pemsa_identity = file_identity(pemsa)
    except OSError as error:
        raise ControlRuntimeError(
            f"could not read compatibility provenance: {error}"
        ) from error

    compat = PicoCompat.from_seed(seed)
    records: list[dict[str, object]] = []
    records.extend(
        _fixed_records(
            observed,
            "rng_first",
            [compat.rnd()],
            api="rnd",
        )
    )
    records.extend(
        _fixed_records(
            observed,
            "rng_limit",
            [compat.rnd(10)],
            api="rnd(10)",
        )
    )
    records.extend(
        _fixed_records(
            observed,
            "numeric_floor",
            [pico_floor(-1.2)],
            api="flr",
        )
    )
    records.extend(
        _fixed_records(
            observed,
            "numeric_ceil",
            [pico_ceil(-1.2)],
            api="ceil",
        )
    )
    records.extend(
        _fixed_records(
            observed,
            "numeric_mid",
            [pico_mid(0, 9, 4)],
            api="mid",
        )
    )
    records.extend(
        _fixed_records(
            observed,
            "numeric_mod",
            [pico_mod(7, 4)],
            api="%",
        )
    )
    records.extend(_text_records(observed, "list_len", "3", api="# / del / add"))
    records.extend(_text_records(observed, "list_1", "1", api="del / add"))
    records.extend(_text_records(observed, "list_2", "3", api="del / add"))
    records.extend(_text_records(observed, "list_3", "4", api="del / add"))
    records.extend(_text_records(observed, "draw", "5", api="camera / pset / pget"))
    records.extend(_input_records(observed.get("input", {})))
    status = (
        "accepted"
        if all(record["status"] == "match" for record in records)
        else "mismatch"
    )
    return {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "status": status,
        "seed": seed,
        "provenance": {
            "source": {
                **source_manifest.to_json(),
                "path": source.name,
            },
            "pemsa": {
                **pemsa_identity.to_json(),
                "path": pemsa.name,
            },
        },
        "records": records,
    }


def write_compatibility_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _fixed_records(
    observed: dict[str, list[str]],
    name: str,
    expected: list[PicoFixed],
    *,
    api: str,
) -> list[dict[str, object]]:
    values = observed.get(name, [])
    records: list[dict[str, object]] = []
    for index, native in enumerate(expected):
        actual = values[index] if index < len(values) else None
        expected_text = native.to_pico_string()
        records.append(
            {
                "name": name,
                "api": api,
                "index": index,
                "observed": actual,
                "expected": expected_text,
                "native": {"raw": native.raw, "value": native.to_double()},
                "status": "match" if actual == expected_text else "mismatch",
            }
        )
    return records


def _text_records(
    observed: dict[str, list[str]], name: str, expected: str, *, api: str
) -> list[dict[str, object]]:
    values = observed.get(name, [])
    actual = values[0] if values else None
    return [
        {
            "name": name,
            "api": api,
            "index": 0,
            "observed": actual,
            "expected": expected,
            "native": {"text": expected},
            "status": "match" if actual == expected else "mismatch",
        }
    ]


def _input_records(observed: dict[str, list[str]]) -> list[dict[str, object]]:
    masks = (32, 1, 0, 6, 8)
    input_state = PicoInput()
    records: list[dict[str, object]] = []
    for frame, mask in enumerate(masks, start=1):
        input_state.advance(mask)
        for index in range(4):
            records.append(
                _input_record(
                    observed,
                    f"input_btn{index}",
                    frame - 1,
                    int(input_state.btn(index)),
                    api="btn",
                )
            )
            records.append(
                _input_record(
                    observed,
                    f"input_btnp{index}",
                    frame - 1,
                    int(input_state.btnp(index)),
                    api="btnp",
                )
            )
        records.append(
            _input_record(
                observed,
                "input_frame",
                frame - 1,
                frame,
                api="_update60",
            )
        )
    return records


def _input_record(
    observed: dict[str, list[str]],
    name: str,
    index: int,
    expected: int,
    *,
    api: str,
) -> dict[str, object]:
    values = observed.get(name, [])
    actual = values[index] if index < len(values) else None
    expected_text = str(expected)
    return {
        "name": name,
        "api": api,
        "index": index,
        "observed": actual,
        "expected": expected_text,
        "native": {"numeric_boolean": expected},
        "status": "match" if actual == expected_text else "mismatch",
    }
