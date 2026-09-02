from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1
_SECTION_RE = re.compile(rb"^__([A-Za-z0-9_]+)__[ \t]*(?:\r?\n|\Z)")
_FUNCTION_RE = re.compile(
    r"^(?P<indent>\s*)(?P<local>local\s+)?function\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_\.]*)\s*\("
)
_ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<local>local\s+)?"
    r"(?P<names>[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*(?P<operator>[+\-*/%]?=)(?!=)"
)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    byte_length: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SectionManifest:
    name: str
    start_line: int
    end_line: int
    byte_length: int
    sha256: str
    line_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "line_count": self.line_count,
        }


@dataclass(frozen=True, slots=True)
class P8Section:
    name: str
    start_line: int
    end_line: int
    marker: bytes
    payload: bytes
    raw: bytes
    payload_lines: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class SymbolManifest:
    kind: str
    name: str
    line: int
    scope: str
    operator: str | None = None

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": self.kind,
            "name": self.name,
            "line": self.line,
            "scope": self.scope,
        }
        if self.operator is not None:
            value["operator"] = self.operator
        return value


@dataclass(frozen=True, slots=True)
class CartridgeManifest:
    schema_version: int
    path: str
    byte_length: int
    line_count: int
    sha256: str
    sections: tuple[SectionManifest, ...]
    functions: tuple[SymbolManifest, ...]
    assignments: tuple[SymbolManifest, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "byte_length": self.byte_length,
            "line_count": self.line_count,
            "sha256": self.sha256,
            "sections": [section.to_json() for section in self.sections],
            "functions": [function.to_json() for function in self.functions],
            "assignments": [assignment.to_json() for assignment in self.assignments],
        }


def file_identity(path: Path) -> FileIdentity:
    data = _read_bytes(path)
    return FileIdentity(
        path=str(path),
        byte_length=len(data),
        sha256=_sha256(data),
    )


def manifest_for_path(path: Path) -> CartridgeManifest:
    data = _read_bytes(path)
    return manifest_for_bytes(data, path=path)


def manifest_for_bytes(
    data: bytes, *, path: Path | str = "<memory>"
) -> CartridgeManifest:
    lines = data.splitlines(keepends=True)
    sections_data = sections_for_bytes(data)
    sections = tuple(
        SectionManifest(
            name=section.name,
            start_line=section.start_line,
            end_line=section.end_line,
            byte_length=len(section.raw),
            sha256=_sha256(section.raw),
            line_count=section.end_line - section.start_line + 1,
        )
        for section in sections_data
    )

    source = data.decode("utf-8")
    lua_section = next((section for section in sections if section.name == "lua"), None)
    functions: list[SymbolManifest] = []
    assignments: list[SymbolManifest] = []
    if lua_section is not None:
        lua_start = lua_section.start_line
        lua_end = lua_section.end_line
        source_lines = source.splitlines()
        for line_number in range(lua_start + 1, lua_end + 1):
            line = source_lines[line_number - 1]
            if function_match := _FUNCTION_RE.match(line):
                functions.append(
                    SymbolManifest(
                        kind="function",
                        name=function_match.group("name"),
                        line=line_number,
                        scope="local" if function_match.group("local") else "global",
                    )
                )
            if assignment_match := _ASSIGNMENT_RE.match(line):
                scope = "local" if assignment_match.group("local") else "global"
                for name in assignment_match.group("names").split(","):
                    assignments.append(
                        SymbolManifest(
                            kind="assignment",
                            name=name.strip(),
                            line=line_number,
                            scope=scope,
                            operator=assignment_match.group("operator"),
                        )
                    )

    return CartridgeManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        path=str(path),
        byte_length=len(data),
        line_count=len(lines),
        sha256=_sha256(data),
        sections=sections,
        functions=tuple(functions),
        assignments=tuple(assignments),
    )


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _with_next(
    markers: list[tuple[int, str]], line_count: int
) -> list[tuple[tuple[int, str], tuple[int, str | None]]]:
    return [
        (
            marker,
            (*markers[index + 1],) if index + 1 < len(markers) else (line_count, None),
        )
        for index, marker in enumerate(markers)
    ]


def sections_for_bytes(data: bytes) -> tuple[P8Section, ...]:
    """Parse PICO-8 section markers while retaining exact source bytes."""
    lines = data.splitlines(keepends=True)
    markers = [
        (index, match.group(1).decode("ascii"))
        for index, line in enumerate(lines)
        if (match := _SECTION_RE.match(line)) is not None
    ]
    if not markers:
        raise ValueError("PICO-8 cartridge contains no sections")
    if len({name for _, name in markers}) != len(markers):
        raise ValueError("PICO-8 cartridge contains duplicate sections")

    sections: list[P8Section] = []
    for index, (marker_index, name) in enumerate(markers):
        next_index = markers[index + 1][0] if index + 1 < len(markers) else len(lines)
        marker = lines[marker_index]
        payload_lines = tuple(lines[marker_index + 1 : next_index])
        sections.append(
            P8Section(
                name=name,
                start_line=marker_index + 1,
                end_line=next_index,
                marker=marker,
                payload=b"".join(payload_lines),
                raw=b"".join(lines[marker_index:next_index]),
                payload_lines=payload_lines,
            )
        )
    return tuple(sections)


def section_for_bytes(data: bytes, name: str) -> P8Section:
    try:
        return next(
            section for section in sections_for_bytes(data) if section.name == name
        )
    except StopIteration as error:
        raise ValueError(f"PICO-8 cartridge has no __{name}__ section") from error
