from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TypeAlias

FIXED_SHIFT = 16
FIXED_ONE = 1 << FIXED_SHIFT
FIXED_MASK = FIXED_ONE - 1
UINT32_MASK = (1 << 32) - 1
RAND_MAX = 2_147_483_647
RAND_DEGREE = 31
RAND_SEPARATOR = 3
RAND_WARMUP = 10 * RAND_DEGREE
BUTTON_MASK_LIMIT = 0b11_1111
PERSISTENT_SLOTS = 64

@dataclass(frozen=True, slots=True)
class PicoFixed:
    """Signed Q16.16 value matching Pemsa's no-rounding fixmath build."""

    raw: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", _i32(self.raw))

    @classmethod
    def from_int(cls, value: int) -> PicoFixed:
        if isinstance(value, bool):
            raise TypeError("PICO-8 numeric values cannot be booleans")
        return cls(_i32(value * FIXED_ONE))

    @classmethod
    def from_float(cls, value: float) -> PicoFixed:
        # fix16_from_float receives a C float and FIXMATH_NO_ROUNDING is set.
        narrowed = _f32(value)
        scaled = _f32(narrowed * _f32(float(FIXED_ONE)))
        return cls(_i32(int(scaled)))

    @classmethod
    def from_raw(cls, value: int) -> PicoFixed:
        return cls(value)

    def to_float(self) -> float:
        return _f32(_f32(float(self.raw)) / _f32(float(FIXED_ONE)))

    def to_double(self) -> float:
        return self.raw / FIXED_ONE

    def to_pico_string(self) -> str:
        sign = "-" if self.raw < 0 else ""
        absolute = abs(self.raw)
        whole, fraction = divmod(absolute, FIXED_ONE)
        digits = (fraction * 10_000) // FIXED_ONE
        if digits == 0:
            return f"{sign}{whole}"
        return f"{sign}{whole}.{digits:04d}".rstrip("0")

    def floor(self) -> PicoFixed:
        return PicoFixed.from_raw(_i32(self.raw & ~FIXED_MASK))

    def ceil(self) -> PicoFixed:
        floor = self.floor()
        return floor if self.raw & FIXED_MASK == 0 else floor + PicoFixed.from_int(1)

    def round(self) -> PicoFixed:
        remainder = self % PicoFixed.from_int(1)
        return self.floor() if remainder < PicoFixed.from_float(0.5) else self.ceil()

    def __add__(self, other: PicoScalar) -> PicoFixed:
        return PicoFixed.from_raw(self.raw + _as_fixed(other).raw)

    def __radd__(self, other: PicoScalar) -> PicoFixed:
        return self + other

    def __sub__(self, other: PicoScalar) -> PicoFixed:
        return PicoFixed.from_raw(self.raw - _as_fixed(other).raw)

    def __rsub__(self, other: PicoScalar) -> PicoFixed:
        return _as_fixed(other) - self

    def __mul__(self, other: PicoScalar) -> PicoFixed:
        product = self.raw * _as_fixed(other).raw
        return PicoFixed.from_raw(product >> FIXED_SHIFT)

    def __rmul__(self, other: PicoScalar) -> PicoFixed:
        return self * other

    def __truediv__(self, other: PicoScalar) -> PicoFixed:
        divisor = _as_fixed(other).raw
        if divisor == 0:
            raise ZeroDivisionError("PICO-8 fixed-point division by zero")
        numerator = self.raw << FIXED_SHIFT
        sign = -1 if (numerator < 0) != (divisor < 0) else 1
        quotient = abs(numerator) // abs(divisor)
        return PicoFixed.from_raw(sign * quotient)

    def __mod__(self, other: PicoScalar) -> PicoFixed:
        divisor = _as_fixed(other).raw
        if divisor == 0:
            raise ZeroDivisionError("PICO-8 fixed-point modulo by zero")
        quotient = abs(self.raw) // abs(divisor)
        if (self.raw < 0) != (divisor < 0):
            quotient = -quotient
        return PicoFixed.from_raw(self.raw - quotient * divisor)

    def __neg__(self) -> PicoFixed:
        return PicoFixed.from_raw(-self.raw)

    def __lt__(self, other: PicoScalar) -> bool:
        return self.raw < _as_fixed(other).raw

    def __le__(self, other: PicoScalar) -> bool:
        return self.raw <= _as_fixed(other).raw

    def __gt__(self, other: PicoScalar) -> bool:
        return self.raw > _as_fixed(other).raw

    def __ge__(self, other: PicoScalar) -> bool:
        return self.raw >= _as_fixed(other).raw

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PicoFixed):
            return self.raw == other.raw
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self == _as_fixed(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:
        return f"PicoFixed(raw={self.raw}, value={self.to_double():g})"


PicoScalar: TypeAlias = int | float | PicoFixed


def pico_floor(value: PicoScalar) -> PicoFixed:
    return _as_fixed(value).floor()


def pico_ceil(value: PicoScalar) -> PicoFixed:
    return _as_fixed(value).ceil()


def pico_round(value: PicoScalar) -> PicoFixed:
    return _as_fixed(value).round()


def pico_mid(first: PicoScalar, second: PicoScalar, third: PicoScalar) -> PicoFixed:
    x = _as_fixed(first)
    y = _as_fixed(second)
    z = _as_fixed(third)
    if x > y:
        x, y = y, x
    return max(x, min(y, z))


def pico_mod(first: PicoScalar, second: PicoScalar) -> PicoFixed:
    return _as_fixed(first) % _as_fixed(second)


@dataclass(slots=True)
class PicoRng:
    """glibc-compatible `rand` stream used by Pemsa on this Linux runner."""

    seed_value: int = 1
    _state: list[int] = field(default_factory=lambda: [0] * RAND_DEGREE)
    _front: int = RAND_SEPARATOR
    _rear: int = 0

    def __post_init__(self) -> None:
        self.seed(self.seed_value)

    def seed(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("PICO-8 srand seed must be an integer")
        seed = value & UINT32_MASK
        if seed == 0:
            seed = 1
        self.seed_value = value
        state = [0] * RAND_DEGREE
        state[0] = seed
        for index in range(1, RAND_DEGREE):
            state[index] = (16_807 * state[index - 1]) % RAND_MAX
        self._state = state
        self._front = RAND_SEPARATOR
        self._rear = 0
        for _ in range(RAND_WARMUP):
            self._next_rand()

    def rand_int(self) -> int:
        return self._next_rand()

    def rnd(self, limit: PicoScalar = 1) -> PicoFixed:
        maximum = _as_fixed(limit).to_float()
        random_float = _f32(_f32(float(self._next_rand())) / _f32(float(RAND_MAX)))
        return PicoFixed.from_float(_f32(random_float * _f32(maximum)))

    def checkpoint(self) -> tuple[int, tuple[int, ...], int, int]:
        return self.seed_value, tuple(self._state), self._front, self._rear

    def restore(self, checkpoint: tuple[int, tuple[int, ...], int, int]) -> None:
        seed, state, front, rear = checkpoint
        if len(state) != RAND_DEGREE:
            raise ValueError("invalid PICO-8 RNG checkpoint")
        self.seed_value = seed
        self._state = list(state)
        self._front = front
        self._rear = rear

    def _next_rand(self) -> int:
        value = (self._state[self._front] + self._state[self._rear]) & UINT32_MASK
        self._state[self._front] = value
        self._front = (self._front + 1) % RAND_DEGREE
        self._rear = (self._rear + 1) % RAND_DEGREE
        return value >> 1


@dataclass(slots=True)
class PicoInput:
    current_mask: int = 0
    previous_mask: int = 0
    mouse_x: PicoFixed = field(default_factory=lambda: PicoFixed.from_int(0))
    mouse_y: PicoFixed = field(default_factory=lambda: PicoFixed.from_int(0))
    mouse_button: PicoFixed = field(default_factory=lambda: PicoFixed.from_int(0))

    def advance(
        self,
        mask: int,
        *,
        mouse_x: PicoScalar = 0,
        mouse_y: PicoScalar = 0,
        mouse_button: PicoScalar = 0,
    ) -> None:
        if isinstance(mask, bool) or not 0 <= mask <= BUTTON_MASK_LIMIT:
            raise ValueError("PICO-8 input mask must be in 0..63")
        self.previous_mask = self.current_mask
        self.current_mask = mask
        self.mouse_x = _as_fixed(mouse_x)
        self.mouse_y = _as_fixed(mouse_y)
        self.mouse_button = _as_fixed(mouse_button)

    def btn(self, index: int) -> bool:
        _check_button(index)
        return bool(self.current_mask & (1 << index))

    def btnp(self, index: int) -> bool:
        _check_button(index)
        bit = 1 << index
        return bool(self.current_mask & bit) and not bool(self.previous_mask & bit)

    def stat(self, index: int) -> PicoFixed | None:
        if index == 32:
            return self.mouse_x
        if index == 33:
            return self.mouse_y
        if index == 34:
            return self.mouse_button
        return None


@dataclass(slots=True)
class PicoPersistentData:
    values: list[PicoFixed] = field(
        default_factory=lambda: [PicoFixed.from_int(0)] * PERSISTENT_SLOTS
    )

    def dget(self, index: int) -> PicoFixed:
        _check_persistent_index(index)
        return self.values[index]

    def dset(self, index: int, value: PicoScalar) -> None:
        _check_persistent_index(index)
        self.values[index] = _as_fixed(value)

    def checkpoint(self) -> tuple[int, ...]:
        return tuple(value.raw for value in self.values)

    def restore(self, checkpoint: tuple[int, ...]) -> None:
        if len(checkpoint) != PERSISTENT_SLOTS:
            raise ValueError("invalid PICO-8 persistent-data checkpoint")
        self.values = [PicoFixed.from_raw(value) for value in checkpoint]


@dataclass(slots=True)
class PicoCompat:
    """Explicit compatibility boundary for native gameplay conversion."""

    rng: PicoRng
    input: PicoInput = field(default_factory=PicoInput)
    persistent: PicoPersistentData = field(default_factory=PicoPersistentData)

    @classmethod
    def from_seed(cls, seed: int) -> PicoCompat:
        return cls(rng=PicoRng(seed))

    def srand(self, seed: int) -> None:
        self.rng.seed(seed)

    def rnd(self, limit: PicoScalar = 1) -> PicoFixed:
        return self.rng.rnd(limit)

    def btn(self, index: int) -> bool:
        return self.input.btn(index)

    def btnp(self, index: int) -> bool:
        return self.input.btnp(index)

    def stat(self, index: int) -> PicoFixed | None:
        return self.input.stat(index)

    def dget(self, index: int) -> PicoFixed:
        return self.persistent.dget(index)

    def dset(self, index: int, value: PicoScalar) -> None:
        self.persistent.dset(index, value)


def _as_fixed(value: PicoScalar) -> PicoFixed:
    if isinstance(value, PicoFixed):
        return value
    if isinstance(value, bool):
        raise TypeError("PICO-8 numeric values cannot be booleans")
    if isinstance(value, int):
        return PicoFixed.from_int(value)
    if isinstance(value, float):
        return PicoFixed.from_float(value)
    raise TypeError(f"unsupported PICO-8 numeric value: {type(value).__name__}")


def _check_button(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 5:
        raise ValueError("PICO-8 button index must be an integer in 0..5")


def _check_persistent_index(index: int) -> None:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < PERSISTENT_SLOTS
    ):
        raise ValueError("PICO-8 persistent-data index must be an integer in 0..63")


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _i32(value: int) -> int:
    value &= UINT32_MASK
    return value if value < (1 << 31) else value - (1 << 32)
