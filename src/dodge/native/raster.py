from __future__ import annotations

from dataclasses import dataclass, field

from dodge.native.assets import GFX_HEIGHT, GFX_WIDTH
from dodge.native.compat import PicoFixed, PicoScalar

RASTER_WIDTH = 128
RASTER_HEIGHT = 128
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT
FULL_FILL_PATTERN = 0x0000
MAX_FILL_PATTERN = 0xFFFF


class RasterError(ValueError):
    """A raster operation cannot be represented by the indexed framebuffer."""


@dataclass(slots=True)
class IndexedRaster:
    """PICO-8-like indexed raster state without a window or GPU dependency."""

    width: int = RASTER_WIDTH
    height: int = RASTER_HEIGHT
    pixels: bytearray = field(default_factory=lambda: bytearray(RASTER_SIZE))
    draw_palette: list[int] = field(default_factory=lambda: list(range(16)))
    screen_palette: list[int] = field(default_factory=lambda: list(range(16)))
    transparent: list[bool] = field(
        default_factory=lambda: [True] + [False] * 15
    )
    draw_color: int = 6
    fill_pattern: int = FULL_FILL_PATTERN
    camera_x: PicoFixed = field(default_factory=lambda: PicoFixed.from_int(0))
    camera_y: PicoFixed = field(default_factory=lambda: PicoFixed.from_int(0))
    clip_x: int = 0
    clip_y: int = 0
    clip_width: int = RASTER_WIDTH
    clip_height: int = RASTER_HEIGHT
    operations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width != RASTER_WIDTH or self.height != RASTER_HEIGHT:
            raise RasterError("PICO-8 raster dimensions must be 128x128")
        if len(self.pixels) != RASTER_SIZE:
            raise RasterError("PICO-8 raster must contain 16384 indexed pixels")
        self._check_palette_state()

    def cls(self, color: int = 0) -> None:
        self._check_color(color)
        mapped = self.draw_palette[color]
        self.pixels[:] = bytes([mapped]) * RASTER_SIZE
        self.operations.append("cls")

    def color(self, color: int = 6) -> None:
        self._check_color(color)
        self.draw_color = color
        self.operations.append("color")

    def camera(self, x: PicoScalar = 0, y: PicoScalar = 0) -> None:
        self.camera_x = _fixed(x)
        self.camera_y = _fixed(y)
        self.operations.append("camera")

    def clip(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if x is None and y is None and width is None and height is None:
            self.clip_x, self.clip_y = 0, 0
            self.clip_width, self.clip_height = self.width, self.height
        elif None in {x, y, width, height}:
            raise RasterError("clip requires all four values or no values")
        else:
            assert x is not None
            assert y is not None
            assert width is not None
            assert height is not None
            if width < 0 or height < 0:
                raise RasterError("clip dimensions must be non-negative")
            self.clip_x, self.clip_y = x, y
            self.clip_width, self.clip_height = width, height
        self.operations.append("clip")

    def pal(
        self,
        source: int | None = None,
        destination: int | None = None,
        mode: int = 0,
    ) -> None:
        if source is None and destination is None:
            self.draw_palette[:] = range(16)
            self.screen_palette[:] = range(16)
        elif source is None or destination is None:
            raise RasterError("pal requires both source and destination colors")
        else:
            self._check_color(source)
            self._check_color(destination)
            if mode == 0:
                self.draw_palette[source] = destination
            elif mode == 1:
                self.screen_palette[source] = destination
            else:
                raise RasterError("PICO-8 palette mode must be 0 or 1")
        self.operations.append("pal")

    def palt(self, color: int | None = None, transparent: bool | None = None) -> None:
        if color is None and transparent is None:
            self.transparent[:] = [True] + [False] * 15
        elif color is None or transparent is None:
            raise RasterError("palt requires both color and transparency")
        else:
            self._check_color(color)
            self.transparent[color] = transparent
        self.operations.append("palt")

    def fillp(self, pattern: int | PicoFixed | None = None) -> None:
        if pattern is None:
            self.fill_pattern = FULL_FILL_PATTERN
        elif isinstance(pattern, PicoFixed):
            self.fill_pattern = (pattern.raw >> 16) & 0xFFFF
        elif isinstance(pattern, int) and not isinstance(pattern, bool):
            if not 0 <= pattern <= MAX_FILL_PATTERN:
                raise RasterError("fill pattern must be a 16-bit value")
            self.fill_pattern = pattern
        else:
            raise RasterError("fill pattern must be a 16-bit integer")
        self.operations.append("fillp")

    def pset(self, x: PicoScalar, y: PicoScalar, color: int | None = None) -> None:
        self._plot_world(x, y, self.draw_color if color is None else color)
        self.operations.append("pset")

    def pget(self, x: PicoScalar, y: PicoScalar) -> int:
        screen_x, screen_y = self._screen_coordinates(x, y)
        if not self._inside(screen_x, screen_y):
            return 0
        return self.pixels[screen_y * self.width + screen_x]

    def line(
        self,
        x0: PicoScalar,
        y0: PicoScalar,
        x1: PicoScalar,
        y1: PicoScalar,
        color: int | None = None,
    ) -> None:
        start_x, start_y = self._screen_coordinates(x0, y0)
        end_x, end_y = self._screen_coordinates(x1, y1)
        dx = abs(end_x - start_x)
        step_x = 1 if start_x < end_x else -1
        dy = -abs(end_y - start_y)
        step_y = 1 if start_y < end_y else -1
        error = dx + dy
        while True:
            value = self.draw_color if color is None else color
            self._plot_screen(start_x, start_y, value)
            if start_x == end_x and start_y == end_y:
                break
            double_error = 2 * error
            if double_error >= dy:
                error += dy
                start_x += step_x
            if double_error <= dx:
                error += dx
                start_y += step_y
        self.operations.append("line")

    def rect(
        self,
        x0: PicoScalar,
        y0: PicoScalar,
        x1: PicoScalar,
        y1: PicoScalar,
        color: int | None = None,
    ) -> None:
        self.line(x0, y0, x1, y0, color)
        self.line(x1, y0, x1, y1, color)
        self.line(x1, y1, x0, y1, color)
        self.line(x0, y1, x0, y0, color)
        self.operations.append("rect")

    def rectfill(
        self,
        x0: PicoScalar,
        y0: PicoScalar,
        x1: PicoScalar,
        y1: PicoScalar,
        color: int | None = None,
    ) -> None:
        start_x, start_y = self._screen_coordinates(x0, y0)
        end_x, end_y = self._screen_coordinates(x1, y1)
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        if start_y > end_y:
            start_y, end_y = end_y, start_y
        value = self.draw_color if color is None else color
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                self._plot_screen(x, y, value)
        self.operations.append("rectfill")

    def circfill(
        self,
        center_x: PicoScalar,
        center_y: PicoScalar,
        radius: PicoScalar,
        color: int | None = None,
    ) -> None:
        x0, y0 = self._screen_coordinates(center_x, center_y)
        radius_pixels = max(0, _coordinate(radius))
        value = self.draw_color if color is None else color
        for y in range(y0 - radius_pixels, y0 + radius_pixels + 1):
            for x in range(x0 - radius_pixels, x0 + radius_pixels + 1):
                if (x - x0) ** 2 + (y - y0) ** 2 <= radius_pixels**2:
                    self._plot_screen(x, y, value)
        self.operations.append("circfill")

    def sprite(
        self,
        sheet: bytes | bytearray,
        sprite_id: int,
        x: PicoScalar,
        y: PicoScalar,
        width: int = 1,
        height: int = 1,
    ) -> None:
        if len(sheet) != GFX_WIDTH * GFX_HEIGHT:
            raise RasterError("sprite sheet must contain 16384 palette indexes")
        if (
            not isinstance(sprite_id, int)
            or isinstance(sprite_id, bool)
            or sprite_id < 0
        ):
            raise RasterError("sprite id must be a non-negative integer")
        if width < 1 or height < 1:
            raise RasterError("sprite dimensions must be positive")
        source_x = (sprite_id % 16) * 8
        source_y = (sprite_id // 16) * 8
        origin_x = _coordinate(x)
        origin_y = _coordinate(y)
        for offset_y in range(height * 8):
            for offset_x in range(width * 8):
                sample_x = source_x + offset_x
                sample_y = source_y + offset_y
                if sample_x >= GFX_WIDTH or sample_y >= GFX_HEIGHT:
                    continue
                source_color = sheet[sample_y * GFX_WIDTH + sample_x]
                self._check_color(source_color)
                if not self.transparent[source_color]:
                    self._plot_world(
                        origin_x + offset_x,
                        origin_y + offset_y,
                        source_color,
                    )
        self.operations.append("sprite")

    def indexed_pixels(self) -> bytes:
        return bytes(self.pixels)

    def display_pixels(self) -> bytes:
        return bytes(self.screen_palette[value] for value in self.pixels)

    def state_json(self) -> dict[str, object]:
        return {
            "draw_color": self.draw_color,
            "fill_pattern": self.fill_pattern,
            "draw_palette": list(self.draw_palette),
            "screen_palette": list(self.screen_palette),
            "transparent": [int(value) for value in self.transparent],
            "camera": [self.camera_x.raw, self.camera_y.raw],
            "clip": [self.clip_x, self.clip_y, self.clip_width, self.clip_height],
            "operations": list(self.operations),
        }

    def _plot_world(self, x: PicoScalar, y: PicoScalar, color: int) -> None:
        screen_x, screen_y = self._screen_coordinates(x, y)
        self._plot_screen(screen_x, screen_y, color)

    def _plot_screen(self, x: int, y: int, color: int) -> None:
        self._check_color(color)
        if not self._inside(x, y) or not self._inside_clip(x, y):
            return
        if not self._pattern_bit(x, y):
            return
        self.pixels[y * self.width + x] = self.draw_palette[color]

    def _screen_coordinates(self, x: PicoScalar, y: PicoScalar) -> tuple[int, int]:
        return (
            _coordinate(x) - _coordinate(self.camera_x),
            _coordinate(y) - _coordinate(self.camera_y),
        )

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _inside_clip(self, x: int, y: int) -> bool:
        return (
            self.clip_x <= x < self.clip_x + self.clip_width
            and self.clip_y <= y < self.clip_y + self.clip_height
        )

    def _pattern_bit(self, x: int, y: int) -> bool:
        bit = 15 - (((y & 3) << 2) | (x & 3))
        return not bool(self.fill_pattern & (1 << bit))

    def _check_palette_state(self) -> None:
        if len(self.draw_palette) != 16 or len(self.screen_palette) != 16:
            raise RasterError("PICO-8 palettes must contain 16 entries")
        if len(self.transparent) != 16:
            raise RasterError("PICO-8 transparency state must contain 16 entries")
        for palette in (self.draw_palette, self.screen_palette):
            for color in palette:
                self._check_color(color)
        self._check_color(self.draw_color)

    @staticmethod
    def _check_color(color: int) -> None:
        if (
            isinstance(color, bool)
            or not isinstance(color, int)
            or not 0 <= color <= 15
        ):
            raise RasterError("PICO-8 palette index must be an integer in 0..15")


def _fixed(value: PicoScalar) -> PicoFixed:
    if isinstance(value, PicoFixed):
        return value
    if isinstance(value, bool):
        raise RasterError("raster coordinates cannot be booleans")
    if isinstance(value, int):
        return PicoFixed.from_int(value)
    if isinstance(value, float):
        return PicoFixed.from_float(value)
    raise RasterError(f"unsupported raster coordinate: {type(value).__name__}")


def _coordinate(value: PicoScalar) -> int:
    return _fixed(value).raw >> 16
