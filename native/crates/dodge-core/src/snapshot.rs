use crate::{
    CORE_SCHEMA_VERSION, CoreError, EnemyState, InputState, LifecycleState, Mode, NativeGame,
    PicoFixed, PlayerState, RngCheckpoint,
};

pub const FRAMEBUFFER_WIDTH: usize = 128;
pub const FRAMEBUFFER_HEIGHT: usize = 128;
pub const FRAMEBUFFER_SIZE: usize = FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT;
pub const PALETTE_SIZE: usize = 16;
pub const SNAPSHOT_WIRE_VERSION: u32 = 1;
pub const CARTRIDGE_SOURCE_SHA256: [u8; 32] = [
    0x74, 0x53, 0xa9, 0x65, 0x8f, 0xd3, 0x25, 0x77, 0x38, 0x5a, 0xd7, 0x26, 0x72, 0xa5, 0x4a, 0xd8,
    0x4f, 0xf7, 0x05, 0x67, 0xfa, 0xdb, 0xde, 0x75, 0xba, 0x66, 0x34, 0xaa, 0x5c, 0xc6, 0x84, 0xa3,
];

const SNAPSHOT_MAGIC: [u8; 4] = *b"DGSN";
const MAX_SNAPSHOT_ENEMIES: u32 = 4_096;
const BACKGROUND_COLOR: u8 = 12;
const SHADOW_COLOR: u8 = 1;
const ENTITY_COLOR: u8 = 7;

/// Provenance carried by every native observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotProvenance {
    pub core_schema_version: u32,
    pub cartridge_sha256: [u8; 32],
}

impl SnapshotProvenance {
    pub const fn current() -> Self {
        Self {
            core_schema_version: CORE_SCHEMA_VERSION,
            cartridge_sha256: CARTRIDGE_SOURCE_SHA256,
        }
    }
}

/// Canonical mutable state exposed to native consumers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FullState {
    pub seed: u32,
    pub lifecycle: LifecycleState,
    pub input: InputState,
    pub rng: RngCheckpoint,
    pub player: PlayerState,
    pub enemies: Vec<EnemyState>,
    pub enemy_timer: PicoFixed,
    pub enemy_est: PicoFixed,
    pub friendly_timer: u32,
    pub enemy_max_size: PicoFixed,
    pub speed: PicoFixed,
    pub freeze_rate: PicoFixed,
    pub pattern_timer: u32,
    pub pattern_active: bool,
    pub score: PicoFixed,
    pub survival_frames: u32,
    pub transition_render_y: i16,
}

/// Indexed raster state needed to reproduce the PICO-8 draw boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderState {
    pub draw_color: u8,
    pub fill_pattern: u16,
    pub draw_palette: [u8; PALETTE_SIZE],
    pub screen_palette: [u8; PALETTE_SIZE],
    pub transparent: [bool; PALETTE_SIZE],
    pub camera_x: PicoFixed,
    pub camera_y: PicoFixed,
    pub clip_x: i16,
    pub clip_y: i16,
    pub clip_width: u16,
    pub clip_height: u16,
    pub transition_y: i16,
}

impl RenderState {
    pub const fn new(transition_y: i16) -> Self {
        Self {
            draw_color: 6,
            fill_pattern: 0,
            draw_palette: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            screen_palette: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            transparent: [
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false,
            ],
            camera_x: PicoFixed::ZERO,
            camera_y: PicoFixed::ZERO,
            clip_x: 0,
            clip_y: 0,
            clip_width: FRAMEBUFFER_WIDTH as u16,
            clip_height: FRAMEBUFFER_HEIGHT as u16,
            transition_y,
        }
    }

    pub fn validate(&self) -> bool {
        if self.draw_color >= PALETTE_SIZE as u8 {
            return false;
        }
        if self.clip_width > FRAMEBUFFER_WIDTH as u16
            || self.clip_height > FRAMEBUFFER_HEIGHT as u16
        {
            return false;
        }
        self.palette_is_valid(&self.draw_palette) && self.palette_is_valid(&self.screen_palette)
    }

    fn palette_is_valid(&self, palette: &[u8; PALETTE_SIZE]) -> bool {
        palette.iter().all(|color| *color < PALETTE_SIZE as u8)
    }
}

impl Default for RenderState {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Row-major PICO-8 palette-index framebuffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedFramebuffer {
    pixels: [u8; FRAMEBUFFER_SIZE],
}

impl IndexedFramebuffer {
    pub const fn blank() -> Self {
        Self {
            pixels: [0; FRAMEBUFFER_SIZE],
        }
    }

    pub fn pixels(&self) -> &[u8; FRAMEBUFFER_SIZE] {
        &self.pixels
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.pixels.as_slice()
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<u8> {
        if x >= FRAMEBUFFER_WIDTH || y >= FRAMEBUFFER_HEIGHT {
            return None;
        }
        self.pixels.get(y * FRAMEBUFFER_WIDTH + x).copied()
    }

    fn clear(&mut self, color: u8, state: &RenderState) {
        if let Some(mapped) = state.draw_palette.get(usize::from(color)).copied() {
            self.pixels.fill(mapped);
        }
    }

    fn rect_fill(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u8, state: &RenderState) {
        let (start_x, end_x) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (start_y, end_y) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                self.plot_world(x, y, color, state);
            }
        }
    }

    fn circle_fill(
        &mut self,
        center_x: i32,
        center_y: i32,
        radius: i32,
        color: u8,
        state: &RenderState,
    ) {
        let radius = radius.max(0);
        let radius_squared = radius * radius;
        for y in center_y - radius..=center_y + radius {
            for x in center_x - radius..=center_x + radius {
                let dx = x - center_x;
                let dy = y - center_y;
                if dx * dx + dy * dy <= radius_squared {
                    self.plot_world(x, y, color, state);
                }
            }
        }
    }

    fn plot_world(&mut self, x: i32, y: i32, color: u8, state: &RenderState) {
        let camera_x = state.camera_x.raw() >> 16;
        let camera_y = state.camera_y.raw() >> 16;
        self.plot_screen(x - camera_x, y - camera_y, color, state);
    }

    fn plot_screen(&mut self, x: i32, y: i32, color: u8, state: &RenderState) {
        if x < 0
            || y < 0
            || x >= FRAMEBUFFER_WIDTH as i32
            || y >= FRAMEBUFFER_HEIGHT as i32
            || x < i32::from(state.clip_x)
            || y < i32::from(state.clip_y)
            || x >= i32::from(state.clip_x) + i32::from(state.clip_width)
            || y >= i32::from(state.clip_y) + i32::from(state.clip_height)
            || !pattern_bit(state.fill_pattern, x, y)
        {
            return;
        }
        let Some(mapped) = state.draw_palette.get(usize::from(color)).copied() else {
            return;
        };
        if let Some(pixel) = self
            .pixels
            .get_mut(y as usize * FRAMEBUFFER_WIDTH + x as usize)
        {
            *pixel = mapped;
        }
    }
}

impl Default for IndexedFramebuffer {
    fn default() -> Self {
        Self::blank()
    }
}

/// One complete native observation from one simulation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    provenance: SnapshotProvenance,
    logical_state: FullState,
    render_state: RenderState,
    framebuffer: IndexedFramebuffer,
}

impl Snapshot {
    pub(crate) fn from_game(game: &NativeGame) -> Self {
        let logical_state = game.full_state();
        let render_state = RenderState::new(logical_state.transition_render_y);
        let framebuffer = render_full_state(&logical_state, &render_state);
        Self {
            provenance: SnapshotProvenance::current(),
            logical_state,
            render_state,
            framebuffer,
        }
    }

    pub fn provenance(&self) -> SnapshotProvenance {
        self.provenance
    }

    pub fn logical_state(&self) -> &FullState {
        &self.logical_state
    }

    pub fn render_state(&self) -> &RenderState {
        &self.render_state
    }

    pub fn framebuffer(&self) -> &IndexedFramebuffer {
        &self.framebuffer
    }

    pub fn pixels(&self) -> &[u8; FRAMEBUFFER_SIZE] {
        self.framebuffer.pixels()
    }

    pub fn pixel_hash(&self) -> u64 {
        stable_hash(self.framebuffer.as_bytes())
    }

    pub fn state_hash(&self) -> u64 {
        let mut writer = Writer::new();
        write_full_state(&mut writer, &self.logical_state);
        stable_hash(writer.as_bytes())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.bytes(&SNAPSHOT_MAGIC);
        writer.u32(SNAPSHOT_WIRE_VERSION);
        writer.u32(self.provenance.core_schema_version);
        writer.bytes(&self.provenance.cartridge_sha256);
        write_full_state(&mut writer, &self.logical_state);
        write_render_state(&mut writer, &self.render_state);
        writer.bytes(self.framebuffer.as_bytes());
        writer.into_bytes()
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        let mut reader = Reader::new(bytes);
        if reader.bytes(4)? != SNAPSHOT_MAGIC {
            return Err(CoreError::InvalidSnapshotMagic);
        }
        let version = reader.u32()?;
        if version != SNAPSHOT_WIRE_VERSION {
            return Err(CoreError::InvalidSnapshotVersion(version));
        }
        let core_schema_version = reader.u32()?;
        if core_schema_version != CORE_SCHEMA_VERSION {
            return Err(CoreError::InvalidSnapshotVersion(core_schema_version));
        }
        let mut cartridge_sha256 = [0_u8; 32];
        cartridge_sha256.copy_from_slice(reader.bytes(32)?);
        if cartridge_sha256 != CARTRIDGE_SOURCE_SHA256 {
            return Err(CoreError::InvalidSnapshotValue);
        }
        let logical_state = read_full_state(&mut reader)?;
        let render_state = read_render_state(&mut reader)?;
        if !render_state.validate() {
            return Err(CoreError::InvalidSnapshotValue);
        }
        let mut pixels = [0_u8; FRAMEBUFFER_SIZE];
        pixels.copy_from_slice(reader.bytes(FRAMEBUFFER_SIZE)?);
        if pixels.iter().any(|pixel| *pixel >= PALETTE_SIZE as u8) {
            return Err(CoreError::InvalidSnapshotValue);
        }
        if !reader.is_finished() {
            return Err(CoreError::InvalidSnapshotTrailingBytes);
        }
        Ok(Self {
            provenance: SnapshotProvenance {
                core_schema_version,
                cartridge_sha256,
            },
            logical_state,
            render_state,
            framebuffer: IndexedFramebuffer { pixels },
        })
    }
}

fn render_full_state(state: &FullState, render: &RenderState) -> IndexedFramebuffer {
    let mut framebuffer = IndexedFramebuffer::blank();
    framebuffer.clear(BACKGROUND_COLOR, render);
    match state.lifecycle.mode {
        Mode::Menu => {}
        Mode::TransitionToGame => {
            if state.transition_render_y > 0 {
                render_game(state, render, &mut framebuffer);
            }
            framebuffer.rect_fill(
                0,
                i32::from(state.transition_render_y),
                (FRAMEBUFFER_WIDTH - 1) as i32,
                i32::from(state.transition_render_y) + (FRAMEBUFFER_HEIGHT - 1) as i32,
                ENTITY_COLOR,
                render,
            );
        }
        Mode::Game | Mode::Terminal => render_game(state, render, &mut framebuffer),
    }
    framebuffer
}

fn render_game(state: &FullState, render: &RenderState, framebuffer: &mut IndexedFramebuffer) {
    for enemy in &state.enemies {
        let x = coordinate(enemy.x);
        let y = coordinate(enemy.y);
        if enemy.personality >= 2 {
            framebuffer.circle_fill(x, y + 1, 4, SHADOW_COLOR, render);
            framebuffer.circle_fill(x, y, 4, ENTITY_COLOR, render);
        } else {
            let size = coordinate(enemy.size).max(0);
            framebuffer.rect_fill(x, y + 1, x + size, y + size + 1, SHADOW_COLOR, render);
            framebuffer.rect_fill(x, y, x + size, y + size, ENTITY_COLOR, render);
        }
    }
    if !state.lifecycle.dead {
        let x = coordinate(state.player.x);
        let y = coordinate(state.player.y);
        let size = coordinate(state.player.size);
        framebuffer.circle_fill(x, y, size, ENTITY_COLOR, render);
    }
}

fn coordinate(value: PicoFixed) -> i32 {
    value.raw() >> 16
}

fn pattern_bit(pattern: u16, x: i32, y: i32) -> bool {
    let bit = 15 - (((y & 3) << 2) | (x & 3));
    pattern & (1_u16 << bit) == 0
}

fn mode_tag(mode: Mode) -> u8 {
    match mode {
        Mode::Menu => 0,
        Mode::TransitionToGame => 1,
        Mode::Game => 2,
        Mode::Terminal => 3,
    }
}

fn mode_from_tag(tag: u8) -> Result<Mode, CoreError> {
    match tag {
        0 => Ok(Mode::Menu),
        1 => Ok(Mode::TransitionToGame),
        2 => Ok(Mode::Game),
        3 => Ok(Mode::Terminal),
        _ => Err(CoreError::InvalidSnapshotValue),
    }
}

fn write_full_state(writer: &mut Writer, state: &FullState) {
    writer.u32(state.seed);
    writer.u32(state.lifecycle.frame);
    writer.u8(mode_tag(state.lifecycle.mode));
    writer.i16(state.lifecycle.transition_y);
    writer.bool(state.lifecycle.started);
    writer.bool(state.lifecycle.game_ready);
    writer.bool(state.lifecycle.dead);
    writer.u8(state.input.current_mask());
    writer.u8(state.input.previous_mask());
    writer.u32(state.rng.seed);
    for value in state.rng.state {
        writer.u32(value);
    }
    writer.u8(state.rng.front);
    writer.u8(state.rng.rear);
    write_player_state(writer, state.player);
    writer.u32(state.enemies.len() as u32);
    for enemy in &state.enemies {
        write_enemy_state(writer, *enemy);
    }
    writer.i32(state.enemy_timer.raw());
    writer.i32(state.enemy_est.raw());
    writer.u32(state.friendly_timer);
    writer.i32(state.enemy_max_size.raw());
    writer.i32(state.speed.raw());
    writer.i32(state.freeze_rate.raw());
    writer.u32(state.pattern_timer);
    writer.bool(state.pattern_active);
    writer.i32(state.score.raw());
    writer.u32(state.survival_frames);
    writer.i16(state.transition_render_y);
}

fn read_full_state(reader: &mut Reader<'_>) -> Result<FullState, CoreError> {
    let seed = reader.u32()?;
    let lifecycle = LifecycleState {
        frame: reader.u32()?,
        mode: mode_from_tag(reader.u8()?)?,
        transition_y: reader.i16()?,
        started: reader.bool()?,
        game_ready: reader.bool()?,
        dead: reader.bool()?,
    };
    let input = InputState::from_masks(reader.u8()?, reader.u8()?)?;
    let rng = read_rng(reader)?;
    let player = read_player_state(reader)?;
    let enemy_count = reader.u32()?;
    if enemy_count > MAX_SNAPSHOT_ENEMIES {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut enemies = Vec::with_capacity(enemy_count as usize);
    for _ in 0..enemy_count {
        enemies.push(read_enemy_state(reader)?);
    }
    Ok(FullState {
        seed,
        lifecycle,
        input,
        rng,
        player,
        enemies,
        enemy_timer: PicoFixed::from_raw(reader.i32()?),
        enemy_est: PicoFixed::from_raw(reader.i32()?),
        friendly_timer: reader.u32()?,
        enemy_max_size: PicoFixed::from_raw(reader.i32()?),
        speed: PicoFixed::from_raw(reader.i32()?),
        freeze_rate: PicoFixed::from_raw(reader.i32()?),
        pattern_timer: reader.u32()?,
        pattern_active: reader.bool()?,
        score: PicoFixed::from_raw(reader.i32()?),
        survival_frames: reader.u32()?,
        transition_render_y: reader.i16()?,
    })
}

fn write_player_state(writer: &mut Writer, player: PlayerState) {
    writer.i32(player.x.raw());
    writer.i32(player.y.raw());
    writer.i32(player.vx.raw());
    writer.i32(player.vy.raw());
    writer.i32(player.size.raw());
}

fn read_player_state(reader: &mut Reader<'_>) -> Result<PlayerState, CoreError> {
    Ok(PlayerState {
        x: PicoFixed::from_raw(reader.i32()?),
        y: PicoFixed::from_raw(reader.i32()?),
        vx: PicoFixed::from_raw(reader.i32()?),
        vy: PicoFixed::from_raw(reader.i32()?),
        size: PicoFixed::from_raw(reader.i32()?),
    })
}

fn write_enemy_state(writer: &mut Writer, enemy: EnemyState) {
    writer.i32(enemy.x.raw());
    writer.i32(enemy.y.raw());
    writer.i32(enemy.vx.raw());
    writer.i32(enemy.vy.raw());
    writer.i32(enemy.size.raw());
    writer.i32(enemy.max_size.raw());
    writer.i8(enemy.personality);
    writer.i32(enemy.speed.raw());
    writer.bool(enemy.inside);
    writer.bool(enemy.is_dying);
}

fn read_enemy_state(reader: &mut Reader<'_>) -> Result<EnemyState, CoreError> {
    Ok(EnemyState {
        x: PicoFixed::from_raw(reader.i32()?),
        y: PicoFixed::from_raw(reader.i32()?),
        vx: PicoFixed::from_raw(reader.i32()?),
        vy: PicoFixed::from_raw(reader.i32()?),
        size: PicoFixed::from_raw(reader.i32()?),
        max_size: PicoFixed::from_raw(reader.i32()?),
        personality: reader.i8()?,
        speed: PicoFixed::from_raw(reader.i32()?),
        inside: reader.bool()?,
        is_dying: reader.bool()?,
    })
}

fn read_rng(reader: &mut Reader<'_>) -> Result<RngCheckpoint, CoreError> {
    let seed = reader.u32()?;
    let mut state = [0_u32; 31];
    for value in &mut state {
        *value = reader.u32()?;
    }
    let front = reader.u8()?;
    let rear = reader.u8()?;
    if front >= 31 || rear >= 31 {
        return Err(CoreError::InvalidRngCheckpoint);
    }
    Ok(RngCheckpoint {
        seed,
        state,
        front,
        rear,
    })
}

fn write_render_state(writer: &mut Writer, state: &RenderState) {
    writer.u8(state.draw_color);
    writer.u16(state.fill_pattern);
    writer.bytes(&state.draw_palette);
    writer.bytes(&state.screen_palette);
    for value in state.transparent {
        writer.bool(value);
    }
    writer.i32(state.camera_x.raw());
    writer.i32(state.camera_y.raw());
    writer.i16(state.clip_x);
    writer.i16(state.clip_y);
    writer.u16(state.clip_width);
    writer.u16(state.clip_height);
    writer.i16(state.transition_y);
}

fn read_render_state(reader: &mut Reader<'_>) -> Result<RenderState, CoreError> {
    let draw_color = reader.u8()?;
    let fill_pattern = reader.u16()?;
    let mut draw_palette = [0_u8; PALETTE_SIZE];
    draw_palette.copy_from_slice(reader.bytes(PALETTE_SIZE)?);
    let mut screen_palette = [0_u8; PALETTE_SIZE];
    screen_palette.copy_from_slice(reader.bytes(PALETTE_SIZE)?);
    let mut transparent = [false; PALETTE_SIZE];
    for value in &mut transparent {
        *value = reader.bool()?;
    }
    Ok(RenderState {
        draw_color,
        fill_pattern,
        draw_palette,
        screen_palette,
        transparent,
        camera_x: PicoFixed::from_raw(reader.i32()?),
        camera_y: PicoFixed::from_raw(reader.i32()?),
        clip_x: reader.i16()?,
        clip_y: reader.i16()?,
        clip_width: reader.u16()?,
        clip_height: reader.u16()?,
        transition_y: reader.i16()?,
    })
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 14_695_981_039_346_656_037_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211_u64);
    }
    hash
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn i8(&mut self, value: i8) {
        self.bytes.push(value as u8);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn i16(&mut self, value: i16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, length: usize) -> Result<&'a [u8], CoreError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(CoreError::InvalidSnapshotTruncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(CoreError::InvalidSnapshotTruncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, CoreError> {
        self.bytes(1)?
            .first()
            .copied()
            .ok_or(CoreError::InvalidSnapshotTruncated)
    }

    fn i8(&mut self) -> Result<i8, CoreError> {
        Ok(self.u8()? as i8)
    }

    fn bool(&mut self) -> Result<bool, CoreError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CoreError::InvalidSnapshotValue),
        }
    }

    fn u16(&mut self) -> Result<u16, CoreError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.bytes(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn i16(&mut self) -> Result<i16, CoreError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.bytes(2)?);
        Ok(i16::from_le_bytes(value))
    }

    fn u32(&mut self) -> Result<u32, CoreError> {
        let mut value = [0_u8; 4];
        value.copy_from_slice(self.bytes(4)?);
        Ok(u32::from_le_bytes(value))
    }

    fn i32(&mut self) -> Result<i32, CoreError> {
        let mut value = [0_u8; 4];
        value.copy_from_slice(self.bytes(4)?);
        Ok(i32::from_le_bytes(value))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{FRAMEBUFFER_SIZE, Snapshot};
    use crate::{Action, NativeConfig, NativeGame, PicoFixed};

    #[test]
    fn reset_snapshot_exposes_typed_state_and_indexed_pixels() {
        let game = NativeGame::new(NativeConfig::default());
        let snapshot = game.snapshot();
        assert_eq!(snapshot.logical_state().lifecycle.frame, 0);
        assert_eq!(snapshot.logical_state().player.x, PicoFixed::from_int(64));
        assert!(snapshot.logical_state().enemies.is_empty());
        assert_eq!(snapshot.pixels().len(), FRAMEBUFFER_SIZE);
        assert_eq!(snapshot.framebuffer().pixel(0, 0), Some(12));
        assert_eq!(snapshot.render_state().draw_palette[12], 12);
    }

    #[test]
    fn snapshot_roundtrip_preserves_bytes_state_and_pixel_hash() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let _ = game.step(Action::Right, 17);
        let snapshot = game.snapshot();
        let bytes = snapshot.canonical_bytes();
        let restored = Snapshot::from_canonical_bytes(&bytes);
        assert!(restored.is_ok());
        if let Ok(restored) = restored {
            assert_eq!(restored.canonical_bytes(), bytes);
            assert_eq!(restored.logical_state(), snapshot.logical_state());
            assert_eq!(restored.render_state(), snapshot.render_state());
            assert_eq!(restored.pixels(), snapshot.pixels());
            assert_eq!(restored.state_hash(), snapshot.state_hash());
            assert_eq!(restored.pixel_hash(), snapshot.pixel_hash());
        }
    }

    #[test]
    fn rendering_does_not_mutate_simulation_state_or_rng() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let _ = game.step(Action::Right, 17);
        let before = game.full_state();
        let first = game.snapshot();
        let second = game.snapshot();

        assert_eq!(game.full_state(), before);
        assert_eq!(first, second);
    }

    #[test]
    fn truncated_snapshot_is_rejected_before_decode() {
        let snapshot = NativeGame::new(NativeConfig::default()).snapshot();
        let bytes = snapshot.canonical_bytes();
        assert!(!bytes.is_empty());
        let truncated = bytes.get(..bytes.len() - 1);
        assert!(truncated.is_some());
        if let Some(truncated) = truncated {
            assert!(Snapshot::from_canonical_bytes(truncated).is_err());
        }
    }
}
