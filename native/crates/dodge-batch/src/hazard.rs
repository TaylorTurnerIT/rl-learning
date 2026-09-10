use dodge_core::{Mode, NativeGame, PicoFixed};
use rayon::prelude::*;

pub const HAZARD_TTC: usize = 0;
pub const HAZARD_ENEMY_PRESENCE: usize = 1;
pub const HAZARD_ENEMY_VX: usize = 2;
pub const HAZARD_ENEMY_VY: usize = 3;
pub const HAZARD_ENEMY_TYPE: usize = 4;
pub const HAZARD_AOE_PRESENCE: usize = 5;
pub const HAZARD_AOE_VX: usize = 6;
pub const HAZARD_AOE_VY: usize = 7;
pub const HAZARD_AOE_STAGE: usize = 8;
pub const HAZARD_AOE_EXPLOSION: usize = 9;
pub const HAZARD_AOE_PATTERN: usize = 10;
pub const HAZARD_SPAWN_CORNER_MASK: usize = 11;
pub const HAZARD_SPAWN_HALO: usize = 12;
pub const HAZARD_PLAYER_PRESENCE: usize = 13;
pub const HAZARD_CHANNELS: usize = 14;
pub const HAZARD_SCALARS: usize = 4;
pub const HAZARD_DEFAULT_HORIZON: u32 = 32;
pub const HAZARD_DEFAULT_SPAWN_HALO_RADIUS: u32 = 1;
pub const HAZARD_MAX_GRID_SIZE: u32 = 128;
pub const HAZARD_OBSERVATION_VERSION: u32 = 1;

const PLAYER_CENTER_MIN: f32 = 2.0;
const PLAYER_CENTER_MAX: f32 = 125.0;
const SCREEN_SIZE: f32 = 128.0;

/// Immutable geometry shared by every lane in one hazard observation batch.
///
/// The game state remains lane-local; only the fixed player-center query
/// points are shared. Keeping this separate from `NativeGame` avoids rebuilding
/// the same Q16.16 centers once per lane without changing TTC semantics.
#[derive(Clone, Debug)]
pub(crate) struct HazardGeometry {
    size: usize,
    centers: Vec<PicoFixed>,
}

impl HazardGeometry {
    pub(crate) fn new(size: usize) -> Option<Self> {
        let Some(cell_count) = size.checked_mul(size) else {
            return None;
        };
        observation_size(size)?;
        let mut centers = Vec::with_capacity(size);
        let width = (PLAYER_CENTER_MAX - PLAYER_CENTER_MIN) / size as f32;
        for column in 0..size {
            centers.push(PicoFixed::from_f32(
                PLAYER_CENTER_MIN + (column as f32 + 0.5) * width,
            ));
        }
        debug_assert_eq!(centers.len() * centers.len(), cell_count);
        Some(Self { size, centers })
    }

    pub(crate) const fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn centers(&self) -> &[PicoFixed] {
        self.centers.as_slice()
    }

    fn cell_for_coordinate(&self, value: f32) -> usize {
        if value <= PLAYER_CENTER_MIN {
            return 0;
        }
        if value >= PLAYER_CENTER_MAX {
            return self.size.saturating_sub(1);
        }
        let normalized = (value - PLAYER_CENTER_MIN) / (PLAYER_CENTER_MAX - PLAYER_CENTER_MIN);
        ((normalized * self.size as f32).floor() as usize).min(self.size.saturating_sub(1))
    }

    fn cell_bounds(&self, x: f32, y: f32, width: f32, height: f32) -> (usize, usize, usize, usize) {
        // Native enemy and pattern rectangles store their top-left corner. The
        // TTC grid stores player-center cells, but footprint painting must use
        // the same rectangle bounds as the native collision/render path.
        (
            self.cell_for_coordinate(y),
            self.cell_for_coordinate(y + height),
            self.cell_for_coordinate(x),
            self.cell_for_coordinate(x + width),
        )
    }
}

/// One slow/reference hazard observation for one native lane.
#[derive(Clone, Debug, PartialEq)]
pub struct HazardObservation {
    pub lane: usize,
    pub seed: u32,
    pub frame: u32,
    pub survival_frames: u32,
    pub done: bool,
    pub mode: Mode,
    pub hazard_observation: Vec<f32>,
    pub ttc_reference: Vec<f32>,
    pub player_position: [f32; 2],
}

pub const fn observation_size(grid_size: usize) -> Option<usize> {
    if grid_size == 0 {
        return None;
    }
    let Some(cell_count) = grid_size.checked_mul(grid_size) else {
        return None;
    };
    let Some(field_values) = cell_count.checked_mul(HAZARD_CHANNELS) else {
        return None;
    };
    field_values.checked_add(HAZARD_SCALARS)
}

pub fn validate_grid_size(grid_size: u32) -> bool {
    (1..=HAZARD_MAX_GRID_SIZE).contains(&grid_size)
        && observation_size(grid_size as usize).is_some()
}

pub fn observe(
    lane: usize,
    game: &NativeGame,
    grid_size: u32,
    horizon: u32,
    spawn_halo_radius: u32,
) -> Result<HazardObservation, dodge_core::CoreError> {
    let geometry = HazardGeometry::new(grid_size as usize)
        .ok_or(dodge_core::CoreError::InvalidSnapshotValue)?;
    observe_with_geometry(lane, game, &geometry, horizon, spawn_halo_radius, false)
}

pub fn observe_parallel(
    lane: usize,
    game: &NativeGame,
    grid_size: u32,
    horizon: u32,
    spawn_halo_radius: u32,
) -> Result<HazardObservation, dodge_core::CoreError> {
    let geometry = HazardGeometry::new(grid_size as usize)
        .ok_or(dodge_core::CoreError::InvalidSnapshotValue)?;
    observe_with_geometry(lane, game, &geometry, horizon, spawn_halo_radius, true)
}

pub(crate) fn observe_with_geometry(
    lane: usize,
    game: &NativeGame,
    geometry: &HazardGeometry,
    horizon: u32,
    spawn_halo_radius: u32,
    parallel_ttc: bool,
) -> Result<HazardObservation, dodge_core::CoreError> {
    let size = geometry.size();
    let cell_count = size
        .checked_mul(size)
        .ok_or(dodge_core::CoreError::InvalidSnapshotValue)?;
    let observation_size =
        observation_size(size).ok_or(dodge_core::CoreError::InvalidSnapshotValue)?;
    let mut values = vec![0.0_f32; observation_size];
    let mut ttc_reference = Vec::with_capacity(cell_count);
    let no_hit = horizon.saturating_add(1) as f32;
    let ttc_at = |cell: usize| -> Result<Option<u32>, dodge_core::CoreError> {
        let row = cell / size;
        let column = cell % size;
        let Some(x) = geometry.centers().get(column).copied() else {
            return Err(dodge_core::CoreError::InvalidSnapshotValue);
        };
        let Some(y) = geometry.centers().get(row).copied() else {
            return Err(dodge_core::CoreError::InvalidSnapshotValue);
        };
        game.time_to_death_at(x, y, horizon)
    };
    let ttc_values = if parallel_ttc {
        (0..cell_count)
            .into_par_iter()
            .map(ttc_at)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..cell_count).map(ttc_at).collect::<Result<Vec<_>, _>>()?
    };
    for (cell, time) in ttc_values.into_iter().enumerate() {
        ttc_reference.push(time.map_or(f32::INFINITY, |frame| frame as f32));
        set_value(
            &mut values,
            HAZARD_TTC,
            cell_count,
            cell,
            time.map_or(no_hit, |frame| frame as f32),
        );
    }

    let mut enemy_velocity_sums = vec![[0.0_f32; 2]; cell_count];
    let mut enemy_velocity_counts = vec![0_u32; cell_count];
    let mut aoe_velocity_sums = vec![[0.0_f32; 2]; cell_count];
    let mut aoe_velocity_counts = vec![0_u32; cell_count];

    for enemy in game.enemies() {
        let width = if enemy.personality >= 2 {
            8.0
        } else {
            enemy.size.to_f32()
        };
        let bounds = geometry.cell_bounds(enemy.x.to_f32(), enemy.y.to_f32(), width, width);
        if enemy.personality == -1 {
            let max_size = enemy.max_size.to_f32();
            let stage = if max_size > 0.0 {
                (enemy.size.to_f32() / max_size).clamp(0.0, 1.0)
            } else {
                0.0
            };
            paint_aoe(
                &mut values,
                cell_count,
                &mut aoe_velocity_sums,
                &mut aoe_velocity_counts,
                bounds,
                size,
                enemy.vx.to_f32(),
                enemy.vy.to_f32(),
                stage,
                true,
                false,
            );
        } else {
            paint_enemy(
                &mut values,
                cell_count,
                &mut enemy_velocity_sums,
                &mut enemy_velocity_counts,
                bounds,
                size,
                enemy.vx.to_f32(),
                enemy.vy.to_f32(),
                enemy_type(enemy.personality),
            );
        }
    }

    if let Some(pattern_index) = game.active_pattern_index()
        && let Some(pattern) = game.patterns().get(pattern_index)
    {
        for rect in &pattern.rects {
            let bounds = geometry.cell_bounds(
                rect.x.to_f32(),
                rect.y.to_f32(),
                rect.width.to_f32(),
                rect.height.to_f32(),
            );
            paint_aoe(
                &mut values,
                cell_count,
                &mut aoe_velocity_sums,
                &mut aoe_velocity_counts,
                bounds,
                size,
                rect.dx.to_f32(),
                rect.dy.to_f32(),
                rect.sh.to_f32() / 2.0,
                false,
                true,
            );
        }
    }

    write_velocity_channels(
        &mut values,
        cell_count,
        &enemy_velocity_sums,
        &enemy_velocity_counts,
        HAZARD_ENEMY_VX,
        HAZARD_ENEMY_VY,
    );
    write_velocity_channels(
        &mut values,
        cell_count,
        &aoe_velocity_sums,
        &aoe_velocity_counts,
        HAZARD_AOE_VX,
        HAZARD_AOE_VY,
    );

    let mut spawn_masks = vec![0_u8; cell_count];
    let mut spawn_halo = vec![false; cell_count];
    for spawn in game.spawns() {
        let spawn_column = geometry.cell_for_coordinate(spawn.x.to_f32());
        let spawn_row = geometry.cell_for_coordinate(spawn.y.to_f32());
        let corner_bit = spawn_corner_bit(spawn.x.to_f32(), spawn.y.to_f32());
        let exact_cell = spawn_row * size + spawn_column;
        if let Some(mask) = spawn_masks.get_mut(exact_cell) {
            *mask |= corner_bit;
        }
        let radius = (spawn_halo_radius as usize).min(size.saturating_sub(1));
        let row_start = spawn_row.saturating_sub(radius);
        let row_end = spawn_row.saturating_add(radius).min(size.saturating_sub(1));
        let column_start = spawn_column.saturating_sub(radius);
        let column_end = spawn_column
            .saturating_add(radius)
            .min(size.saturating_sub(1));
        for row in row_start..=row_end {
            for column in column_start..=column_end {
                let cell = row * size + column;
                if let Some(value) = spawn_halo.get_mut(cell) {
                    *value = true;
                }
            }
        }
    }
    for cell in 0..cell_count {
        set_value(
            &mut values,
            HAZARD_SPAWN_CORNER_MASK,
            cell_count,
            cell,
            spawn_masks.get(cell).copied().unwrap_or(0) as f32,
        );
        set_value(
            &mut values,
            HAZARD_SPAWN_HALO,
            cell_count,
            cell,
            f32::from(spawn_halo.get(cell).copied().unwrap_or(false)),
        );
    }

    let player = game.player();
    let player_cell = cell_for_coordinate(player.x.to_f32(), size)
        + cell_for_coordinate(player.y.to_f32(), size) * size;
    set_value(
        &mut values,
        HAZARD_PLAYER_PRESENCE,
        cell_count,
        player_cell,
        1.0,
    );
    let scalar_offset = HAZARD_CHANNELS * cell_count;
    let player_scalars = [
        player.x.to_f32() / SCREEN_SIZE,
        player.y.to_f32() / SCREEN_SIZE,
        player.vx.to_f32() / SCREEN_SIZE,
        player.vy.to_f32() / SCREEN_SIZE,
    ];
    for (offset, value) in player_scalars.into_iter().enumerate() {
        if let Some(slot) = values.get_mut(scalar_offset + offset) {
            *slot = value;
        }
    }

    let lifecycle = game.lifecycle();
    Ok(HazardObservation {
        lane,
        seed: game.seed(),
        frame: lifecycle.frame,
        survival_frames: game.survival_frames(),
        done: lifecycle.dead,
        mode: lifecycle.mode,
        hazard_observation: values,
        ttc_reference,
        player_position: [player.x.to_f32(), player.y.to_f32()],
    })
}

pub(crate) fn centered_points(size: usize) -> Vec<PicoFixed> {
    HazardGeometry::new(size).map_or_else(Vec::new, |geometry| geometry.centers)
}

fn enemy_type(personality: i8) -> f32 {
    (i32::from(personality).clamp(0, 4) + 1) as f32 / 5.0
}

fn paint_enemy(
    values: &mut [f32],
    cell_count: usize,
    velocity_sums: &mut [[f32; 2]],
    velocity_counts: &mut [u32],
    (top, bottom, left, right): (usize, usize, usize, usize),
    size: usize,
    vx: f32,
    vy: f32,
    kind: f32,
) {
    for row in top..=bottom {
        for column in left..=right {
            let cell = row * size + column;
            set_value(values, HAZARD_ENEMY_PRESENCE, cell_count, cell, 1.0);
            update_max(values, HAZARD_ENEMY_TYPE, cell_count, cell, kind);
            add_velocity(
                velocity_sums,
                velocity_counts,
                cell,
                vx / SCREEN_SIZE,
                vy / SCREEN_SIZE,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_aoe(
    values: &mut [f32],
    cell_count: usize,
    velocity_sums: &mut [[f32; 2]],
    velocity_counts: &mut [u32],
    (top, bottom, left, right): (usize, usize, usize, usize),
    size: usize,
    vx: f32,
    vy: f32,
    stage: f32,
    explosion: bool,
    pattern: bool,
) {
    for row in top..=bottom {
        for column in left..=right {
            let cell = row * size + column;
            set_value(values, HAZARD_AOE_PRESENCE, cell_count, cell, 1.0);
            update_max(values, HAZARD_AOE_STAGE, cell_count, cell, stage);
            if explosion {
                set_value(values, HAZARD_AOE_EXPLOSION, cell_count, cell, 1.0);
            }
            if pattern {
                set_value(values, HAZARD_AOE_PATTERN, cell_count, cell, 1.0);
            }
            add_velocity(
                velocity_sums,
                velocity_counts,
                cell,
                vx / SCREEN_SIZE,
                vy / SCREEN_SIZE,
            );
        }
    }
}

fn add_velocity(sums: &mut [[f32; 2]], counts: &mut [u32], cell: usize, vx: f32, vy: f32) {
    if let Some(sum) = sums.get_mut(cell) {
        sum[0] += vx;
        sum[1] += vy;
    }
    if let Some(count) = counts.get_mut(cell) {
        *count = count.saturating_add(1);
    }
}

fn write_velocity_channels(
    values: &mut [f32],
    cell_count: usize,
    sums: &[[f32; 2]],
    counts: &[u32],
    vx_channel: usize,
    vy_channel: usize,
) {
    for cell in 0..cell_count {
        let Some(count) = counts.get(cell).copied() else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let Some(sum) = sums.get(cell).copied() else {
            continue;
        };
        set_value(values, vx_channel, cell_count, cell, sum[0] / count as f32);
        set_value(values, vy_channel, cell_count, cell, sum[1] / count as f32);
    }
}

fn set_value(values: &mut [f32], channel: usize, cell_count: usize, cell: usize, value: f32) {
    if let Some(slot) = values.get_mut(channel * cell_count + cell) {
        *slot = value;
    }
}

fn update_max(values: &mut [f32], channel: usize, cell_count: usize, cell: usize, value: f32) {
    if let Some(slot) = values.get_mut(channel * cell_count + cell) {
        *slot = slot.max(value);
    }
}

fn spawn_corner_bit(x: f32, y: f32) -> u8 {
    let horizontal = if x <= (PLAYER_CENTER_MIN + PLAYER_CENTER_MAX) / 2.0 {
        0
    } else {
        1
    };
    let vertical = if y <= (PLAYER_CENTER_MIN + PLAYER_CENTER_MAX) / 2.0 {
        0
    } else {
        1
    };
    match (horizontal, vertical) {
        (0, 0) => 1,
        (1, 0) => 2,
        (0, 1) => 4,
        (1, 1) => 8,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HAZARD_CHANNELS, HAZARD_PLAYER_PRESENCE, HAZARD_SPAWN_CORNER_MASK, HAZARD_SPAWN_HALO,
        HAZARD_TTC, HazardGeometry, observation_size, observe,
    };
    use dodge_core::{NativeConfig, NativeGame};

    #[test]
    fn reference_observation_has_finite_fixed_shape_and_spawn_labels() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        game.reset_ml();
        for _ in 0..13 {
            assert!(game.advance_frame_ml(0, 0).is_ok());
        }
        let observation = observe(0, &game, 4, 32, 1)
            .unwrap_or_else(|_| unreachable!("reference hazard observation should succeed"));
        assert_eq!(
            observation.hazard_observation.len(),
            observation_size(4).unwrap_or(0)
        );
        assert!(
            observation
                .hazard_observation
                .iter()
                .all(|value| value.is_finite())
        );
        let cell_count = 4 * 4;
        assert_eq!(
            observation
                .hazard_observation
                .get(HAZARD_TTC * cell_count)
                .copied(),
            Some(33.0)
        );
        assert_eq!(
            observation
                .hazard_observation
                .get(HAZARD_SPAWN_CORNER_MASK * cell_count)
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            observation
                .hazard_observation
                .get(HAZARD_SPAWN_HALO * cell_count + 1)
                .copied(),
            Some(1.0)
        );
        assert!(
            observation
                .hazard_observation
                .get(HAZARD_PLAYER_PRESENCE * cell_count..)
                .is_some()
        );
        assert_eq!(HAZARD_CHANNELS, 14);
    }

    #[test]
    fn observation_size_overflow_is_rejected() {
        assert_eq!(observation_size(0), None);
        assert!(observation_size(usize::MAX).is_none());
    }

    #[test]
    fn aoe_bounds_use_native_top_left_rectangle_coordinates() {
        // This is the frame-2122-sized death box from the reproduced trial-5
        // trajectory, expressed on the 16x16 frozen-center grid.
        let geometry = HazardGeometry::new(16)
            .unwrap_or_else(|| unreachable!("valid hazard geometry should be constructible"));
        assert_eq!(
            geometry.cell_bounds(82.0, 100.0, 30.0, 30.0),
            (12, 15, 10, 14)
        );
    }

    #[test]
    fn centered_geometry_is_reusable_without_changing_query_points() {
        let geometry = HazardGeometry::new(4)
            .unwrap_or_else(|| unreachable!("valid hazard geometry should be constructible"));
        let first = geometry.centers().to_vec();
        let second = geometry.centers().to_vec();
        assert_eq!(first, second);
        assert_eq!(first, super::centered_points(4));
    }
}
