from __future__ import annotations

import numpy as np
import pytest

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment, _raw_state_from_snapshot
from dodge.neat.state import PlayerState, RawState
from dodge.ng.waypoint import (
    PLAYER_CENTER_MAX,
    PLAYER_CENTER_MIN,
    WaypointController,
    WaypointFeasibilityConfig,
    WaypointGrid,
    evaluate_waypoint_oracle,
)

pytest.importorskip("dodge_native")


def _state(x: float, y: float) -> RawState:
    return RawState(
        frame=13,
        player=PlayerState(x, y, 0.0, 0.0, 4.0),
        enemies=(),
        aoes=(),
    )


@pytest.mark.parametrize(
    ("spacing", "expected_axis"),
    [
        (
            8,
            (
                2.0,
                10.0,
                18.0,
                26.0,
                34.0,
                42.0,
                50.0,
                58.0,
                66.0,
                74.0,
                82.0,
                90.0,
                98.0,
                106.0,
                114.0,
                122.0,
                125.0,
            ),
        ),
        (16, (2.0, 18.0, 34.0, 50.0, 66.0, 82.0, 98.0, 114.0, 125.0)),
        (32, (2.0, 34.0, 66.0, 98.0, 125.0)),
    ],
)
def test_waypoint_grid_respects_native_player_bounds(
    spacing: int, expected_axis: tuple[float, ...]
) -> None:
    grid = WaypointGrid(spacing)

    assert grid.axis_points == expected_axis
    assert all(
        PLAYER_CENTER_MIN <= point <= PLAYER_CENTER_MAX for point in grid.axis_points
    )
    assert grid.shape == (len(expected_axis), len(expected_axis))
    assert grid.point_count == len(expected_axis) ** 2


def test_waypoint_quantization_is_clamped_and_tie_breaks_toward_lower_cell() -> None:
    grid = WaypointGrid(32)

    assert grid.nearest_cell(-100, 1000) == (0, 4)
    assert grid.nearest_cell(50, 50) == (1, 1)
    assert grid.nearest_cell(50, 66) == (1, 2)


def test_waypoint_controller_maps_relative_targets_to_native_actions() -> None:
    controller = WaypointController(WaypointGrid(32))

    right = controller.decide(_state(66, 66), ACTION_CHOICES.index("right"))
    diagonal = controller.decide(_state(66, 66), ACTION_CHOICES.index("up_left"))
    wall = controller.decide(_state(2, 2), ACTION_CHOICES.index("left"))

    assert right.target == (98.0, 66.0)
    assert right.native_action == "right"
    assert right.target_reached is False
    assert diagonal.target == (34.0, 34.0)
    assert diagonal.native_action == "up_left"
    assert wall.target == (2.0, 2.0)
    assert wall.native_action == "neutral"
    assert wall.target_reached is True


def test_waypoint_grid_can_ban_corner_targets_without_teleporting() -> None:
    grid = WaypointGrid(32, ban_corner_nodes=True)

    current, target, point = grid.target_for_action(
        34, 34, ACTION_CHOICES.index("up_left")
    )

    assert current == (1, 1)
    assert target == current
    assert point == (34.0, 34.0)
    assert grid.is_corner((0, 0))
    assert not grid.is_corner((1, 1))


def test_arrival_latching_holds_neutral_until_the_next_waypoint_decision() -> None:
    controller = WaypointController(
        WaypointGrid(32),
        tolerance=2.0,
        arrival_latching=True,
    )

    correcting = controller.steer_position(31.0, 66.0, (1, 2), arrived=False)
    latched = controller.steer_position(31.0, 66.0, (1, 2), arrived=True)

    assert correcting.native_action == "right"
    assert latched.native_action == "neutral"
    assert latched.target_reached is True


@pytest.mark.parametrize("tolerance", [0.0, 2.0, 7.5])
def test_waypoint_hot_path_matches_full_decision_for_all_targets_and_actions(
    tolerance: float,
) -> None:
    grid = WaypointGrid(32)
    controller = WaypointController(grid, tolerance=tolerance)
    positions = (2.0, 18.0, 50.0, 66.0, 82.0, 125.0)
    target_cells = ((0, 0), (1, 2), (4, 4))

    assert grid.axis_points is grid.axis_points
    for x in positions:
        for y in positions:
            for target_cell in target_cells:
                for action_index in range(len(ACTION_CHOICES)):
                    decision = controller.steer_position(
                        x,
                        y,
                        target_cell,
                        action_index,
                    )
                    assert (
                        controller.native_action_index_for_position(
                            x,
                            y,
                            target_cell,
                        )
                        == decision.native_action_index
                    )


def test_waypoint_target_cell_hot_path_matches_full_target_contract() -> None:
    grid = WaypointGrid(16)
    for x, y in ((2.0, 2.0), (50.0, 66.0), (124.0, 124.0)):
        for action_index in range(len(ACTION_CHOICES)):
            assert (
                grid.target_cell_for_action(x, y, action_index)
                == grid.target_for_action(
                    x,
                    y,
                    action_index,
                )[1]
            )


def test_waypoint_controller_native_actions_preserve_game_hashes() -> None:
    controller = WaypointController(WaypointGrid(16))
    controlled = NativeBatchEnvironment(step_frames=4, full_state=True, board=True)
    direct = NativeBatchEnvironment(step_frames=4, full_state=True, board=True)
    seeds = np.asarray([13, 27], dtype=np.uint32)
    controlled.reset_batch(seeds)
    direct.reset_batch(seeds)

    for step in range(40):
        actions: list[int] = []
        for snapshot in controlled.observe_full_state():
            decision = controller.decide(
                _raw_state_from_snapshot(snapshot),
                (step + snapshot.seed) % len(ACTION_CHOICES),
            )
            actions.append(decision.native_action_index)
            assert decision.native_action in ACTION_CHOICES
        controlled_result = controlled.step_batch(actions)
        direct_result = direct.step_batch(actions)
        np.testing.assert_array_equal(
            controlled_result.state_hashes,
            direct_result.state_hashes,
        )
        np.testing.assert_array_equal(
            controlled_result.pixel_hashes,
            direct_result.pixel_hashes,
        )
        if bool(np.any(controlled_result.done)):
            break

    controlled.close()
    direct.close()


def test_waypoint_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WaypointGrid(0)
    with pytest.raises(ValueError, match="outside the nine-action"):
        WaypointGrid(16).neighbor_cell((0, 0), 9)
    with pytest.raises(ValueError, match="below half spacing"):
        WaypointController(WaypointGrid(8), tolerance=4)


def test_v44_waypoint_oracle_decodes_native_snapshots() -> None:
    result = evaluate_waypoint_oracle(
        [13, 27],
        16,
        config=WaypointFeasibilityConfig(
            max_episode_steps=2,
            lookahead_steps=1,
            hold_decisions=1,
            native_lanes=2,
        ),
    )

    assert result["spacing"] == 16
    assert result["grid_shape"] == [9, 9]
    assert result["point_count"] == 81
    assert result["summary"]["count"] == 2


def test_v45_waypoint_oracle_resets_inactive_done_lanes() -> None:
    result = evaluate_waypoint_oracle(
        [30100, 30118],
        16,
        config=WaypointFeasibilityConfig(
            max_episode_steps=80,
            lookahead_steps=1,
            hold_decisions=1,
            native_lanes=2,
        ),
    )

    assert result["summary"]["count"] == 2
