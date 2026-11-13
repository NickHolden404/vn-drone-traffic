import numpy as np
import pandas as pd
import pytest

from cvt.analytics.maps import grid_maps, render_map


def test_grid_maps_counts_seconds_and_speed():
    fps, mpp = 10.0, 0.1                       # frame 200x100 px = 20 m x 10 m, 2 m cells
    mover = pd.DataFrame({"track_id": 1, "frame": range(50), "label": 1,
                          "xs_m": np.linspace(1, 19, 50), "ys_m": 1.0, "speed_kmh": 36.0})
    parked = pd.DataFrame({"track_id": 2, "frame": range(100), "label": 1,
                           "xs_m": 5.0, "ys_m": 9.0, "speed_kmh": 0.5})
    m = grid_maps(pd.concat([mover, parked]), mpp, (200, 100), fps, 2.0, 3.0)
    assert m["density"].shape == (5, 10)
    assert m["density"][4, 2] == pytest.approx(10.0)          # 100 frames at 10 fps
    assert m["stationary"][4, 2] == pytest.approx(10.0) and m["stationary"][0].sum() == 0
    assert m["density"][0].sum() == pytest.approx(5.0)
    assert m["speed"][0, 3] == pytest.approx(36.0) and np.isnan(m["speed"][2, 2])


def test_render_map_keeps_size_and_leaves_empty_cells_clear():
    ref = np.full((100, 200, 3), 90, np.uint8)
    grid = np.full((5, 10), np.nan)
    grid[0, :3] = [10.0, 20.0, 30.0]
    out = render_map(ref, grid, "speed", "km/h")
    assert out.shape == ref.shape and out.dtype == np.uint8
    assert (out[70:80, 20:60] == 90).all()                    # an empty cell away from the legend
    assert not (out[5:15, 5:15] == 90).all()


def test_render_map_places_cells_at_cell_px():
    ref = np.full((100, 100, 3), 90, np.uint8)
    grid = np.full((3, 3), np.nan)
    grid[2, 0] = 5.0                                          # third row: y 80..120, cropped
    painted = (render_map(ref, grid, "", "s", cell_px=40.0) != 90).any(axis=2)
    assert painted[80:100, 0:40].all()
    assert painted[:, :70].sum() == 20 * 40                   # nothing outside the cell
    assert not painted[76:, 40:].any()                        # (the legend sits at x > 70)
    stretched = (render_map(ref, grid, "", "s") != 90).any(axis=2)
    assert stretched[67:80, 0:30].all()                       # the old stretch: 3 rows over 100 px
