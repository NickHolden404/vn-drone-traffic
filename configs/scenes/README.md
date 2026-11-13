# Scene files

One yaml per sequence, read by `cvt.track.count.load_scene`:

```yaml
seq_id: M0101              # sequence name as in the MOT GT (existence is checked by the caller)
set_name: T-UAVDT          # manifest set the sequence belongs to
ref_frame: 0               # index into the sequence's sorted frames (0 = first), not a frame number
lines:
  - {name: main_road, p1: [120, 400], p2: [1700, 420]}   # [x, y] pixels
```

- Coordinates are pixels of the stabilization **reference frame**, the same frame the stabilized
  tracks live in.
- `ref_frame` is an **index** into the sequence's frame list sorted by frame number (0 = the first
  frame), not a frame number: with frames 1, 2, 3, 5, `ref_frame: 3` is frame 5. For the demo
  (`cvt demo`, `scripts/demo.ps1`) it is an index into the segment's frames counted from
  `--start` (0 = the first frame at `--start`).
- Lines must not intersect any ignore region of the reference frame (ignore regions are GT
  metadata, so looking at them is allowed); tracker boxes inside ignore regions are dropped
  before counting.
- Lines are drawn before any tracker output exists, so they cannot be tuned to a tracker.
- Each line needs a name and two distinct 2-D points. A track counts as crossing when its anchor
  moves from one side of the line to the other and the step intersects the segment p1..p2.
- Direction: with d = p2 - p1 and v = point - p1, `pos` is a move from the negative to the
  positive side of cross(d, v) = d.x*v.y - d.y*v.x (image y points down); `neg` is the reverse.
