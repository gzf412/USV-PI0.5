#!/usr/bin/env python3
"""Build a geometry-checked two- or three-cylinder scene, without collecting data.

The boat crosses the nearest cylinder on one side and the farther cylinder on
the other.  The smooth reference is privileged teacher metadata; the fixed
instruction, front image, and existing boat/buoy state are the policy inputs.
This script's preflight does not establish that the simulated boat can execute
the route; an actual recording and physical gate-by-gate review are required.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from usv_vla.core import TaskConfig
from usv_vla.route import RouteTaskEvaluator
from usv_vla.scene_plan import validate_scene, xy_from
from usv_vla.world import ORIGIN, build_world


def _smoothstep(t):
    t = max(0., min(1., t))
    return t * t * (3. - 2. * t)


def _make_scene_once(seed=0, first_side="left", heading_offset_deg=15., distance_m=27., obstacles=2, layout_seed=None):
    """Build one candidate; the public function retries deterministic layouts."""
    if first_side not in ("left", "right"):
        raise ValueError("First side must be left or right")
    if not math.isfinite(distance_m) or not 22. <= distance_m <= 29.:
        raise ValueError("Target distance must be between 22 and 29 m")
    if obstacles not in (2, 3):
        raise ValueError("Choose two or three required obstacles")
    if obstacles == 3 and distance_m < 28.:
        raise ValueError("Three obstacles need at least 28 m of route length")
    if not math.isfinite(heading_offset_deg) or abs(heading_offset_deg) > 22.:
        raise ValueError("Absolute initial heading offset must be at most 22 degrees")
    rng = random.Random(int(seed) if layout_seed is None else int(layout_seed))
    # A small water-side translation and route yaw give reproducible variation
    # while leaving enough space in the actual shore collision mesh.
    reference = ORIGIN[3] + rng.uniform(-.08, .08)
    start = [ORIGIN[0] + rng.uniform(-1., 1.), ORIGIN[1] + rng.uniform(.5, 2.),
             ORIGIN[2], reference + math.radians(heading_offset_deg)]
    side_sign = 1 if first_side == "left" else -1
    other_side = "right" if first_side == "left" else "left"
    fractions = ([.29, .66] if obstacles == 2 else [.20, .50, .80])
    # Keep protected entry/exit intervals while varying relative obstacle
    # spacing, and retain generous collision and camera-frustum preflight.
    variation = .022 if obstacles == 2 else .012
    gate_xs = [distance_m * (fraction + rng.uniform(-variation, variation))
               for fraction in fractions]
    # Slight changes to the object location and planned crossing clearance
    # produce different trajectories without changing the required sides.
    offsets = [rng.uniform(1.23, 1.37) for _ in range(obstacles)]
    clearances = [rng.uniform(2.08, 2.15) for _ in range(obstacles)]
    crossings = [clearances[i] - offsets[i] for i in range(obstacles)]
    local_cylinders = [(x, -side_sign * (-1)**i * offsets[i])
                       for i, x in enumerate(gate_xs)]
    cylinders = []
    gates = []
    for idx, (forward, left) in enumerate(local_cylinders):
        x, y = xy_from(start, reference, forward, left)
        cylinders.append(dict(id=idx + 1, x=x, y=y, radius_m=.405))
        sign = side_sign * (-1)**idx
        low, high = (2., 3.7) if sign == 1 else (-3.7, -2.)
        gates.append(dict(center=[x, y], yaw_rad=reference, left_min_m=low,
                          left_max_m=high, exit_forward_m=2.))
    # The reference is tangent to the gate centerlines at each physical
    # crossing and through each exit; transitions use cubic smoothstep.
    entries = [x-2. for x in gate_xs]
    exits = [x+2.5 for x in gate_xs]
    if entries[0] <= 0 or any(b <= a for a, b in zip(exits, entries[1:])) or exits[-1] >= distance_m - 2.1:
        raise ValueError("Route is too short for the required protected exits")

    def left_at(x):
        ys = [side_sign * (-1)**i * crossings[i] for i in range(obstacles)]
        if x < entries[0]:
            return ys[0] * _smoothstep(x / entries[0])
        for i in range(obstacles):
            if x <= exits[i]:
                return ys[i]
            if i+1 < obstacles and x < entries[i+1]:
                return ys[i] + (ys[i+1]-ys[i]) * _smoothstep((x-exits[i])/(entries[i+1]-exits[i]))
        return ys[-1]

    n = math.ceil(exits[-1] / .3)
    xs = sorted({exits[-1]*i/n for i in range(1, n+1)} | set(entries) | set(gate_xs) | set(exits))
    waypoints = [xy_from(start, reference, x, left_at(x)) for x in xs]
    target = xy_from(start, reference, distance_m, 0.)
    route = dict(kind="cylinder_pass", side=first_side, reference_yaw_rad=reference,
                 cruise_speed_mps=.65, maneuver_speed_mps=.5435685442559686,
                 waypoints=waypoints, waypoint_radius_m=.85, gates=gates,
                 cylinder_models=[f"usv_cylinder_{i+1}" for i in range(obstacles)],
                 prompt_include_phase=False,
                 instruction=((f"First pass on the {first_side} side of the initially nearer orange cylinder, "
                               f"then pass on the {other_side} side of the initially farther orange cylinder; "
                               if obstacles == 2 else
                               f"First pass on the {first_side} side of the initially nearest orange cylinder, "
                               f"then pass on the {other_side} side of the initially middle-distance orange cylinder, "
                               f"then pass on the {first_side} side of the initially farthest orange cylinder; ") +
                              "then approach the red round buoy and stop 1.9 to 2.3 meters away. "
                              "Left and right refer to the initial direction from the boat to the buoy."))
    scene = dict(schema_version=1, category=f"around_{first_side}", index=0, seed=int(seed),
                 generation_attempt=0, initial_pose=start, target_xy=target,
                 cylinders=cylinders, route=route,
                 pilot_description=dict(purpose=f"{obstacles} sequential cylinder gates",
                                        first_side=first_side, second_side=other_side,
                                        heading_offset_deg=heading_offset_deg,
                                        target_distance_m=distance_m,
                                        planned_gate_center_clearance_m=clearances,
                                        layout_seed=int(seed) if layout_seed is None else int(layout_seed)))
    scene["initial_visibility"] = validate_scene(scene)
    # Exercise exactly the SDF/world construction and obstacle metadata used by
    # collection. Temporary files are deleted; no simulation is started.
    with tempfile.TemporaryDirectory(prefix="usv-multi-preflight-") as tmp:
        spec = build_world(Path(tmp)/"world.sdf", scene=scene)
        RouteTaskEvaluator(TaskConfig(**spec["task"]),
                           {row["slot"]: (row["x"], row["y"]) for row in spec["buoys"]},
                           start[:2], start[3], route=route, obstacles=spec["obstacles"])
    return scene


def make_scene(seed=0, first_side="left", heading_offset_deg=15., distance_m=27., obstacles=2):
    """Generate a reproducible varied scene with deterministic safe retries.

    Preflight checks are geometric only; real recording and independent
    review are still required for every demonstration.
    """
    errors = []
    for attempt in range(24):
        # Preserve the old seed's initial placement as the first attempt.
        layout_seed = int(seed) if attempt == 0 else int(seed) * 1009 + attempt
        try:
            scene = _make_scene_once(seed, first_side, heading_offset_deg,
                                     distance_m, obstacles, layout_seed)
            scene["generation_attempt"] = attempt
            return scene
        except ValueError as exc:
            message = str(exc)
            errors.append(message)
            if message.startswith(("First side must", "Target distance must", "Choose two", "Three obstacles need", "Absolute initial heading")):
                raise
    raise ValueError(f"Could not build safe multi-obstacle scene for seed {seed}: {errors[-1]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-side", choices=("left", "right"), default="left")
    parser.add_argument("--heading-offset-deg", type=float, default=15.)
    parser.add_argument("--distance-m", type=float, default=27.)
    parser.add_argument("--obstacles", type=int, choices=(2, 3), default=2)
    parser.add_argument("--output", type=Path, help="Write a new scene JSON; default is stdout")
    args = parser.parse_args()
    scene = make_scene(args.seed, args.first_side, args.heading_offset_deg, args.distance_m, args.obstacles)
    content = json.dumps(scene, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite a previous plan or data file.
        with args.output.open("x", encoding="utf-8") as output:
            output.write(content)


if __name__ == "__main__":
    main()
