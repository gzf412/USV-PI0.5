import json
import math

import numpy as np

from usv_vla.core import TaskConfig
from usv_vla.green_shore import _obstacles, generate_green_shore, review_green_shore
from usv_vla.scene_plan import shore_clearance


def test_scenes_cover_both_kinds_and_both_banks():
    scenes = [generate_green_shore(i, 3100+i) for i in range(4)]
    assert scenes[0] == generate_green_shore(0, 3100)
    assert {(s["shore_obstacle"]["bank_side"],
             s["shore_obstacle"]["obstacle_model"].startswith("v2_buoy_")) for s in scenes} == {
                 ("left", False), ("left", True), ("right", False), ("right", True)}
    assert {len(s["cylinders"]) for s in scenes} == {1, 2, 3, 4}
    for scene in scenes:
        assert scene["target_slot"] == 4
        assert len({b["slot"] for b in scene["buoys"]}) == len(scene["buoys"])
        assert scene["route"]["prompt_include_phase"] is False
        assert 25 <= math.dist(scene["initial_pose"][:2], scene["target_xy"]) <= 29
        assert min(shore_clearance(point) for point in scene["route"]["waypoints"]) >= 2.5


def make_fixture(path):
    scene = generate_green_shore(0, 3100)
    task = vars(TaskConfig(target=4))
    metadata = dict(scene_plan=scene, route=scene["route"], task=task, obstacles=_obstacles(scene))
    (path / "episode.json").write_text(json.dumps(metadata))
    (path / "summary.json").write_text(json.dumps(dict(success=True, route_complete=True)))
    start = np.asarray(scene["initial_pose"][:2])
    target = np.asarray(scene["target_xy"])
    stop = target+(start-target)/np.linalg.norm(start-target)*2.1
    vertices = [start, *map(np.asarray, scene["route"]["waypoints"]), stop]
    positions = []
    for a, b in zip(vertices, vertices[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(b-a))/.05))
        positions.extend(a+(b-a)*i/n for i in range(n))
    positions.extend([stop]*23)
    rows = []
    for i, p in enumerate(positions):
        d = positions[min(i+1, len(positions)-1)]-positions[max(0, i-1)]
        yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-9 else scene["route"]["reference_yaw_rad"]
        rows.append(dict(index=i, decision_sim_time_ns=i*100_000_000,
                         pose_xyzrpy=[*p.tolist(), 0., 0., 0., yaw],
                         speed_mps=0. if i >= len(positions)-23 else .5,
                         prompt=scene["route"]["instruction"], success=i == len(positions)-1))
    steps = path / "steps.jsonl"
    steps.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return scene, rows


def test_review_checks_recorded_trajectory_not_only_success_flag(tmp_path):
    scene, rows = make_fixture(tmp_path)
    result = review_green_shore(tmp_path)
    assert result["passed"], result["issues"]
    assert result["crossing"] is not None
    # A stationary last sample outside the requested distance still cannot
    # pass, even if the saved simulator summary claims success.
    rows[-1]["pose_xyzrpy"][:2] = scene["target_xy"]
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    result = review_green_shore(tmp_path)
    assert not result["passed"]
    assert any("stop" in issue for issue in result["issues"])
