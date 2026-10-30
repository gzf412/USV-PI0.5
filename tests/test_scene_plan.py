"""Collection-scene constraints and world integration without Gazebo."""

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

from usv_vla.scene_plan import CATEGORIES, front_camera_geometry, generate_scene, initial_visibility, shore_clearance, validate_scene, xy_from
from usv_vla.world import build_world, from_arguments


class ScenePlanTests(unittest.TestCase):
    def test_reproducible_diverse_scenes_and_complete_initial_cylinder_images(self):
        locations, headings, counts = set(), set(), set()
        for category_index, category in enumerate(CATEGORIES):
            for index in range(8):
                seed = 4600+100*category_index+index
                scene = generate_scene(category, index, seed)
                self.assertEqual(scene, generate_scene(category, index, seed))
                self.assertEqual(scene, json.loads(json.dumps(scene)))
                pose, target = scene["initial_pose"], scene["target_xy"]
                distance = math.dist(pose[:2], target)
                self.assertGreaterEqual(distance, 4.)
                self.assertLessEqual(distance, 30. if category == "shore" else 18.)
                if category == "shore":
                    self.assertGreaterEqual(distance, 25.)
                    self.assertGreaterEqual(shore_clearance(pose[:2]), 2.5)
                    self.assertLessEqual(shore_clearance(pose[:2]), 4.)
                self.assertGreater(math.cos(pose[3])*(target[0]-pose[0])+math.sin(pose[3])*(target[1]-pose[1]), 0.)
                locations.add(tuple(pose[:2])); headings.add(pose[3]); counts.add(len(scene["cylinders"]))
                visibility = initial_visibility(scene)
                if category in ("around_left", "around_right", "between"):
                    self.assertTrue(all(c["fully_visible"] for c in visibility["cylinders"]))
                    self.assertTrue(scene["route"]["gates"])
                    required = len(scene["route"]["cylinder_models"])
                    distances = [math.dist(pose[:2], [c["x"], c["y"]]) for c in scene["cylinders"]]
                    if len(distances) > required:
                        self.assertGreaterEqual(min(distances[required:])-max(distances[:required]), 1.)
                if category == "approach_visible":
                    self.assertTrue(visibility["target"]["fully_visible"])
                    self.assertFalse(visibility["target"]["occluded_by_cylinder"])
                if category == "approach_offscreen":
                    self.assertFalse(visibility["target"]["fully_visible"])
                    self.assertTrue(visibility["target"]["entirely_outside_horizontal"])
                self.assertEqual(visibility, validate_scene(scene))
        self.assertEqual(counts, {1, 2, 3, 4})
        self.assertGreater(len(locations), 40)
        self.assertGreater(len(headings), 40)

    def test_visibility_reads_the_actual_camera_and_rechecks_measured_pose(self):
        geometry = front_camera_geometry()
        self.assertAlmostEqual(geometry["horizontal_fov_rad"], math.radians(80.), places=6)
        self.assertEqual((geometry["width"], geometry["height"]), (1280, 720))
        scene = generate_scene("between", 2, 400)
        self.assertTrue(all(c["fully_visible"] for c in initial_visibility(scene)["cylinders"]))
        changed = scene["initial_pose"].copy(); changed[3] += math.pi/2
        self.assertFalse(all(c["fully_visible"] for c in initial_visibility(scene, changed)["cylinders"]))

    def test_scene_world_uses_true_moved_pose_objects_routes_and_rates(self):
        scene = generate_scene("between", 2, 520)
        scene["real_time_factor"] = 3.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scene.sdf"
            spec = build_world(path, count=8, target=7, scene=scene,
                               front_rate_hz=10, aerial_rate_hz=5)
            tree = ET.parse(path)
            self.assertEqual(len(spec["buoys"]), 1)
            self.assertEqual((spec["buoys"][0]["color"], spec["buoys"][0]["shape"]), ("red", "round"))
            self.assertEqual([spec["buoys"][0]["x"], spec["buoys"][0]["y"]], scene["target_xy"])
            self.assertEqual(spec["initial_pose"], scene["initial_pose"])
            self.assertEqual(spec["scene_plan"], scene)
            self.assertEqual(spec["route"], scene["route"])
            self.assertEqual(len(spec["obstacles"]), 5)
            self.assertEqual(spec["aerial_camera"]["rate_hz"], 5)
            self.assertEqual(tree.findtext('.//sensor[@name="front_camera"]/update_rate'), '10')
            self.assertEqual(float(tree.findtext('.//physics/real_time_factor')), 3.)
            camera = spec["aerial_camera"]
            px, py, pz, _, _, yaw = camera["pose"]
            for point in [scene["initial_pose"][:2], *scene["route"]["waypoints"]]:
                dx, dy = point[0]-px, point[1]-py
                forward, left = math.cos(yaw)*dx+math.sin(yaw)*dy, -math.sin(yaw)*dx+math.cos(yaw)*dy
                self.assertLess(abs(forward)+2, (pz-3)*math.tan(camera["vertical_fov_rad"]/2))
                self.assertLess(abs(left)+2, (pz-3)*math.tan(camera["horizontal_fov_rad"]/2))
            scene_file = Path(directory)/"scene.json"; scene_file.write_text(json.dumps(scene))
            args = SimpleNamespace(buoys=1, target="red-round", destination=None, task="approach", side="left",
                                   bearing_deg=0., distance_m=8., seed=0, cylinders=3, scene_json=scene_file,
                                   camera_rate_hz=10, aerial_rate_hz=5, videos=True)
            self.assertEqual(from_arguments(path, args)["scene_plan"], scene)

    def test_shore_geometry_and_bad_layouts_are_rejected(self):
        self.assertGreater(shore_clearance((-532., 162.)), 5.)
        self.assertEqual(shore_clearance((-530., 155.)), 0.)
        scene = generate_scene("between", 0, 500)
        behind = copy.deepcopy(scene); behind["target_xy"] = behind["initial_pose"][:2]
        with self.assertRaisesRegex(ValueError, "ahead"):
            validate_scene(behind)
        collision = copy.deepcopy(scene)
        collision["cylinders"][0]["x"], collision["cylinders"][0]["y"] = collision["initial_pose"][:2]
        with self.assertRaisesRegex(ValueError, "clear"):
            validate_scene(collision)
        self.assertEqual(generate_scene("shore", 0, 500)["route"]["kind"], "bank_follow")

    def test_long_obstacle_target_is_allowed_within_position_scale(self):
        scene = generate_scene("between", 0, 500)
        scene["target_xy"] = xy_from(scene["initial_pose"], scene["route"]["reference_yaw_rad"], 29., 0.)
        validate_scene(scene)
        scene["target_xy"] = xy_from(scene["initial_pose"], scene["route"]["reference_yaw_rad"], 31., 0.)
        with self.assertRaisesRegex(ValueError, "distance"):
            validate_scene(scene)

    def test_world_preserves_immutable_scene_payload_while_rechecking_visibility(self):
        scene = generate_scene("between", 0, 501)
        # Older saved plans lack fields introduced by a later projection tool.
        scene["initial_visibility"]["target"].pop("entirely_outside_horizontal")
        original_json = json.dumps(scene, sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            spec = build_world(Path(directory)/"scene.sdf", scene=scene)
        self.assertEqual(json.dumps(scene, sort_keys=True), original_json)
        self.assertEqual(json.dumps(spec["scene_plan"], sort_keys=True), original_json)
        self.assertEqual(json.dumps(spec["scene"], sort_keys=True), original_json)
        self.assertIn("entirely_outside_horizontal", spec["initial_visibility"]["target"])


if __name__ == "__main__":
    unittest.main()
