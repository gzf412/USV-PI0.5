"""Observer-camera geometry tests; no Gazebo or GPU is required."""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

from usv_vla.world import AERIAL_IMAGE_TOPIC, ORIGIN, POINTED_BUOY_RAISE_M, build_world, from_arguments


class AerialWorldTests(unittest.TestCase):
    def make(self, directory, **kwargs):
        path = Path(directory)/"world.sdf"
        spec = build_world(path, **kwargs)
        return spec, ET.parse(path).getroot().find("world")

    def test_observer_is_invisible_and_uses_separate_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, world = self.make(directory)
        observer = world.find('model[@name="usv_aerial_observer"]')
        self.assertEqual(observer.findtext("static"), "true")
        self.assertIsNone(observer.find(".//visual"))
        self.assertIsNone(observer.find(".//collision"))
        self.assertIsNone(observer.find(".//plugin"))
        sensor = observer.find('link/sensor[@type="camera"]')
        self.assertEqual(sensor.findtext("topic"), AERIAL_IMAGE_TOPIC)
        self.assertEqual(sensor.findtext("gz_frame_id"), spec["aerial_camera"]["frame_id"])
        optical = observer.find('frame[@name="usv_aerial_camera_optical"]')
        self.assertEqual(optical.get("attached_to"), "camera_link")
        self.assertEqual(optical.find("pose").get("relative_to"), "camera_link")
        self.assertEqual([float(x) for x in optical.findtext("pose").split()],
                         [0., 0., 0., -math.pi/2, 0., -math.pi/2])
        self.assertEqual(sensor.findtext("update_rate"), "10")
        self.assertEqual(sensor.findtext("camera/image/format"), "R8G8B8")
        self.assertEqual(sensor.findtext("camera/image/width"), "1280")
        self.assertEqual(sensor.findtext("camera/image/height"), "720")
        self.assertEqual(sensor.findtext("camera/camera_info_topic"),
                         "/usv/aerial_camera/camera_info")
        boat_cameras = world.findall('model[@name="small_cora_float"]/.//sensor[@type="camera"]')
        self.assertEqual(len(boat_cameras), 1)
        self.assertEqual(boat_cameras[0].findtext("topic"), "/small_cora/camera/image")
        self.assertEqual(spec["state_dim"], 33)
        self.assertEqual(spec["aerial_camera"]["purpose"], "recording_only_not_policy_input")

    def test_pointed_models_raise_visual_and_collision_together(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, world = self.make(directory, count=8, cylinders=0)
        for buoy in spec['buoys']:
            z = float(world.findtext(f'model[@name="{buoy["model"]}"]/pose').split()[2])
            self.assertEqual(z, POINTED_BUOY_RAISE_M if buoy['shape'] == 'pointed' else 0.)
            self.assertEqual(z, buoy['z_m'])

    def test_camera_points_down_and_covers_geometry_and_task_regions(self):
        cases = [dict(count=1, cylinders=0, distance_m=4., bearing_deg=-180.),
                 dict(count=8, cylinders=3, distance_m=16., bearing_deg=125., seed=4),
                 dict(count=8, cylinders=3, kind="pass", side="left", bearing_deg=-90., seed=1),
                 dict(count=8, cylinders=3, kind="pass", side="right", bearing_deg=90., seed=2),
                 dict(count=8, cylinders=3, kind="pass_then_approach", side="left",
                      destination=7, distance_m=16., bearing_deg=170., seed=3),
                 dict(count=8, cylinders=3, kind="pass_then_approach", side="right",
                      destination=7, distance_m=16., bearing_deg=-170., seed=5)]
        for kwargs in cases:
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as directory:
                spec, world = self.make(directory, **kwargs)
                camera = spec["aerial_camera"]
                px, py, pz, roll, pitch, yaw = camera["pose"]
                self.assertEqual(roll, 0.)
                self.assertAlmostEqual(math.cos(yaw)*math.cos(pitch), 0.)
                self.assertAlmostEqual(math.sin(yaw)*math.cos(pitch), 0.)
                self.assertAlmostEqual(-math.sin(pitch), -1.)
                self.assertEqual([float(x) for x in world.findtext(
                    'model[@name="usv_aerial_observer"]/pose').split()], camera["pose"])
                # At Z=3 even the top of a mast remains within the framing.
                hfov, vfov = camera["horizontal_fov_rad"], camera["vertical_fov_rad"]
                half_forward, half_left = (pz-3)*math.tan(vfov/2), (pz-3)*math.tan(hfov/2)

                def assert_visible(x, y):
                    dx, dy = x-px, y-py
                    forward = math.cos(yaw)*dx+math.sin(yaw)*dy
                    left = -math.sin(yaw)*dx+math.cos(yaw)*dy
                    self.assertLessEqual(abs(forward)+camera["view"]["margin_m"], half_forward+1e-8)
                    self.assertLessEqual(abs(left)+camera["view"]["margin_m"], half_left+1e-8)

                assert_visible(*ORIGIN[:2])
                for obstacle in spec["obstacles"]:
                    radius = obstacle["radius_m"]
                    for angle in (i*math.pi/4 for i in range(8)):
                        assert_visible(obstacle["x"]+radius*math.cos(angle),
                                       obstacle["y"]+radius*math.sin(angle))
                target = next(buoy for buoy in spec["buoys"] if buoy["slot"] == spec["task"]["target"])
                if spec["task"]["kind"] != "approach":
                    sign = 1 if spec["task"]["side"] == "left" else -1
                    for forward in (-3, 0, 2, 4):
                        for left in (2*sign, 4*sign):
                            assert_visible(target["x"]+math.cos(yaw)*forward-math.sin(yaw)*left,
                                           target["y"]+math.sin(yaw)*forward+math.cos(yaw)*left)
                if spec["task"]["kind"] != "pass":
                    slot = spec["task"]["destination"]
                    endpoint = target if slot is None else next(b for b in spec["buoys"] if b["slot"] == slot)
                    for angle in (0, math.pi/2, math.pi, 3*math.pi/2):
                        assert_visible(endpoint["x"]+3*math.cos(angle), endpoint["y"]+3*math.sin(angle))

    def test_no_video_omits_observer_and_default_arguments_enable_it(self):
        arguments = dict(buoys=1, target="red-round", destination=None, task="approach",
                         side="left", bearing_deg=0., distance_m=8., seed=0, cylinders=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"world.sdf"
            spec = from_arguments(path, SimpleNamespace(**arguments, videos=False))
            self.assertFalse(spec["aerial_camera"]["enabled"])
            self.assertIsNone(spec["aerial_camera"]["pose"])
            self.assertIsNone(ET.parse(path).find('.//model[@name="usv_aerial_observer"]'))
            spec = from_arguments(path, SimpleNamespace(**arguments))
            self.assertTrue(spec["aerial_camera"]["enabled"])


if __name__ == "__main__":
    unittest.main()
