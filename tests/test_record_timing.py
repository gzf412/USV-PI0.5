import unittest

from usv_vla.core import TaskConfig, TaskEvaluator
from usv_vla.record import policy_task_instruction, scheduled_decision
from usv_vla.route import RouteTaskEvaluator


class DecisionTimingTests(unittest.TestCase):
    def test_policy_instruction_hides_route_evaluator_phase(self):
        instruction = 'Approach the red buoy and stop.'
        route = {'kind': 'direct', 'instruction': instruction, 'reference_yaw_rad': 0.}
        task = RouteTaskEvaluator(TaskConfig(), {0: (15., 0.)}, (0., 0.), route=route)
        self.assertIn('Current phase:', task.context())
        self.assertEqual(policy_task_instruction(task), instruction)
        task.phase = 'finish'
        self.assertEqual(policy_task_instruction(task), instruction)
        task.route['instruction'] = ''
        with self.assertRaises(ValueError):
            policy_task_instruction(task)

    def test_policy_instruction_for_legacy_non_route_task(self):
        task = TaskEvaluator(TaskConfig(), {0: (15., 0.)}, (0., 0.))
        expected = 'Approach the red round buoy and stop 1.9 to 2.3 meters away.'
        self.assertEqual(policy_task_instruction(task), expected)
        task.phase = 'finished'
        self.assertEqual(policy_task_instruction(task), expected)

    def test_transport_jitter_keeps_ticks_causal_and_regular(self):
        previous = 2_510_000_000
        ticks = []
        for actual in (2_620_000_000, 2_710_000_000, 2_830_000_000):
            tick = scheduled_decision(actual, previous)
            self.assertLessEqual(tick, actual)
            self.assertEqual(tick-previous, 100_000_000)
            ticks.append(tick)
            previous = tick
        self.assertEqual(ticks, [2_610_000_000, 2_710_000_000, 2_810_000_000])

    def test_early_ticks_do_not_publish_duplicate_commands(self):
        self.assertIsNone(scheduled_decision(2_490_000_000, -1))
        self.assertIsNone(scheduled_decision(2_600_000_000, 2_510_000_000))

    def test_missed_interval_fails_instead_of_creating_a_catchup_row(self):
        with self.assertRaisesRegex(RuntimeError, 'lateness budget'):
            scheduled_decision(2_670_000_000, 2_510_000_000)


if __name__ == '__main__':
    unittest.main()
