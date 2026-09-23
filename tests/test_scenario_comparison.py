import unittest

import pandas as pd

from models.scenario_cost import compare_schedules, default_scenario_params, generate_demo_plans


class ScenarioComparisonTests(unittest.TestCase):
    def setUp(self):
        self.params = default_scenario_params()
        self.plans = generate_demo_plans()

    def test_same_energy_and_endpoints_allow_cost_difference(self):
        result = compare_schedules(self.plans, self.params)
        self.assertTrue(result['comparable'])
        self.assertEqual(len(result['summaries']), 2)
        self.assertEqual(set(result['details']), set(self.plans))
        self.assertAlmostEqual(result['summaries'].iloc[0]['net_mwh'], result['summaries'].iloc[1]['net_mwh'])
        self.assertAlmostEqual(result['cost_difference_yuan'], result['summaries'].iloc[1]['included_cost_yuan'] - result['summaries'].iloc[0]['included_cost_yuan'])

    def test_different_energy_suppresses_cost_ranking(self):
        plans = self.plans.copy()
        changed = plans['负荷调整计划'].copy()
        changed.loc[10, 'planned_gross_mw'] += 10
        plans['负荷调整计划'] = changed
        result = compare_schedules(plans, self.params)
        self.assertFalse(result['comparable'])
        self.assertIsNone(result['cost_difference_yuan'])
        self.assertIn('net MWh', result['reason'])

    def test_different_endpoint_suppresses_cost_ranking(self):
        plans = self.plans.copy()
        changed = plans['负荷调整计划'].copy()
        changed.loc[10, 'planned_gross_mw'] += 10
        changed.loc[12, 'planned_gross_mw'] += 10
        changed.loc[95, 'planned_gross_mw'] -= 20
        plans['负荷调整计划'] = changed
        result = compare_schedules(plans, self.params)
        self.assertFalse(result['comparable'])
        self.assertIsNone(result['cost_difference_yuan'])
        self.assertIn('endpoint', result['reason'])

    def test_invalid_plan_and_count_fail(self):
        with self.assertRaisesRegex(ValueError, 'two'):
            compare_schedules({'only': next(iter(self.plans.values()))}, self.params)
        plans = self.plans.copy()
        plans['负荷调整计划'] = plans['负荷调整计划'].iloc[:-1]
        with self.assertRaisesRegex(ValueError, '负荷调整计划'):
            compare_schedules(plans, self.params)

    def test_different_dates_are_not_directly_comparable(self):
        plans = self.plans.copy()
        changed = plans['负荷调整计划'].copy()
        changed['timestamp'] = pd.to_datetime(changed['timestamp']) + pd.Timedelta(days=31)
        plans['负荷调整计划'] = changed
        result = compare_schedules(plans, self.params)
        self.assertFalse(result['comparable'])
        self.assertIsNone(result['cost_difference_yuan'])
        self.assertIn('timestamp', result['reason'])

    def test_different_per_period_exogenous_inputs_prevent_ranking(self):
        for field, a_value, b_value in [('aux_rate_pct', 6.0, 7.0), ('carbon_intensity_t_per_mwh', 0.8, 1.0)]:
            with self.subTest(field=field):
                plans = {name: frame.copy() for name, frame in self.plans.items()}
                plans['平稳计划'][field] = a_value
                plans['负荷调整计划'][field] = b_value
                result = compare_schedules(plans, self.params)
                self.assertFalse(result['comparable'])
                self.assertIsNone(result['cost_difference_yuan'])
                self.assertIn(field, result['reason'])


if __name__ == '__main__':
    unittest.main()
