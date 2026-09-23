import unittest

import pandas as pd

from models.scenario_cost import calculate_schedule, default_scenario_params, generate_demo_plans


class ScenarioCostTests(unittest.TestCase):
    def setUp(self):
        self.params = default_scenario_params()
        self.plan = generate_demo_plans()['平稳计划']

    def test_demo_plans_are_deterministic_and_labelled(self):
        plans = generate_demo_plans()
        self.assertEqual(set(plans), {'平稳计划', '负荷调整计划'})
        self.assertEqual(self.params['source_type'], 'synthetic')
        self.assertEqual(self.params['rated_mw'], 600)
        self.assertEqual(self.params['lhv_mj_kg'], 21)
        for plan in plans.values():
            self.assertEqual(len(plan), 96)
            self.assertIn('oil_kg_h', plan)
            self.assertTrue((plan['oil_kg_h'] == 0).all())
        pd.testing.assert_frame_equal(self.plan, generate_demo_plans()['平稳计划'])

    def test_rejects_malformed_time_grid(self):
        variants = [
            (self.plan.iloc[:-1].copy(), '96'),
            (self.plan.assign(timestamp=self.plan['timestamp'].where(self.plan.index != 3, self.plan['timestamp'].iloc[2])), 'timestamp'),
            (self.plan.assign(timestamp=self.plan['timestamp'].where(self.plan.index != 3, self.plan['timestamp'].iloc[3] + pd.Timedelta(minutes=1))), 'timestamp'),
            (self.plan.iloc[::-1].reset_index(drop=True), 'timestamp'),
            (self.plan.assign(timestamp=self.plan['timestamp'].where(self.plan.index != 95, self.plan['timestamp'].iloc[95] + pd.Timedelta(days=1))), 'timestamp'),
        ]
        for frame, message in variants:
            with self.subTest(message=message, first=frame['timestamp'].iloc[0]):
                with self.assertRaisesRegex(ValueError, message):
                    calculate_schedule(frame, self.params)

    def test_rejects_mixed_timezone_timestamps_as_validation_error(self):
        frame = self.plan.copy()
        frame['timestamp'] = frame['timestamp'].astype(str)
        frame.loc[3, 'timestamp'] = '2026-01-01T00:45:00+08:00'
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            calculate_schedule(frame, self.params)

    def test_rejects_bad_load_and_ramps(self):
        variants = [
            (self.plan.assign(planned_gross_mw=self.plan['planned_gross_mw'].where(self.plan.index != 0, 180)), 'initial'),
            (self.plan.assign(planned_gross_mw=self.plan['planned_gross_mw'].where(self.plan.index != 8, 600)), 'ramp'),
            (self.plan.assign(planned_gross_mw=self.plan['planned_gross_mw'].where(self.plan.index != 8, 170)), 'load'),
            (self.plan.assign(planned_gross_mw=self.plan['planned_gross_mw'].where(self.plan.index != 8, 610)), 'load'),
            (self.plan.assign(planned_gross_mw=self.plan['planned_gross_mw'].where(self.plan.index != 8, float('nan'))), 'planned_gross_mw'),
        ]
        for frame, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    calculate_schedule(frame, self.params)

    def test_rejects_implicit_zero_or_bad_oil(self):
        with self.assertRaisesRegex(ValueError, 'oil_kg_h'):
            calculate_schedule(self.plan.drop(columns='oil_kg_h'), self.params)
        for value in [-1, float('nan')]:
            frame = self.plan.copy()
            frame.loc[5, 'oil_kg_h'] = value
            with self.assertRaisesRegex(ValueError, 'oil_kg_h'):
                calculate_schedule(frame, self.params)
        frame = self.plan.copy()
        frame.loc[4, 'planned_gross_mw'] = 360
        frame.loc[5, 'planned_gross_mw'] = 234  # 39%, with valid adjacent ramps
        frame.loc[6, 'planned_gross_mw'] = 360
        with self.assertRaisesRegex(ValueError, 'oil_kg_h'):
            calculate_schedule(frame, self.params)

    def test_rejects_missing_or_invalid_auxiliary_and_carbon(self):
        for key in ['aux_rate_pct', 'carbon_intensity_t_per_mwh']:
            params = dict(self.params)
            params.pop(key)
            with self.assertRaisesRegex(ValueError, key):
                calculate_schedule(self.plan, params)
        frame = self.plan.assign(aux_rate_pct=[6.0] * 96)
        frame.loc[6, 'aux_rate_pct'] = 101
        with self.assertRaisesRegex(ValueError, 'aux_rate_pct'):
            calculate_schedule(frame, self.params)
        frame = self.plan.assign(carbon_intensity_t_per_mwh=[-0.1] * 96)
        with self.assertRaisesRegex(ValueError, 'carbon_intensity_t_per_mwh'):
            calculate_schedule(frame, self.params)

    def test_coal_and_cost_units_and_totals(self):
        frame = self.plan.copy()
        frame['planned_gross_mw'] = 600.0
        frame.loc[0, 'planned_gross_mw'] = 420.0  # obey initial; inspect row 1
        frame.loc[1, 'planned_gross_mw'] = 555.0  # obey ramp, then 600
        frame.loc[10, 'oil_kg_h'] = 10.0
        result = calculate_schedule(frame, self.params)
        row = result['detail_df'].iloc[3]
        self.assertAlmostEqual(row['coal_g_kwh'], 290.1)
        self.assertAlmostEqual(row['gross_mwh'], 150)
        self.assertAlmostEqual(row['coal_tons'], 290.1 * 150 / 1000)
        self.assertAlmostEqual(row['fuel_cost_yuan'] / row['gross_mwh'], 232.08)
        self.assertAlmostEqual(row['carbon_cost_yuan'], row['carbon_intensity_t_per_mwh'] * 150 * 62.36)
        self.assertAlmostEqual(row['net_mwh'], 150 * 0.94)
        self.assertAlmostEqual(result['detail_df'].iloc[10]['oil_cost_yuan'], 10 * 6.18 * 0.25)
        self.assertAlmostEqual(result['summary']['included_cost_yuan'], result['detail_df']['included_cost_yuan'].sum())
        self.assertAlmostEqual(result['summary']['net_mwh'], result['detail_df']['net_mwh'].sum())
        self.assertIn('not included', ' '.join(result['assumptions']).lower())

    def test_invalid_basis_or_grade_fails(self):
        for key, value in [('coal_price_basis', 'standard'), ('lhv_mj_kg', 22)]:
            params = dict(self.params, **{key: value})
            with self.assertRaisesRegex(ValueError, key):
                calculate_schedule(self.plan, params)

    def test_assumptions_identify_demo_parameter_sources(self):
        notes = ' '.join(calculate_schedule(self.plan, self.params)['assumptions'])
        self.assertIn('coal_price: built-in demonstration value', notes)
        self.assertIn('aux_rate: synthetic', notes)
        self.assertIn('schedule: synthetic', notes)


if __name__ == '__main__':
    unittest.main()
