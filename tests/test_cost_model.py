import copy
import importlib
import unittest
from pathlib import Path


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert Path('models/cost_model.py').exists(), 'Unified cost module is missing'
        cls.core = importlib.import_module('models.cost_model')

    def params(self, **changes):
        p = self.core.default_params()
        p.update(calc_strategy='table', coal_price_basis='actual')
        p.update(changes)
        return p

    def test_table_baseline_and_seven_load_export(self):
        r = self.core.run_cost_calculation(self.params())
        self.assertEqual(r['燃料成本_元_MWh'], 232.08)
        self.assertEqual(r['单位成本参考_元_MWh'], 288.88)
        self.assertEqual(len(r['detail_df']), 7)

    def test_default_dynamic_matches_original(self):
        r = self.core.run_cost_calculation(self.core.default_params())
        self.assertEqual(r['实时煤耗_g_kWh'], 366.24)
        self.assertTrue(r['losses_breakdown'])

    def test_static_and_direct_methods_available(self):
        r = self.core.run_cost_calculation(self.params(calc_strategy='counter_balance', coal_price_basis='standard'))
        self.assertGreater(r['实时煤耗_g_kWh'], 366)
        r = self.core.run_cost_calculation(self.params(calc_strategy='direct_balance', coal_price_basis='standard'))
        self.assertEqual(r['实时煤耗_g_kWh'], 207.8)

    def test_input_is_not_modified_by_load_sweep(self):
        p = self.core.default_params()
        before = copy.deepcopy(p)
        self.core.run_cost_calculation(p)
        self.assertEqual(p, before)

    def test_carbon_year_matches_current_and_curve(self):
        r = self.core.run_cost_calculation(self.params(carbon_year=2030))
        self.assertEqual(r['碳成本_元_MWh'], 65)
        self.assertEqual(r['detail_df'].iloc[0]['碳成本 (元/MWh)'], 65)

    def test_output_energy_uses_station_rating(self):
        r = self.core.run_cost_calculation(self.params(rated_power_mw=300, operating_hours=2))
        self.assertEqual(r['计算发电量_MWh'], 600)
        self.assertAlmostEqual(r['总成本_元'], 173328)

    def test_unspecified_energy_does_not_display_false_zero_total(self):
        self.assertIsNone(self.core.run_cost_calculation(self.params())['总成本_元'])

    def test_optional_legacy_fatigue_is_transmitted_and_warned(self):
        r = self.core.run_cost_calculation(self.params(include_legacy_fatigue=True, deep_peaking_cycles=1, power_mwh=100))
        self.assertEqual(r['疲劳示例成本_元'], 5000000)
        self.assertTrue(any('疲劳' in w for w in r['warnings']))
        self.assertEqual(r['总成本_元'], 5028888)

    def test_coal_and_price_basis_must_match(self):
        with self.assertRaisesRegex(ValueError, '口径'):
            self.core.run_cost_calculation(self.params(coal_price_basis='standard'))

    def test_invalid_inputs_rejected(self):
        for change in [dict(load_pct=0), dict(load_pct=101), dict(coal_price_yuan_per_ton=-1), dict(lhv=0), dict(power_mwh=float('nan'))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.core.run_cost_calculation(self.params(**change))

    def test_impossible_temperature_rejected(self):
        p = self.core.default_params()
        p['counter_balance_input']['exhaust_temp'] = 0
        with self.assertRaises(ValueError):
            self.core.run_cost_calculation(p)

    def test_failed_predictor_never_becomes_zero_coal(self):
        def unavailable(_):
            raise ValueError('Missing model feature')
        with self.assertRaisesRegex(ValueError, 'Missing model feature'):
            self.core.run_cost_calculation(self.params(calc_strategy='lgbm', coal_basis='standard', coal_price_basis='standard'), unavailable)

    def test_custom_predictor_is_used_for_current_and_curve(self):
        r = self.core.run_cost_calculation(self.params(calc_strategy='lgbm', coal_basis='standard', coal_price_basis='standard'), lambda row: 300)
        self.assertEqual(r['燃料成本_元_MWh'], 240)
        self.assertTrue((r['detail_df']['实时煤耗 (g/kWh)'] == 300).all())


if __name__ == '__main__':
    unittest.main()
