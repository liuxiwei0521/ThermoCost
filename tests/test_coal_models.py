import importlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class CoalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert Path('models/coal_models.py').exists(), 'Unified model module is missing'
        cls.ml = importlib.import_module('models.coal_models')

    def test_training_and_holdout_are_disjoint(self):
        data = self.ml.generate_demo_data(80)
        report = self.ml.train_model(data, 'thermal', 'standard')
        self.assertEqual(len(report['holdout']), 24)
        self.assertFalse(set(report['train_indices']) & set(report['test_indices']))
        self.assertEqual(report['bundle']['coal_basis'], 'standard')
        self.assertNotIn('timestamp', report['bundle']['features'])

    def test_imputer_fits_training_only_and_labels_stay_aligned(self):
        data = self.ml.generate_demo_data(80)
        data['ambient_temp'] = 10.0
        data.loc[10, 'ambient_temp'] = np.nan
        data.loc[56:, 'ambient_temp'] = 999.0
        data.loc[5, 'coal_consumption'] = np.nan
        report = self.ml.train_model(data, 'thermal', 'standard')
        features = report['bundle']['features']
        value = report['bundle']['pipeline'].named_steps['imputer'].statistics_[features.index('ambient_temp')]
        self.assertEqual(value, 10)
        self.assertNotIn(5, report['train_indices'])

    def test_missing_feature_is_not_silently_filled_with_zero(self):
        data = self.ml.generate_demo_data(30).drop(columns='lhv')
        with self.assertRaisesRegex(ValueError, 'lhv'):
            self.ml.prepare_features(data, 'thermal')

    def test_time_order_split_sorts_timestamp(self):
        data = self.ml.generate_demo_data(30).iloc[::-1].reset_index(drop=True)
        report = self.ml.train_model(data, 'thermal', 'standard')
        self.assertLess(report['train_end'], report['test_start'])

    def test_small_data_is_not_evaluated_on_training_set(self):
        with self.assertRaises(ValueError):
            self.ml.train_model(self.ml.generate_demo_data(10), 'thermal', 'standard')

    def test_zero_targets_make_mape_subset_explicit(self):
        m = self.ml.regression_metrics([0, 10], [1, 8])
        self.assertEqual(m['MAE'], 1.5)
        self.assertEqual(m['MAPE (%)'], 20)
        self.assertEqual(m['MAPE样本数'], 1)

    def test_legacy_models_keep_their_own_contract(self):
        if any(not (self.ml.MODEL_DIR / filename).is_file() for filename in self.ml.LEGACY_FILES.values()):
            self.skipTest('optional local legacy PKL files are not present')
        a = self.ml.load_legacy('legacy1', trusted=True)
        b = self.ml.load_legacy('legacy2', trusted=True)
        self.assertIn('hour_sin', a['features'])
        self.assertNotIn('hour_sin', b['features'])
        with self.assertRaises(ValueError):
            self.ml.predict_bundle(a, {'load_pct': 100})
        with self.assertRaises(ValueError):
            self.ml.load_legacy('legacy1', trusted=False)

    def test_generated_data_is_reproducible_and_labelled(self):
        a, b = self.ml.generate_demo_data(30), self.ml.generate_demo_data(30)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a['data_source'] == 'synthetic').all())
        self.assertTrue((a['coal_basis'] == 'standard').all())

    def test_feature_builder_does_not_mutate_input(self):
        data = self.ml.generate_demo_data(30)
        before = data.copy(deep=True)
        self.ml.prepare_features(data, 'thermal')
        pd.testing.assert_frame_equal(data, before)

    def test_comparison_skips_incompatible_coal_basis(self):
        data = self.ml.generate_demo_data(25)
        summary, detail = self.ml.evaluate_methods(data, 'standard')
        self.assertNotIn('table', set(summary['method']))
        self.assertIn('direct_balance', set(summary['method']))
        self.assertEqual(len(detail), 25)

    def test_bundle_prediction_can_run_on_complete_row(self):
        data = self.ml.generate_demo_data(30)
        report = self.ml.train_model(data, 'thermal', 'standard')
        value = self.ml.predict_bundle(report['bundle'], data.iloc[-1].to_dict())
        self.assertTrue(np.isfinite(value))

    def test_both_legacy_models_predict_with_complete_contract(self):
        if any(not (self.ml.MODEL_DIR / filename).is_file() for filename in self.ml.LEGACY_FILES.values()):
            self.skipTest('optional local legacy PKL files are not present')
        row = self.ml.generate_demo_data(30).iloc[-1].to_dict()
        for source in ('legacy1', 'legacy2'):
            with self.subTest(source=source):
                bundle = self.ml.load_legacy(source, trusted=True)
                self.assertGreater(self.ml.predict_bundle(bundle, row), 0)

    def test_duplicate_timestamps_cannot_cross_holdout_boundary(self):
        data = self.ml.generate_demo_data(30)
        data.loc[29, 'timestamp'] = data.loc[0, 'timestamp']
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.ml.train_model(data, 'thermal', 'standard')

    def test_invalid_evaluation_rows_do_not_turn_into_zero_predictions(self):
        data = self.ml.generate_demo_data(30)
        data.loc[0, 'exhaust_temp'] = -100
        summary, detail = self.ml.evaluate_methods(data, 'standard')
        self.assertNotIn('dynamic_cb_Pred', detail)
        state = summary.set_index('method').loc['dynamic_cb', '状态']
        self.assertIn('不可评估', state)

    def test_evaluation_uses_supplied_efficiencies(self):
        from models.cost_model import default_params, coal_at_load
        data = self.ml.generate_demo_data(30)
        data['boiler_eff'], data['pipe_eff'] = 0.8, 0.9
        labels = []
        for _, row in data.iterrows():
            p = default_params()
            p.update(calc_strategy='counter_balance', load_pct=row['load_pct'],
                     boiler_eff=row['boiler_eff'], pipe_eff=row['pipe_eff'])
            p['counter_balance_input'].update(row.to_dict())
            labels.append(coal_at_load(p, p['load_pct'])['coal'])
        data['coal_consumption'] = labels
        summary, details = self.ml.evaluate_methods(data, 'standard')
        self.assertEqual(summary.set_index('method').loc['counter_balance', 'MAE'], 0)
        self.assertEqual(list(details['counter_balance_Pred']), labels)

    def test_missing_evaluation_parameters_are_explicit(self):
        summary, _ = self.ml.evaluate_methods(self.ml.generate_demo_data(30), 'standard')
        assumptions = summary.set_index('method').loc['counter_balance', '参数假设']
        self.assertIn('boiler_eff=0.92', assumptions)
        self.assertIn('pipe_eff=0.98', assumptions)


if __name__ == '__main__':
    unittest.main()
