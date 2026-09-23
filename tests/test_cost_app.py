import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest
import tests.app_test_compat  # noqa: F401 - AppTest segmented-control compatibility

APP = Path(__file__).resolve().parents[1] / 'app' / '成本测算.py'


class AppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.radio(key='workflow').set_value('single').run()
        self.assertFalse(app.exception)
        return app

    def test_only_available_methods_are_offered_in_single_page(self):
        app = self.app()
        self.assertNotIn('LightGBM', app.selectbox(key='method').options)
        self.assertEqual(len(app.selectbox(key='method').options), 4)
        self.assertEqual(app.selectbox(key='method').label, 'Coal-consumption estimation method (experiment)')

    def test_ramp_allowance_is_not_presented_as_market_risk_model(self):
        app = self.app()
        app.button(key='calculate').click().run()
        self.assertFalse(app.exception)
        labels = {item.label for item in app.metric}
        self.assertIn('Example fixed ramp-bin allowance (CNY/MWh)', labels)
        self.assertNotIn('Example risk allowance (CNY/MWh)', labels)

    def test_english_default_and_language_switch_preserve_result(self):
        app = self.app()
        self.assertTrue(any('Thermal Unit Cost Estimation Prototype' in title.value for title in app.title))
        app.selectbox(key='method').set_value('table').run()
        app.button(key='calculate').click().run()
        result = app.session_state['cost_result']
        app.button_group(key='language_switcher').set_value(['中文']).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['cost_result'], result)
        self.assertEqual(app.selectbox(key='method').value, 'table')
        self.assertTrue(any('火电机组成本测算' in title.value for title in app.title))
        self.assertFalse(any('上次测算结果' in item.value for item in app.info))

    def test_calculation_and_result_persist_after_rerun(self):
        app = self.app()
        app.button(key='calculate').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == 'Unit cost reference (CNY/MWh)' for m in app.metric))
        app.run()
        self.assertTrue(any(m.label == 'Unit cost reference (CNY/MWh)' for m in app.metric))

    def test_table_calculation_matches_baseline(self):
        app = self.app()
        app.selectbox(key='method').set_value('table').run()
        app.button(key='calculate').click().run()
        self.assertFalse(app.exception)
        values = {m.label: m.value for m in app.metric}
        self.assertEqual(values['Unit cost reference (CNY/MWh)'], '288.88')

    def test_training_and_holdout_evaluation_survive_widget_change(self):
        app = self.app()
        app.checkbox(key='use_demo').check().run()
        app.button(key='train').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['training_report']['holdout']), 150)
        app.button(key='evaluate').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['evaluation'])
        app.selectbox(key='eval_method').set_value('direct_balance').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['evaluation'])

    def test_invalid_thermal_input_does_not_show_false_success(self):
        app = self.app()
        app.number_input(key='exhaust_temp').set_value(0.0).run()
        app.button(key='calculate').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(app.metric)

    def test_unexpected_training_failure_uses_safe_english_message(self):
        with patch('models.coal_models.train_model', side_effect=RuntimeError('private backend detail')):
            app = self.app()
            app.checkbox(key='use_demo').check().run()
            app.button(key='train').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Unexpected error' in item.value for item in app.error))
        self.assertFalse(any('private backend detail' in item.value for item in app.error))

    def test_all_non_ml_methods_calculate(self):
        for method in ['counter_balance', 'direct_balance']:
            with self.subTest(method=method):
                app = self.app()
                app.selectbox(key='method').set_value(method).run()
                app.button(key='calculate').click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(app.metric)

    def test_unverified_pickle_sources_are_not_in_public_ui(self):
        app = self.app()
        self.assertFalse(any(item.key == 'model_source' for item in app.selectbox))
        self.assertFalse(any(item.key == 'legacy_trusted' for item in app.checkbox))
        self.assertFalse(any('111' in item.value for item in app.caption))

    def test_synthetic_demo_is_not_described_as_uploaded_training_data(self):
        app = self.app()
        app.checkbox(key='use_demo').check().run()
        self.assertFalse(app.exception)
        self.assertFalse(any('uploaded data' in item.value for item in app.caption))

    def test_evaluation_without_training_does_not_claim_lightgbm_model(self):
        app = self.app()
        app.checkbox(key='use_demo').check().run()
        app.button(key='evaluate').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No LightGBM model' in item.value for item in app.caption))

    def test_prior_legacy_evaluation_is_not_relabelled_as_session_model(self):
        app = self.app()
        app.session_state['evaluation'] = {
            'source': 'legacy1', 'basis': 'standard', 'note': 'diagnostic',
            'summary': pd.DataFrame(), 'details': pd.DataFrame(),
        }
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(any('Coal-method evaluation' in item.value for item in app.subheader))

    def test_prior_legacy_cost_result_is_not_presented_without_its_model_source(self):
        app = self.app()
        app.button(key='calculate').click().run()
        self.assertTrue(app.metric)
        app.session_state['cost_snapshot']['source'] = 'legacy1'
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.metric)

    def test_lgbm_is_only_offered_for_active_session_training(self):
        app = self.app()
        self.assertNotIn('LightGBM', app.selectbox(key='method').options)
        app.checkbox(key='use_demo').check().run()
        app.button(key='train').click().run()
        self.assertFalse(app.exception)
        self.assertIn('LightGBM', app.selectbox(key='method').options)
        app.selectbox(key='method').set_value('lgbm').run()
        app.checkbox(key='use_demo').uncheck().run()
        self.assertFalse(app.exception)
        self.assertNotIn('LightGBM', app.selectbox(key='method').options)
        self.assertNotEqual(app.selectbox(key='method').value, 'lgbm')

    def test_dataset_change_marks_model_result_stale(self):
        app = self.app()
        app.checkbox(key='use_demo').check().run()
        app.button(key='train').click().run()
        app.selectbox(key='method').set_value('lgbm').run()
        app.button(key='calculate').click().run()
        self.assertFalse(app.error)
        app.checkbox(key='use_demo').uncheck().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('previous result' in m.value for m in app.info))


if __name__ == '__main__':
    unittest.main()
