import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import tests.app_test_compat  # noqa: F401 - AppTest segmented-control compatibility
import models.coal_models as coal_models


APP = Path(__file__).resolve().parents[1] / 'app' / '成本测算.py'


class ScenarioAppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_96_period_workflow_is_default_and_single_point_lab_is_available(self):
        app = self.app()
        self.assertEqual(app.radio(key='workflow').value, 'scenario')
        self.assertTrue(any('Thermal Unit Cost Estimation Prototype' in item.value for item in app.title))
        self.assertIn('96-Period Schedule Cost Estimation', app.radio(key='workflow').options[0])
        self.assertTrue(any('96-Period Schedule Cost Estimation' in item.value for item in app.subheader))
        self.assertTrue(any(button.key == 'scenario_calculate' for button in app.button))
        self.assertFalse(any(selectbox.key == 'method' for selectbox in app.selectbox))
        app.radio(key='workflow').set_value('single').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['workflow_saved'], 'single')
        self.assertIn('Single-Point Coal and Cost Methods Lab', app.radio(key='workflow').options[1])
        self.assertTrue(any(selectbox.key == 'method' for selectbox in app.selectbox))

    def test_legacy_chinese_workflow_state_routes_to_scenario(self):
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state['workflow'] = '96时段运行情景'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key='workflow').value, 'scenario')

    def test_synthetic_96_point_scenario_runs_without_upload(self):
        app = self.app()
        app.radio(key='workflow').set_value('scenario').run()
        self.assertEqual(app.session_state['workflow_saved'], 'scenario')
        self.assertFalse(app.exception)
        app.button(key='scenario_calculate').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == '96-period included cost (CNY)' for m in app.metric))
        self.assertTrue(any(d.label == 'Download 96-period cost detail' for d in app.get('download_button')))

    def test_assumption_copy_is_self_contained_without_external_table_references(self):
        app = self.app()
        options = app.selectbox(key='scenario_carbon_source').options
        self.assertIn('Built-in synthetic emission-intensity curve (varies by load)', options)
        app.button(key='scenario_calculate').click().run()
        visible = ' '.join(
            str(item.value)
            for kind in ('caption', 'info', 'warning', 'markdown')
            for item in app.get(kind)
        )
        self.assertIn('built-in synthetic operating points', visible.lower())
        self.assertNotIn('Table 2', visible)
        self.assertNotIn('Table 5', visible)
        self.assertNotIn('source document', visible.lower())

    def test_different_energy_demo_warns_instead_of_ranking(self):
        app = self.app()
        app.radio(key='workflow').set_value('scenario').run()
        app.checkbox(key='demo_different_energy').check().run()
        app.button(key='scenario_calculate').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('not directly comparable' in m.value for m in app.warning))
        self.assertFalse(any(m.label == 'Plan B − A cost difference (CNY)' for m in app.metric))

    def test_public_clone_without_local_pickles_keeps_new_workflow(self):
        with TemporaryDirectory() as temporary, patch.object(coal_models, 'MODEL_DIR', Path(temporary)):
            app = self.app()
            self.assertFalse(any(item.key == 'model_source' for item in app.selectbox))
            app.button(key='scenario_calculate').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label == '96-period included cost (CNY)' for m in app.metric))
            app.radio(key='workflow').set_value('single').run()
            self.assertFalse(app.exception)
            self.assertNotIn('LightGBM', app.selectbox(key='method').options)

    def test_edited_demo_price_is_labelled_user_scenario(self):
        app = self.app()
        app.radio(key='workflow').set_value('scenario').run()
        app.number_input(key='scenario_coal_price').set_value(900.0).run()
        app.button(key='scenario_calculate').click().run()
        self.assertFalse(app.exception)
        notes = ' '.join(app.session_state['scenario_result']['assumptions'])
        self.assertIn('coal_price: user_scenario', notes)

    def test_language_switch_preserves_scenario_result_and_inputs(self):
        app = self.app()
        app.radio(key='workflow').set_value('scenario').run()
        self.assertEqual(app.session_state['workflow_saved'], 'scenario')
        app.selectbox(key='scenario_carbon_source').set_value('自定义固定值').run()
        app.checkbox(key='demo_different_energy').check().run()
        app.number_input(key='scenario_coal_price').set_value(900.0).run()
        app.button(key='scenario_calculate').click().run()
        before = app.session_state['scenario_result']['summaries'].copy(deep=True)
        app.button_group(key='language_switcher').set_value(['中文']).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key='workflow').value, 'scenario')
        self.assertEqual(app.number_input(key='scenario_coal_price').value, 900.0)
        self.assertEqual(app.selectbox(key='scenario_carbon_source').value, '自定义固定值')
        self.assertTrue(app.checkbox(key='demo_different_energy').value)
        self.assertTrue(any(m.label == '96时段含项总成本 (元)' for m in app.metric))
        self.assertTrue(app.session_state['scenario_result']['summaries'].equals(before))
        self.assertFalse(any('上次结果' in item.value for item in app.warning))


if __name__ == '__main__':
    unittest.main()
