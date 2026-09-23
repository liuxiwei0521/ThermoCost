"""Regression tests for the application's bilingual display contract."""

import importlib.util
import ast
from pathlib import Path
from string import Formatter
import unittest
from unittest.mock import patch

import pandas as pd

from app import i18n


class TranslationTests(unittest.TestCase):
    def test_shared_translation_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('app.i18n'))

    def test_new_session_uses_english(self):
        self.assertTrue(callable(getattr(i18n, 'get_language', None)))
        with patch.object(i18n.st, 'session_state', {}):
            self.assertEqual(i18n.get_language(), 'en')

    def test_switcher_value_overrides_stored_language(self):
        self.assertTrue(callable(getattr(i18n, 'get_language', None)))
        with patch.object(i18n.st, 'session_state', {'language': 'zh', 'language_switcher': 'EN'}):
            self.assertEqual(i18n.get_language(), 'en')
        with patch.object(i18n.st, 'session_state', {'language': 'en', 'language_switcher': '中文'}):
            self.assertEqual(i18n.get_language(), 'zh')

    def test_dictionary_is_complete_and_templates_match(self):
        self.assertTrue(i18n.TRANSLATIONS['en'])
        self.assertEqual(set(i18n.TRANSLATIONS['en']), set(i18n.TRANSLATIONS['zh']))
        for key in i18n.TRANSLATIONS['en']:
            def fields(template):
                return {name for _, name, _, _ in Formatter().parse(template) if name}
            self.assertEqual(fields(i18n.TRANSLATIONS['en'][key]), fields(i18n.TRANSLATIONS['zh'][key]), key)

    def test_fixed_ui_copy_has_english_entries(self):
        app_dir = Path(__file__).resolve().parents[1] / 'app'
        for filename in ('成本测算.py', 'scenario_view.py'):
            tree = ast.parse((app_dir / filename).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ui':
                    if node.args and isinstance(node.args[0], ast.Constant):
                        self.assertIn(node.args[0].value, i18n.UI_EN, (filename, node.lineno))

    def test_upload_widget_identity_does_not_depend_on_language(self):
        app_dir = Path(__file__).resolve().parents[1] / 'app'
        for filename in ('成本测算.py', 'scenario_view.py'):
            tree = ast.parse((app_dir / filename).read_text(encoding='utf-8'))
            upload_calls = [node for node in ast.walk(tree)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'file_uploader']
            self.assertTrue(upload_calls, filename)
            for node in upload_calls:
                self.assertIsInstance(node.args[0], ast.Constant)
                self.assertTrue(any(keyword.arg == 'label_visibility' and isinstance(keyword.value, ast.Constant)
                                    and keyword.value.value == 'collapsed' for keyword in node.keywords))

    def test_missing_key_does_not_fall_back_to_chinese(self):
        self.assertTrue(callable(getattr(i18n, 't', None)))
        with self.assertRaises(KeyError):
            i18n.t('absent.translation.key', 'en')

    def test_core_title_and_dynamic_text(self):
        self.assertTrue(callable(getattr(i18n, 't', None)))
        self.assertEqual(i18n.t('app.title', 'en'), 'Thermal Unit Cost Estimation Prototype')
        self.assertEqual(i18n.t('app.title', 'zh'), '火电机组成本测算研究原型')
        self.assertEqual(i18n.t('data.loaded', 'en', count=7), 'Loaded 7 records; verify source and measurement units.')

    def test_display_table_is_localized_without_changing_raw_data(self):
        self.assertTrue(callable(getattr(i18n, 'localize_table', None)))
        raw = pd.DataFrame({'负荷率 (%)': [70.0], '稳燃状态': ['无需投油']})
        before = raw.copy(deep=True)
        display = i18n.localize_table(raw, 'single_cost', 'en')
        self.assertEqual(list(display.columns), ['Load (%)', 'Stabilization state'])
        self.assertEqual(display.iloc[0, 1], 'No oil required')
        pd.testing.assert_frame_equal(raw, before)

    def test_legacy_risk_column_is_described_as_fixed_ramp_allowance_on_screen(self):
        raw = pd.DataFrame({'风险加成 (元/MWh)': [3.0]})
        self.assertEqual(
            list(i18n.localize_table(raw, 'single_cost', 'zh').columns),
            ['固定爬坡档位加成 (元/MWh)'],
        )
        self.assertEqual(
            list(i18n.localize_table(raw, 'single_cost', 'en').columns),
            ['Fixed ramp-bin allowance (CNY/MWh)'],
        )
        self.assertEqual(list(raw.columns), ['风险加成 (元/MWh)'])

    def test_scenario_display_table_translates_known_plan_names(self):
        self.assertTrue(callable(getattr(i18n, 'localize_table', None)))
        raw = pd.DataFrame({'plan': ['平稳计划'], 'planned_gross_mw': [420.0]})
        display = i18n.localize_table(raw, 'scenario_detail', 'en')
        self.assertEqual(display.loc[0, 'Plan'], 'Steady Plan')
        self.assertEqual(display.loc[0, 'Scheduled gross power (MW)'], 420.0)
        self.assertEqual(raw.loc[0, 'plan'], '平稳计划')

    def test_reason_and_assumption_localize_without_changing_codes(self):
        self.assertTrue(callable(getattr(i18n, 'localize_reason', None)))
        self.assertTrue(callable(getattr(i18n, 'localize_assumption', None)))
        reason = i18n.localize_reason('timestamp axes differ; delivered net MWh differs', 'zh')
        self.assertIn('时间轴不同', reason)
        self.assertIn('净电量不同', reason)
        assumption = i18n.localize_assumption('source_type: synthetic; neither example values nor output are measured-plant validation', 'zh')
        self.assertIn('模拟', assumption)
        self.assertIn('实测', assumption)
        self.assertIn('发票', i18n.localize_assumption('coal_price: document_example, not a purchase invoice', 'zh'))

    def test_internal_source_codes_are_explained_without_external_document_references(self):
        coal = i18n.localize_assumption(
            'coal_g_kwh: simulated-description table 2, piecewise-linear interpolation; actual-coal basis',
            'zh',
        )
        carbon = i18n.localize_assumption(
            'carbon_intensity_t_per_mwh: illustrative table-5 piecewise-linear curve',
            'zh',
        )
        source = i18n.localize_assumption(
            'coal_curve: document_example (simulated description, table 2)',
            'zh',
        )
        visible = ' '.join([coal, carbon, source])
        self.assertIn('内置模拟煤耗曲线', coal)
        self.assertIn('内置模拟排放强度曲线', carbon)
        self.assertIn('分段线性插值', visible)
        self.assertNotIn('文档', visible)
        self.assertNotIn('表 2', visible)
        self.assertNotIn('表 5', visible)

    def test_warning_and_known_errors_follow_selected_language(self):
        self.assertTrue(callable(getattr(i18n, 'localize_warning', None)))
        self.assertTrue(callable(getattr(i18n, 'localize_error', None)))
        warning = '油耗、碳成本、损耗和风险加成采用旧版示例表或经验式，未按真实机组标定。'
        self.assertIn('not calibrated', i18n.localize_warning(warning, 'en'))
        self.assertIn('exhaust temperature', i18n.localize_error(ValueError('排烟温度不能低于环境温度；请检查测点与单位'), 'calculation', 'en').lower())
        self.assertIn('第 5 行', i18n.localize_error(ValueError('row 5: oil_kg_h must be positive below 40% load'), 'scenario', 'zh'))
        self.assertIn('oil_kg_h', i18n.localize_error(ValueError('row 5: oil_kg_h must be positive below 40% load'), 'scenario', 'zh'))

    def test_known_validation_and_model_warnings_are_actionable_in_english(self):
        self.assertIn('finite', i18n.localize_error(ValueError('load_pct 超出有效范围或不是有限值'), 'calculate', 'en'))
        self.assertIn('96', i18n.localize_error(ValueError('平稳计划: expected 96 timestamp rows, received 95'), 'scenario', 'en'))
        self.assertIn('load', i18n.localize_error(ValueError('平稳计划: row 2: load outside min_load_pct/max_load_pct'), 'scenario', 'en'))
        self.assertIn('fixed', i18n.localize_warning('多负荷曲线固定其他输入，仅改变负荷及相关效率；不是机组实测负荷—成本曲线。', 'en'))

    def test_evaluation_status_and_assumptions_are_display_only(self):
        raw = pd.DataFrame({'method': ['table'], '状态': ['诊断结果；不代表真实电厂验证'], '参数假设': ['默认情景：rated_power_mw=600.0']})
        display = i18n.localize_table(raw, 'evaluation_summary', 'en')
        self.assertIn('Diagnostic', display.loc[0, 'Status'])
        self.assertIn('Default', display.loc[0, 'Parameter assumptions'])
        self.assertEqual(raw.loc[0, '状态'], '诊断结果；不代表真实电厂验证')

    def test_unknown_error_uses_safe_english_fallback(self):
        self.assertTrue(callable(getattr(i18n, 'localize_error', None)))
        self.assertEqual(i18n.localize_error(RuntimeError('未知插件故障'), 'calculation', 'en'), 'Unexpected error. Check inputs and logs.')


if __name__ == '__main__':
    unittest.main()
