"""Single integrated cost prototype; run with Streamlit."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from models.cost_model import METHODS, CORRECTIONS, STEAM_DEFAULTS, LEGACY_DEFAULTS, default_params, run_cost_calculation
from models.coal_models import train_model, predict_bundle, evaluate_methods, generate_demo_data, regression_metrics
from app.scenario_view import render as render_scenario
from app.i18n import localize_error, localize_table, localize_warning, render_language_switcher, stable_selectbox, t, ui

logger = logging.getLogger(__name__)


def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def _remember_workflow():
    st.session_state['workflow_saved'] = st.session_state['workflow']


@st.cache_data
def demo_data():
    return generate_demo_data(500)


def model_panel():
    with st.sidebar:
        st.header(ui('煤耗建模与数据诊断（实验）'))
        st.caption(ui('LightGBM 如在本页训练，仅保留在当前会话；此页面不加载来源未核实的历史模型。'))
        demo = st.checkbox(ui('使用人工仿真样例（仅演示）'), key='use_demo')
        st.write(ui('训练/诊断数据 CSV'))
        upload = st.file_uploader('Training/diagnostic CSV', type=['csv'], key='dataset', label_visibility='collapsed')
        data, fingerprint = None, None
        try:
            if demo:
                data = demo_data()
                fingerprint = 'synthetic-500-v1'
                st.warning(ui('标签由旧机理公式生成，不属于真实电厂实绩。'))
                st.download_button(ui('下载人工仿真数据'), csv_bytes(data), 'synthetic_plant_demo.csv', 'text/csv')
            elif upload is not None:
                raw = upload.getvalue()
                data = pd.read_csv(BytesIO(raw))
                fingerprint = sha256(raw).hexdigest()
                st.caption(t('data.loaded', count=len(data)))
        except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
            st.error(t('error.csv_read') + ' ' + localize_error(exc, 'csv'))
        except Exception as exc:
            logger.exception('Unexpected training CSV read failure')
            st.error(t('error.csv_read') + ' ' + localize_error(exc, 'csv'))
        if st.session_state.get('data_fingerprint') != fingerprint:
            st.session_state['data_fingerprint'] = fingerprint
            st.session_state.pop('training_report', None)
            st.session_state.pop('evaluation', None)
        schema = stable_selectbox(st, ui('新模型特征集合'), ['thermal', 'legacy'], format_func=lambda x: ui('热力/煤流量特征') if x == 'thermal' else ui('旧版运行/日历特征'), key='schema')
        basis = stable_selectbox(st, ui('数据标签煤耗口径'), ['standard', 'actual'], format_func=lambda x: ui('标准煤耗') if x == 'standard' else ui('实际煤耗'), key='label_basis')
        st.caption(ui('有 timestamp 时先排序再按70/30划分；否则按文件行序划分，不声称时间泛化效果。缺失值只由训练段拟合填充。'))
        if st.button(ui('训练新模型并评价留出集'), key='train', disabled=data is None):
            try:
                report = train_model(data, schema, basis)
                report['fingerprint'] = fingerprint
                st.session_state['training_report'] = report
                st.session_state.pop('evaluation', None)
                st.session_state['model_generation'] = st.session_state.get('model_generation', 0) + 1
                st.success(ui('训练完成，仅用独立留出集报告误差。'))
            except (ValueError, KeyError, TypeError) as exc:
                st.error(t('error.training') + ' ' + localize_error(exc, 'training'))
            except Exception as exc:
                logger.exception('Unexpected model training failure')
                st.error(t('error.training') + ' ' + localize_error(exc, 'training'))
        st.session_state['model_source'] = 'session'
        report = st.session_state.get('training_report')
        bundle = report['bundle'] if report is not None else None
        if st.button(ui('比较煤耗方法'), key='evaluate', disabled=data is None):
            try:
                # A freshly trained model is compared only on its untouched holdout.
                if report is not None:
                    eval_data = report['holdout']
                    eval_basis = report['bundle']['coal_basis']
                    note = '统一使用新模型的独立留出集；仿真样本仍不能验证真实业务效果。'
                else:
                    eval_data, eval_basis = data, basis
                    note = '上传/仿真数据上的方法诊断；未训练模型时不声称样本外效果。'
                summary, details = evaluate_methods(eval_data, eval_basis, bundle)
                if bundle is not None and bundle['coal_basis'] != eval_basis:
                    note += ' 所选模型与标签口径不同，LightGBM未参加比较。'
                st.session_state['evaluation'] = {'summary': summary, 'details': details, 'note': note, 'basis': eval_basis, 'source': 'session' if bundle is not None else 'none'}
            except (ValueError, KeyError, TypeError) as exc:
                st.error(t('error.evaluation') + ' ' + localize_error(exc, 'evaluation'))
            except Exception as exc:
                logger.exception('Unexpected model evaluation failure')
                st.error(t('error.evaluation') + ' ' + localize_error(exc, 'evaluation'))
        report = st.session_state.get('training_report')
        if report is not None:
            st.caption(t('split.timestamp' if '按时间' in report['split_note'] else 'split.row_order'))
            st.caption(t('training.counts', train=len(report['train_indices']), test=len(report['test_indices'])))
            st.write(localize_table(pd.DataFrame([report['metrics']]), 'evaluation_summary'))
            st.download_button(t('download.holdout'), csv_bytes(report['holdout']), 'holdout_predictions.csv', 'text/csv')
    return bundle


def parameter_panel(bundle):
    p = default_params()
    available_methods = [name for name in METHODS if name != 'lgbm' or bundle is not None]
    if st.session_state.get('method') not in available_methods:
        st.session_state['method'] = available_methods[0]
    if st.session_state.get('_selected_method') not in available_methods:
        st.session_state['_selected_method'] = available_methods[0]
    method = stable_selectbox(st, ui('煤耗估算方法（实验）'), available_methods, format_func=lambda x: t('method.' + x), key='method')
    p['calc_strategy'] = method
    basis = 'actual' if method == 'table' else ('standard' if method != 'lgbm' else (bundle['coal_basis'] if bundle is not None else st.session_state['label_basis']))
    p['coal_basis'] = basis
    # Reset price basis only when switching between actual/standard coal methods.
    if st.session_state.get('method_basis') != basis:
        st.session_state['price_basis'] = basis
        st.session_state['_selected_price_basis'] = basis
        st.session_state['method_basis'] = basis
    tabs = st.tabs([ui(x) for x in ['运行与成本', '热力工况', '运行/日历特征（实验）', '公式边界']])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        p['load_pct'] = c1.number_input(ui('机组负荷率 (%)'), min_value=30.0, max_value=100.0, value=100.0, key='load')
        p['rated_power_mw'] = c2.number_input(ui('机组额定功率 (MW)'), min_value=1.0, value=600.0, key='rating')
        p['load_ramp'] = stable_selectbox(c3, ui('爬坡率 (%/min)'), [0.5, 1.5, 3.0, 5.0], index=1, key='ramp')
        p['lhv'] = stable_selectbox(c1, ui('煤 LHV (MJ/kg)'), [18, 21, 24], index=1, key='lhv')
        p['coal_price_basis'] = stable_selectbox(c2, ui('煤价口径'), ['actual', 'standard'], format_func=lambda x: ui('实际煤价格') if x == 'actual' else ui('标准煤折算价格'), key='price_basis')
        p['coal_price_yuan_per_ton'] = c3.number_input(ui('同口径煤价 (元/吨)'), min_value=0.0, value=800.0, key='coal_price')
        st.caption(t('coal.basis_note', basis=t('coal.actual' if basis == 'actual' else 'coal.standard')))
        p['coal_correction'] = stable_selectbox(c1, ui('煤质经验修正'), list(CORRECTIONS), format_func=lambda x: t('correction.' + x), key='correction')
        p['carbon_year'] = stable_selectbox(c2, ui('碳成本示例年份'), [2025, 2030], key='year')
        p['power_mwh'] = c1.number_input(ui('发电量 (MWh；0表示未提供)'), min_value=0.0, value=0.0, key='energy')
        p['operating_hours'] = c2.number_input(ui('运行时长 (h；0表示未提供)'), min_value=0.0, value=0.0, key='hours')
        st.caption(ui('油耗与碳成本仍使用旧版固定情景表，不提供不会影响结果的油价、碳价控件。'))
        with st.expander(ui('未经标定的疲劳成本旧公式示例')):
            p['include_legacy_fatigue'] = st.checkbox(ui('将旧疲劳成本示例计入总成本'), key='fatigue_enabled')
            st.warning(ui('旧式为次数×转子成本×比例，缺少冷启动寿命损伤基准；比例不能解释为实际转子寿命消耗。默认不计入。'))
            p['deep_peaking_cycles'] = st.number_input(ui('示例调峰次数'), min_value=0, value=0, key='cycles', disabled=not p['include_legacy_fatigue'])
            p['rotor_replacement_yuan'] = st.number_input(ui('示例转子重置成本 (元)'), min_value=0.0, value=1.5e8, key='rotor', disabled=not p['include_legacy_fatigue'])
            p['deep_cycle_damage_ratio'] = stable_selectbox(st, ui('旧公式示例比例'), [1 / 50, 1 / 30, 1 / 20], index=1, format_func=lambda x: f'1/{round(1 / x)}', key='damage_ratio', disabled=not p['include_legacy_fatigue'])
    with tabs[1]:
        st.caption(ui('流量统一按 t/h 输入，焓值按 kJ/kg 输入。示例热平衡边界与系数尚未对真实机组标定。'))
        c1, c2, c3 = st.columns(3)
        steam = p['counter_balance_input']
        steam['exhaust_temp'] = c1.number_input(ui('排烟温度 (℃)'), value=128.5, key='exhaust_temp')
        steam['o2_content'] = c2.number_input(ui('烟气含氧量 (%)'), min_value=1.0, max_value=10.0, value=3.2, key='oxygen')
        steam['ambient_temp'] = c3.number_input(ui('环境温度 (℃)'), value=20.0, key='ambient_temp')
        labels = {'ms': '主汽', 'rh': '热再热', 'pw': '给水', 'oh': '冷再热', 'sh': '过热减温水', 'rh_des': '再热减温水'}
        with st.expander(ui('流量与焓值'), expanded=False):
            c1, c2 = st.columns(2)
            for suffix, label in labels.items():
                g, h = 'g_' + suffix, 'h_' + suffix
                steam[g] = c1.number_input(ui(label) + t('steam.flow_suffix'), min_value=0.0, value=STEAM_DEFAULTS[g], key=g)
                steam[h] = c2.number_input(ui(label) + t('steam.enthalpy_suffix'), min_value=0.0, value=STEAM_DEFAULTS[h], key=h)
        p['boiler_eff'] = c1.number_input(ui('静态锅炉效率'), min_value=0.01, max_value=1.0, value=0.92, key='boiler_eff')
        p['pipe_eff'] = c2.number_input(ui('管道效率'), min_value=0.01, max_value=1.0, value=0.98, key='pipe_eff')
        flow = c3.number_input(ui('入炉煤流量 (t/h)'), min_value=0.0, value=174.0, key='coal_flow')
        p['direct_balance_input'] = {'actual_coal_flow_t_h': flow, 'base_load_pct': p['load_pct']}
        p['lgbm_input'].update(steam, actual_coal_flow_t_h=flow)
    with tabs[2]:
        st.caption(ui('仅供当前会话使用运行/日历特征训练 LightGBM 的实验。缩写沿用历史代码，实际业务定义和测点单位需核实。'))
        c1, c2 = st.columns(2)
        for i, (key, value) in enumerate(LEGACY_DEFAULTS.items()):
            p['lgbm_input'][key] = (c1 if i % 2 == 0 else c2).number_input(key, value=float(value), key='legacy_' + key)
    with tabs[3]:
        st.write(ui('查表煤耗采用最近负荷档位；35%按30%档位。动态锅炉效率沿用旧系数与70%下限。机组损耗为线性经验式。'))
        st.write(ui('低于40%且不低于35%使用脉冲投油成本；低于35%使用持续投油成本。风险加成是固定爬坡档位表，不是完整风控模型。'))
        st.write(ui('负荷扫描固定其他测点；正平衡法沿用0.95经验外推指数，不能当作真实机组工况曲线。'))
        st.write(ui('本单点模式未实现三煤源动态加权、容量补偿与辅助服务抵扣，也没有完成真实电厂验证。顶部另有独立的96时段模拟成本流程，两者口径不可直接拼接。'))
    return p


def show_cost_result(result):
    st.subheader(ui('成本测算结果'))
    c1, c2, c3 = st.columns(3)
    c1.metric(ui('单位成本参考 (元/MWh)'), f"{result['单位成本参考_元_MWh']:.2f}")
    c2.metric(ui('煤耗 (g/kWh)'), f"{result['实时煤耗_g_kWh']:.2f}")
    c3.metric(ui('燃料成本 (元/MWh)'), f"{result['燃料成本_元_MWh']:.2f}")
    c1.metric(ui('机组损耗示例 (元/MWh)'), f"{result['机组损耗_元_MWh']:.2f}")
    c2.metric(ui('投油成本示例 (元/MWh)'), f"{result['投油成本_元_MWh']:.2f}")
    c3.metric(ui('碳成本示例 (元/MWh)'), f"{result['碳成本_元_MWh']:.2f}")
    c1.metric(ui('固定爬坡档位加成示例 (元/MWh)'), f"{result['运行风险溢价_元_MWh']:.2f}")
    c2.metric(ui('计算电量 (MWh)'), f"{result['计算发电量_MWh']:.2f}")
    c3.metric(ui('总成本 (元)'), ui('未提供电量/时长') if result['总成本_元'] is None else f"{result['总成本_元']:,.2f}")
    st.caption(t('result.stabilization', state=localize_table(pd.DataFrame([{'稳燃状态': result['稳燃状态']}]), 'single_cost').iloc[0, 0], oil=result['耗油率_kg_h'], fatigue=f"{result['疲劳示例成本_元']:,.2f}"))
    for warning in result['warnings']:
        st.warning(localize_warning(warning))
    if result['losses_breakdown']:
        st.write(t('result.boiler_efficiency', value=result['dynamic_efficiency']))
        st.dataframe(localize_table(pd.DataFrame({'热损失项': result['losses_breakdown'].keys(), '损失 (%)': result['losses_breakdown'].values()}), 'single_loss'), width='stretch')
    detail = result['detail_df']
    st.dataframe(localize_table(detail, 'single_cost'), width='stretch')
    st.plotly_chart(px.line(detail.sort_values('负荷率 (%)'), x='负荷率 (%)', y='单位成本参考 (元/MWh)', markers=True, title=t('chart.load_cost'), labels={'负荷率 (%)': ui('机组负荷率 (%)'), '单位成本参考 (元/MWh)': ui('单位成本参考 (元/MWh)')}), use_container_width=True)
    st.download_button(ui('下载多负荷成本明细'), csv_bytes(detail), 'cost_detail.csv', 'text/csv')
    scalar = {k: v for k, v in result.items() if k not in ('detail_df', 'warnings', 'losses_breakdown')}
    st.download_button(ui('下载当前成本结果'), csv_bytes(pd.DataFrame([scalar])), 'current_cost.csv', 'text/csv')


def show_evaluation():
    evaluation = st.session_state.get('evaluation')
    if evaluation is None:
        return
    if evaluation.get('source') not in ('session', 'none'):
        st.session_state.pop('evaluation', None)
        return
    st.subheader(ui('煤耗方法评估与诊断'))
    st.warning(t('evaluation.note.' + ('holdout' if evaluation['source'] == 'session' and '独立留出集' in evaluation['note'] else 'diagnostic')) + (t('evaluation.basis_mismatch') if '口径不同' in evaluation['note'] else ''))
    model_source = ui('当前会话训练的 LightGBM') if evaluation['source'] == 'session' else ui('未使用 LightGBM')
    st.caption(t('evaluation.caption', basis=t('coal.actual' if evaluation['basis'] == 'actual' else 'coal.standard'), source=model_source))
    summary, details = evaluation['summary'], evaluation['details']
    st.dataframe(localize_table(summary, 'evaluation_summary'), width='stretch')
    available = [c.removesuffix('_Pred') for c in details if c.endswith('_Pred')]
    if not available:
        return
    if st.session_state.get('eval_method') not in available:
        st.session_state['eval_method'] = available[0]
    method = stable_selectbox(st, ui('诊断方法'), available, format_func=lambda x: t('method.' + x), key='eval_method')
    prediction = method + '_Pred'
    scatter = px.scatter(details, x='Load_pct', y=['True_Coal', prediction], title=t('chart.coal_comparison'), labels={'Load_pct': t('chart.load'), 'value': ui('煤耗 (g/kWh)'), 'variable': t('chart.series')})
    for trace in scatter.data:
        original = trace.name
        translated = t('chart.observed') if original == 'True_Coal' else t('method.' + method) + t('chart.prediction_suffix')
        trace.name = translated
        trace.legendgroup = translated
        trace.hovertemplate = trace.hovertemplate.replace(original, translated)
    st.plotly_chart(scatter, use_container_width=True)
    error_df = pd.DataFrame({'Error (g/kWh)': details[prediction] - details['True_Coal']})
    st.plotly_chart(px.histogram(error_df, x='Error (g/kWh)', title=t('chart.error_distribution'), labels={'count': t('chart.sample_count'), 'Error (g/kWh)': t('chart.error')}), use_container_width=True)
    segments = pd.cut(details['Load_pct'], [0, 50, 75, 90, 100], right=True, labels=['≤50%', '(50%,75%]', '(75%,90%]', '(90%,100%]'])
    rows = []
    for label in segments.cat.categories:
        mask = segments == label
        if mask.any():
            rows.append({'负荷区间': label, **regression_metrics(details.loc[mask, 'True_Coal'], details.loc[mask, prediction])})
    st.dataframe(localize_table(pd.DataFrame(rows), 'evaluation_summary'), width='stretch')
    st.caption(ui('MAPE只使用非零标签，同时报告其有效样本数；错误预测不会作为零值参与指标。'))
    st.download_button(ui('下载评估汇总'), csv_bytes(summary), 'coal_metrics.csv', 'text/csv')
    st.download_button(ui('下载逐样本预测'), csv_bytes(details), 'coal_predictions.csv', 'text/csv')


def main():
    st.set_page_config(page_title=ui('火电机组成本测算研究原型'), layout='wide')
    render_language_switcher()
    st.title(ui('火电机组成本测算研究原型'))
    st.caption(ui('96时段流程仅测算含项成本；单工况方法为独立实验。不生成市场报价或交易策略。'))
    if st.session_state.get('workflow') in ('单点成本与煤耗模型', '96时段运行情景'):
        st.session_state['workflow'] = 'single' if st.session_state['workflow'] == '单点成本与煤耗模型' else 'scenario'
    if 'workflow_saved' not in st.session_state:
        st.session_state['workflow_saved'] = st.session_state.get('workflow', 'scenario')
    if st.session_state.get('workflow') != st.session_state['workflow_saved']:
        st.session_state['workflow'] = st.session_state['workflow_saved']
    workflow = st.radio(ui('功能'), ['scenario', 'single'], index=0 if st.session_state['workflow_saved'] == 'scenario' else 1, format_func=lambda x: ui('单工况煤耗与成本方法实验' if x == 'single' else '96时段运行计划成本估算与对比'), horizontal=True, key='workflow', on_change=_remember_workflow)
    if workflow == 'scenario':
        render_scenario()
        return
    bundle = model_panel()
    params = parameter_panel(bundle)
    snapshot = {'params': params, 'source': st.session_state['model_source'], 'generation': st.session_state.get('model_generation', 0), 'data': st.session_state.get('data_fingerprint'), 'bundle_ready': bundle is not None}
    if st.button(ui('开始成本测算'), type='primary', key='calculate'):
        try:
            predictor = (lambda row: predict_bundle(bundle, row)) if bundle is not None else None
            result = run_cost_calculation(params, predictor)
            st.session_state['cost_result'] = result
            st.session_state['cost_snapshot'] = snapshot
        except (ValueError, KeyError, TypeError) as exc:
            st.session_state.pop('cost_result', None)
            st.error(t('error.calculate') + ' ' + localize_error(exc, 'calculate'))
        except Exception as exc:
            logger.exception('Unexpected cost calculation failure')
            st.session_state.pop('cost_result', None)
            st.error(t('error.calculate') + ' ' + localize_error(exc, 'calculate'))
    previous_snapshot = st.session_state.get('cost_snapshot')
    if previous_snapshot is not None and previous_snapshot.get('source') != 'session':
        st.session_state.pop('cost_result', None)
        st.session_state.pop('cost_snapshot', None)
    result = st.session_state.get('cost_result')
    if result is not None:
        if st.session_state.get('cost_snapshot') != snapshot:
            st.info(ui('下面是上次测算结果；当前输入或模型已变化，请重新测算。'))
        show_cost_result(result)
    show_evaluation()


if __name__ == '__main__':
    main()
