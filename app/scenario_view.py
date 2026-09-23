"""Streamlit view for the illustrative, non-bidding 96-period cost workflow."""

from hashlib import sha256
from io import BytesIO
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from models.scenario_cost import compare_schedules, default_scenario_params, generate_demo_plans
from app.i18n import localize_assumption, localize_error, localize_plan_name, localize_reason, localize_table, stable_selectbox, t, ui

logger = logging.getLogger(__name__)


def _download_csv(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


def render():
    st.subheader(ui('96时段运行计划成本估算与对比'))
    st.info(ui('这是成本研究原型，不生成报价、交易策略或利润。演示数据和参数不是电厂实测。'))
    defaults = default_scenario_params()
    params = default_scenario_params()
    demo = st.checkbox(ui('使用内置人工计划（可直接运行）'), value=True, key='scenario_demo')
    different_energy = st.checkbox(ui('演示不可比情景：第二计划第11时段增加10 MW'), key='demo_different_energy', disabled=not demo)
    uploads = None
    demo_plans = None
    if demo:
        demo_plans = generate_demo_plans()
        if different_energy:
            demo_plans['负荷调整计划'].loc[10, 'planned_gross_mw'] += 10
        left, right = st.columns(2)
        left.download_button(ui('下载人工计划A CSV'), _download_csv(demo_plans['平稳计划']), 'demo_plan_a.csv', 'text/csv')
        right.download_button(ui('下载人工计划B CSV'), _download_csv(demo_plans['负荷调整计划']), 'demo_plan_b.csv', 'text/csv')
    else:
        st.caption(ui('上传两个各含96行的 CSV。必填列：timestamp、planned_gross_mw、oil_kg_h；可选逐时段 aux_rate_pct、carbon_intensity_t_per_mwh。时间须为同一天00:00—23:45、每15分钟一条；数据来源和业务含义由上传者核实。'))
        left, right = st.columns(2)
        left.write(ui('计划A CSV'))
        right.write(ui('计划B CSV'))
        uploads = (
            left.file_uploader('Plan A CSV', type=['csv'], key='scenario_plan_a', label_visibility='collapsed'),
            right.file_uploader('Plan B CSV', type=['csv'], key='scenario_plan_b', label_visibility='collapsed'),
        )
        params['source_type'] = 'user_scenario'
        params['parameter_sources']['schedule'] = 'user-uploaded scenario; provenance not verified by this app'

    with st.expander(ui('情景参数与来源'), expanded=False):
        st.caption(ui('默认煤耗和排放强度由内置模拟工况点生成，仅用于功能演示；价格和厂用电率也是演示参数。修改后的参数属于用户情景，不代表实测。'))
        c1, c2, c3 = st.columns(3)
        params['rated_mw'] = c1.number_input(ui('额定功率 (MW)'), min_value=1.0, value=600.0, key='scenario_rated')
        params['initial_mw'] = c2.number_input(ui('计划前初始出力 (MW)'), min_value=0.0, value=420.0, key='scenario_initial')
        params['ramp_limit_pct_per_min'] = c3.number_input(ui('最大爬坡率 (%额定功率/分钟)'), min_value=0.0, value=1.5, key='scenario_ramp')
        params['min_load_pct'] = c1.number_input(ui('最低负荷率 (%)'), min_value=30.0, max_value=99.0, value=30.0, key='scenario_min_load')
        params['max_load_pct'] = c2.number_input(ui('最高负荷率 (%)'), min_value=31.0, max_value=100.0, value=100.0, key='scenario_max_load')
        params['lhv_mj_kg'] = stable_selectbox(c3, ui('实际煤低位热值 (MJ/kg)'), [18, 21, 24], index=1, key='scenario_lhv')
        params['coal_price_yuan_per_ton'] = c1.number_input(ui('实际煤价 (元/吨)'), min_value=0.0, value=800.0, key='scenario_coal_price')
        params['oil_price_yuan_per_kg'] = c2.number_input(ui('油价 (元/kg)'), min_value=0.0, value=6.18, key='scenario_oil_price')
        params['carbon_price_yuan_per_ton'] = c3.number_input(ui('碳价 (元/吨CO₂)'), min_value=0.0, value=62.36, key='scenario_carbon_price')
        params['aux_rate_pct'] = c1.number_input(ui('统一厂用电率 (%)'), min_value=0.0, max_value=99.9, value=6.0, key='scenario_aux')
        carbon_source = stable_selectbox(c2, ui('排放强度输入'), ['内置模拟排放强度曲线（随负荷率变化）', '自定义固定值'], format_func=ui, key='scenario_carbon_source')
        if carbon_source == '自定义固定值':
            params['carbon_intensity_t_per_mwh'] = c3.number_input(ui('固定排放强度 (吨CO₂/毛MWh)'), min_value=0.0, max_value=3.0, value=0.842, key='scenario_carbon_intensity')
        st.caption(ui('煤耗按“实际煤 g/kWh × 实际煤价 元/吨”；不自动折算标准煤。毛发电量扣除厂用电率后为净电量；厂用电没有单独虚构电费。'))

    source_keys = {
        'rated_mw': 'rated_power', 'initial_mw': 'initial_power',
        'ramp_limit_pct_per_min': 'ramp_limit', 'min_load_pct': 'load_bounds',
        'max_load_pct': 'load_bounds', 'lhv_mj_kg': 'lhv',
        'coal_price_yuan_per_ton': 'coal_price', 'oil_price_yuan_per_kg': 'oil_price',
        'carbon_price_yuan_per_ton': 'carbon_price', 'aux_rate_pct': 'aux_rate',
        'carbon_intensity_t_per_mwh': 'carbon_curve',
    }
    for field, source_key in source_keys.items():
        if params[field] != defaults[field]:
            params['parameter_sources'][source_key] = 'user_scenario (edited in UI; not independently verified)'
            params['source_type'] = 'user_scenario'

    upload_digests = tuple(sha256(item.getvalue()).hexdigest() if item is not None else None for item in uploads) if uploads else None
    snapshot = (demo, different_energy, upload_digests, tuple((key, str(value)) for key, value in params.items() if key != 'parameter_sources'))
    if st.button(ui('计算96时段成本'), type='primary', key='scenario_calculate'):
        try:
            if demo:
                plans = demo_plans
            else:
                if uploads is None or any(item is None for item in uploads):
                    raise ValueError('请上传计划A和计划B两个 CSV，或使用人工演示计划')
                plans = {
                    '计划A': pd.read_csv(BytesIO(uploads[0].getvalue())),
                    '计划B': pd.read_csv(BytesIO(uploads[1].getvalue())),
                }
            result = compare_schedules(plans, params)
            st.session_state['scenario_result'] = result
            st.session_state['scenario_snapshot'] = snapshot
        except (ValueError, KeyError, TypeError, pd.errors.ParserError, UnicodeError) as exc:
            st.session_state.pop('scenario_result', None)
            st.error(t('error.scenario') + ' ' + localize_error(exc, 'scenario'))
        except Exception as exc:
            logger.exception('Unexpected scenario calculation failure')
            st.session_state.pop('scenario_result', None)
            st.error(t('error.scenario') + ' ' + localize_error(exc, 'scenario'))

    result = st.session_state.get('scenario_result')
    if result is None:
        return
    if st.session_state.get('scenario_snapshot') != snapshot:
        st.warning(ui('输入已改变；以下为上次结果，请重新计算。'))
    summaries = result['summaries']
    st.subheader(ui('成本及服务量'))
    st.dataframe(localize_table(summaries, 'scenario_summary'), width='stretch', hide_index=True)
    first = summaries.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(ui('96时段含项总成本 (元)'), f"{first['included_cost_yuan']:,.2f}")
    c2.metric(ui('计划A净发电量 (MWh)'), f"{first['net_mwh']:,.2f}")
    c3.metric(ui('计划A单位成本参考 (元/净MWh)'), f"{first['unit_cost_yuan_per_net_mwh']:,.2f}")
    if result['comparable']:
        st.success(ui('两计划日期、净电量、起终点及外生输入口径一致；这里只比较当前含项成本。时段出力不同，不代表同等市场交付服务或市场最优。'))
        st.metric(ui('方案B－A成本差额 (元)'), f"{result['cost_difference_yuan']:,.2f}")
    else:
        st.warning(t('scenario.not_comparable', reason=localize_reason(result['reason'])))

    combined = pd.concat([detail.assign(plan=name) for name, detail in result['details'].items()], ignore_index=True)
    chart_data = combined.assign(plan=combined['plan'].map(localize_plan_name))
    st.plotly_chart(px.line(chart_data, x='timestamp', y='planned_gross_mw', color='plan', title=t('chart.scheduled_power'), labels={'timestamp': t('chart.timestamp'), 'planned_gross_mw': t('chart.gross_power'), 'plan': t('chart.plan')}), use_container_width=True)
    st.plotly_chart(px.line(chart_data, x='timestamp', y='unit_cost_yuan_per_net_mwh', color='plan', title=t('chart.included_unit_cost'), labels={'timestamp': t('chart.timestamp'), 'unit_cost_yuan_per_net_mwh': t('chart.unit_cost'), 'plan': t('chart.plan')}), use_container_width=True)
    st.dataframe(localize_table(combined, 'scenario_detail'), width='stretch', hide_index=True)
    st.download_button(ui('下载96时段成本明细'), _download_csv(combined), 'thermal_cost_96_detail.csv', 'text/csv')
    st.download_button(ui('下载两计划汇总'), _download_csv(summaries), 'thermal_cost_96_summary.csv', 'text/csv')
    st.download_button(ui('下载假设与排除项'), _download_csv(pd.DataFrame({'assumption': result['assumptions']})), 'thermal_cost_96_assumptions.csv', 'text/csv')
    with st.expander(ui('口径与限制')):
        for note in result['assumptions']:
            st.write(f'- {localize_assumption(note)}')
        st.write(ui('原始参考资料中的部分示例数值存在口径差异。本程序仅采用能够明确核算的煤耗、投油和排放参数，不补入来源或计算方式无法确认的成本项。'))
