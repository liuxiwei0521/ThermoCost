"""Display-only localization; model inputs and CSV contracts stay unchanged."""

import re

import streamlit as st

from app.translations import (
    TRANSLATIONS, UI_EN, EN_COLUMNS, ZH_COLUMNS, PLAN_NAMES, STABILIZATION,
    REASON_ZH, WARNING_EN, ERROR_EN, SCENARIO_ERROR_ZH, SOURCE_ZH,
)

DEFAULT_LANGUAGE = 'en'


def get_language():
    selected = st.session_state.get('language_switcher')
    if selected in ('中文', 'EN'):
        return 'zh' if selected == '中文' else 'en'
    stored = st.session_state.get('language', DEFAULT_LANGUAGE)
    return stored if stored in ('zh', 'en') else DEFAULT_LANGUAGE


def t(key, language=None, **kwargs):
    selected = language or get_language()
    return TRANSLATIONS[selected][key].format(**kwargs)


def ui(chinese, **kwargs):
    """Translate a fixed application-owned label; missing English copy is an error."""
    language = get_language()
    template = chinese if language == 'zh' else UI_EN[chinese]
    return template.format(**kwargs)


def render_language_switcher(key='language_switcher'):
    _, right = st.columns([5, 1])
    with right:
        selected = st.segmented_control('Language / 语言', ['EN', '中文'], default='中文' if get_language() == 'zh' else 'EN',
                                        key=key, label_visibility='collapsed')
    language = 'zh' if selected == '中文' else 'en'
    st.session_state['language'] = language
    return language


def stable_selectbox(container, label, options, **kwargs):
    """Keep an internal selected value when translated option labels change."""
    key = kwargs['key']
    options = list(options)
    saved_key = '_selected_' + key
    default_index = kwargs.pop('index', 0)
    if saved_key not in st.session_state:
        st.session_state[saved_key] = st.session_state.get(key, options[default_index])
    if st.session_state[saved_key] not in options:
        st.session_state[saved_key] = options[default_index]

    def remember():
        st.session_state[saved_key] = st.session_state[key]

    return container.selectbox(label, options, index=options.index(st.session_state[saved_key]),
                               on_change=remember, **kwargs)


def localize_plan_name(value, language=None):
    selected = language or get_language()
    return PLAN_NAMES.get(str(value), {}).get(selected, str(value))


def localize_table(frame, kind, language=None):
    """Return a display copy; raw results and CSV exports retain their fields."""
    selected = language or get_language()
    mappings = EN_COLUMNS if selected == 'en' else ZH_COLUMNS
    if kind not in mappings:
        raise ValueError(f'Unknown display table kind: {kind}')
    display = frame.copy(deep=True)
    if 'plan' in display:
        display['plan'] = display['plan'].map(lambda value: localize_plan_name(value, selected))
    if '稳燃状态' in display:
        display['稳燃状态'] = display['稳燃状态'].map(lambda value: STABILIZATION.get(value, {}).get(selected, value))
    if '煤耗口径' in display:
        display['煤耗口径'] = display['煤耗口径'].map(lambda value: t('coal.actual' if value == 'actual' else 'coal.standard', selected) if value in ('actual', 'standard') else value)
    if '热损失项' in display and selected == 'en':
        names = {'q2_排烟热损失': 'q2 exhaust-gas loss', 'q3_气体未燃烧损失': 'q3 unburned-gas loss',
                 'q4_固体未燃烧损失': 'q4 unburned-solid loss', 'q5_散热损失': 'q5 heat-dissipation loss',
                 'q6_灰渣热损失': 'q6 ash loss'}
        display['热损失项'] = display['热损失项'].map(lambda value: names.get(value, value))
    if kind == 'evaluation_summary':
        if 'method' in display:
            display['method'] = display['method'].map(lambda value: t('method.' + value, selected) if 'method.' + str(value) in TRANSLATIONS[selected] else value)
        if '状态' in display:
            display['状态'] = display['状态'].map(lambda value: localize_status(value, selected))
        if '参数假设' in display:
            display['参数假设'] = display['参数假设'].map(lambda value: localize_parameter_assumptions(value, selected))
    if kind == 'evaluation_detail':
        display = display.rename(columns={column: t('method.' + column.removesuffix('_Pred'), selected) + (' prediction' if selected == 'en' else '预测') for column in display if column.endswith('_Pred') and 'method.' + column.removesuffix('_Pred') in TRANSLATIONS[selected]})
    return display.rename(columns=mappings[kind])


def localize_status(value, language=None):
    selected = language or get_language()
    if selected == 'zh':
        return value
    if value == '诊断结果；不代表真实电厂验证':
        return 'Diagnostic result; not validated on a real plant.'
    if str(value).startswith('不可评估: '):
        return 'Cannot evaluate: ' + localize_error(str(value).split(': ', 1)[1], 'evaluation', selected)
    return t('error.unexpected', selected)


def localize_parameter_assumptions(value, language=None):
    selected = language or get_language()
    if selected == 'zh':
        return value
    if value == '使用提供的参数':
        return 'Using supplied parameters.'
    if value == '无缺省情景参数':
        return 'No default scenario parameters.'
    prefix = '默认情景：' if str(value).startswith('默认情景：') else ''
    body = str(value)[len(prefix):]
    return ('Default scenario: ' if prefix else '') + body.replace('；', '; ')


def localize_reason(reason, language=None):
    selected = language or get_language()
    if selected == 'en':
        return reason
    if reason in REASON_ZH:
        return REASON_ZH[reason]
    return '；'.join(REASON_ZH.get(part.strip(), '比较条件未满足') for part in reason.split(';'))


def localize_assumption(note, language=None):
    selected = language or get_language()
    if note.startswith('source_type: '):
        source = note.split(': ', 1)[1].split(';', 1)[0]
        if selected == 'en':
            label = {'synthetic': 'synthetic', 'user_scenario': 'user scenario'}.get(source, 'unverified')
            return f'Data source: {label}; example values and outputs are not measured-plant validation.'
        label = {'synthetic': '模拟', 'user_scenario': '用户情景'}.get(source, '未核实')
        return f'数据来源：{label}；示例值和输出均不构成真实机组实测验证。'
    if note.startswith('coal_g_kwh: '):
        return ('Coal consumption: piecewise-linear interpolation of built-in synthetic operating points; '
                'calculated on an actual-coal basis.' if selected == 'en' else
                '煤耗：内置模拟煤耗曲线采用分段线性插值；按实际煤口径计算。')
    if note.startswith('aux_rate_pct: '):
        if 'uploaded per-period' in note:
            return ('Auxiliary rate: per-period values from the uploaded file.' if selected == 'en' else
                    '厂用电率：按上传文件的逐时段数值。')
        value = re.search(r'\(([^)]+)\)', note)
        amount = value.group(1) if value else ('unspecified' if selected == 'en' else '未指定')
        return (f'Auxiliary rate: fixed scenario value ({amount}%).' if selected == 'en' else
                f'厂用电率：固定情景值（{amount}%）。')
    if note.startswith('carbon_intensity_t_per_mwh: '):
        if 'uploaded per-period' in note:
            return ('Emission intensity: per-period values from the uploaded file.' if selected == 'en' else
                    '排放强度：按上传文件的逐时段数值。')
        if 'table-5' in note or 'built-in synthetic operating points' in note:
            return ('Emission intensity: piecewise-linear interpolation of the built-in synthetic '
                    'emission-intensity curve.' if selected == 'en' else
                    '排放强度：内置模拟排放强度曲线采用分段线性插值。')
        value = re.search(r'\(([^)]+)\)', note)
        amount = value.group(1) if value else ('unspecified' if selected == 'en' else '未指定')
        return (f'Emission intensity: fixed scenario value ({amount} tCO2/gross MWh).' if selected == 'en' else
                f'排放强度：固定情景值（{amount} 吨CO₂/毛MWh）。')
    if note.startswith('included costs: '):
        return ('Included costs: actual-coal fuel, explicitly entered oil and carbon costs; the auxiliary '
                'rate only adjusts delivered net energy.' if selected == 'en' else
                '纳入成本：实际煤燃料、明确输入的投油和碳成本；厂用电率只调整交付净电量。')
    if note.startswith('not included: '):
        return ('Not included: fixed O&M, startup, fatigue, capacity, risk, market revenue or bidding.'
                if selected == 'en' else '未纳入：固定运维、启停、疲劳、容量、风险、市场收益或报价。')
    name, _, source = note.partition(': ')
    labels_zh = {
        'rated_power': '额定功率', 'initial_power': '初始出力', 'load_bounds': '负荷边界',
        'ramp_limit': '爬坡率', 'lhv': '煤低位热值', 'coal_curve': '煤耗曲线',
        'carbon_curve': '排放曲线', 'coal_price': '煤价', 'oil_price': '油价',
        'carbon_price': '碳价', 'aux_rate': '厂用电率', 'schedule': '运行计划',
    }
    labels_en = {
        'rated_power': 'Rated power', 'initial_power': 'Initial power', 'load_bounds': 'Load bounds',
        'ramp_limit': 'Ramp limit', 'lhv': 'Coal LHV', 'coal_curve': 'Coal-consumption curve',
        'carbon_curve': 'Emission-intensity curve', 'coal_price': 'Coal price', 'oil_price': 'Oil price',
        'carbon_price': 'Carbon price', 'aux_rate': 'Auxiliary rate', 'schedule': 'Schedule',
    }
    labels = labels_en if selected == 'en' else labels_zh
    if name in labels:
        if selected == 'zh' and source in SOURCE_ZH:
            detail = SOURCE_ZH[source]
        elif source.startswith('user_scenario') or source.startswith('user-uploaded'):
            detail = ('user scenario input; not independently verified' if selected == 'en' else
                      '用户情景输入，未经独立核实')
        elif source.startswith('built-in'):
            detail = (source if selected == 'en' else '内置模拟或演示参数，非实测')
        elif source.startswith('document_example'):
            detail = ('built-in demonstration value; not measured' if selected == 'en' else
                      '内置演示参数，非实测')
        elif source.startswith('synthetic'):
            detail = ('synthetic value' if selected == 'en' else '人工生成的模拟值')
        elif source.startswith('scenario assumption'):
            detail = ('scenario assumption; not calibrated to a plant' if selected == 'en' else
                      '未经机组标定的情景假设')
        else:
            detail = ('source not verified' if selected == 'en' else '来源尚未核实')
        return f'{labels[name]}: {detail}.' if selected == 'en' else f'{labels[name]}：{detail}。'
    return 'Other scenario note: source not verified.' if selected == 'en' else '其他情景说明：来源尚未核实。'


def localize_warning(note, language=None):
    selected = language or get_language()
    if selected == 'zh':
        return note
    return WARNING_EN.get(note, t('error.unexpected', selected))


def localize_error(error, context, language=None):
    """Translate known validation failures; never leak unknown raw text."""
    selected = language or get_language()
    message = str(error)
    if message == '请上传计划A和计划B两个 CSV，或使用人工演示计划':
        return ('Upload both Plan A and Plan B CSV files, or use the synthetic demo plans.'
                if selected == 'en' else message)
    plan_prefix = ''
    if ': ' in message:
        prefix, rest = message.split(': ', 1)
        if prefix in PLAN_NAMES:
            plan_prefix, message = localize_plan_name(prefix, selected) + ': ', rest
    if message in ERROR_EN:
        return plan_prefix + (ERROR_EN[message] if selected == 'en' else message)
    if message in SCENARIO_ERROR_ZH:
        return plan_prefix + (message if selected == 'en' else SCENARIO_ERROR_ZH[message])
    match = re.fullmatch(r'(.+?) (必须是数值|超出有效范围或不是有限值)', message)
    if match:
        field, problem = match.groups()
        if selected == 'zh':
            return plan_prefix + message
        return plan_prefix + (f'{field} must be numeric.' if problem == '必须是数值' else f'{field} is outside the allowed range or not finite.')
    match = re.fullmatch(r'(缺少模型特征|缺少评估测点): (.+)', message)
    if match:
        return plan_prefix + ((f'Missing {"model features" if match.group(1) == "缺少模型特征" else "evaluation measurements"}: {match.group(2)}.') if selected == 'en' else message)
    match = re.fullmatch(r'缺少 (actual_coal_flow_t_h)', message)
    if match:
        return plan_prefix + (f'Missing {match.group(1)}.' if selected == 'en' else message)
    match = re.fullmatch(r'expected 96 timestamp rows, received (\d+)', message)
    if match:
        return plan_prefix + (message if selected == 'en' else f'计划需要 96 行时间戳，实际收到 {match.group(1)} 行。')
    match = re.fullmatch(r'(?:row|timestamp row) (\d+): (.+)', message)
    if match:
        row, detail = match.groups()
        if detail == 'oil_kg_h must be positive below 40% load':
            return plan_prefix + t('error.low_oil', selected, row=row)
        if detail.startswith('invalid '):
            field = detail.removeprefix('invalid ')
            return plan_prefix + (f'Row {row}: invalid {field}.' if selected == 'en' else f'第 {row} 行：{field} 无效。')
        if detail == 'load outside min_load_pct/max_load_pct':
            return plan_prefix + (f'Row {row}: load outside min_load_pct/max_load_pct.' if selected == 'en' else f'第 {row} 行：负荷超出最小/最大负荷率。')
        if detail == 'expected consecutive same-day 15-minute points':
            return plan_prefix + (f'Timestamp row {row}: expected consecutive same-day 15-minute points.' if selected == 'en' else f'第 {row} 行：须为同一天连续 15 分钟时间点。')
        if 'exceeds' in detail:
            return plan_prefix + (f'Row {row}: {detail}.' if selected == 'en' else f'第 {row} 行：爬坡超出 {detail.split("exceeds ", 1)[1]}。')
    match = re.fullmatch(r'(missing required column|missing)\: (.+)', message)
    if match:
        return plan_prefix + (message + '.' if selected == 'en' else f'缺少必填列：{match.group(2)}。')
    match = re.fullmatch(r'missing (\w+): supply a column or explicit scenario value', message)
    if match:
        field = match.group(1)
        return plan_prefix + (f'Missing {field}: supply a column or explicit scenario value.' if selected == 'en' else f'缺少 {field}：请提供逐时段列或明确的情景参数。')
    match = re.fullmatch(r'(.+?) (must be a finite number|is outside the allowed range or not finite)', message)
    if match:
        return plan_prefix + (message + '.' if selected == 'en' else f'{match.group(1)} 须为有效数值。')
    return t('error.unexpected', selected)
