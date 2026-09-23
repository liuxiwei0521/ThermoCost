"""Existing prototype assumptions, shared by the single UI and evaluation code.

Tables and heat-loss coefficients are illustrative, not calibrated plant data.
No claim is made that unit cost plus a risk allowance is a strict marginal cost.
"""
from copy import deepcopy
import math

import pandas as pd

STANDARD_COAL_LHV = 29307.6  # kJ/kg
LOADS = (100, 85, 70, 55, 40, 35, 30)
TABLE_COAL = {
    18: {100: 338.5, 85: 352, 70: 372, 55: 398, 40: 432, 30: 465},
    21: {100: 290.1, 85: 302, 70: 318, 55: 342, 40: 372, 30: 398},
    24: {100: 253.8, 85: 264, 70: 278, 55: 298, 40: 325, 30: 343},
}
TABLE_CARBON = {
    2025: {100: 51, 75: 53.5, 50: 57.2, 30: 61},
    2030: {100: 65, 75: 68, 50: 73, 30: 78},
}
TABLE_RISK = {0.5: 2, 1.5: 5, 3.0: 12, 5.0: 25}
CORRECTIONS = {'不修正': 1.0, '劣质煤(1.15~1.18)': 1.165, '优质煤(0.88~0.90)': 0.89}
METHODS = {
    'dynamic_cb': '动态反平衡法（示例机理）',
    'table': '查表法（基础版）',
    'counter_balance': '静态反平衡法',
    'direct_balance': '正平衡法',
    'lgbm': 'LightGBM',
}
STEAM_DEFAULTS = {
    'g_ms': 1025.0, 'h_ms': 3400.0, 'g_rh': 980.0, 'h_rh': 3500.0,
    'g_pw': 1000.0, 'h_pw': 1000.0, 'g_oh': 0.0, 'h_oh': 2800.0,
    'g_sh': 0.0, 'h_sh': 800.0, 'g_rh_des': 0.0, 'h_rh_des': 800.0,
    'exhaust_temp': 128.5, 'o2_content': 3.2, 'ambient_temp': 20.0,
}
LEGACY_DEFAULTS = {
    'ep': 600.0, 'lpq': 410.0, 'mpq': 0.0, 'rhp': 3.5, 'bp': 12.0,
    'wt': 280.0, 'at': 538.0, 'rht': 538.0, 'hour': 12, 'weekday': 2, 'month': 6,
}


def default_params():
    return {
        'calc_strategy': 'dynamic_cb', 'load_pct': 100.0, 'load_ramp': 1.5,
        'rated_power_mw': 600.0, 'lhv': 21.0, 'coal_correction': '不修正',
        'coal_basis': 'standard', 'coal_price_basis': 'standard',
        'coal_price_yuan_per_ton': 800.0, 'carbon_year': 2025,
        'boiler_eff': 0.92, 'pipe_eff': 0.98,
        'power_mwh': 0.0, 'operating_hours': 0.0,
        'include_legacy_fatigue': False, 'deep_peaking_cycles': 0,
        'rotor_replacement_yuan': 1.5e8, 'deep_cycle_damage_ratio': 1 / 30,
        'counter_balance_input': deepcopy(STEAM_DEFAULTS),
        'direct_balance_input': {'actual_coal_flow_t_h': 174.0, 'base_load_pct': 100.0},
        'lgbm_input': dict(STEAM_DEFAULTS, **LEGACY_DEFAULTS, actual_coal_flow_t_h=174.0),
    }


def number(value, name, lower=None, upper=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} 必须是数值') from None
    if not math.isfinite(value) or (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f'{name} 超出有效范围或不是有限值')
    return value


def dynamic_efficiency(load_pct, exhaust_temp, o2_content, ambient_temp):
    """Retain legacy coefficients and 70% floor, with invalid-input rejection."""
    load = number(load_pct, 'load_pct', 20, 100)
    oxygen = number(o2_content, 'o2_content', 1, 10)
    exhaust = number(exhaust_temp, 'exhaust_temp')
    ambient = number(ambient_temp, 'ambient_temp')
    if exhaust < ambient:
        raise ValueError('排烟温度不能低于环境温度；请检查测点与单位')
    losses = {
        'q2_排烟热损失': 0.035 * (21 / (21 - oxygen)) * (exhaust - ambient),
        'q3_气体未燃烧损失': 0.1,
        'q4_固体未燃烧损失': 1 + 2.5 * ((100 - load) / 100) ** 2,
        'q5_散热损失': 0.5 * (100 / load),
        'q6_灰渣热损失': 0.2,
    }
    efficiency = max(70.0, 100 - sum(losses.values())) / 100
    return efficiency, {k: round(v, 2) for k, v in losses.items()}


def coal_at_load(params, load_pct, predictor=None):
    p = deepcopy(params)
    load = number(load_pct, 'load_pct', 30, 100)
    rating = number(p['rated_power_mw'], 'rated_power_mw', 0.001)
    lhv = number(p['lhv'], 'lhv', 0.001)
    method = p['calc_strategy']
    if method not in METHODS:
        raise ValueError('未知煤耗计算方法')
    if p['coal_correction'] not in CORRECTIONS:
        raise ValueError('未知煤质修正选项')
    corr = CORRECTIONS[p['coal_correction']]
    efficiency, losses = None, {}
    basis = 'actual' if method == 'table' else 'standard'
    if method == 'table':
        if lhv not in TABLE_COAL:
            raise ValueError('查表法只支持 LHV 18、21、24 MJ/kg')
        key = min(TABLE_COAL[lhv], key=lambda x: abs(x - load))
        if 35 <= load < 40:
            key = 30
        coal = TABLE_COAL[lhv][key] * corr
    elif method in ('dynamic_cb', 'counter_balance'):
        inputs = p['counter_balance_input']
        values = {k: number(inputs[k], k, 0) for k in STEAM_DEFAULTS if k.startswith(('g_', 'h_'))}
        # Flow is t/h; enthalpy kJ/kg. Numerically HR is MJ/h and HR/LHV is t/h.
        hr = values['g_ms'] * values['h_ms'] + values['g_rh'] * values['h_rh'] - sum(
            values[g] * values[h] for g, h in [('g_pw', 'h_pw'), ('g_oh', 'h_oh'), ('g_sh', 'h_sh'), ('g_rh_des', 'h_rh_des')]
        )
        if hr <= 0:
            raise ValueError('热耗为零或负值；请核对流量、焓值和热平衡边界')
        if method == 'dynamic_cb':
            efficiency, losses = dynamic_efficiency(load, inputs['exhaust_temp'], inputs['o2_content'], inputs['ambient_temp'])
        else:
            efficiency = number(p['boiler_eff'], 'boiler_eff', 0.01, 1)
        pipe_eff = number(p['pipe_eff'], 'pipe_eff', 0.01, 1)
        coal = round(hr / (STANDARD_COAL_LHV * efficiency * pipe_eff) * 1000 / (rating * load / 100), 2) * corr
    elif method == 'direct_balance':
        flow = number(p['direct_balance_input']['actual_coal_flow_t_h'], 'actual_coal_flow_t_h', 0)
        base = number(p['direct_balance_input']['base_load_pct'], 'base_load_pct', 30, 100)
        if corr != 1:
            raise ValueError('正平衡法使用实测煤流量，不叠加未经标定的煤质修正')
        # Retain the original 0.95 extrapolation exponent; not a calibrated curve.
        estimated_flow = flow * (load / base) ** 0.95
        coal = estimated_flow * (lhv * 1000) / STANDARD_COAL_LHV * 1000 / (rating * load / 100)
    else:
        if predictor is None:
            raise ValueError('请选择模型并提供有效的 LightGBM 输入')
        basis = p.get('coal_basis')
        row = dict(p['lgbm_input'], load_pct=load, lhv=lhv)
        coal = float(predictor(row)) * corr
    coal = number(coal, '煤耗预测', 0)
    if basis not in ('actual', 'standard'):
        raise ValueError('必须明确模型输出的煤耗口径')
    if p['coal_price_basis'] != basis:
        raise ValueError('煤耗与煤价口径不一致；必须同为实际煤或标准煤，程序不自动折算价格')
    return {'coal': round(coal, 2), 'basis': basis, 'efficiency': efficiency, 'losses': losses}


def cost_row(params, load_pct, predictor=None):
    coal = coal_at_load(params, load_pct, predictor)
    price = number(params['coal_price_yuan_per_ton'], 'coal_price', 0)
    year = params['carbon_year']
    ramp = number(params['load_ramp'], 'load_ramp', 0)
    if year not in TABLE_CARBON or ramp not in TABLE_RISK:
        raise ValueError('年份或爬坡档位不在示例表中')
    key = 100 if load_pct >= 95 else (75 if load_pct >= 62 else (50 if load_pct >= 35 else 30))
    oil_cost, state, rate = (0, '无需投油', 0) if load_pct >= 40 else ((25.01, '脉冲投油', 850) if load_pct >= 35 else (75.53, '持续投油', 2200))
    fuel = round(coal['coal'] * price / 1000, 2)
    loss = round(0.8 + (100 - load_pct) * 0.12, 2)
    carbon, risk = TABLE_CARBON[year][key], TABLE_RISK[ramp]
    return {
        '负荷率 (%)': load_pct, '实时煤耗 (g/kWh)': coal['coal'],
        '锅炉效率 (%)': None if coal['efficiency'] is None else round(coal['efficiency'] * 100, 2),
        '燃料成本 (元/MWh)': fuel, '机组损耗 (元/MWh)': loss,
        '投油成本 (元/MWh)': oil_cost, '碳成本 (元/MWh)': carbon,
        '风险加成 (元/MWh)': risk, '单位成本参考 (元/MWh)': round(fuel + loss + oil_cost + carbon + risk, 2),
        '煤耗口径': coal['basis'], '稳燃状态': state, '耗油率 (kg/h)': rate,
        'losses': coal['losses'],
    }


def run_cost_calculation(params, predictor=None):
    p = deepcopy(params)
    load = number(p['load_pct'], 'load_pct', 30, 100)
    power = number(p.get('power_mwh', 0), 'power_mwh', 0)
    hours = number(p.get('operating_hours', 0), 'operating_hours', 0)
    current = cost_row(p, load, predictor)
    detail = [cost_row(p, lp, predictor) for lp in LOADS]
    energy = power if power > 0 else number(p['rated_power_mw'], 'rated_power_mw', 0.001) * load / 100 * hours
    warnings = ['油耗、碳成本、损耗和风险加成采用旧版示例表或经验式，未按真实机组标定。']
    if p['calc_strategy'] in ('dynamic_cb', 'counter_balance', 'lgbm'):
        warnings.append('多负荷曲线固定其他输入，仅改变负荷及相关效率；不是机组实测负荷—成本曲线。')
    if p['calc_strategy'] == 'direct_balance':
        warnings.append('正平衡负荷扫描采用旧版0.95经验指数外推，尚未标定。')
    cycles = number(p.get('deep_peaking_cycles', 0), 'deep_peaking_cycles', 0)
    if not cycles.is_integer():
        raise ValueError('深度调峰次数必须为整数')
    fatigue = 0.0
    if p.get('include_legacy_fatigue', False):
        fatigue = cycles * number(p['rotor_replacement_yuan'], 'rotor_replacement_yuan', 0) * number(p['deep_cycle_damage_ratio'], 'deep_cycle_damage_ratio', 0, 1)
        warnings.append('疲劳成本仅保留旧公式示例；相对冷启动损伤不等于转子寿命比例，不能用于真实寿命成本。')
    if power > 0 and hours > 0:
        warnings.append('同时填写电量与时长时，以填写的电量计算总成本。')
    return {
        '实时煤耗_g_kWh': current['实时煤耗 (g/kWh)'],
        '煤耗口径': current['煤耗口径'], '煤耗计算策略': p['calc_strategy'],
        '燃料成本_元_MWh': current['燃料成本 (元/MWh)'],
        '机组损耗_元_MWh': current['机组损耗 (元/MWh)'],
        '投油成本_元_MWh': current['投油成本 (元/MWh)'],
        '碳成本_元_MWh': current['碳成本 (元/MWh)'],
        '运行风险溢价_元_MWh': current['风险加成 (元/MWh)'],
        '单位成本参考_元_MWh': current['单位成本参考 (元/MWh)'],
        '稳燃状态': current['稳燃状态'], '耗油率_kg_h': current['耗油率 (kg/h)'],
        '疲劳示例成本_元': round(fatigue, 2), '计算发电量_MWh': energy,
        '总成本_元': round(current['单位成本参考 (元/MWh)'] * energy + fatigue, 2) if energy > 0 else None,
        'losses_breakdown': current['losses'], 'dynamic_efficiency': current['锅炉效率 (%)'],
        'detail_df': pd.DataFrame([{k: v for k, v in row.items() if k != 'losses'} for row in detail]),
        'warnings': warnings,
    }
