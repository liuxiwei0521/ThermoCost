"""Illustrative 96-period thermal-unit cost calculations, not market bids.

The built-in coal and carbon curves are synthetic operating points retained
for a reproducible demonstration. They are not plant calibration.
"""

import math

import numpy as np
import pandas as pd


PERIOD_HOURS = 0.25
COAL_LOADS = np.array([30, 40, 55, 70, 85, 100], dtype=float)
COAL_G_KWH = {
    18: [488.2, 435.6, 392.4, 362.8, 345.2, 338.5],
    21: [418.5, 373.4, 336.3, 310.9, 295.8, 290.1],
    24: [366.2, 326.7, 294.3, 272.1, 258.8, 253.8],
}
CARBON_LOADS = np.array([30, 50, 75, 100], dtype=float)
CARBON_T_MWH = np.array([1.050, 0.915, 0.842, 0.795], dtype=float)


def default_scenario_params():
    """Return fully labelled demo inputs, never measured-plant defaults."""
    return {
        'source_type': 'synthetic',
        'rated_mw': 600.0,
        'initial_mw': 420.0,
        'min_load_pct': 30.0,
        'max_load_pct': 100.0,
        'ramp_limit_pct_per_min': 1.5,
        'lhv_mj_kg': 21,
        'coal_price_basis': 'actual',
        'coal_price_yuan_per_ton': 800.0,
        'oil_price_yuan_per_kg': 6.18,
        'carbon_price_yuan_per_ton': 62.36,
        'aux_rate_pct': 6.0,
        'carbon_intensity_t_per_mwh': 'built_in_curve',
        'parameter_sources': {
            'rated_power': 'built-in synthetic 600 MW unit',
            'initial_power': 'synthetic 70% initial setpoint',
            'load_bounds': 'built-in synthetic 30%-100% range',
            'ramp_limit': 'scenario assumption (1.5% rated MW/min; not calibrated)',
            'lhv': 'built-in synthetic 21 MJ/kg grade',
            'coal_curve': 'built-in synthetic operating points, piecewise-linear interpolation',
            'carbon_curve': 'built-in synthetic operating points, piecewise-linear interpolation',
            'coal_price': 'built-in demonstration value, not a purchase invoice',
            'oil_price': 'built-in demonstration value, not a purchase invoice',
            'carbon_price': 'built-in demonstration value, not an actual allowance cost',
            'aux_rate': 'synthetic constant 6%, not a plant curve',
            'schedule': 'synthetic deterministic examples',
        },
    }


def generate_demo_plans():
    """Two same-energy plans; comparison is illustrative, not optimization."""
    times = pd.date_range('2026-01-01', periods=96, freq='15min')
    base = np.full(96, 420.0)
    shifting = base.copy()
    shifting[1:95] = np.tile([360.0, 480.0], 47)
    return {
        '平稳计划': pd.DataFrame({'timestamp': times, 'planned_gross_mw': base, 'oil_kg_h': 0.0}),
        '负荷调整计划': pd.DataFrame({'timestamp': times, 'planned_gross_mw': shifting, 'oil_kg_h': 0.0}),
    }


def _number(value, field, minimum=None, maximum=None, upper_open=False):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field} must be a finite number') from None
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (
        maximum is not None and (result >= maximum if upper_open else result > maximum)
    ):
        raise ValueError(f'{field} is outside the allowed range or not finite')
    return result


def _column(frame, field, minimum=None, maximum=None, upper_open=False):
    if field not in frame:
        raise ValueError(f'missing required column: {field}')
    values = pd.to_numeric(frame[field], errors='coerce')
    invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
    if minimum is not None:
        invalid |= values < minimum
    if maximum is not None:
        invalid |= values >= maximum if upper_open else values > maximum
    if invalid.any():
        row = int(np.flatnonzero(invalid.to_numpy())[0]) + 1
        raise ValueError(f'row {row}: invalid {field}')
    return values.to_numpy(dtype=float)


def _optional_series(frame, params, field, minimum=None, maximum=None, upper_open=False):
    if field in frame:
        return _column(frame, field, minimum, maximum, upper_open), f'{field}: uploaded per-period values'
    if field not in params:
        raise ValueError(f'missing {field}: supply a column or explicit scenario value')
    raw = params[field]
    if field == 'carbon_intensity_t_per_mwh' and raw == 'built_in_curve':
        return None, 'carbon_intensity_t_per_mwh: built-in synthetic operating points, piecewise-linear interpolation'
    value = _number(raw, field, minimum, maximum, upper_open)
    return np.full(len(frame), value), f'{field}: fixed scenario value ({value:g})'


def calculate_schedule(frame, params):
    """Validate and cost one complete day of gross-MW setpoints.

    Returns full-precision rows/totals. Unit costs divide included costs by
    delivered net MWh. No auxiliary CNY charge or market revenue is invented.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('schedule must be a DataFrame')
    if len(frame) != 96:
        raise ValueError(f'expected 96 timestamp rows, received {len(frame)}')
    if 'timestamp' not in frame:
        raise ValueError('missing required column: timestamp')
    try:
        times = pd.to_datetime(frame['timestamp'], errors='raise')
    except (ValueError, TypeError):
        raise ValueError('invalid timestamp value') from None
    if times.isna().any() or getattr(times.dt, 'tz', None) is not None:
        raise ValueError('timestamp must be a timezone-naive 15-minute local schedule')
    if times.iloc[0].time() != pd.Timestamp('00:00').time():
        raise ValueError('timestamp row 1 must be 00:00 for a full-day schedule')
    expected = pd.date_range(times.iloc[0], periods=96, freq='15min')
    mismatch = np.flatnonzero((times.to_numpy() != expected.to_numpy()))
    if len(mismatch):
        raise ValueError(f'timestamp row {int(mismatch[0]) + 1}: expected consecutive same-day 15-minute points')

    rated = _number(params.get('rated_mw'), 'rated_mw', 0.001)
    initial = _number(params.get('initial_mw'), 'initial_mw', 0)
    min_pct = _number(params.get('min_load_pct'), 'min_load_pct', 0, 100)
    max_pct = _number(params.get('max_load_pct'), 'max_load_pct', 0, 100)
    if min_pct >= max_pct or min_pct < 30 or max_pct > 100:
        raise ValueError('load bounds must satisfy 30 <= min_load_pct < max_load_pct <= 100')
    if not rated * min_pct / 100 <= initial <= rated * max_pct / 100:
        raise ValueError('initial_mw is outside the specified load bounds')
    ramp = _number(params.get('ramp_limit_pct_per_min'), 'ramp_limit_pct_per_min', 0)
    lhv = params.get('lhv_mj_kg')
    if lhv not in COAL_G_KWH:
        raise ValueError('lhv_mj_kg must be 18, 21, or 24 (supported built-in synthetic grades only)')
    if params.get('coal_price_basis') != 'actual':
        raise ValueError('coal_price_basis must be actual to match the actual-coal consumption curve')
    coal_price = _number(params.get('coal_price_yuan_per_ton'), 'coal_price_yuan_per_ton', 0)
    oil_price = _number(params.get('oil_price_yuan_per_kg'), 'oil_price_yuan_per_kg', 0)
    carbon_price = _number(params.get('carbon_price_yuan_per_ton'), 'carbon_price_yuan_per_ton', 0)

    mw = _column(frame, 'planned_gross_mw', 0)
    load = mw / rated * 100
    bad_load = np.flatnonzero((load < min_pct - 1e-10) | (load > max_pct + 1e-10))
    if len(bad_load):
        raise ValueError(f'row {int(bad_load[0]) + 1}: load outside min_load_pct/max_load_pct')
    change = np.diff(np.r_[initial, mw])
    max_change = rated * ramp / 100 * 15
    bad_ramp = np.flatnonzero(np.abs(change) > max_change + 1e-9)
    if len(bad_ramp):
        row = int(bad_ramp[0]) + 1
        label = 'initial ramp' if row == 1 else 'ramp'
        raise ValueError(f'row {row}: {label} exceeds {ramp:g}% rated MW/min')
    oil = _column(frame, 'oil_kg_h', 0)
    unsafe_oil = np.flatnonzero((load < 40) & (oil <= 0))
    if len(unsafe_oil):
        raise ValueError(f'row {int(unsafe_oil[0]) + 1}: oil_kg_h must be positive below 40% load')
    aux, aux_note = _optional_series(frame, params, 'aux_rate_pct', 0, 100, upper_open=True)
    carbon, carbon_note = _optional_series(frame, params, 'carbon_intensity_t_per_mwh', 0, 3)
    if carbon is None:
        carbon = np.interp(load, CARBON_LOADS, CARBON_T_MWH)

    coal = np.interp(load, COAL_LOADS, COAL_G_KWH[lhv])
    gross = mw * PERIOD_HOURS
    net = gross * (1 - aux / 100)
    coal_tons = coal * gross / 1000
    fuel_cost = coal_tons * coal_price
    oil_cost = oil * oil_price * PERIOD_HOURS
    carbon_cost = carbon * gross * carbon_price
    included_cost = fuel_cost + oil_cost + carbon_cost
    detail = pd.DataFrame({
        'timestamp': times.to_numpy(),
        'planned_gross_mw': mw,
        'load_pct': load,
        'ramp_mw_per_min': change / 15,
        'gross_mwh': gross,
        'aux_rate_pct': aux,
        'net_mwh': net,
        'coal_g_kwh': coal,
        'coal_tons': coal_tons,
        'oil_kg_h': oil,
        'carbon_intensity_t_per_mwh': carbon,
        'fuel_cost_yuan': fuel_cost,
        'oil_cost_yuan': oil_cost,
        'carbon_cost_yuan': carbon_cost,
        'included_cost_yuan': included_cost,
        'unit_cost_yuan_per_net_mwh': included_cost / net,
    })
    gross_total = float(gross.sum())
    net_total = float(net.sum())
    cost_total = float(included_cost.sum())
    summary = {
        'gross_mwh': gross_total,
        'net_mwh': net_total,
        'fuel_cost_yuan': float(fuel_cost.sum()),
        'oil_cost_yuan': float(oil_cost.sum()),
        'carbon_cost_yuan': float(carbon_cost.sum()),
        'included_cost_yuan': cost_total,
        'unit_cost_yuan_per_net_mwh': cost_total / net_total,
        'hours_below_40_pct': float(((load < 40).sum()) * PERIOD_HOURS),
        'max_ramp_mw_per_min': float(np.abs(change / 15).max()),
        'initial_mw': initial,
        'final_mw': float(mw[-1]),
    }
    assumptions = [
        f"source_type: {params.get('source_type', 'user_scenario')}; neither example values nor output are measured-plant validation",
        'coal_g_kwh: built-in synthetic operating points, piecewise-linear interpolation; actual-coal basis',
        aux_note,
        carbon_note,
        'included costs: actual-coal fuel, explicit oil and carbon only; auxiliary rate adjusts delivered net MWh',
        'not included: fixed O&M, startup, fatigue, capacity, risk, market revenue or bidding',
    ]
    assumptions.extend(f'{name}: {source}' for name, source in params.get('parameter_sources', {}).items())
    return {'detail_df': detail, 'summary': summary, 'assumptions': assumptions}


def compare_schedules(plans, params):
    """Compare two schedules only when delivered service and endpoints match."""
    if not isinstance(plans, dict) or len(plans) != 2:
        raise ValueError('comparison requires exactly two named plans')
    names = list(plans)
    if any(not isinstance(name, str) or not name.strip() for name in names) or names[0].strip() == names[1].strip():
        raise ValueError('two distinct nonempty plan names are required')
    evaluated = {}
    for name, frame in plans.items():
        try:
            evaluated[name] = calculate_schedule(frame, params)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'{name}: {exc}') from exc
    summaries = pd.DataFrame([{'plan': name, **evaluated[name]['summary']} for name in names])
    first, second = (evaluated[name]['summary'] for name in names)
    reasons = []
    detail_a, detail_b = (evaluated[name]['detail_df'] for name in names)
    if not detail_a['timestamp'].equals(detail_b['timestamp']):
        reasons.append('timestamp axes differ')
    for field in ('aux_rate_pct', 'carbon_intensity_t_per_mwh'):
        supplied_a, supplied_b = (field in plans[name] for name in names)
        if supplied_a != supplied_b or (supplied_a and not np.allclose(detail_a[field], detail_b[field], rtol=0, atol=1e-10)):
            reasons.append(f'{field} input assumptions differ')
    if not math.isclose(first['net_mwh'], second['net_mwh'], rel_tol=1e-9, abs_tol=1e-6):
        reasons.append('delivered net MWh differs')
    for field in ('initial_mw', 'final_mw'):
        if not math.isclose(first[field], second[field], rel_tol=0, abs_tol=1e-6):
            reasons.append(f'{field} endpoint differs')
    comparable = not reasons
    return {
        'summaries': summaries,
        'details': {name: result['detail_df'] for name, result in evaluated.items()},
        'assumptions': evaluated[names[0]]['assumptions'],
        'comparable': comparable,
        'reason': 'same date, daily net MWh, endpoints and input convention; temporal service may still differ' if comparable else '; '.join(reasons),
        'cost_difference_yuan': second['included_cost_yuan'] - first['included_cost_yuan'] if comparable else None,
    }
