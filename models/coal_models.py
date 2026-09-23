"""Feature contracts, legacy inference and honest holdout diagnostics."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models.cost_model import STANDARD_COAL_LHV, default_params, coal_at_load, dynamic_efficiency

MODEL_DIR = Path(__file__).resolve().parent
LEGACY_FILES = {'legacy1': 'coal_consumption_lgbm.pkl', 'legacy2': 'coal_consumption_lgbm_v2.pkl'}
FEATURE_SCHEMAS = {
    'thermal': ['load_pct', 'lhv', 'exhaust_temp', 'o2_content', 'ambient_temp', 'g_ms', 'h_ms', 'g_rh', 'h_rh', 'g_pw', 'actual_coal_flow_t_h'],
    'legacy': ['load_pct', 'lhv', 'ep', 'lpq', 'mpq', 'rhp', 'bp', 'wt', 'at', 'rht', 'hour', 'weekday', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
}


def add_calendar_features(data):
    df = data.copy()
    for source, period in [('hour', 24), ('month', 12)]:
        if source in df:
            values = pd.to_numeric(df[source], errors='raise')
            df[f'{source}_sin'] = np.sin(2 * np.pi * values / period)
            df[f'{source}_cos'] = np.cos(2 * np.pi * values / period)
    return df


def prepare_features(data, schema=None, features=None):
    df = add_calendar_features(data)
    if features is None:
        if schema not in FEATURE_SCHEMAS:
            raise ValueError('未知特征集合')
        features = FEATURE_SCHEMAS[schema]
    missing = [f for f in features if f not in df]
    if missing:
        raise ValueError('缺少模型特征: ' + ', '.join(missing))
    try:
        X = df.loc[:, features].apply(pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('模型特征必须为数值，不能含文本标签') from exc
    if np.isinf(X.to_numpy()).any():
        raise ValueError('模型特征包含无穷值')
    return X, list(features)


def regression_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if len(actual) == 0 or actual.shape != prediction.shape or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError('评价样本为空、形状不一致或含无效数值')
    error = prediction - actual
    nonzero = np.abs(actual) > 1e-8
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        '样本数': len(actual), 'MAE': float(np.abs(error).mean()),
        'RMSE': float(np.sqrt((error ** 2).mean())), 'Bias': float(error.mean()),
        'MAPE (%)': float((np.abs(error[nonzero] / actual[nonzero])).mean() * 100) if nonzero.any() else None,
        'MAPE样本数': int(nonzero.sum()),
        'R²': float(1 - np.sum(error ** 2) / denominator) if len(actual) > 1 and denominator > 0 else None,
    }


def train_model(data, schema, coal_basis):
    if coal_basis not in ('actual', 'standard'):
        raise ValueError('必须明确训练标签的煤耗口径')
    df = data.copy()
    if not df.index.is_unique:
        raise ValueError('训练数据索引必须唯一')
    if 'coal_basis' in df and not (df['coal_basis'] == coal_basis).all():
        raise ValueError('训练标签煤耗口径与选择不一致')
    if 'coal_consumption' not in df:
        raise ValueError('缺少 coal_consumption 标签')
    y = pd.to_numeric(df['coal_consumption'], errors='raise')
    df['coal_consumption'] = y
    df = df.loc[np.isfinite(y) & (y >= 0)].copy()
    if len(df) < 20:
        raise ValueError('至少需要20条有效标签记录；不使用训练集代替测试集')
    if 'timestamp' in df:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
        if df['timestamp'].isna().any() or df['timestamp'].duplicated().any():
            raise ValueError('timestamp 必须有效且唯一，避免同一时刻进入训练与测试')
        df = df.sort_values('timestamp', kind='stable')
    X, features = prepare_features(df, schema)
    split = int(len(df) * 0.7)
    train_X, test_X = X.iloc[:split], X.iloc[split:]
    if train_X.isna().all().any():
        raise ValueError('训练段存在完全缺失的特征，无法估计填充值')
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median').set_output(transform='pandas')),
        ('model', lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31, max_depth=6, min_child_samples=5, random_state=42, n_jobs=1, verbosity=-1)),
    ])
    # No test-set early stopping; imputer is fitted only on the training slice.
    pipeline.fit(train_X, df['coal_consumption'].iloc[:split])
    prediction = pipeline.predict(test_X)
    holdout = df.iloc[split:].copy()
    holdout['prediction'] = prediction
    return {
        'bundle': {'pipeline': pipeline, 'features': features, 'schema': schema, 'coal_basis': coal_basis, 'source': 'session-trained'},
        'holdout': holdout, 'metrics': regression_metrics(holdout['coal_consumption'], prediction),
        'train_indices': list(df.index[:split]), 'test_indices': list(df.index[split:]),
        'train_end': df['timestamp'].iloc[split - 1] if 'timestamp' in df else split - 1,
        'test_start': df['timestamp'].iloc[split] if 'timestamp' in df else split,
        'split_note': '按时间排序70/30划分' if 'timestamp' in df else '按文件行序70/30划分；不等同于时间泛化验证',
    }


def load_legacy(source, trusted=False):
    if not trusted:
        raise ValueError('仅能加载已确认可信来源的本地 pickle 模型')
    if source not in LEGACY_FILES:
        raise ValueError('未知旧模型来源')
    path = MODEL_DIR / LEGACY_FILES[source]
    model, features = joblib.load(path)
    if not isinstance(features, (list, tuple)) or not all(isinstance(f, str) for f in features):
        raise ValueError('旧模型特征合同无效')
    return {'model': model, 'features': list(features), 'source': source, 'coal_basis': None}


def predict_bundle(bundle, row):
    X, _ = prepare_features(pd.DataFrame([row]), features=bundle['features'])
    if 'pipeline' in bundle:
        value = bundle['pipeline'].predict(X)[0]
    else:
        if X.isna().any().any():
            raise ValueError('旧模型缺少训练时填充器；输入特征必须完整')
        value = bundle['model'].predict(X)[0]
    if not np.isfinite(value) or value < 0:
        raise ValueError('模型输出非有限值或负煤耗；不自动截断或替换为零')
    return float(value)


def evaluate_methods(data, coal_basis, bundle=None):
    if coal_basis not in ('standard', 'actual'):
        raise ValueError('请选择评估标签的煤耗口径')
    if 'coal_basis' in data and not (data['coal_basis'] == coal_basis).all():
        raise ValueError('评估数据口径与选择不一致')
    required = ['load_pct', 'lhv', 'coal_consumption']
    if any(k not in data for k in required):
        raise ValueError('评估集需包含 load_pct、lhv、coal_consumption')
    truth = pd.to_numeric(data['coal_consumption'], errors='raise')
    if not np.isfinite(truth).all() or (truth < 0).any() or len(data) < 2:
        raise ValueError('评估集至少需要两条有效非负标签')
    details = pd.DataFrame({'Load_pct': data['load_pct'], 'True_Coal': truth})
    methods = ['table'] if coal_basis == 'actual' else ['dynamic_cb', 'counter_balance', 'direct_balance']
    if bundle is not None and bundle['coal_basis'] == coal_basis:
        methods.append('lgbm')
    summaries = []
    for method in methods:
        predictions, error_message = [], None
        parameter_keys = ['rated_power_mw'] if method in ('dynamic_cb', 'counter_balance', 'direct_balance') else []
        if method in ('dynamic_cb', 'counter_balance'):
            parameter_keys.append('pipe_eff')
        if method == 'counter_balance':
            parameter_keys.append('boiler_eff')
        defaults = default_params()
        assumptions = [f'{key}={defaults[key]}' for key in parameter_keys if key not in data]
        for _, row in data.iterrows():
            try:
                p = default_params()
                p.update(calc_strategy=method, load_pct=float(row['load_pct']), lhv=float(row['lhv']), coal_price_basis=coal_basis, coal_basis=coal_basis)
                if method in ('dynamic_cb', 'counter_balance'):
                    keys = [k for k in p['counter_balance_input'] if method == 'dynamic_cb' or k.startswith(('g_', 'h_'))]
                    missing = [k for k in keys if k not in row or pd.isna(row[k])]
                    if missing:
                        raise ValueError('缺少评估测点: ' + ', '.join(missing))
                    p['counter_balance_input'].update({k: float(row[k]) for k in keys})
                if method == 'direct_balance':
                    if 'actual_coal_flow_t_h' not in row or pd.isna(row['actual_coal_flow_t_h']):
                        raise ValueError('缺少 actual_coal_flow_t_h')
                    p['direct_balance_input'] = {'actual_coal_flow_t_h': float(row['actual_coal_flow_t_h']), 'base_load_pct': p['load_pct']}
                if method == 'lgbm':
                    p['lgbm_input'] = row.to_dict()
                for key in parameter_keys:
                    if key in row:
                        p[key] = float(row[key])
                predictor = (lambda x: predict_bundle(bundle, x)) if method == 'lgbm' else None
                predictions.append(coal_at_load(p, p['load_pct'], predictor)['coal'])
            except (ValueError, KeyError, TypeError) as exc:
                error_message = str(exc)
                break
        if error_message is not None:
            summaries.append({'method': method, '状态': '不可评估: ' + error_message, '参数假设': '；'.join(assumptions) or '使用提供的参数'})
            continue
        details[f'{method}_Pred'] = predictions
        summaries.append({'method': method, '状态': '诊断结果；不代表真实电厂验证', '参数假设': '默认情景：' + '；'.join(assumptions) if assumptions else '无缺省情景参数', **regression_metrics(truth, predictions)})
    return pd.DataFrame(summaries), details


def generate_demo_data(n=500):
    """Synthetic labels from the old mechanism: useful only for UI/tests."""
    if n < 1:
        raise ValueError('仿真样本数必须为正')
    rng = np.random.default_rng(42)
    rows = []
    for i, load in enumerate(rng.uniform(30, 100, n)):
        p = default_params()
        steam = p['counter_balance_input'].copy()
        for key in ('g_ms', 'g_rh', 'g_pw'):
            steam[key] *= load / 100
        steam.update(exhaust_temp=120 + (100 - load) * 0.15, o2_content=2.8 + (100 - load) * 0.03, ambient_temp=20.0)
        eff, _ = dynamic_efficiency(load, steam['exhaust_temp'], steam['o2_content'], steam['ambient_temp'])
        hr = steam['g_ms'] * steam['h_ms'] + steam['g_rh'] * steam['h_rh'] - steam['g_pw'] * steam['h_pw']
        standard_flow = hr / (STANDARD_COAL_LHV * eff * p['pipe_eff'])
        coal = standard_flow * 1000 / (600 * load / 100)
        row = dict(steam, **{k: v for k, v in p['lgbm_input'].items() if k not in steam})
        row.update(load_pct=load, lhv=21.0, actual_coal_flow_t_h=standard_flow * STANDARD_COAL_LHV / 21000 + rng.normal(0, 1.5), coal_consumption=coal)
        for key, noise in [('g_ms', 5), ('g_rh', 5), ('g_pw', 5), ('exhaust_temp', 2), ('o2_content', 0.2), ('ambient_temp', 2)]:
            row[key] += rng.normal(0, noise)
        stamp = pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i)
        row.update(timestamp=stamp, hour=stamp.hour, weekday=stamp.weekday(), month=stamp.month,
                   rated_power_mw=600.0, coal_basis='standard', data_source='synthetic')
        rows.append(row)
    return pd.DataFrame(rows)
