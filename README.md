# ThermoCost

## 👉 Online Demo [Click here to launch the app](https://thermal-cost-prototype.streamlit.app/)

**A reproducible research prototype for thermal power unit cost estimation and day-ahead schedule comparison.**

> **Scope statement:** This repository contains a sanitized, independently reproducible prototype derived from cost-estimation work conducted within a broader industry collaboration. It is not the original enterprise codebase, a complete electricity-trading platform, or evidence of production deployment. No confidential source code, raw plant data, proprietary market records, or externally sourced model artifacts are required to run the public workflow.

## Overview

ThermoCost is a bilingual Streamlit application for exploring how the operating schedule of a coal-fired generating unit affects selected variable costs. Its primary workflow evaluates a complete day of 96 fifteen-minute dispatch intervals and can compare two schedules under controlled assumptions.

The repository also includes a separate single-operating-point laboratory for comparing coal-consumption estimation methods and training a session-only LightGBM model on user-supplied data.

The default interface language is English. Users can switch between English and Chinese without resetting inputs or results.

## Key Features

- Full-day validation and cost estimation for 96 × 15-minute schedules
- Side-by-side comparison of two schedules when their delivered service and assumptions are comparable
- Explicit fuel, oil-support, and carbon-cost components
- Gross-energy and net-delivered-energy accounting
- Built-in synthetic operating curves for reproducible demonstrations
- CSV upload, detailed result inspection, charts, and CSV export
- Separate single-point estimation methods: lookup, dynamic counter-balance, static counter-balance, and direct balance
- Optional LightGBM training and holdout diagnostics within the current application session
- English/Chinese interface and automated tests for the principal calculations and interactions

## Quick Start

Python 3.10–3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/成本测算.py
```

On Windows, the included launcher searches the project `.venv`, available Conda environments, and system Python installations for a compatible runtime:

```powershell
.\启动成本测算.bat
```

To check the environment without starting Streamlit:

```powershell
.\启动成本测算.bat --check
```

The launcher does not install dependencies automatically and binds the application to the local machine.

## Workflow 1: 96-Period Schedule Cost Estimation

The default workflow evaluates one or two full-day gross-power schedules. Two deterministic example schedules are included, so the application can be explored without external data.

### Schedule CSV Format

Each uploaded schedule must contain exactly 96 consecutive local timestamps from `00:00` to `23:45` at 15-minute intervals.

Required columns:

- `timestamp`: timezone-naive local timestamp
- `planned_gross_mw`: scheduled gross output in MW
- `oil_kg_h`: oil-support rate in kg/h

Optional per-period columns:

- `aux_rate_pct`: auxiliary-power rate in percent
- `carbon_intensity_t_per_mwh`: carbon intensity in tCO2 per gross MWh

Example:

```csv
timestamp,planned_gross_mw,oil_kg_h
2026-01-01 00:00:00,420,0
2026-01-01 00:15:00,420,0
```

The example above shows only two rows; a valid file must contain all 96 intervals. If a schedule operates below 40% load, `oil_kg_h` must be positive for those intervals.

The validator also checks:

- timestamp completeness and ordering;
- minimum and maximum load limits;
- the ramp from the initial setpoint and all interval-to-interval ramps;
- finite numeric values and valid parameter ranges;
- consistency between the actual-coal consumption curve and coal-price basis.

### Included Cost Method

For each 15-minute interval:

```text
Gross energy (MWh) = scheduled gross power (MW) × 0.25 h
Actual coal (t) = coal consumption (g/kWh) × gross energy (MWh) ÷ 1000
Fuel cost (CNY) = actual coal (t) × actual-coal price (CNY/t)
Oil cost (CNY) = oil rate (kg/h) × oil price (CNY/kg) × 0.25 h
Carbon cost (CNY) = carbon intensity (tCO2/gross MWh)
                    × gross energy (MWh) × carbon price (CNY/tCO2)
Net energy (MWh) = gross energy (MWh) × (1 - auxiliary rate)
Included unit cost (CNY/net MWh) = (fuel + oil + carbon cost) ÷ net energy
```

“Included cost” means fuel, explicit oil support, and carbon only. Auxiliary consumption reduces delivered net energy; it is not represented as an invented additional electricity charge.

### Built-In Demonstration Curves

The public workflow uses transparent synthetic operating points with piecewise-linear interpolation. They support repeatable software demonstrations but are not calibrated plant curves.

Coal-consumption load nodes are `30, 40, 55, 70, 85, 100%`:

- LHV 18 MJ/kg: `488.2, 435.6, 392.4, 362.8, 345.2, 338.5 g/kWh`
- LHV 21 MJ/kg: `418.5, 373.4, 336.3, 310.9, 295.8, 290.1 g/kWh`
- LHV 24 MJ/kg: `366.2, 326.7, 294.3, 272.1, 258.8, 253.8 g/kWh`

Carbon-intensity load nodes are `30, 50, 75, 100%`, with corresponding values of `1.050, 0.915, 0.842, 0.795 tCO2/gross MWh`.

Users may replace the default auxiliary rate and carbon intensity with explicit per-period CSV values. Replacing the coal curve requires a code-level model change and appropriate validation.

### Schedule Comparison Guardrails

A cost difference is reported only when the two schedules have:

- the same timestamp axis;
- the same delivered net MWh;
- the same initial and final MW;
- consistent per-period auxiliary-rate and carbon-intensity input conventions.

These checks improve accounting comparability, but they do not establish that two schedules provide identical temporal market service. A lower included cost is not, by itself, proof of a better market bid or an optimal dispatch decision.

## Workflow 2: Single-Point Method Laboratory

The second workflow is intentionally separate from the 96-period model. It supports exploratory comparison of:

- actual-coal lookup at predefined load points;
- dynamic counter-balance estimation;
- static counter-balance estimation;
- direct-balance estimation from coal flow;
- a LightGBM regressor trained during the current session.

These methods use different assumptions and measurement requirements. Their outputs should not be concatenated with the 96-period results as if they formed one calibrated plant model.

### Session-Trained LightGBM

Training data must include:

- `coal_consumption` as the target in g/kWh;
- every feature required by the selected feature schema;
- at least 20 valid, non-negative target records;
- an explicit actual-coal or standard-coal basis.

If `timestamp` is provided, records are sorted chronologically and split 70%/30% into training and holdout sets. Without timestamps, the original row order is used and the result is not presented as temporal-generalization evidence. Median imputation is fitted on the training partition only.

Reported diagnostics include MAE, RMSE, MAPE, Bias, R-squared, sample counts, and load-segment results where available. MAPE excludes zero-valued targets. Synthetic demo data are provided only for interface exploration and software testing; they do not validate model accuracy on real generating units.

The trained model remains in the current Streamlit session and is not written to disk. Historical local `.pkl` files are excluded by `.gitignore` because their publication provenance has not been established.

## Outputs

Depending on the workflow, the application provides:

- daily cost and energy summaries;
- interval-level cost details;
- scheduled-power and unit-cost charts;
- schedule-comparison diagnostics;
- model holdout predictions and evaluation summaries;
- downloadable CSV files for further analysis.

## Repository Structure

```text
app/
  成本测算.py          # Streamlit entry point
  scenario_view.py    # 96-period interface and result presentation
  translations.py     # English/Chinese interface text
  i18n.py             # Localization helpers
models/
  scenario_cost.py    # Schedule validation, costing, and comparison
  cost_model.py       # Single-point mechanism and lookup methods
  coal_models.py      # Feature contracts, training, and evaluation
tests/                # Unit and Streamlit interaction tests
requirements.txt      # Runtime dependencies
启动成本测算.bat       # Windows launcher
```

## Testing

Run the complete test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover calculation units, data contracts, schedule comparability, model splitting and evaluation, localization, interface behavior, and launcher checks. Passing tests demonstrate software consistency; they do not constitute physical-model calibration or commercial validation.

## Limitations

This prototype does **not** currently implement or claim:

- dynamic marginal-cost curves or defensible bid-price floors;
- market-price forecasting, bid submission, market clearing, or automated trading;
- unit-commitment or economic-dispatch optimization;
- startup, shutdown, fixed O&M, fatigue, capacity-payment, risk, or ancillary-service economics in the primary workflow;
- dynamic blending of multiple coal sources;
- calibration against a named generating unit or independent validation on confidential plant data;
- verified production use or quantified commercial savings.

The numerical outputs should therefore be treated as research and software demonstrations, not as operational, investment, regulatory, or bidding advice.

## Collaboration and Publication Boundary

The broader collaboration involved research on cost analysis and bidding-decision support. This public repository reproduces only the cost-estimation component that can be shared independently. It does not contain the complete collaborative system and should not be used to infer ownership, implementation status, or performance of non-public components.

Before publishing a fork, review the staged files and do not force-add local model artifacts, internal reference files, credentials, databases, or private datasets excluded by `.gitignore`.

## License

No open-source license is currently included. Unless a license is added by the repository owner, the code remains under default copyright protection and should not be redistributed or reused beyond what applicable law permits.
