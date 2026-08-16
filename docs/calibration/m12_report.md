# M12 Calibration Report

## Goals

M12 validates the canonical Maya v1 simulator after adding exact financial charges, richer world events, canonical engine construction, and deterministic calibration diagnostics.

## Baseline M11 Failure

The post-M11 architecture could fail when M4 affordability reasoned over total liquid resources while an M5 direct monetary effect targeted only one account. Perceived and actual costs could also disagree. The reproducible targeted baseline is `financial.bank_balance` underflow from a direct monetary delta even when other liquid accounts exist.

A full old-code cohort was not run from this branch; keeping that isolated historical run would require checking out pre-M12 code and catalogs outside the runtime package. The targeted regression is retained in `tests/test_consequences.py::test_monetary_underflow_fails_without_partial_mutation`.

## Structural Financial Fix

M12 keeps strict direct monetary effects for legacy/state-delta semantics, but ordinary event costs now use explicit `FinancialChargeDefinition` records. Charges settle exact `Decimal` amounts across a declared funding order, audit every transfer, and use either `require_full` or `arrear` shortfall policy. Scheduled financial charges preserve decision/event/outcome provenance and execute exactly once.

## Event Additions

The starter world expands from 5 to 12 events with modest city, finance, social, health, housing, technology, bureaucracy, education, and refund opportunities. The global world event probability remains 0.45 and max events per week remains 1.

## Parameter Change Log

- `pay_for_faster_transport` estimated cost: 14.00 -> 16.00, aligned with actual charge.
- `handle_immediately` estimated cost: 22.00 -> 24.00, aligned with actual charge.
- `accept_invitation` estimated cost: 18.00 -> 20.00, aligned with actual charge.
- Direct event-cost bank deltas for transport, small expenses, and social invitations were replaced by explicit financial charges.
- Added behavior tags to starter event options for M11 evidence.
- Added 7 starter events: phone/device problem, household maintenance issue, free local activity, bureaucratic errand, minor health setback, university admin deadline, and small refund opportunity.
- No M8 employment, M7 routine, M9 development, M10 social, M11 adaptation rates, or global M3 event probability were tuned in this hardening pass.

## 200-Seed Calibration Summary

- requested runs: 200
- successful runs: 200
- failures: 0
- hard invariant failures: 0
- warnings: ROUTINE_LOCK_IN, DEVELOPMENT_LOCK_IN, STATE_BOUNDARY_SATURATION, SOCIAL_LOCK_IN

### Finance Distributions
- final liquid: p05 0.00, p50 2750.60, p95 10384.34, mean 3398.93
- maximum arrears: p05 0.00, p50 1440.60, p95 7475.59, mean 2323.28
- final arrears: p05 0.00, p50 0.00, p95 5080.87, mean 719.60
- final debt: p05 0.00, p50 0.00, p95 0.00, mean 1.75
- arrear incidence rate: 0.81
- arrear recovery rate: 0.75
- liquid p05/p50/p95 by checkpoint: {'12': '0.00/82.00/1340.76', '26': '0.00/0.00/1072.48', '52': '0.00/0.00/2067.20', '156': '0.00/2750.60/10384.34'}

### Employment Funnel
- ever employed by checkpoint: {'12': 0.605, '26': 0.91, '52': 0.985, '156': 1.0}
- current employed by checkpoint: {'12': 0.58, '26': 0.65, '52': 0.83, '156': 0.99}
- first employment week: p05 4.00, p50 11.00, p95 34.25, mean 13.88
- application -> interview rate: 0.00
- interview -> offer rate: 0.00
- offer -> acceptance rate: 0.00

### Education / Development
- graduation rate by checkpoint: {'12': 0.0, '26': 0.0, '52': 0.0, '156': 0.13}
- graduation week: p05 144.75, p50 153.00, p95 156.00, mean 152.08
- development profile shares: {'balanced_study': 0.9967307692307692, 'light_self_development': 0.003269230769230769}

### Health / Mental Saturation
- boundary saturation rate: 0.3010
- saturated fields: ['health.physical_health', 'mental.recovery_need', 'mental.mood', 'mental.loneliness']

### Social
- new connection incidence rate: 0.90
- final connections: p05 2.00, p50 4.00, p95 7.00, mean 4.12
- social outcome counts: {'awkward': 340, 'friction': 2827, 'light': 1, 'limited': 493, 'neutral': 9617, 'promising': 146, 'supportive': 891, 'unavailable': 324, 'warm': 12942}

### Routine
- profile shares: {'austerity_home_week': 0.0034935897435897437, 'balanced_week': 0.0, 'low_cost_active_week': 0.0, 'recovery_focus_week': 0.996474358974359, 'social_week': 3.205128205128205e-05}
- dominant profile/share: recovery_focus_week / 1.00

### Events
- event-week rate: 0.40
- dominant event/share: study_pressure / 0.21
- dominant category/share: education / 0.28

### Adaptation
- habits formed: p05 3.00, p50 4.00, p95 5.00, mean 3.99
- max habit familiarity: p05 0.16, p50 0.16, p95 0.16, mean 0.16
- max weekly personality delta: p05 0.00, p50 0.00, p95 0.00, mean 0.00
- max anchor displacement: p05 0.05, p50 0.05, p95 0.05, mean 0.05

## 50-Seed Holdout Comparison

- requested runs: 50
- successful runs: 50
- failures: 0
- hard invariant failures: 0
- warnings: ROUTINE_LOCK_IN, DEVELOPMENT_LOCK_IN, STATE_BOUNDARY_SATURATION, SOCIAL_LOCK_IN

### Finance Distributions
- final liquid: p05 0.00, p50 2927.60, p95 10793.47, mean 3741.12
- maximum arrears: p05 0.00, p50 1105.20, p95 7194.13, mean 2025.50
- final arrears: p05 0.00, p50 0.00, p95 4115.18, mean 569.65
- final debt: p05 0.00, p50 0.00, p95 0.00, mean 0.80
- arrear incidence rate: 0.78
- arrear recovery rate: 0.80
- liquid p05/p50/p95 by checkpoint: {'12': '0.00/444.20/1488.76', '26': '0.00/0.00/1469.10', '52': '0.00/0.00/2947.78', '156': '0.00/2927.60/10793.47'}

### Employment Funnel
- ever employed by checkpoint: {'12': 0.68, '26': 0.94, '52': 1.0, '156': 1.0}
- current employed by checkpoint: {'12': 0.66, '26': 0.62, '52': 0.84, '156': 1.0}
- first employment week: p05 4.00, p50 9.00, p95 26.55, mean 11.06
- application -> interview rate: 0.00
- interview -> offer rate: 0.00
- offer -> acceptance rate: 0.00

### Education / Development
- graduation rate by checkpoint: {'12': 0.0, '26': 0.0, '52': 0.0, '156': 0.18}
- graduation week: p05 145.40, p50 152.00, p95 156.00, mean 151.00
- development profile shares: {'balanced_study': 0.9942307692307693, 'light_self_development': 0.0057692307692307696}

### Health / Mental Saturation
- boundary saturation rate: 0.3005
- saturated fields: ['health.physical_health', 'mental.recovery_need', 'mental.mood', 'mental.loneliness']

### Social
- new connection incidence rate: 0.88
- final connections: p05 2.00, p50 4.00, p95 6.00, mean 3.82
- social outcome counts: {'awkward': 76, 'friction': 693, 'limited': 124, 'neutral': 2347, 'promising': 38, 'supportive': 225, 'unavailable': 64, 'warm': 3249}

### Routine
- profile shares: {'austerity_home_week': 0.004358974358974359, 'balanced_week': 0.0, 'low_cost_active_week': 0.0, 'recovery_focus_week': 0.9955128205128205, 'social_week': 0.0001282051282051282}
- dominant profile/share: recovery_focus_week / 1.00

### Events
- event-week rate: 0.41
- dominant event/share: study_pressure / 0.20
- dominant category/share: education / 0.28

### Adaptation
- habits formed: p05 3.00, p50 4.00, p95 5.00, mean 3.96
- max habit familiarity: p05 0.16, p50 0.16, p95 0.16, mean 0.16
- max weekly personality delta: p05 0.00, p50 0.00, p95 0.00, mean 0.00
- max anchor displacement: p05 0.05, p50 0.05, p95 0.05, mean 0.05

## Remaining Warnings

- calibration: ROUTINE_LOCK_IN, DEVELOPMENT_LOCK_IN, STATE_BOUNDARY_SATURATION, SOCIAL_LOCK_IN
- holdout: ROUTINE_LOCK_IN, DEVELOPMENT_LOCK_IN, STATE_BOUNDARY_SATURATION, SOCIAL_LOCK_IN
- warnings are diagnostics only; no M8 employment, M7 routine, M9 development, M10 social, or M11 adaptation tuning was performed in M12.

## Canonical Seed-42 Validation

Canonical input paths are frozen below. Validation requires 12/26/52-week runs to match the corresponding prefixes of the 156-week run, and two 156-week runs with seed 42 to produce identical complete JSON.

- 156-week canonical run completed with 157 serialized states.
- 12-week, 26-week, and 52-week runs matched the corresponding 156-week state prefixes.
- two independent 156-week runs produced identical complete JSON.
- canonical JSON SHA-256: `23f31d68cdb12d2a9da3cdad34167ebea91cb7197f0d289a8a50b0f79decab8a`

## Known Limitations

- Indefinite starter jobs can make current week-156 employment sticky after acquisition. Employment warnings therefore use acquisition funnel timing/rates rather than current employment alone.
- Calibration diagnostics are broad warnings, not automatic tuning instructions.
- No layoffs, career ladders, relationship life-cycle arcs, or long-term macroeconomic systems are implemented in M12.

## Frozen Canonical Input Paths

- `configs/canonical/maya_v1.toml`
- `configs/scenarios/maya_start.toml`
- `configs/events/starter.toml`
- `configs/consequences/starter.toml`
- `configs/routines/starter.toml`
- `configs/employment/starter.toml`
- `configs/development/starter.toml`
- `configs/social/starter.toml`
- `configs/adaptation/starter.toml`
