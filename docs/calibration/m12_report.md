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

## Pre-Tuning Hardened Baseline

After the employment/social instrumentation and bounded-state diagnostics were hardened, but before the final routine/development profile tuning, a 200-seed x 156-week cohort still showed demonstrated lock-in and saturation:

- routine lock: recovery_focus_week 0.9970, austerity_home_week 0.0030, social_week 0.0000, balanced_week 0.0000, low_cost_active_week 0.0000.
- development lock: balanced_study 0.9483, light_self_development 0.0517.
- boundary saturation: 0.1934, concentrated in mental.mood and mental.loneliness.
- education: graduation by week 156 was 0.790; final efficiency p05/p50/p95 was 0.1438/0.1772/0.3315.
- corrected social tracking already showed broad voluntary action rather than a true no-opportunity lock: connect 0.9235, seek_support 0.0509, engage 0.0255, keep_social_light 0.0002, dominant contact share 0.4989.

The final calibrated cohorts below are the freeze criteria. Employment funnel metrics are reported from the corrected M8 tracker in the final calibration and holdout sections.

## Parameter Change Log

- `pay_for_faster_transport` estimated cost: 14.00 -> 16.00, aligned with actual charge.
- `handle_immediately` estimated cost: 22.00 -> 24.00, aligned with actual charge.
- `accept_invitation` estimated cost: 18.00 -> 20.00, aligned with actual charge.
- Direct event-cost bank deltas for transport, small expenses, and social invitations were replaced by explicit financial charges.
- Added behavior tags to starter event options for M11 evidence.
- Added 7 starter events: phone/device problem, household maintenance issue, free local activity, bureaucratic errand, minor health setback, university admin deadline, and small refund opportunity.
- Corrected calibration instrumentation for M8 employment stages. Skipped openings no longer count as applications; submitted applications, invitations, attended interviews, produced offers, accepted offers, and declined paths are tracked separately. Observed effect: impossible zero funnel rates disappeared while employment starts remained coherent.
- Corrected calibration instrumentation for M10 social choices. Each executed focal decision contributes one canonical voluntary choice (`connect`, `seek_support`, `engage`, or `keep_social_light`), no-opportunity weeks are tracked separately, and named contact shares are deterministic. Observed effect: SOCIAL_LOCK_IN cleared without tuning M10.
- Added a deterministic routine repeat penalty for the currently repeated routine profile: 0.00 -> min(1.20, repeated_weeks * 0.12), applied through the weekly routine event option values. Reason: M7 routine choice had become an attractor despite state responsiveness. Observed effect: dominant routine share moved from 0.9970 recovery_focus_week to 0.6906 in calibration and 0.6855 in holdout.
- Added boundary-sensitive damping for ordinary recurring bounded-state effects in M7 routine, M8 work, M9 development, and M10 social: raw ordinary delta -> raw_delta * distance_to_relevant_boundary / (distance + 120.0). Food-security shortfall and sleep-debt floor semantics remain exact. Reason: recurring ordinary life effects were pinning bounded health/mental fields. Observed effect: boundary saturation moved from 0.1934 to 0.0133 in calibration and 0.0130 in holdout.
- Tuned starter M7 routine profile values. Balanced_week estimated_cost 95.00 -> 82.00 and slightly lower social/health/comfort values; austerity_home_week estimated_cost 50.00 -> 46.00 and comfort 0.35 -> 0.20; low_cost_active_week estimated_cost 62.00 -> 58.00 with higher short/future/social/health values; recovery_focus_week health/comfort 0.45/0.55 -> 0.65/0.75 but lower autonomy; social_week estimated_cost 145.00 -> 134.00 with small signal adjustments. Reason: preserve plausible niches without random variety. Observed effect: low_cost_active_week reached 0.2884/0.2962 share while recovery stayed below the 0.75 warning threshold.
- Tuned starter M9 development profile values. Balanced_study energy_cost 24.0 -> 21.0 and learning 0.55 -> 0.46; intensive_study energy_cost 42.0 -> 34.0 and learning/future 0.78/0.68 -> 0.96/0.86; reduced_study energy_cost 10.0 -> 8.0 with higher recovery-compatible value; admin_skill_focus energy_cost 26.0 -> 21.0, future/learning 0.48/0.62 -> 0.62/0.72, and added finance goal tag; language_skill_focus energy_cost 24.0 -> 20.0, future/learning 0.42/0.58 -> 0.56/0.70, added social_pressure 0.10 and social/city goal tags. Reason: make profile tradeoffs real under generic M4 scoring. Observed effect: dominant development share moved from 0.9483 balanced_study to 0.6067 admin_skill_focus in calibration and 0.6132 in holdout.
- Re-evaluated education after lock-in and saturation fixes. `progress_per_full_study_week` remains 1.6; a temporary 1.3 candidate produced 0.000 graduation in a smoke cohort and was rejected. Observed effect at freeze: graduation by week 156 is common but not universal, 0.755 calibration and 0.840 holdout.
- M8 employment parameters, M10 social catalog probabilities/signals, M11 adaptation rates, M3 event probability, finance starting money, arrear settlement, and debt behavior were inspected but not tuned in this final calibration pass.

## 200-Seed Calibration Summary

- requested runs: 200
- successful runs: 200
- failures: 0
- hard invariant failures: 0
- warnings: none

### Finance Distributions
- final liquid: p05 0.00, p50 3090.30, p95 10845.79, mean 3736.49
- maximum arrears: p05 0.00, p50 1423.00, p95 7494.42, mean 2294.37
- final arrears: p05 0.00, p50 0.00, p95 4789.91, mean 658.20
- final debt: p05 0.00, p50 0.00, p95 0.00, mean 1.50
- arrear incidence rate: 0.815
- arrear recovery rate: 0.765
- liquid p05/p50/p95 by checkpoint: {'12': '0.00/96.40/1380.76', '26': '0.00/0.00/1177.62', '52': '0.00/0.00/2226.70', '156': '0.00/3090.30/10845.79'}

### Employment Funnel
- ever employed by checkpoint: {'12': 0.605, '26': 0.915, '52': 0.985, '156': 1.0}
- current employed by checkpoint: {'12': 0.58, '26': 0.65, '52': 0.82, '156': 0.99}
- submissions/skips/invitations/attended/offers/accepted: 2683 / 0 / 1073 / 1073 / 393 / 393
- first employment week: p05 4.00, p50 11.00, p95 34.25, mean 13.77
- application submission -> interview invitation rate: 0.400
- interview attended -> offer rate: 0.366
- offer -> acceptance rate: 1.000

### Education / Development
- progress p05/p50/p95 by checkpoint: {'12': '49.16/50.37/51.78', '26': '53.80/55.80/58.03', '52': '61.92/65.21/68.58', '156': '96.44/100.00/100.00'}
- graduation rate by checkpoint: {'12': 0.0, '26': 0.0, '52': 0.0, '156': 0.755}
- graduation week: p05 132.50, p50 146.00, p95 155.00, mean 145.58
- development profile shares: {'admin_skill_focus': 0.6067307692307692, 'language_skill_focus': 0.3428525641025641, 'light_self_development': 0.050416666666666665}
- final efficiency: p05 0.1980, p50 0.2484, p95 0.3531, mean 0.2573
- efficiency factors: {'energy_factor': 'p05/p50/p95 0.4475/0.5313/0.6910', 'stress_factor': 'p05/p50/p95 0.6754/0.7343/0.7816', 'mental_load_factor': 'p05/p50/p95 0.6914/0.7165/0.7851', 'recovery_factor': 'p05/p50/p95 0.8382/0.8826/0.9531', 'workload_factor': 'p05/p50/p95 1.0000/1.0000/1.0000'}

### Health / Mental Saturation
- boundary saturation rate: 0.0133
- saturated fields: ['mental.loneliness']
- boundary direction by field: {'health.energy': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'health.physical_health': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.stress': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.mental_load': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.recovery_need': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.mood': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.loneliness': '0 weeks p50/p95 10.5/40.0; 100 weeks p50/p95 0.0/0.0', 'health.sleep_debt': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0'}

### Social
- focal opportunity rate: 0.891
- no-opportunity rate: 0.109
- new connection incidence rate: 0.900
- final connections: p05 2.00, p50 4.00, p95 7.00, mean 4.00
- choice shares: {'connect': 0.8895123003884333, 'engage': 0.025032369443245578, 'keep_social_light': 0.00010789814415192059, 'seek_support': 0.08534743202416918}
- dominant contact/share: lina / 0.504
- social outcome counts: {'awkward': 297, 'friction': 2782, 'light': 3, 'limited': 723, 'neutral': 9207, 'promising': 162, 'supportive': 1206, 'unavailable': 444, 'warm': 12980}

### Routine
- profile shares: {'austerity_home_week': 0.021025641025641025, 'balanced_week': 0.0, 'low_cost_active_week': 0.2883974358974359, 'recovery_focus_week': 0.6905769230769231, 'social_week': 0.0}
- dominant profile/share: recovery_focus_week / 0.691

### Events
- event-week rate: 0.399
- dominant event/share: study_pressure / 0.195
- dominant category/share: education / 0.261

### Adaptation
- habits formed: p05 6.00, p50 7.00, p95 7.00, mean 6.92
- final habit strength: p05 14.74, p50 45.20, p95 88.45, mean 54.36
- max habit familiarity: p05 0.1409, p50 0.1430, p95 0.1460, mean 0.1432
- max weekly personality delta: p05 0.000490, p50 0.000564, p95 0.000635, mean 0.000561
- max anchor displacement: p05 0.037536, p50 0.039041, p95 0.040533, mean 0.038988

## 50-Seed Holdout Comparison

- requested runs: 50
- successful runs: 50
- failures: 0
- hard invariant failures: 0
- warnings: none

### Finance Distributions
- final liquid: p05 0.00, p50 3427.70, p95 11270.77, mean 4126.45
- maximum arrears: p05 0.00, p50 1093.20, p95 7108.48, mean 1975.98
- final arrears: p05 0.00, p50 0.00, p95 3770.43, mean 522.16
- final debt: p05 0.00, p50 0.00, p95 0.00, mean 0.50
- arrear incidence rate: 0.780
- arrear recovery rate: 0.818
- liquid p05/p50/p95 by checkpoint: {'12': '0.00/459.20/1524.26', '26': '0.00/0.00/1557.80', '52': '0.00/0.00/3106.48', '156': '0.00/3427.70/11270.77'}

### Employment Funnel
- ever employed by checkpoint: {'12': 0.68, '26': 0.94, '52': 1.0, '156': 1.0}
- current employed by checkpoint: {'12': 0.66, '26': 0.64, '52': 0.84, '156': 1.0}
- submissions/skips/invitations/attended/offers/accepted: 628 / 0 / 256 / 256 / 99 / 99
- first employment week: p05 4.00, p50 9.00, p95 26.55, mean 11.22
- application submission -> interview invitation rate: 0.408
- interview attended -> offer rate: 0.387
- offer -> acceptance rate: 1.000

### Education / Development
- progress p05/p50/p95 by checkpoint: {'12': '49.26/50.36/51.81', '26': '53.95/55.86/58.32', '52': '62.32/65.71/69.82', '156': '97.30/100.00/100.00'}
- graduation rate by checkpoint: {'12': 0.0, '26': 0.0, '52': 0.0, '156': 0.84}
- graduation week: p05 127.05, p50 148.00, p95 153.00, mean 144.38
- development profile shares: {'admin_skill_focus': 0.6132051282051282, 'language_skill_focus': 0.3242307692307692, 'light_self_development': 0.06256410256410257}
- final efficiency: p05 0.2028, p50 0.2539, p95 0.3523, mean 0.2619
- efficiency factors: {'energy_factor': 'p05/p50/p95 0.4546/0.5383/0.6895', 'stress_factor': 'p05/p50/p95 0.6701/0.7348/0.7764', 'mental_load_factor': 'p05/p50/p95 0.6909/0.7170/0.7854', 'recovery_factor': 'p05/p50/p95 0.8399/0.8910/0.9601', 'workload_factor': 'p05/p50/p95 1.0000/1.0000/1.0000'}

### Health / Mental Saturation
- boundary saturation rate: 0.0130
- saturated fields: ['mental.loneliness']
- boundary direction by field: {'health.energy': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'health.physical_health': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.stress': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.mental_load': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.recovery_need': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.mood': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0', 'mental.loneliness': '0 weeks p50/p95 12.5/41.0; 100 weeks p50/p95 0.0/0.0', 'health.sleep_debt': '0 weeks p50/p95 0.0/0.0; 100 weeks p50/p95 0.0/0.0'}

### Social
- focal opportunity rate: 0.888
- no-opportunity rate: 0.112
- new connection incidence rate: 0.840
- final connections: p05 2.00, p50 4.00, p95 6.00, mean 4.00
- choice shares: {'connect': 0.9026574234546505, 'engage': 0.024263431542461005, 'keep_social_light': 0.00014442518775274407, 'seek_support': 0.07293471981513576}
- dominant contact/share: lina / 0.518
- social outcome counts: {'awkward': 68, 'friction': 694, 'light': 1, 'limited': 159, 'neutral': 2306, 'promising': 40, 'supportive': 265, 'unavailable': 81, 'warm': 3310}

### Routine
- profile shares: {'austerity_home_week': 0.018333333333333333, 'balanced_week': 0.0, 'low_cost_active_week': 0.29615384615384616, 'recovery_focus_week': 0.6855128205128205, 'social_week': 0.0}
- dominant profile/share: recovery_focus_week / 0.686

### Events
- event-week rate: 0.403
- dominant event/share: study_pressure / 0.188
- dominant category/share: education / 0.262

### Adaptation
- habits formed: p05 6.00, p50 7.00, p95 7.00, mean 6.86
- final habit strength: p05 16.20, p50 45.83, p95 88.42, mean 54.73
- max habit familiarity: p05 0.1407, p50 0.1425, p95 0.1456, mean 0.1427
- max weekly personality delta: p05 0.000494, p50 0.000567, p95 0.000671, mean 0.000567
- max anchor displacement: p05 0.037041, p50 0.039012, p95 0.040983, mean 0.038997

## Remaining Warnings

- calibration: none
- holdout: none
- warnings are diagnostics only. The final calibrated and holdout cohorts have no remaining soft warnings.
- financial pressure remains intentionally visible even without a warning: arrear incidence is high, but final arrears commonly recover and no negative-money hard invariant failed. This reflects Maya's starting vulnerability rather than hidden debt forgiveness or free money.

## Canonical Seed-42 Validation

Canonical input paths are frozen below. Validation requires 12/26/52-week runs to match the corresponding prefixes of the 156-week run, and two 156-week runs with seed 42 to produce identical complete JSON.

- 156 completed: yes, final week 156 with 157 serialized states including week 0.
- prefix equality: 12-week, 26-week, and 52-week runs all match the corresponding 156-week state prefixes.
- same-seed complete JSON equality: true.
- same-seed SHA-256: `c84696ff87bb548bfd6f7c496b080530d04c1bb2a102548e97ed0b3033d21089`.
- seed-42 final state: Maya completed Urban Studies BA, is employed as Retail Assistant at North Arcade Books, has no arrears, and has final liquid bank balance 6035.00 EUR.

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
