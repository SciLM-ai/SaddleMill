# Exact Sella benchmark configs used in the active cohort-0 package

These files were copied verbatim from `active_cohort0_arm_configs_20260929_234617_v2`.

Arms:

- `SEL00M24_SELLA_PRFO`: native Sella P-RFO control (`ablation = normal`, `maxiter = 24`).
- `SEL03M24_SELLA_PRFO_FIXED_TRUST`: P-RFO trust-adaptation-isolation arm (`ablation = trust_adaptation_isolation`, `maxiter = 24`).
- `SEL04M24_SELLA_QN`: Sella QN arm (`method = qn`, `ablation = qn_newton_safe_false`, `maxiter = 24`).

Each arm contains the exact dataset-specific configs for `oc20`, `oc22`, `mp20bat`, and `lemat`. Dataset-specific calculator/task fields differ, so use the matching dataset config rather than collapsing them into one file.

The configs retain campaign placeholders such as `__SET_BY_CAMPAIGN_GENERATOR__`; they are reference configs from the benchmark campaign, not a universal standalone run wrapper.
