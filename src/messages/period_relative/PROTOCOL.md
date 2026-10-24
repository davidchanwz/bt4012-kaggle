# Period-relative feature experiment — 18 October 2026

Written before fitting new candidates. Use only supplied competition data. No Kaggle upload is performed.

Hypothesis: supplementing absolute features with within-period percentiles can help transfer across changing feature distributions.

Inputs: original 165 float32 features, plus 165 average-tie percentile ranks, calculated separately for each period over supplied known/query rows. Past unknown rows do not enter the ranking population. Future query features can enter unsupervised same-period ranks, as they would at test time. Fraud class values never enter the transformation. Unknown status determines the supplied query population, not its fraud class. Keep every original feature.

Controls: (1) the exact original raw-feature recency reference; (2) 330 inputs with each raw feature duplicated once, to distinguish extra representation from changing feature-subsampling coverage; (3) raw plus ranks, also 330 inputs. Same LightGBM settings, tau-5 recency and seeds 0,1,2. This duplicate control is added before seeing candidate scores. It is an implementation/capacity control, not a new feature hypothesis.

Current incumbent: reproduce 80% unknown-row auxiliary blend plus 20% masked six-relation graph probabilities. The unknown-row blend itself is 80/20 reference/auxiliary in log-odds space. Reuse saved predictions when their exact fold and seeds match; retrain controls for new windows. Assert agreement with overlapping existing predictions.

Primary windows: 25→26–30,28→29–32,30→31–35. Additional windows fixed here: 25→26–35 and 21→22–35. Both candidate and controls at cutoff25 predict to35, so one fitted model supports two windows. Report pooled AUC, per-period AUC and forecast-distance bins1–3,4–7,8–14. The labels and windows overlap earlier work and are not independent confirmation.

Candidate formulas: standalone raw-plus-ranks, and one fixed 50/50 probability blend with the current incumbent. No coefficient grid. Apply the same formulas to the duplicate control for interpretation. Report each model versus the raw reference and versus the stronger incumbent.

Decision: examine consistency across the three recent windows, the two longer forecasts, and the matched duplicate control. Disclose losses. A near-zero saturated-fold gain alone does not justify a public submission. Lock any finalist and its formula before an extra 32→33–35 check; this check overlaps other windows and cannot supply independence. Final fits average seeds0–4. Reload every saved model, reproduce all test probabilities, and check sample order, unique indices, finite bounded values and CSV round-trip before delivering a submission.

Meaningful implementation checks: ties and constant features behave correctly; changing fraud class values cannot change ranks; modifying a later period cannot change earlier period features; changing training-unknown vectors cannot change ranks; test missing-label status does not exclude supplied query vectors. Save input hashes, configuration, predictions, model importances, timing and all negative results.
