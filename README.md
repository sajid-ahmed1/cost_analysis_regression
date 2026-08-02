# cost_analysis_regression
This project is a pure Data Science and Machine Learning repository to host the SMAPI (Vodafone internal tool) projection for events forecasting that determines the cost for users. We need to better predict the events data per Quarter


"Because install_id resets on app updates, we treated install_ids not as static unique users, but as an indicator of active installations and update frequency. Combining install_ids with cookie_ids allowed TabFM and LightGBM to explicitly learn how app deployment cycles impact total SDK telemetry volume."

## FAQ / decision log

Running log of questions raised while building this project and the reasoning behind the
answer. Add to this whenever a non-obvious modelling or evaluation decision gets made —
newest entries at the bottom.

### Does dropping `fullVisitorId` from the model mean the row loses its identity?

No. Every row keeps all its own feature values (sessions, events, platform, market, lags,
etc.) — dropping the ID only means it's excluded from the matrix passed to `.fit()`. It stays
in the dataframe as a label column so predictions can be joined back to a visitor afterward,
e.g. `X = df.drop(columns=["fullVisitorId", "quarter", "target"])`.

### If we encode `quarter` as a running integer (2016Q3=1, 2016Q4=2, 2017Q1=3, 2017Q2=4, 2017Q3=5...), does that give the model seasonality?

No — that's a **trend** feature (time passing), not a **seasonality** feature. Q4 2016 and Q4
2017 get unrelated numbers (2 and 6), so the model can't learn "the 4th quarter in the cycle
behaves differently." Seasonality needs a feature that *repeats* every year: `quarter_of_year
∈ {1,2,3,4}` (via `.dt.quarter`), as a categorical/one-hot or cyclic sin/cos encoding. Use
**both** the running index (trend) and quarter-of-year (seasonality) as separate columns.

### What's the target column?

Next quarter's `total_events` (hits) per visitor — shift `total_events` forward by one quarter
within each visitor group. This matches the project scope: forecasting event volume, not
revenue. The last observed quarter per visitor has no future value to predict and needs to be
dropped or held out separately from the lag-NaN handling (see below).

### Do we train separate models per platform/device/market, or one pooled model?

Decision: **one pooled model** (TabFM and LightGBM) with platform/device/market as categorical
features, rather than separate models per segment. Segment-specific models could fit
segment patterns better but multiply the evaluation surface and split already-limited data.
Traffic medium/source is suspected to be low-value but is included in the pooled model and
kept or dropped based on feature importance afterward, rather than removed upfront on a hunch.

### Why impute 0 for missing lag features (e.g. `total_events_prev_q`) instead of dropping or using another fill value?

0 encodes "no observed history" (e.g. a new SDK install with no prior quarter), which is a
real, meaningful state — not a missing/unknown value. This must stay consistent between
training and any future scoring pipeline (same imputation logic in both places) to avoid
train/serve skew.

### Should we add a quarter-of-year feature for seasonality (e.g. December/summer promotions)?

Yes — `quarter_of_year` (1-4, via `.dt.quarter`) is added alongside the running `quarter`
trend index. It repeats every year, so the model can learn "Q4 behaves differently" in a way
the running index can't (see the quarter-encoding FAQ above). Implemented in
`src/features/engineering.py::add_quarter_columns`.

### Where does the feature engineering logic live?

`notebooks/02_feature_engineering.ipynb` keeps the logic inline, cell by cell, since that's
clearer for exploration and review. The same logic is mirrored as a reusable pipeline of
functions in `src/features/engineering.py` (`flatten_raw_columns`, `add_quarter_columns`,
`aggregate_visitor_quarter`, `add_lag_and_target_features`, `build_quarterly_active_users`,
composed by `build_visitor_quarter_features()`) for reuse by a training script or a future
scoring pipeline, and so it can be unit tested. Keep the two in sync when the logic changes.

### The goal is a single per-quarter company-wide events number, so why predict at the visitor grain?

Decision: **bottom-up forecasting** - train and predict at the visitor x quarter grain (richer
features, far more training rows, supports LightGBM/TabFM), then sum each quarter's per-visitor
predictions to get the quarter total. This is the only viable option here: a pure quarter-level
time series would only have ~8 rows (one per quarter) to train on, nowhere near enough for a
model like LightGBM or TabFM, so it's not a real alternative in practice.

This has consequences for evaluation: don't just evaluate per-visitor prediction error in
isolation - also evaluate the **summed** prediction against the actual quarter total, since
that's the number that actually matters for the SMAPI cost projection. Per-visitor errors can
partially cancel out (over- and under-predictions offsetting) when summed, so the aggregate
metric can look better or worse than the per-row metrics suggest - track both.

### For small-N time series (e.g. ~8 quarters or ~24 months), why walk-forward instead of LOOCV?

Decision: **walk-forward / rolling-origin validation**, not leave-one-out. LOOCV assumes each
held-out point is independent of the rest of the training set - true for i.i.d. rows, false for
a time series, where adjacent periods are autocorrelated. Leaving out April but training on
both March and May lets the model see April's neighbors on both sides, which is temporal
leakage - the model would never have that "future" information when actually forecasting.
Walk-forward only ever trains on the past and predicts the next period (train on 1..k, predict
k+1, slide forward), matching how the model is actually used in production. This applies
whether the series is quarterly (~8 points) or monthly (~24 points) - going to monthly gives
more folds and a less noisy average, but doesn't remove the autocorrelation problem, so the
validation scheme still needs to change either way. Whatever grain training happens at, the
final reported error should still be evaluated at the quarter grain, since that's what SMAPI
cost reporting actually consumes.

### Install IDs reset on app version updates - does that undermine the bottom-up (per-visitor lag features) approach for real SMAPI data?

Yes, partially, and worth being explicit about it rather than glossing over it. The bottom-up
approach's lag features (`total_events_prev_q`, `total_events_qoq_change`) assume a visitor's
identifier persists across quarters so "previous quarter" is a real prior observation. If
install IDs reset on every app update, a chunk of the population effectively becomes a "new
visitor" with every release - not just at the start of the historical window (which the 0
lag-impute already handles) but repeatedly, on an ongoing basis. That means a meaningful
fraction of rows carry little or no real lag signal even though they aren't the visitor's
*true* first quarter - the model can't tell "genuinely new" from "same person, ID reset" apart.

This doesn't invalidate the approach - install ID churn is itself an intentional signal (see
the top-of-README note on treating install_id as an indicator of deployment/update activity
rather than a static unique user), and it's still a legitimate, demonstrable technique for a
showcase. But for a production-faithful SMAPI model, the fix is to reduce identity churn before
modelling: combine `install_id` with a more persistent identifier (e.g. `cookie_id`) to bridge
across resets where possible, and treat the remaining reset-driven "cold starts" as expected
noise rather than a modelling bug - not something a bigger N or a different validation scheme
alone can fix.