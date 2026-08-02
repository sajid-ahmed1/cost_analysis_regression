"""Visitor x quarter feature engineering for event-volume forecasting."""

import json

import pandas as pd

RAW_COLUMNS = ["fullVisitorId", "date", "device", "trafficSource", "totals", "geoNetwork"]


def flatten_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the JSON string columns and pull out the fields we need."""
    df = df.copy()
    device = df["device"].apply(json.loads)
    traffic = df["trafficSource"].apply(json.loads)
    totals = df["totals"].apply(json.loads)
    geo = df["geoNetwork"].apply(json.loads)

    df["platform"] = device.apply(lambda d: d.get("operatingSystem"))
    df["device_category"] = device.apply(lambda d: d.get("deviceCategory"))
    df["traffic_medium"] = traffic.apply(lambda d: d.get("medium"))
    df["traffic_source"] = traffic.apply(lambda d: d.get("source"))
    # market proxy - geography stands in for "market" since there's no product market field
    df["market"] = geo.apply(lambda d: d.get("country"))

    # Sparse fields (only present when non-zero) - default to 0
    df["sessions"] = totals.apply(lambda d: int(d.get("visits", 0)))
    df["events"] = totals.apply(lambda d: int(d.get("hits", 0)))
    df["pageviews"] = totals.apply(lambda d: int(d.get("pageviews", 0)))

    return df.drop(columns=["device", "trafficSource", "totals", "geoNetwork"])


def add_quarter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the calendar quarter and quarter-of-year (seasonality) columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["quarter"] = df["date"].dt.to_period("Q")
    # quarter_of_year (1-4) repeats every year, so it can capture seasonal effects
    # (e.g. Dec promotions, summer spikes) that the running `quarter` index can't -
    # `quarter` only captures a linear trend over time.
    df["quarter_of_year"] = df["date"].dt.quarter
    return df


def aggregate_visitor_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw rows to one row per visitor per quarter - the model's training grain."""
    features = (
        df.groupby(["fullVisitorId", "quarter"])
        .agg(
            quarter_of_year=("quarter_of_year", "first"),
            total_sessions=("sessions", "sum"),
            total_events=("events", "sum"),
            total_pageviews=("pageviews", "sum"),
            # most common category per visitor-quarter
            platform=("platform", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            device_category=("device_category", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            traffic_medium=("traffic_medium", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            traffic_source=("traffic_source", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            market=("market", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        )
        .reset_index()
    )

    # average events per session, guarding against divide-by-zero
    features["events_per_session"] = features["total_events"] / features["total_sessions"].replace(0, pd.NA)
    return features


def add_lag_and_target_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add prior-quarter lag features and the forecasting target (next quarter's events).

    Computed per visitor, sorted by quarter, so each row can only see its own history.
    """
    features = features.sort_values(["fullVisitorId", "quarter"]).reset_index(drop=True)
    grouped = features.groupby("fullVisitorId")

    features["total_events_prev_q"] = grouped["total_events"].shift(1)
    # 0 = "no observed history" (e.g. a visitor/SDK's first quarter), a real state rather
    # than a missing value - keep this imputation identical in any future scoring pipeline
    # to avoid train/serve skew.
    features["total_events_prev_q"] = features["total_events_prev_q"].fillna(0)
    features["total_events_qoq_change"] = features["total_events"] - features["total_events_prev_q"]

    # Target: next quarter's total_events. A visitor's most recent quarter has no future
    # quarter yet, so its target is NaN - drop/hold these out at modelling time, don't impute.
    features["target_next_q_events"] = grouped["total_events"].shift(-1)

    return features


def build_quarterly_active_users(features: pd.DataFrame) -> pd.DataFrame:
    """Quarter-level (not per-visitor) feature: distinct active visitors per quarter."""
    return features.groupby("quarter")["fullVisitorId"].nunique().rename("active_users").reset_index()


def build_visitor_quarter_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline: raw rows -> visitor x quarter features + target.

    Returns (features, quarterly_active_users).
    """
    df = flatten_raw_columns(raw_df)
    df = add_quarter_columns(df)
    features = aggregate_visitor_quarter(df)
    features = add_lag_and_target_features(features)
    quarterly_active_users = build_quarterly_active_users(features)
    return features, quarterly_active_users
