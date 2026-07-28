# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a data science / ML repository for projecting SMAPI (Vodafone internal tool) events
data to forecast per-quarter costs for users. The goal is regression-based cost forecasting
from events data.

## Purpose and how to work here

This repo is both a work deliverable and a personal learning project for the user. When
building or discussing regression models, don't just hand over code — explain the model
evaluation reasoning alongside it (which metrics apply and why, how to interpret them,
overfitting/leakage/cross-validation pitfalls, baseline comparisons). Evaluation is the area
the user most wants to strengthen, so treat it as a first-class part of every answer, not an
afterthought.

## Current state

This repository is a fresh scaffold: `src/`, `scripts/`, `tests/`, `notebooks/`, and `data/` all
exist but are currently empty, and `requirement.txt` has no pinned dependencies yet. There is no
build, lint, or test tooling configured yet — a `.venv` exists but only `pip` is installed.

When adding the first real code, set up dependencies in `requirement.txt` (or migrate to
`pyproject.toml` if preferred) and establish the actual project layout — check with the user
before assuming a structure, since none exists to infer from yet.

## Directory intent (from repo scaffold)

- `src/` — library/pipeline code (feature engineering, model training, forecasting logic)
- `scripts/` — standalone entry points (e.g. running a forecast, data pulls)
- `notebooks/` — exploratory analysis
- `tests/` — test suite
- `data/` — local data files (check `.gitignore` before adding data — most data patterns are
  likely excluded from version control)
