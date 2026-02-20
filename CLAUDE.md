# CLAUDE.md — OpSource Containment Division Calculator

## CRITICAL: Always Push After Committing
**Every `git commit` MUST be followed immediately by `git push origin main`.**
Streamlit Cloud deploys from GitHub — commits that are not pushed are invisible to the user.
Never end a work session without confirming `git push` ran successfully.

## Environment
- Python: `C:/Users/ryanr/AppData/Local/Programs/Python/Python312/python.exe`
- Compile check: `C:/Users/ryanr/AppData/Local/Programs/Python/Python312/python.exe -m py_compile <file>`
- Git bash paths: use `/c/3PC-Financials` (Unix-style), not `C:/3PC-Financials`
- Streamlit Cloud repo: `https://github.com/ryanrajnath/3PC-Financials`
- Always run py_compile on both `engine.py` and `app.py` before committing

## Workflow Pattern
1. Read the file before editing (required by Edit tool)
2. Make changes with Edit tool
3. Compile: `python -m py_compile engine.py && python -m py_compile app.py`
4. `git add <files> && git commit -m "..."`
5. `git push origin main`  ← NEVER SKIP THIS

## Sub-Agent Pattern
- General-purpose agents: use Read/Write/Edit tools to modify files — they cannot run Bash
- Main session handles: py_compile verification, git commit, git push
- Always verify py_compile passes before committing agent output

## Streamlit Anti-Patterns to Avoid
- **Never use `key=` on a selectbox that calls `st.rerun()` inside its if-block** — the key persists the selected value in session_state, causing an infinite rerun loop on every subsequent render. Solution: omit the key (widget resets to default on rerun naturally).
- **Never call `del st.session_state["widget_key"]` after the widget is instantiated** in the same script run — raises `StreamlitAPIException`.
- **Always call `run_and_store()` when loading a preset** so the model updates immediately without requiring a manual Run click.
- Month/period labels must use `"M1"`, `"M2"` format (not `"YYYY-MM"`) — Plotly parses date strings and shifts them to 1970 epoch when used with `add_vline`.

## Project Structure
- `engine.py` — all financial model logic (weekly engine, monthly rollup, sensitivity)
- `app.py` — Streamlit UI only; imports from engine.py
- `export.py` — Excel export helper
- `.streamlit/config.toml` — dark theme config

## Key Design Decisions
- All assumptions live in `st.session_state.assumptions` (dict)
- Headcount plan is `st.session_state.headcount_plan` (list of 120 ints, one per month)
- Model runs via `run_and_store()` which calls `run_model()` and stores results in session_state
- Universal date range slider stored in `st.session_state.global_range_lo/hi`
- Presets defined in `PRESETS` dict in app.py; use `_build_hc(rules)` for headcount
- Per-inspector overhead scaling: `software_per_inspector`, `insurance_per_inspector`, `travel_per_inspector`, `recruiting_per_inspector` (all default 0.0)

## Default Assumptions (from business context)
- ST bill rate: $39/hr | OT premium: 1.5x (passthrough mode)
- Inspector wage: $20/hr | Burden: 30%
- Team lead ratio: 1 per 12 inspectors
- Net days: 60 | LOC APR: 8.5% | Max LOC: $1M
- GM fully loaded: $117K/yr | Management burden: 25%
