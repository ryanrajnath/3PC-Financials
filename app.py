"""
Containment Division Calculator — OpSource
"""
from __future__ import annotations
import hashlib, json, warnings
from datetime import date, datetime
from math import ceil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from engine import (
    default_assumptions, default_headcount,
    run_model, run_sensitivity, find_breakeven_inspectors,
)
from export import build_excel

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Containment Division Calculator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & Layout ─────────────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; height: 0; }
.block-container { padding-top: 0.75rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Tab Navigation ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #E2E8F0 !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1D4ED8 !important;
    background: #F1F5F9 !important;
}
.stTabs [aria-selected="true"][data-baseweb="tab"] {
    color: #1D4ED8 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #1D4ED8 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── Section Headers ─────────────────────────────────────────────────── */
.sec-hdr {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748B;
    padding: 0.5rem 0 0.25rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 0.75rem;
    margin-top: 0.5rem;
}

/* ── KPI Cards ───────────────────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 2px;
}

/* ── App Header ──────────────────────────────────────────────────────── */
.app-header {
    padding: 12px 0 16px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 12px;
}
.app-title {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.3px;
}
.app-subtitle {
    font-size: 12px;
    color: #64748B;
    margin-top: 2px;
}

/* ── Status Indicators ───────────────────────────────────────────────── */
.status-stale {
    font-size: 12px;
    color: #D97706;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
}
.status-ok {
    font-size: 12px;
    color: #059669;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
}

/* ── Info/callout boxes ──────────────────────────────────────────────── */
.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 16px 20px;
    color: #1E40AF;
    font-size: 14px;
    margin: 12px 0;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #1D4ED8 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1E40AF !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

/* ── Metric widgets ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

TPL = "plotly_white"
PC  = ["#1D4ED8", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2", "#EA580C", "#65A30D"]

# Plotly chart config — enables scroll-to-zoom and pan on all charts
_CHART_CONFIG = {"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ── Session state ─────────────────────────────────────────────────────────────
if "assumptions"    not in st.session_state: st.session_state.assumptions    = default_assumptions()
if "headcount_plan" not in st.session_state: st.session_state.headcount_plan = default_headcount()
if "results"        not in st.session_state: st.session_state.results        = None
if "run_hash"       not in st.session_state: st.session_state.run_hash       = None
if "run_ts"         not in st.session_state: st.session_state.run_ts         = None
if "bootstrapped"   not in st.session_state: st.session_state.bootstrapped   = False
if "preset_version" not in st.session_state: st.session_state.preset_version  = 0
if "view_mode"      not in st.session_state: st.session_state.view_mode       = "Level 1 — Investor"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash_inputs():
    d = json.dumps({
        "a": {k: str(v) for k, v in st.session_state.assumptions.items()},
        "h": st.session_state.headcount_plan,
    }, sort_keys=True)
    return hashlib.md5(d.encode()).hexdigest()

def fmt_dollar(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def kpi(col, label, value, sub=None):
    sub_h = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_h}</div>',
        unsafe_allow_html=True
    )

def section(label):
    st.markdown(f'<div class="sec-hdr">{label}</div>', unsafe_allow_html=True)

# ── Valuation curve (EV/EBITDA multiples by size tier) ────────────────────
# Calibrated from: UHY 2024, First Page Sage 2025, Raincatcher, Kroll M&A data
# Reflects quality inspection / containment staffing (above plain light industrial,
# below pure professional staffing). Midpoint of market range at each tier.
_VALUATION_CURVE = [
    (0,          2.5),   # EBITDA < $250K  — main-street sale, high key-person risk
    (250_000,    3.5),   # $250K – $500K   — small business; limited buyer pool
    (500_000,    4.5),   # $500K – $1M     — lower MM entry; M&A process viable
    (1_000_000,  5.0),   # $1M – $2M       — light industrial / inspection LMM range
    (2_000_000,  6.0),   # $2M – $5M       — mid-market; PE add-on / platform interest
    (5_000_000,  7.5),   # $5M+            — strategic value; recurring revenue premium
]

def _ev_multiple(ebitda: float) -> float:
    """Interpolated EV/EBITDA multiple based on trailing EBITDA size."""
    if ebitda <= 0:
        return _VALUATION_CURVE[0][1]
    floors = [c[0] for c in _VALUATION_CURVE]
    mults  = [c[1] for c in _VALUATION_CURVE]
    if ebitda >= floors[-1]:
        return mults[-1]
    for i in range(len(floors) - 1):
        if floors[i] <= ebitda < floors[i + 1]:
            t = (ebitda - floors[i]) / (floors[i + 1] - floors[i])
            return mults[i] + t * (mults[i + 1] - mults[i])
    return mults[0]

def run_and_store():
    with st.spinner("Calculating…"):
        try:
            w, m, q = run_model(st.session_state.assumptions, st.session_state.headcount_plan)
            st.session_state.results  = (w, m, q)
            st.session_state.run_hash = _hash_inputs()
            st.session_state.run_ts   = datetime.now().strftime("%I:%M %p")
        except Exception as e:
            st.error(f"Calculation failed: {e}")
            import traceback; st.code(traceback.format_exc())
            st.session_state.results = None

def results_ready(): return st.session_state.results is not None
def is_stale():      return results_ready() and st.session_state.run_hash != _hash_inputs()

def _select(df, cols):
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

def _line(df, x, ys, names, title, pct_y=False):
    fig = go.Figure()
    for y, nm, c in zip(ys, names, PC):
        if y in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[y], name=nm, mode="lines",
                                     line=dict(color=c, width=2)))
    fig.update_layout(template=TPL, title=title, height=300,
                      margin=dict(l=10, r=10, t=36, b=60),
                      legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
                      yaxis=dict(tickformat=".0%" if pct_y else "$,.0f"))
    fig.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
    return fig

def _bar(df, x, ys, names, title, tickformat="$,.0f"):
    fig = go.Figure()
    for y, nm, c in zip(ys, names, PC):
        if y in df.columns:
            fig.add_trace(go.Bar(x=df[x], y=df[y], name=nm, marker_color=c))
    fig.update_layout(template=TPL, title=title, barmode="stack", height=300,
                      margin=dict(l=10, r=10, t=36, b=60),
                      legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
                      yaxis=dict(tickformat=tickformat))
    return fig

def _fmt_table(df, dollar_cols=None, pct_cols=None, highlight_neg=None,
               highlight_loc=None, max_loc_val=None, height=380):
    fmt = {}
    for c in (dollar_cols or []):
        if c in df.columns: fmt[c] = "${:,.0f}"
    for c in (pct_cols or []):
        if c in df.columns: fmt[c] = "{:.1%}"
    styled = df.style.format(fmt)

    def _row_style(row):
        styles = [""] * len(row)
        if highlight_neg and highlight_neg in row.index and pd.notna(row[highlight_neg]) and row[highlight_neg] < 0:
            styles[row.index.get_loc(highlight_neg)] = "color: #E05252; font-weight: bold"
        if highlight_loc and highlight_loc in row.index and max_loc_val and pd.notna(row[highlight_loc]):
            if row[highlight_loc] > max_loc_val * 0.8:
                styles[row.index.get_loc(highlight_loc)] = "background-color: #3D2D00; color: #F0A843"
        return styles

    if highlight_neg or highlight_loc:
        styled = styled.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, height=height)

def _apply_range(mo):
    """Apply the universal date range stored in session state to a monthly dataframe."""
    lo = st.session_state.get("global_range_lo", 1)
    hi = st.session_state.get("global_range_hi", len(mo))
    return mo[(mo["month_idx"] >= lo - 1) & (mo["month_idx"] <= hi - 1)]


def _render_loc_chart(mo, wdf, a):
    """Build and render the Credit Line / Cash / AR chart (shared by L1 and Brief tab)."""
    def _first_month(df, col):
        rows = df[df[col] > 0]
        if not len(rows):
            return None, None
        idx = rows.index[0]
        label = rows.iloc[0]["period"]
        return idx, label

    m_tl, m_tl_lbl = _first_month(mo, "team_leads_avg")
    m_oc, m_oc_lbl = _first_month(mo, "n_opscoord")
    m_fs, m_fs_lbl = _first_month(mo, "n_fieldsup")
    m_rm, m_rm_lbl = _first_month(mo, "n_regionalmgr")

    fig_loc = go.Figure()
    fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["loc_end"], name="Credit Line Balance",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)", line=dict(color=PC[3], width=2)))
    fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["ar_end"], name="Accounts Receivable",
        line=dict(color=PC[0], width=2)))
    fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["cash_end"], name="Cash on Hand",
        line=dict(color=PC[1], width=2)))
    fig_loc.add_hline(y=float(a["max_loc"]), line_dash="dot", line_color=PC[3],
                      annotation_text="Credit Limit", annotation_font_color=PC[3])

    _milestones = [
        (m_tl, m_tl_lbl, "1st Team Lead", PC[0]),
        (m_oc, m_oc_lbl, "1st Ops Coord", PC[4]),
        (m_fs, m_fs_lbl, "1st Field Sup", PC[2]),
        (m_rm, m_rm_lbl, "1st Reg. Mgr",  PC[5]),
    ]
    for _mi, _mp, _ml, _mc in _milestones:
        if _mi is not None:
            fig_loc.add_shape(
                type="line", x0=_mp, x1=_mp, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(dash="dot", color=_mc, width=1)
            )
            fig_loc.add_annotation(
                x=_mp, y=0.98, xref="x", yref="paper",
                text=f"{_ml} ({_mp})", font=dict(color=_mc, size=10),
                showarrow=False, textangle=-90,
                xanchor="right", yanchor="top"
            )

    fig_loc.update_layout(template=TPL, height=340, margin=dict(l=10, r=10, t=10, b=60),
                          legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"), yaxis=dict(tickformat="$,.0f"),
                          xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
    st.plotly_chart(fig_loc, use_container_width=True, config=_CHART_CONFIG)


def _render_ebitda_chart(mo_in):
    """Build and render the EBITDA & Implied Company Valuation chart (shared by L1 and Brief tab)."""
    mo_cf = mo_in.copy()
    mo_cf["ttm_ebitda"] = mo_cf["ebitda"].rolling(12, min_periods=1).sum()
    mo_cf["implied_ev"] = mo_cf["ttm_ebitda"].apply(
        lambda e: e * _ev_multiple(e) if e > 0 else 0
    )

    fig_cf = go.Figure()

    fig_cf.add_trace(go.Scatter(
        x=mo_cf["period"], y=mo_cf["ttm_ebitda"],
        name="Trailing 12-Mo EBITDA", mode="lines",
        line=dict(color=PC[0], width=2),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        yaxis="y1",
    ))

    fig_cf.add_trace(go.Scatter(
        x=mo_cf["period"], y=mo_cf["implied_ev"],
        name="Implied Company Value (EV)", mode="lines",
        line=dict(color=PC[2], width=2.5, dash="dot"),
        yaxis="y2",
    ))

    fig_cf.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1, yref="y1")

    _pos_rows = mo_cf[mo_cf["ttm_ebitda"] > 0]
    if len(_pos_rows):
        _cross_period = _pos_rows.iloc[0]["period"]
        fig_cf.add_shape(
            type="line", x0=_cross_period, x1=_cross_period, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color=PC[1], width=1)
        )
        fig_cf.add_annotation(
            x=_cross_period, y=0.98, xref="x", yref="paper",
            text=f"TTM EBITDA positive: {_cross_period}", font=dict(color=PC[1], size=10),
            showarrow=False, textangle=-90,
            xanchor="right", yanchor="top"
        )

    fig_cf.update_layout(
        template=TPL, height=340, margin=dict(l=10, r=10, t=10, b=60),
        legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
        yaxis=dict(tickformat="$,.0f", title="Trailing 12-Mo EBITDA ($)", side="left"),
        yaxis2=dict(
            tickformat="$,.0f", title="Implied EV ($)",
            overlaying="y", side="right", showgrid=False,
        ),
    )
    st.plotly_chart(fig_cf, use_container_width=True, config=_CHART_CONFIG)

    _ss_ttm = mo_cf[mo_cf["ttm_ebitda"] > 0]["ttm_ebitda"].iloc[-3:].mean() if mo_cf["ttm_ebitda"].gt(0).any() else 0
    _ss_ev  = _ss_ttm * _ev_multiple(_ss_ttm) if _ss_ttm > 0 else 0
    _ss_mult = _ev_multiple(_ss_ttm) if _ss_ttm > 0 else 0
    if _ss_ev > 0:
        st.caption(
            f"At steady state (~${_ss_ttm:,.0f} TTM EBITDA), implied exit value is "
            f"**${_ss_ev:,.0f}** at {_ss_mult:.1f}x trailing EBITDA. "
            f"Multiple expands as EBITDA grows — calibrated from M&A market data for "
            f"containment/inspection staffing (UHY, First Page Sage, 2024–25)."
        )


# ── Scenario presets ──────────────────────────────────────────────────────────
def _build_hc(rules):
    hc = [0] * 120
    for start, end, val in rules:
        for i in range(start - 1, min(end, 120)):
            hc[i] = val
    return hc

PRESETS = {
    "Conservative": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate": 35.0, "ot_hours": 8, "burden": 0.35,
            "team_lead_ratio": 12, "lead_wage": 22.0, "lead_ot_hours": 8,
            "net_days": 120, "apr": 0.095, "max_loc": 750_000,
            "cash_buffer": 25_000, "initial_cash": 0,
            "software_monthly": 300, "recruiting_monthly": 500,
            "insurance_monthly": 1_000, "travel_monthly": 300},
        "headcount": _build_hc([(1,1,5),(2,2,10),(3,3,15),(4,6,15),(7,120,15)]),
    },
    "Base Case": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate": 39.0, "ot_hours": 10, "burden": 0.30,
            "team_lead_ratio": 12, "lead_wage": 25.0, "lead_ot_hours": 10,
            "net_days": 90, "apr": 0.085, "max_loc": 1_000_000,
            "cash_buffer": 25_000, "initial_cash": 0,
            "software_monthly": 500, "recruiting_monthly": 1_000,
            "insurance_monthly": 1_500, "travel_monthly": 500},
        "headcount": _build_hc([(1,1,5),(2,2,10),(3,3,15),(4,4,20),(5,5,25),(6,120,30)]),
    },
    "Aggressive": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate": 42.0, "ot_hours": 15, "burden": 0.28,
            "inspector_wage": 21.0, "lead_wage": 26.0, "lead_ot_hours": 15,
            "team_lead_ratio": 12,
            "gm_loaded_annual": 130_000, "opscoord_base": 70_000,
            "fieldsup_base": 75_000, "regionalmgr_base": 120_000,
            "net_days": 60, "apr": 0.075, "max_loc": 2_000_000,
            "cash_buffer": 50_000, "initial_cash": 50_000,
            "software_monthly": 1_000, "recruiting_monthly": 3_000,
            "insurance_monthly": 2_500, "travel_monthly": 1_500},
        "headcount": _build_hc([(1,2,25),(3,4,50),(5,6,75),(7,9,100),
                                 (10,12,125),(13,24,150),(25,60,175)]),
    },
    "25–500 Employees over 60 Months": {
        "assumptions": {**default_assumptions(),
            # Billing — at scale, slightly above base rate; OT stays meaningful
            "st_bill_rate": 42.0,
            "ot_hours": 10,
            "ot_bill_mode": "passthrough",
            # Labor costs — scale justifies tighter burden via safety maturity
            "inspector_wage": 21.0,
            "burden": 0.28,
            "lead_wage": 26.0,
            "lead_st_hours": 40,
            "lead_ot_hours": 10,
            "team_lead_ratio": 12,
            "lead_bill_premium": 1.0,
            # Management — enterprise-grade compensation
            "gm_loaded_annual": 150_000,
            "opscoord_base": 72_000,
            "fieldsup_base": 78_000,
            "regionalmgr_base": 125_000,
            "mgmt_burden": 0.25,
            # AR / cash
            "net_days": 120,
            "apr": 0.08,
            "max_loc": 10_000_000,   # $10M LOC needed for 500-person AR float
            "cash_buffer": 100_000,
            "initial_cash": 50_000,
            # Fixed overhead baseline (per-inspector components handle the scaling)
            "software_monthly": 1_000,
            "recruiting_monthly": 2_000,
            "insurance_monthly": 3_000,
            "travel_monthly": 1_000,
            # Per-inspector overhead scaling — these scale all overhead with headcount
            "software_per_inspector": 18.0,     # $18/inspector/mo — workforce mgmt (e.g. Bullhorn, scheduling systems)
            "insurance_per_inspector": 12.0,    # $12/inspector/mo — GL/umbrella above workers comp
            "travel_per_inspector": 8.0,        # $8/inspector/mo — supervisor site visits, regional travel
            "recruiting_per_inspector": 15.0,   # $15/inspector/mo — job boards, agency fees, turnover at scale
            # Turnover / onboarding (higher-volume onboarding program)
            "inspector_onboarding_cost": 600.0,
            "inspector_turnover_rate": 1.2,     # 120% annual — high turnover is structural at this scale
            "mgmt_winddown_weeks": 12,
            # Bad debt — tighter collections team at scale brings this down
            "bad_debt_pct": 0.005,              # 0.5% — enterprise AR function tightens collections
            # Corp alloc — at 500 inspectors, OpSource corporate overhead allocation increases
            "corp_alloc_mode": "pct_revenue",
            "corp_alloc_pct": 0.02,             # 2% of revenue to parent company
            "corp_alloc_fixed": 0.0,
        },
        # Headcount ramp: 25 → 500 inspectors by M60, gradual S-curve
        "headcount": _build_hc([
            (1,  2,   25),   # M1-2:   Initial deployment — 25 inspectors
            (3,  4,   35),   # M3-4:   First growth phase
            (5,  6,   50),   # M5-6:   Expand to 2nd client
            (7,  8,   70),   # M7-8:   3rd client, first field supervisor
            (9,  10,  90),   # M9-10:  Approaching 100, first ops coordinator
            (11, 12, 115),   # M11-12: End of year 1 — 115 inspectors
            (13, 15, 140),   # Q1 Y2
            (16, 18, 170),   # Q2 Y2
            (19, 21, 200),   # Q3 Y2 — 200 milestone
            (22, 24, 235),   # Q4 Y2
            (25, 27, 270),   # Q1 Y3
            (28, 30, 305),   # Q2 Y3 — first regional manager
            (31, 33, 340),   # Q3 Y3
            (34, 36, 375),   # Q4 Y3 — 375 inspectors end of year 3
            (37, 40, 410),   # Q1 Y4
            (41, 44, 445),   # Q2 Y4
            (45, 48, 470),   # Q3 Y4
            (49, 52, 485),   # Q4 Y4
            (53, 56, 495),   # Q1 Y5
            (57, 60, 500),   # Q2 Y5 — 500 inspectors milestone
            (61,120, 500),   # Hold at 500
        ]),
    },
}

# ── Scenario drift detection ───────────────────────────────────────────────
if "active_preset" not in st.session_state: st.session_state.active_preset = "Base Case"
if "preset_assumptions" not in st.session_state: st.session_state.preset_assumptions = None

def _is_modified():
    """Returns True if current assumptions differ from the loaded preset's assumptions."""
    if st.session_state.preset_assumptions is None:
        return False
    cur = {k: str(v) for k, v in st.session_state.assumptions.items()}
    ref = {k: str(v) for k, v in st.session_state.preset_assumptions.items()}
    return cur != ref

# ── Auto-bootstrap on first load ──────────────────────────────────────────────
if not st.session_state.bootstrapped:
    p = PRESETS["Base Case"]
    st.session_state.assumptions       = p["assumptions"].copy()
    st.session_state.headcount_plan    = p["headcount"].copy()
    st.session_state.active_preset     = "Base Case"
    st.session_state.preset_assumptions = p["assumptions"].copy()
    st.session_state.bootstrapped      = True
    run_and_store()


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Header + KPIs
# ════════════════════════════════════════════════════════════════════════════

# ── Header + controls ─────────────────────────────────────────────────────
st.markdown("""
<div class="app-hdr">
  <div class="app-hdr-title">Containment Division Calculator</div>
  <div class="app-hdr-sub">OpSource &nbsp;·&nbsp; Containment Division &nbsp;·&nbsp; Weekly Cash Model</div>
</div>
""", unsafe_allow_html=True)

def _on_preset_change():
    choice = st.session_state._preset_sel
    if choice != "— load preset —":
        p = PRESETS[choice]
        st.session_state.assumptions        = p["assumptions"].copy()
        st.session_state.headcount_plan     = p["headcount"].copy()
        st.session_state.active_preset      = choice
        st.session_state.preset_assumptions = p["assumptions"].copy()
        st.session_state.preset_version     = st.session_state.get("preset_version", 0) + 1
        run_and_store()

# ── View mode toggle + controls ────────────────────────────────────────────
_vm_col, ctrl1, ctrl2, ctrl3 = st.columns([2, 3, 1, 2])
with _vm_col:
    _vm = st.radio(
        "View",
        ["Level 1 — Investor", "Level 2 — Operator"],
        index=0 if st.session_state.view_mode == "Level 1 — Investor" else 1,
        horizontal=True,
        label_visibility="collapsed",
        help="Level 1: investor summary only. Level 2: full model detail.",
    )
    st.session_state.view_mode = _vm
with ctrl1:
    _preset_label = st.session_state.get("active_preset", "Base Case")
    if _is_modified():
        _preset_label = f"{_preset_label} (Modified)"
    st.selectbox(
        "Scenario",
        options=["— load preset —"] + list(PRESETS.keys()),
        index=0,
        key="_preset_sel",
        on_change=_on_preset_change,
        help=f"Active: {_preset_label}",
    )
    if _is_modified():
        st.caption(f"*{_preset_label}*")
with ctrl2:
    if st.button("▶  Run", type="primary", use_container_width=True):
        run_and_store()
        st.rerun()
with ctrl3:
    if is_stale():
        st.markdown('<div class="status-stale">⚡ Inputs changed — click <b>Run</b></div>', unsafe_allow_html=True)
    elif st.session_state.run_ts:
        st.markdown(f'<div class="status-ok">✓ Calculated at {st.session_state.run_ts}</div>', unsafe_allow_html=True)

if results_ready():
    _, mo_h, _ = st.session_state.results

    # ── Universal date range slider (controls ALL tabs) ────────────────────
    _active = mo_h[mo_h["revenue"] > 0]
    _last   = int(_active["month_idx"].max()) + 1 if not _active.empty else 12
    _hi_def = min(_last + 3, len(mo_h))
    if "global_range_lo" not in st.session_state:
        st.session_state.global_range_lo = 1
    if "global_range_hi" not in st.session_state:
        st.session_state.global_range_hi = _hi_def

    # Clamp stored range to valid bounds (protects against preset switches changing dataset length)
    _n_opts = len(mo_h)
    st.session_state.global_range_lo = max(1, min(st.session_state.global_range_lo, _n_opts))
    st.session_state.global_range_hi = max(st.session_state.global_range_lo, min(st.session_state.global_range_hi, _n_opts))

    _rlo, _rhi = st.select_slider(
        "Date Range — applies to all tabs",
        options=list(range(1, len(mo_h) + 1)),
        value=(st.session_state.global_range_lo, st.session_state.global_range_hi),
        key="global_range_slider",
        help="Slide to zoom any range of months. Every chart, table, and KPI card across all tabs updates instantly.",
        label_visibility="collapsed",
    )
    st.session_state.global_range_lo = _rlo
    st.session_state.global_range_hi = _rhi

    # Period label for the selected range
    _sel_months = _rhi - _rlo + 1
    if _sel_months <= 24:
        _rng_label = f"M{_rlo} – M{_rhi} ({_sel_months} months)"
    elif _sel_months % 12 == 0:
        _rng_label = f"M{_rlo} – M{_rhi} ({_sel_months // 12} years)"
    else:
        _rng_label = f"M{_rlo} – M{_rhi} ({_sel_months // 12}Y {_sel_months % 12}M)"
    st.caption(f"📅 Showing **{_rng_label}** across all tabs")

    # Determine view mode
    _L1 = (st.session_state.view_mode == "Level 1 — Investor")

    # Top-level KPI bar — scoped to selected range (hidden in L1 mode)
    if not _L1:
        mo_top = _apply_range(mo_h)
        k1, k2, k3, k4, k5 = st.columns(5)
        peak_mo = mo_top.loc[mo_top["loc_end"].idxmax(), "period"] if mo_top["loc_end"].max() > 0 else "—"
        _n_mo_top = len(mo_top)
        _lbl_top  = f"{_n_mo_top}-Mo" if _n_mo_top <= 24 else (f"{_n_mo_top//12}-Yr" if _n_mo_top % 12 == 0 else f"{_n_mo_top//12}Y{_n_mo_top%12}M")
        kpi(k1, "Peak Credit Line",          fmt_dollar(mo_top["loc_end"].max()),                     peak_mo)
        kpi(k2, f"{_lbl_top} Revenue",       fmt_dollar(mo_top["revenue"].sum()),                     "billed (accrual)")
        kpi(k3, f"{_lbl_top} Net Income",    fmt_dollar(mo_top["ebitda_after_interest"].sum()),        "after interest")
        kpi(k4, "Total Borrowing Cost",      fmt_dollar(mo_top["interest"].sum()),                    "credit line interest")
        yr1 = mo_top[mo_top["month_idx"].between(_rlo - 1, _rlo + 10)]
        kpi(k5, "Year 1 Net Income",         fmt_dollar(yr1["ebitda_after_interest"].sum()),           "first 12 months of range")
    st.divider()


# ════════════════════════════════════════════════════════════════════════════
# VIEW MODE GATE
# ════════════════════════════════════════════════════════════════════════════
_L1 = (st.session_state.view_mode == "Level 1 — Investor")

if _L1:
    # ── Level 1 — Investor View ────────────────────────────────────────────
    if not results_ready():
        st.markdown('<div class="info-box">Select a scenario above and click <b>Run</b> to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a = st.session_state.assumptions
    mo = _apply_range(mo_full)

    # ── Verdict + KPI cards ────────────────────────────────────────────────
    _mo_peak_idx = int(mo["loc_end"].idxmax()) if len(mo) else 0
    _mo_peak_period = mo.loc[_mo_peak_idx, "period"] if len(mo) else "—"
    _peak_rng = float(mo["loc_end"].max()) if len(mo) else 0
    _ss_mo = mo.tail(12)
    _ss_ebitda = float(_ss_mo["ebitda_after_interest"].mean())
    _ss_annual = _ss_ebitda * 12
    _total_interest = float(mo["interest"].sum())
    _ropc = (_ss_annual / _peak_rng) if _peak_rng > 100 else 0
    _profitable_rows = mo[mo["ebitda_after_interest"] > 0]
    _be_month = _profitable_rows.iloc[0]["period"] if len(_profitable_rows) else "Not in range"

    _avg_fccr = float(mo[mo["fccr"] < 99]["fccr"].mean()) if "fccr" in mo.columns and len(mo[mo["fccr"] < 99]) else 0
    _min_fccr = float(mo[mo["fccr"] < 99]["fccr"].min()) if "fccr" in mo.columns and len(mo[mo["fccr"] < 99]) else 0
    _max_loc = float(a.get("max_loc", 1_000_000))
    n_loc = weekly_df["warn_loc_maxed"].sum() if "warn_loc_maxed" in weekly_df.columns else 0
    _burden = float(a.get("burden", 0.30))

    # Verdict banner
    if _ss_ebitda > 0 and _peak_rng > 0:
        st.success(
            f"**This scenario requires {fmt_dollar(_peak_rng)} in credit** (peaks at {_mo_peak_period}), "
            f"reaches profitability at **{_be_month}**, and generates "
            f"**{fmt_dollar(_ss_ebitda)}/month** at steady state — "
            f"a **{_ropc:.1f}x annualized return** on peak capital."
        )
    else:
        st.error("No profitability in selected range. Adjust bill rate, headcount, or burden % and re-run.")

    # 4 KPI cards
    _l1k1, _l1k2, _l1k3, _l1k4 = st.columns(4)
    kpi(_l1k1, "Peak Credit Required",  fmt_dollar(_peak_rng),   f"due by {_mo_peak_period}")
    kpi(_l1k2, "First Profitable Month", _be_month,              "net income after interest")
    kpi(_l1k3, "Steady-State / Month",  fmt_dollar(_ss_ebitda),  "avg last 12 months of range")
    kpi(_l1k4, "Return on Peak Capital", f"{_ropc:.1f}x",        "annualized EBITDA / peak credit")
    st.divider()

    # ── 2 charts side by side ─────────────────────────────────────────────
    _c1, _c2 = st.columns(2)
    with _c1:
        section("Credit Line, Cash & Accounts Receivable")
        _render_loc_chart(mo, weekly_df, a)
    with _c2:
        section("EBITDA & Implied Company Valuation")
        _render_ebitda_chart(mo)

    st.divider()

    # ── Risk callouts ─────────────────────────────────────────────────────
    _has_risks = n_loc or _peak_rng > _max_loc or _burden > 0.35 or (_min_fccr > 0 and _min_fccr < 1.1)
    if _has_risks:
        section("Risk Flags")
        if _peak_rng > _max_loc:
            st.error(
                f"**Peak draw of {fmt_dollar(_peak_rng)} exceeds your {fmt_dollar(_max_loc)} credit facility.** "
                "The business is underfunded at these assumptions — increase the credit line or reduce ramp speed."
            )
        elif n_loc:
            st.warning(
                f"**Credit line hit its maximum in {n_loc} week(s).** "
                "Consider a larger facility or slower headcount ramp."
            )
        if _burden > 0.35:
            st.warning(
                f"**Burden rate {_burden:.0%} is above 35%.** "
                "Verify workers comp and benefits assumptions — this compresses gross margin significantly."
            )
        if _min_fccr > 0 and _min_fccr < 1.1:
            st.warning(
                f"**Minimum FCCR {_min_fccr:.2f}x is below lender threshold of 1.1x.** "
                "Tighten terms or increase headcount/rates before approaching lenders."
            )

    # ── The Four Questions ────────────────────────────────────────────────
    st.divider()
    section("The Four Questions")
    qa1, qa2 = st.columns(2)
    with qa1:
        if _ss_ebitda > 0:
            st.success(
                f"**Does it work?** Yes — at steady state this division generates "
                f"{fmt_dollar(_ss_ebitda)}/month ({fmt_dollar(_ss_annual)}/year) net of all costs and interest."
            )
        else:
            st.error(
                "**Does it work?** Not at current assumptions. "
                "Adjust bill rate, headcount, or payment terms."
            )
        _max_insp = max(st.session_state.headcount_plan)
        _max_ttm  = mo_full["ebitda"].rolling(12, min_periods=1).sum().max() if len(mo_full) else 0
        _max_ev   = _max_ttm * _ev_multiple(_max_ttm) if _max_ttm > 0 else 0
        if _max_ev > 0:
            st.info(
                f"**How big can this become?** At {_max_insp} inspectors, "
                f"TTM EBITDA reaches {fmt_dollar(_max_ttm)} with an implied exit value of "
                f"{fmt_dollar(_max_ev)} ({_ev_multiple(_max_ttm):.1f}x EBITDA)."
            )
    with qa2:
        _nd_val = int(a.get("net_days", 60))
        st.info(
            f"**What capital is required?** A {fmt_dollar(_max_loc)} credit facility sized for "
            f"Net {_nd_val} payment terms. Peak draw: {fmt_dollar(_peak_rng)} at {_mo_peak_period}."
        )
        st.info(
            "**Can the team execute?** The GM runs day-to-day. Supervisor and manager layers trigger "
            "automatically as headcount scales. William Renfrow holds capital and credit — "
            "no operating role required."
        )
    # L1 view is complete — stop here so tab code below does not execute
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# TABS  (Level 2 — Operator View only)
# ════════════════════════════════════════════════════════════════════════════
tab_brief, tab_inputs, tab_detail = st.tabs([
    "◎  Investor Brief", "⚙  Inputs", "≋  Detail"
])


# ════════════════════════════════════════════════════════════════════════════
# INPUTS — 8 key levers upfront, everything else collapsed
# ════════════════════════════════════════════════════════════════════════════
with tab_inputs:
    a  = st.session_state.assumptions
    pv = st.session_state.get("preset_version", 0)
    st.caption("Adjust any value below, then click **Run** above to update results.")

    # ── 8 Key Inputs — always visible ──────────────────────────────────────
    section("Key Assumptions")
    _key_l, _key_r = st.columns(2)

    with _key_l:
        a["st_bill_rate"] = st.slider(
            "Bill Rate ($/hr)",
            min_value=28.0, max_value=65.0, value=float(a["st_bill_rate"]),
            step=0.50, format="$%.2f",
            help="Hourly rate charged to the customer per inspector during regular work hours.",
            key=f"kl_bill_rate_{pv}"
        )
        _burden_pct = st.slider(
            "Burden (%)", min_value=15, max_value=55,
            value=int(round(a["burden"] * 100)), step=1, format="%d%%",
            help="Employer cost on top of wages: FICA, unemployment, workers' comp, benefits. ~30% typical.",
            key=f"kl_burden_{pv}"
        )
        a["burden"] = _burden_pct / 100.0
        a["net_days"] = st.slider(
            "Net Payment Terms (days)",
            min_value=15, max_value=180, value=int(a["net_days"]), step=5,
            help="How long until customers pay after you send the invoice. Net 60 is standard in containment.",
            key=f"kl_net_days_{pv}"
        )

    with _key_r:
        a["max_loc"] = st.number_input(
            "Max Credit Line ($)", min_value=0.0,
            value=float(a["max_loc"]), step=50_000., format="%.0f",
            help="Your bank's credit limit. The model warns when cash needs exceed this.",
            key=f"kl_max_loc_{pv}"
        )
        a["inspector_wage"] = st.number_input(
            "Inspector Wage ($/hr)", min_value=10.0, max_value=50.0,
            value=float(a["inspector_wage"]), step=0.25, format="%.2f",
            help="Base hourly pay before employer taxes and benefits.",
            key=f"kl_insp_wage_{pv}"
        )
        a["start_date"] = st.date_input(
            "Start Date", value=a["start_date"],
            help="First day of the model. All months and years calculate forward from here.",
            key=f"kl_start_date_{pv}"
        )

    # Per-inspector margin caption
    if a.get("ot_bill_mode", "passthrough") == "passthrough":
        _ot_bill = a["st_bill_rate"] + (a["inspector_wage"] * (a.get("ot_pay_multiplier", 1.5) - 1.0) * (1.0 + a["burden"]))
    else:
        _ot_bill = a["st_bill_rate"] * a["ot_bill_premium"]
    _util    = float(a.get("inspector_utilization", 1.0))
    _ob_wk   = float(a.get("inspector_onboarding_cost", 500)) / max(1, int(a.get("inspector_avg_tenure_weeks", 26)))
    _wk_rev  = _util * (a["st_hours"] * a["st_bill_rate"] + a["ot_hours"] * _ot_bill)
    _wk_cost = (a["st_hours"] * a["inspector_wage"] * (1 + a["burden"]) +
                a["ot_hours"] * a["inspector_wage"] * a["ot_pay_multiplier"] * (1 + a["burden"]) + _ob_wk)
    _margin  = (_wk_rev - _wk_cost) / _wk_rev if _wk_rev else 0
    st.caption(f"Per inspector/week: **${_wk_rev:,.0f} rev** · **${_wk_cost:,.0f} cost** · **{_margin:.0%} margin**")

    st.divider()

    # ── Headcount Ramp ─────────────────────────────────────────────────────
    section("Headcount Ramp Plan")
    hc = st.session_state.headcount_plan

    c1, c2, c3, c4 = st.columns(4)
    fv = c1.number_input("Inspectors", 0, 10_000, 25, step=5, key="inp_fv",
        help="Number of inspectors to staff during this period.")
    ff = c2.number_input("From Month", 1, 120, 1, step=1, key="inp_ff",
        help="First month to fill. Month 1 = your model start date.")
    ft = c3.number_input("To Month",   1, 120, 12, step=1, key="inp_ft",
        help="Last month to fill (inclusive). Month 12 = end of Year 1.")
    if c4.button("Apply", use_container_width=True, key="inp_apply",
                 help="Sets inspector count for the selected month range."):
        for i in range(int(ff) - 1, int(ft)):
            hc[i] = int(fv)
        st.session_state.headcount_plan = hc
        st.rerun()

    month_labels = [f"M{i+1}" for i in range(120)]
    lo_hc, hi_hc = st.select_slider("Show months", options=list(range(1, 121)), value=(1, 24), key="inp_hc_rng")
    hc_prev = pd.DataFrame({"period": month_labels, "inspectors": hc}).iloc[lo_hc - 1:hi_hc]
    fig_hc_inp = px.bar(hc_prev, x="period", y="inspectors", template=TPL,
                        title="Inspectors Staffed per Month", color_discrete_sequence=[PC[0]])
    fig_hc_inp.update_layout(height=200, margin=dict(l=10,r=10,t=36,b=10))
    st.plotly_chart(fig_hc_inp, use_container_width=True, config=_CHART_CONFIG)

    hc_df  = pd.DataFrame({"Period": month_labels, "Inspectors": hc})
    edited = st.data_editor(hc_df, column_config={
        "Period":     st.column_config.TextColumn(disabled=True),
        "Inspectors": st.column_config.NumberColumn(min_value=0, max_value=10_000, step=1),
    }, use_container_width=True, height=400, num_rows="fixed")
    st.session_state.headcount_plan = edited["Inspectors"].tolist()

    st.divider()

    # ── Advanced Settings (collapsed expanders) ────────────────────────────
    section("Advanced Settings")

    with st.expander("Billing & Overtime", expanded=False):
        a["ot_hours"] = st.slider(
            "Planned OT Hours / Inspector / Week",
            min_value=0, max_value=25, value=int(a["ot_hours"]), step=1,
            help="Average overtime hours per inspector per week. OT billed at 1.5x regular rate.",
            key=f"adv_ot_hours_{pv}"
        )
        _ot_mode_options = ["Pass-through (cost+ recommended)", "Markup (1.5x bill rate)"]
        _ot_mode_idx = 0 if a.get("ot_bill_mode", "passthrough") == "passthrough" else 1
        _ot_mode_sel = st.radio(
            "OT Billing Method", _ot_mode_options, index=_ot_mode_idx, horizontal=True,
            help="Pass-through: OT bill = ST bill + (wage x 0.5 x burden). Markup: OT bill = ST bill x 1.5x. "
                 "Pass-through is how most containment contracts work.",
            key=f"adv_ot_mode_{pv}"
        )
        a["ot_bill_mode"] = "passthrough" if "Pass-through" in _ot_mode_sel else "markup"
        a["ot_pay_multiplier"] = st.number_input(
            "OT Pay Multiplier", min_value=1.0, max_value=3.0,
            value=float(a["ot_pay_multiplier"]), step=0.25, format="%.2f",
            help="Inspectors earn this multiple of base pay for overtime."
        )
        a["ot_bill_premium"] = st.number_input(
            "OT Billing Multiplier", min_value=1.0, max_value=3.0,
            value=float(a["ot_bill_premium"]), step=0.25, format="%.2f",
            help="Overtime billed at this multiple of the regular rate. (Only used in Markup mode)",
            key=f"adv_ot_bill_premium_{pv}"
        )
        _billing_options = ["Monthly (standard)", "Weekly (faster cash flow)"]
        _billing_idx = 0 if a.get("billing_frequency", "monthly") == "monthly" else 1
        _billing_sel = st.radio(
            "Invoice Frequency", _billing_options, index=_billing_idx, horizontal=True,
            help="Weekly billing reduces peak credit line need by $200-400K at 60 headcount.",
            key=f"adv_billing_mode_{pv}"
        )
        a["billing_frequency"] = "monthly" if _billing_sel == "Monthly (standard)" else "weekly"
        a["lead_bill_premium"] = st.number_input(
            "Team Lead Bill Rate Multiplier", min_value=1.0, max_value=2.0,
            value=float(a.get("lead_bill_premium", 1.0)), step=0.05, format="%.2f",
            help="Team leads billed at this multiple of the inspector rate. 1.0 = same rate.",
            key=f"adv_lead_bill_premium_{pv}"
        )
        a["st_hours"] = st.number_input(
            "Regular Hours / Inspector / Week", min_value=20, max_value=60,
            value=int(a["st_hours"]), step=1, format="%d",
            help="Standard work hours per inspector per week (not counting OT). Typically 40.",
            key=f"adv_st_hours_{pv}"
        )

    with st.expander("Team Leads", expanded=False):
        st.caption("Hourly supervisors billed to the client alongside inspectors.")
        a["team_lead_ratio"] = st.slider(
            "Inspectors per Team Lead",
            min_value=5, max_value=25, value=int(a["team_lead_ratio"]), step=1,
            help="One team lead per N inspectors (rounded up).",
            key=f"adv_tl_ratio_{pv}"
        )
        a["lead_wage"] = st.number_input(
            "Team Lead Hourly Wage ($/hr)", min_value=10.0, max_value=60.0,
            value=float(a["lead_wage"]), step=0.25, format="%.2f",
            help="Base hourly pay for team leads before burden.",
            key=f"adv_lead_wage_{pv}"
        )
        a["lead_ot_hours"] = st.number_input(
            "Team Lead OT Hours / Week", min_value=0, max_value=25,
            value=int(a["lead_ot_hours"]), step=1, format="%d",
            key=f"adv_lead_ot_hours_{pv}"
        )
        a["lead_st_hours"] = st.number_input(
            "Team Lead Regular Hours / Week", min_value=20, max_value=60,
            value=int(a["lead_st_hours"]), step=1, format="%d",
            key=f"adv_lead_st_hours_{pv}"
        )
        # Compute team lead margin vs inspector margin
        _tl_bill_st = a["st_bill_rate"] * float(a.get("lead_bill_premium", 1.0))
        _tl_cost_st = a["lead_wage"] * (1.0 + a["burden"])
        _insp_cost_st = a["inspector_wage"] * (1.0 + a["burden"])
        _tl_margin = (_tl_bill_st - _tl_cost_st) / _tl_bill_st if _tl_bill_st > 0 else 0
        _insp_margin = (a["st_bill_rate"] - _insp_cost_st) / a["st_bill_rate"] if a["st_bill_rate"] > 0 else 0
        if float(a.get("lead_bill_premium", 1.0)) < 1.15 and a["lead_wage"] > a["inspector_wage"]:
            st.warning(
                f"Team leads are margin-dilutive at current settings. "
                f"Inspector margin: **{_insp_margin:.0%}** -- Team lead margin: **{_tl_margin:.0%}**. "
                f"Consider setting Team Lead Bill Rate Multiplier > 1.10 or reducing lead wage."
            )

    with st.expander("Management Salaries", expanded=False):
        st.caption("Roles added automatically as inspector count grows. Salary persists during project gaps.")
        a["gm_loaded_annual"] = st.number_input(
            "GM — Total Annual Cost ($)", min_value=0.0,
            value=float(a["gm_loaded_annual"]), step=1_000., format="%.0f",
            help="Fully loaded GM cost (salary + bonus + benefits). Active from Month 1.",
            key=f"adv_gm_{pv}"
        )
        st.markdown("**Operations Coordinator**")
        _oc1, _oc2 = st.columns(2)
        a["opscoord_base"] = _oc1.number_input(
            "Base Salary ($)", min_value=0.0, value=float(a["opscoord_base"]),
            step=1_000., format="%.0f", key=f"adv_oc_sal_{pv}"
        )
        a["opscoord_span"] = _oc2.number_input(
            "Per N inspectors", min_value=10, value=int(a["opscoord_span"]),
            step=5, format="%d", key=f"adv_oc_sp_{pv}"
        )
        st.markdown("**Field Supervisor**")
        _fs1, _fs2 = st.columns(2)
        a["fieldsup_base"] = _fs1.number_input(
            "Base Salary ($)", min_value=0.0, value=float(a["fieldsup_base"]),
            step=1_000., format="%.0f", key=f"adv_fs_sal_{pv}"
        )
        a["fieldsup_span"] = _fs2.number_input(
            "Per N inspectors", min_value=5, value=int(a["fieldsup_span"]),
            step=5, format="%d", key=f"adv_fs_sp_{pv}",
            help="1 per 25 inspectors (each supervisor oversees one crew)."
        )
        st.markdown("**Regional Manager**")
        _rm1, _rm2 = st.columns(2)
        a["regionalmgr_base"] = _rm1.number_input(
            "Base Salary ($)", min_value=0.0, value=float(a["regionalmgr_base"]),
            step=1_000., format="%.0f", key=f"adv_rm_sal_{pv}"
        )
        a["regionalmgr_span"] = _rm2.number_input(
            "Per N inspectors", min_value=50, value=int(a["regionalmgr_span"]),
            step=10, format="%d", key=f"adv_rm_sp_{pv}"
        )
        _mgmt_burden_pct = st.slider(
            "Management Burden Rate (%)", min_value=10, max_value=40,
            value=int(round(a["mgmt_burden"] * 100)), step=1, format="%d%%",
            key=f"adv_mgmt_burden_{pv}"
        )
        a["mgmt_burden"] = _mgmt_burden_pct / 100.0
        a["mgmt_winddown_weeks"] = st.number_input(
            "Management Wind-Down Lag (weeks)", min_value=0, max_value=26,
            value=int(a.get("mgmt_winddown_weeks", 8)), step=1, format="%d",
            help="After inspectors drop to zero, salaried management stays on payroll for this many weeks.",
            key=f"adv_mgmt_winddown_{pv}"
        )
        def _loaded(base): return base * (1 + a["mgmt_burden"])
        st.caption(
            f"Fully loaded: OC **${_loaded(a['opscoord_base']):,.0f}** · "
            f"FS **${_loaded(a['fieldsup_base']):,.0f}** · "
            f"RM **${_loaded(a['regionalmgr_base']):,.0f}**"
        )

    with st.expander("Management Turnover", expanded=False):
        st.caption("Ongoing cost of replacing management roles — recruiting, screening, ramp-up productivity loss.")
        _oc_to_pct = st.slider("Ops Coordinator — Annual Turnover (%)", min_value=10, max_value=70,
            value=int(round(a.get("opscoord_turnover", 0.35) * 100)), step=1, format="%d%%", key=f"adv_oc_to_{pv}")
        a["opscoord_turnover"] = _oc_to_pct / 100.0
        a["opscoord_replace_cost"] = st.number_input(
            "Ops Coordinator — Replacement Cost ($)", min_value=0.0,
            value=float(a.get("opscoord_replace_cost", 8_000)), step=500., format="%.0f", key=f"adv_oc_rc_{pv}")
        _fs_to_pct = st.slider("Field Supervisor — Annual Turnover (%)", min_value=10, max_value=60,
            value=int(round(a.get("fieldsup_turnover", 0.25) * 100)), step=1, format="%d%%", key=f"adv_fs_to_{pv}")
        a["fieldsup_turnover"] = _fs_to_pct / 100.0
        a["fieldsup_replace_cost"] = st.number_input(
            "Field Supervisor — Replacement Cost ($)", min_value=0.0,
            value=float(a.get("fieldsup_replace_cost", 12_000)), step=500., format="%.0f", key=f"adv_fs_rc_{pv}")
        _rm_to_pct = st.slider("Regional Manager — Annual Turnover (%)", min_value=5, max_value=40,
            value=int(round(a.get("regionalmgr_turnover", 0.18) * 100)), step=1, format="%d%%", key=f"adv_rm_to_{pv}")
        a["regionalmgr_turnover"] = _rm_to_pct / 100.0
        a["regionalmgr_replace_cost"] = st.number_input(
            "Regional Manager — Replacement Cost ($)", min_value=0.0,
            value=float(a.get("regionalmgr_replace_cost", 25_000)), step=1_000., format="%.0f", key=f"adv_rm_rc_{pv}")

    with st.expander("Credit Line Settings", expanded=False):
        st.caption("Funds payroll while waiting for customers to pay.")
        _apr_pct = st.slider(
            "Annual Interest Rate (%)", min_value=4.0, max_value=20.0,
            value=round(float(a["apr"]) * 100, 2), step=0.25, format="%.2f%%",
            help="APR on the outstanding credit line balance.",
            key=f"adv_apr_{pv}"
        )
        a["apr"] = _apr_pct / 100.0
        a["initial_cash"] = st.number_input(
            "Starting Cash ($)", min_value=0.0,
            value=float(a["initial_cash"]), step=5_000., format="%.0f",
            key=f"adv_initial_cash_{pv}"
        )
        a["cash_buffer"] = st.number_input(
            "Minimum Cash Reserve ($)", min_value=0.0,
            value=float(a["cash_buffer"]), step=5_000., format="%.0f",
            help="Model keeps at least this much cash on hand, drawing credit line when needed.",
            key=f"adv_cash_buffer_{pv}"
        )
        a["auto_paydown"] = st.checkbox(
            "Auto-repay credit line when cash exceeds reserve",
            value=bool(a["auto_paydown"]),
            key=f"adv_auto_paydown_{pv}"
        )
        a["use_borrowing_base"] = st.checkbox(
            "Enable AR-based borrowing base (ABL facility)",
            value=bool(a.get("use_borrowing_base", False)),
            help="Limits credit draws to advance_rate x eligible AR balance.",
            key=f"adv_use_bb_{pv}"
        )
        if a["use_borrowing_base"]:
            _adv_pct = st.slider(
                "Advance Rate on AR (%)", min_value=70, max_value=92,
                value=int(round(float(a.get("ar_advance_rate", 0.85)) * 100)), step=1, format="%d%%",
                help="Lender advances this % of eligible AR (<90 days). Standard: 80-85% for staffing ABL.",
                key=f"adv_ar_advance_{pv}"
            )
            a["ar_advance_rate"] = _adv_pct / 100.0
            st.caption(f"Borrowing base = **{a['ar_advance_rate']:.0%} x AR balance** (capped at credit limit)")
        _mo_int = (a["apr"] / 12) * a["max_loc"]
        st.caption(f"Interest at full draw: **${_mo_int:,.0f}/month**")

    with st.expander("Risk & Quality", expanded=False):
        _util_pct = st.slider(
            "Inspector Utilization Rate (%)", min_value=50, max_value=100,
            value=int(round(float(a.get("inspector_utilization", 1.0)) * 100)), step=1, format="%d%%",
            help="Fraction of scheduled hours actually billed to client.",
            key=f"adv_util_{pv}"
        )
        a["inspector_utilization"] = _util_pct / 100.0
        _bd_pct = st.slider(
            "Bad Debt / Write-Off Rate (%)", min_value=0, max_value=5,
            value=max(0, int(round(float(a.get("bad_debt_pct", 0.01)) * 100))), step=1, format="%d%%",
            help="% of billed revenue never collected.",
            key=f"adv_bd_{pv}"
        )
        a["bad_debt_pct"] = _bd_pct / 100.0
        _insp_to_pct = st.slider(
            "Inspector Annual Turnover Rate (%)", min_value=50, max_value=200,
            value=int(round(float(a.get("inspector_turnover_rate", 1.0)) * 100)), step=10, format="%d%%",
            help="Annual churn rate for hourly inspectors. Containment industry: 80-150%.",
            key=f"adv_insp_to_{pv}"
        )
        a["inspector_turnover_rate"] = _insp_to_pct / 100.0
        a["inspector_onboarding_cost"] = st.number_input(
            "Onboarding Cost per Hire ($)", min_value=0.0,
            value=float(a.get("inspector_onboarding_cost", 500.0)), step=50., format="%.0f",
            help="One-time cost per new inspector: background check, drug screen, PPE, orientation.",
            key=f"adv_onboard_{pv}"
        )
        a["inspector_avg_tenure_weeks"] = st.number_input(
            "Average Inspector Tenure (weeks)", min_value=4, max_value=260,
            value=int(a.get("inspector_avg_tenure_weeks", 26)), step=4, format="%d",
            help="How long the average inspector stays. 26 weeks (~6 months) is typical.",
            key=f"adv_tenure_{pv}"
        )

    with st.expander("Fixed Overhead", expanded=False):
        st.caption("Charged every month regardless of headcount.")
        a["software_monthly"]   = st.number_input("Software & Technology ($/mo)", min_value=0.0,
            value=float(a["software_monthly"]), step=100., format="%.0f", key=f"adv_sw_{pv}")
        a["recruiting_monthly"] = st.number_input("Inspector Recruiting ($/mo)", min_value=0.0,
            value=float(a["recruiting_monthly"]), step=100., format="%.0f", key=f"adv_rec_{pv}")
        a["insurance_monthly"]  = st.number_input("Insurance ($/mo)", min_value=0.0,
            value=float(a["insurance_monthly"]), step=100., format="%.0f", key=f"adv_ins_{pv}")
        a["travel_monthly"]     = st.number_input("Travel & Field Expenses ($/mo)", min_value=0.0,
            value=float(a["travel_monthly"]), step=100., format="%.0f", key=f"adv_trav_{pv}")
        ca_mode = st.radio("Corporate Overhead Allocation",
            ["Fixed monthly amount", "Percentage of revenue"], horizontal=True,
            index=0 if a["corp_alloc_mode"] == "fixed" else 1, key=f"adv_ca_mode_{pv}")
        a["corp_alloc_mode"] = "fixed" if ca_mode == "Fixed monthly amount" else "pct_revenue"
        if a["corp_alloc_mode"] == "fixed":
            a["corp_alloc_fixed"] = st.number_input("Corporate Allocation ($/mo)", min_value=0.0,
                value=float(a["corp_alloc_fixed"]), step=500., format="%.0f", key=f"adv_ca_fixed_{pv}")
        else:
            _ca_pct_val = st.number_input("Corporate Allocation (% of revenue)", min_value=0.0,
                max_value=20.0, value=float(a["corp_alloc_pct"]) * 100, step=0.5, format="%.1f", key=f"adv_ca_pct_{pv}")
            a["corp_alloc_pct"] = _ca_pct_val / 100.0
        _total_fixed = (a["software_monthly"] + a["recruiting_monthly"] +
                        a["insurance_monthly"] + a["travel_monthly"] +
                        (a["corp_alloc_fixed"] if a["corp_alloc_mode"] == "fixed" else 0))
        st.caption(f"Total fixed overhead: **${_total_fixed:,.0f}/month**")

        st.markdown("**Per-Inspector Overhead Scaling** *(scales with active headcount)*")
        c1, c2 = st.columns(2)
        with c1:
            a["software_per_inspector"] = st.number_input(
                "Software $/inspector/mo", min_value=0.0, max_value=100.0,
                value=float(a.get("software_per_inspector", 0.0)), step=1.0,
                help="Workforce mgmt, scheduling, QA tools (e.g. Bullhorn, ClockShark)",
                key=f"adv_sw_pi_{pv}"
            )
            a["insurance_per_inspector"] = st.number_input(
                "Insurance $/inspector/mo", min_value=0.0, max_value=100.0,
                value=float(a.get("insurance_per_inspector", 0.0)), step=1.0,
                help="GL/umbrella above workers comp (WC is already in burden %)",
                key=f"adv_ins_pi_{pv}"
            )
        with c2:
            a["travel_per_inspector"] = st.number_input(
                "Travel $/inspector/mo", min_value=0.0, max_value=100.0,
                value=float(a.get("travel_per_inspector", 0.0)), step=1.0,
                help="Supervisor site visits, regional travel scales with field count",
                key=f"adv_trav_pi_{pv}"
            )
            a["recruiting_per_inspector"] = st.number_input(
                "Recruiting $/inspector/mo", min_value=0.0, max_value=100.0,
                value=float(a.get("recruiting_per_inspector", 0.0)), step=1.0,
                help="Ongoing job boards, agency fees — scales with headcount at volume",
                key=f"adv_rec_pi_{pv}"
            )

    with st.expander("Tax Rates", expanded=False):
        st.caption(
            "OpSource is a pass-through entity (S-corp or LLC) -- taxes are paid at the owner level. "
            "**Pre-tax net income is the correct metric for evaluating this division.** "
            "Tax provision shown below is an estimate of the owner's personal obligation."
        )
        _sc_pct = st.slider("SC State Tax Rate (%)", min_value=0, max_value=15,
            value=int(round(float(a.get("sc_state_tax_rate", 0.059)) * 100)),
            step=1, format="%d%%", key=f"adv_sc_tax_{pv}")
        a["sc_state_tax_rate"] = _sc_pct / 100.0
        _fed_pct = st.slider("Federal Tax Rate (%)", min_value=0, max_value=40,
            value=int(round(float(a.get("federal_tax_rate", 0.21)) * 100)), step=1, format="%d%%", key=f"adv_fed_tax_{pv}")
        a["federal_tax_rate"] = _fed_pct / 100.0
        _combined = a["sc_state_tax_rate"] + a["federal_tax_rate"]
        st.caption(f"Combined rate: **{_combined:.1%}**  ·  On $100K income: **${_combined * 100_000:,.0f} in taxes**")

    st.session_state.assumptions = a


# ════════════════════════════════════════════════════════════════════════════
# INVESTOR BRIEF — the landing page
# ════════════════════════════════════════════════════════════════════════════
with tab_brief:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Run</b> above to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a = st.session_state.assumptions

    # ── Risk callouts (only show if triggered) ─────────────────────────────
    n_loc  = weekly_df["warn_loc_maxed"].sum()  if "warn_loc_maxed"  in weekly_df.columns else 0
    n_mgmt = weekly_df["warn_mgmt_no_insp"].sum() if "warn_mgmt_no_insp" in weekly_df.columns else 0
    _burden = float(a.get("burden", 0.30))
    _nd     = int(a.get("net_days", 60))
    _peak   = float(mo_full["loc_end"].max()) if len(mo_full) else 0
    _max_loc = float(a.get("max_loc", 1_000_000))
    _yr1    = mo_full[mo_full["month_idx"] < 12]
    _margin_yr1 = float(_yr1["ebitda_margin"].mean()) if len(_yr1) else 0

    # Apply universal date range
    mo = _apply_range(mo_full)
    lo_idx = mo["month_idx"].min(); hi_idx = mo["month_idx"].max()
    qdf = qdf_full[(qdf_full["quarter_idx"] >= lo_idx // 3) & (qdf_full["quarter_idx"] <= hi_idx // 3)].copy()
    wdf = weekly_df[(weekly_df["month_idx"] >= lo_idx) & (weekly_df["month_idx"] <= hi_idx)].copy()

    # ── 1. Does the math work? (Verdict + KPIs) ───────────────────────────
    _n_months = len(mo)
    if _n_months <= 24:
        _period_label = f"{_n_months}-Month"
    elif _n_months % 12 == 0:
        _period_label = f"{_n_months // 12}-Year"
    else:
        _period_label = f"{_n_months // 12}Y {_n_months % 12}M"

    _mo_peak_idx = int(mo["loc_end"].idxmax()) if len(mo) else 0
    _mo_peak_period = mo.loc[_mo_peak_idx, "period"] if len(mo) else "—"

    _post_peak = mo.loc[_mo_peak_idx:]
    _self_fund_rows = _post_peak[_post_peak["loc_end"] <= 1]
    _self_fund = _self_fund_rows.iloc[0]["period"] if len(_self_fund_rows) > 0 else None

    _ss_mo = mo.tail(12)
    _ss_ebitda = float(_ss_mo["ebitda_after_interest"].mean())
    _ss_annual = _ss_ebitda * 12
    _total_interest = float(mo["interest"].sum())
    _peak_rng = float(mo["loc_end"].max()) if len(mo) else 0
    _int_coverage = (_ss_ebitda * 12) / max(1, _total_interest)
    _ropc = (_ss_annual / _peak_rng) if _peak_rng > 100 else 0

    _profitable_rows = mo[mo["ebitda_after_interest"] > 0]
    _be_month = _profitable_rows.iloc[0]["period"] if len(_profitable_rows) else "Not in range"

    if _ss_ebitda > 0 and _peak_rng > 0:
        st.success(
            f"**Selected range ({_period_label}): peak credit required is {fmt_dollar(_peak_rng)}** ({_mo_peak_period}), "
            f"first profitable month **{_be_month}**, "
            f"and steady-state net income **{fmt_dollar(_ss_ebitda)}/month** after interest "
            f"-- **{_ropc:.1f}x annualized return** on peak capital deployed."
        )
    elif _ss_ebitda <= 0:
        st.error(
            f"**No profitability in selected range.** "
            f"Net income at end of range: {fmt_dollar(_ss_ebitda)}/month. "
            "Extend the range, increase bill rate, or reduce burden."
        )

    # KPI row
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    kpi(v1, "Peak Credit Needed",      fmt_dollar(_peak_rng),                            _mo_peak_period)
    kpi(v2, "First Profitable Month",  _be_month,                                        "net after interest")
    kpi(v3, "Steady-State / Month",    fmt_dollar(_ss_ebitda),                           "net income (last 12 mo of range)")
    kpi(v4, f"{_period_label} Net Income", fmt_dollar(mo["ebitda_after_interest"].sum()), "after interest -- selected range")
    kpi(v5, "Total Interest Cost",     fmt_dollar(_total_interest),                      "credit line cost -- selected range")
    kpi(v6, "Return on Peak Capital",  f"{_ropc:.1f}x",                                  "ann. EBITDA / peak LOC")

    if _self_fund:
        st.caption(f"Credit line fully repaid (division self-funding): **{_self_fund}**  --  EBITDA/Interest coverage: **{_int_coverage:.1f}x**")

    # Working capital row
    section("Working Capital & Credit Metrics")
    wc1, wc2, wc3, wc4, wc5 = st.columns(5)

    _avg_dso = float(mo["dso"].mean()) if "dso" in mo.columns else 0
    _max_dso = float(mo["dso"].max()) if "dso" in mo.columns else 0
    _avg_loc_util = float(mo["loc_utilization"].mean()) if "loc_utilization" in mo.columns else 0
    _peak_loc_util = float(mo["loc_utilization"].max()) if "loc_utilization" in mo.columns else 0
    _avg_cash = float(mo["cash_end"].mean()) if len(mo) else 0
    _avg_monthly_out = float((mo["hourly_labor"] + mo["salaried_cost"] + mo["overhead"] + mo["interest"]).mean()) if len(mo) else 1
    _days_cash = (_avg_cash / (_avg_monthly_out / 30)) if _avg_monthly_out > 0 else 0
    _avg_fccr = float(mo[mo["fccr"] < 99]["fccr"].mean()) if "fccr" in mo.columns and len(mo[mo["fccr"] < 99]) else 0
    _min_fccr = float(mo[mo["fccr"] < 99]["fccr"].min()) if "fccr" in mo.columns and len(mo[mo["fccr"] < 99]) else 0
    _use_bb = bool(a.get("use_borrowing_base", False))
    _loc_headroom_avg = float(mo_full["loc_headroom"].mean()) if "loc_headroom" in mo_full.columns else None

    kpi(wc1, "Avg DSO",           f"{_avg_dso:.0f} days",        f"max {_max_dso:.0f} days in range")
    kpi(wc2, "Peak LOC Util.",    f"{_peak_loc_util:.0%}",        "of credit facility used")
    kpi(wc3, "Days Cash on Hand", f"{_days_cash:.0f} days",       "avg cash / daily burn")
    kpi(wc4, "Avg FCCR",          f"{_avg_fccr:.1f}x",           f"min {_min_fccr:.1f}x (1.1x = lender threshold)")
    if _use_bb and _loc_headroom_avg is not None:
        kpi(wc5, "Avg BB Headroom", fmt_dollar(_loc_headroom_avg), "available credit above LOC balance")
    else:
        _bad_debt_annual = float(mo["revenue"].sum()) * float(a.get("bad_debt_pct", 0.01)) / max(1, len(mo)) * 12
        kpi(wc5, "Annual Bad Debt Est.", fmt_dollar(_bad_debt_annual), f"{a.get('bad_debt_pct', 0.01):.1%} of revenue")

    st.divider()

    # ── 2. How much credit and when? (LOC chart) ──────────────────────────
    _brief_c1, _brief_c2 = st.columns(2)

    with _brief_c1:
        section("Credit Line, Cash & AR")
        _render_loc_chart(mo, weekly_df, a)

    # ── 3. EBITDA trajectory + implied company valuation ──────────────────
    with _brief_c2:
        section("EBITDA & Implied Company Valuation")
        _render_ebitda_chart(mo)


    # ── 4. Risk callouts (only show if triggered) ─────────────────────────
    if n_loc or n_mgmt or _burden > 0.35 or _nd > 90 or (0 < _margin_yr1 < 0.10) or (_min_fccr > 0 and _min_fccr < 1.1) or _peak_loc_util > 0.85:
        section("Risk Callouts")
        if n_loc:
            st.error(
                f"**Credit line maxed in {n_loc} week(s).** "
                f"Peak draw of {fmt_dollar(_peak)} exceeds your {fmt_dollar(_max_loc)} facility. "
                "Consider a larger line, faster payment terms, or reducing headcount ramp speed."
            )
        elif _peak > _max_loc * 0.85:
            st.warning(
                f"**Credit line near limit.** Peak draw {fmt_dollar(_peak)} is >85% of your "
                f"{fmt_dollar(_max_loc)} facility -- limited headroom for delays or surprises."
            )
        if n_mgmt:
            st.warning(
                f"**Paying salaried management with no inspectors in {n_mgmt} week(s).** "
                "Check your headcount plan for gaps between projects."
            )
        if _burden > 0.35:
            st.warning(
                f"**Burden rate {_burden:.0%} is above 35%.** This compresses margins significantly. "
                "Verify workers comp classification and benefit assumptions."
            )
        if _nd > 90:
            st.warning(
                f"**Net {_nd} payment terms.** At this lag, LOC peaks at {fmt_dollar(_peak)}. "
                "Confirm your credit facility can support this draw before launch."
            )
        if _margin_yr1 < 0.10 and _margin_yr1 > 0:
            st.warning(
                f"**Year 1 operating margin is {_margin_yr1:.1%}** -- thin margin leaves little room "
                "for pricing pressure, cost surprises, or payment delays."
            )
        if _min_fccr > 0 and _min_fccr < 1.1:
            st.warning(f"**FCCR drops to {_min_fccr:.2f}x** -- below the typical lender covenant of 1.10x.")
        elif _peak_loc_util > 0.85:
            st.warning(f"**Credit line peaks at {_peak_loc_util:.0%} utilization** -- limited headroom.")

    # ── Headcount chart (small, at bottom of brief) ───────────────────────
    section("Staffing by Role")
    st.plotly_chart(_bar(mo, "period",
        ["inspectors_avg","team_leads_avg","n_opscoord","n_fieldsup","n_regionalmgr"],
        ["Inspectors","Team Leads","Ops Coordinators","Field Supervisors","Regional Managers"],
        "Average Monthly Headcount", tickformat=",.0f"), use_container_width=True, config=_CHART_CONFIG)

    st.divider()

    # ── Export ─────────────────────────────────────────────────────────────
    section("Export to Excel")
    if st.button("Build Excel Report"):
        with st.spinner("Building..."):
            xlsx = build_excel(a, st.session_state.headcount_plan, weekly_df, mo_full, qdf_full)
        st.download_button(
            "Download Excel", data=xlsx,
            file_name="containment_division_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ════════════════════════════════════════════════════════════════════════════
# DETAIL — sub-tabs for Financials, Headcount, Summary, Sensitivity
# ════════════════════════════════════════════════════════════════════════════
with tab_detail:
    d1, d2, d3, d4 = st.tabs(["Financials", "Headcount", "Summary", "Sensitivity"])

    # ── Financials sub-tab ────────────────────────────────────────────────
    with d1:
        if not results_ready():
            st.markdown('<div class="info-box">Click <b>Run</b> above to generate results.</div>', unsafe_allow_html=True)
            st.stop()

        weekly_df, mo_full, qdf_full = st.session_state.results
        a = st.session_state.assumptions

        mo = _apply_range(mo_full)
        lo_idx = mo["month_idx"].min(); hi_idx = mo["month_idx"].max()
        qdf = qdf_full[(qdf_full["quarter_idx"] >= lo_idx // 3) & (qdf_full["quarter_idx"] <= hi_idx // 3)].copy()
        wdf = weekly_df[(weekly_df["month_idx"] >= lo_idx) & (weekly_df["month_idx"] <= hi_idx)].copy()

        f1, f2, f3 = st.tabs(["Monthly", "Quarterly", "Weekly Detail"])

        with f1:
            dcols = ["period","inspectors_avg","team_leads_avg","revenue",
                     "hourly_labor","salaried_cost","overhead","total_labor",
                     "ebitda","ebitda_margin","interest","ebitda_after_interest","ebitda_ai_margin",
                     "collections","ar_end","loc_end","cash_end"]
            _col_rename = {
                "period":"Period","inspectors_avg":"Inspectors","team_leads_avg":"Team Leads",
                "revenue":"Revenue","hourly_labor":"Hourly Labor","salaried_cost":"Salaried Mgmt",
                "overhead":"Overhead","total_labor":"Total Labor","ebitda":"Oper. Profit",
                "ebitda_margin":"Oper. %","interest":"Interest","ebitda_after_interest":"Net Income",
                "ebitda_ai_margin":"Net %","collections":"Collections","ar_end":"AR Balance",
                "loc_end":"Credit Line","cash_end":"Cash",
            }
            _fmt_table(_select(mo, dcols).rename(columns=_col_rename),
                dollar_cols=["Revenue","Hourly Labor","Salaried Mgmt","Overhead","Total Labor",
                             "Oper. Profit","Interest","Net Income","Collections","AR Balance","Credit Line","Cash"],
                pct_cols=["Oper. %","Net %"],
                highlight_neg="Net Income",
                highlight_loc="Credit Line", max_loc_val=float(a["max_loc"]))

        with f2:
            qcols = ["yr_q","revenue","hourly_labor","salaried_cost","overhead","total_labor",
                     "ebitda","ebitda_margin","interest","ebitda_after_interest","ebitda_ai_margin",
                     "ar_end","loc_end","cash_end"]
            _qcol_rename = {
                "yr_q":"Quarter","revenue":"Revenue","hourly_labor":"Hourly Labor",
                "salaried_cost":"Salaried Mgmt","overhead":"Overhead","total_labor":"Total Labor",
                "ebitda":"Oper. Profit","ebitda_margin":"Oper. %","interest":"Interest",
                "ebitda_after_interest":"Net Income","ebitda_ai_margin":"Net %",
                "ar_end":"AR Balance","loc_end":"Credit Line","cash_end":"Cash",
            }
            _fmt_table(_select(qdf, qcols).rename(columns=_qcol_rename),
                dollar_cols=["Revenue","Hourly Labor","Salaried Mgmt","Overhead","Total Labor",
                             "Oper. Profit","Interest","Net Income","AR Balance","Credit Line","Cash"],
                pct_cols=["Oper. %","Net %"],
                highlight_neg="Net Income",
                highlight_loc="Credit Line", max_loc_val=float(a["max_loc"]))

        with f3:
            w_neg = int(wdf["warn_neg_ebitda"].sum()) if "warn_neg_ebitda" in wdf.columns else 0
            if w_neg:
                st.markdown(f'<div class="warn-box">Negative operating profit in {w_neg} week(s) in selected range.</div>', unsafe_allow_html=True)
            wt1, wt2, wt3, wt4 = st.tabs(["Headcount & Revenue","Labor & Profit","Invoicing & Payments","Cash & Credit Line"])
            with wt1:
                _fmt_table(_select(wdf, ["week_start","week_end","inspectors","team_leads",
                    "n_opscoord","n_fieldsup","n_regionalmgr","insp_st_hrs","insp_ot_hrs",
                    "insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"]).rename(columns={
                    "week_start":"Week Start","week_end":"Week End","inspectors":"Inspectors",
                    "team_leads":"Team Leads","n_opscoord":"Ops Coord","n_fieldsup":"Field Sup",
                    "n_regionalmgr":"Reg. Mgr","insp_st_hrs":"Insp ST Hrs","insp_ot_hrs":"Insp OT Hrs",
                    "insp_rev_st":"Insp Rev ST","insp_rev_ot":"Insp Rev OT",
                    "lead_rev_st":"Lead Rev ST","lead_rev_ot":"Lead Rev OT","revenue_wk":"Revenue",
                }),
                    dollar_cols=["Insp Rev ST","Insp Rev OT","Lead Rev ST","Lead Rev OT","Revenue"])
            with wt2:
                _fmt_table(_select(wdf, ["week_start","inspectors","team_leads",
                    "insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                    "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"]).rename(columns={
                    "week_start":"Week Start","inspectors":"Inspectors","team_leads":"Team Leads",
                    "insp_labor_st":"Insp Labor ST","insp_labor_ot":"Insp Labor OT",
                    "lead_labor_st":"Lead Labor ST","lead_labor_ot":"Lead Labor OT",
                    "hourly_labor":"Hourly Labor","salaried_wk":"Salaried Mgmt",
                    "overhead_wk":"Overhead","revenue_wk":"Revenue","ebitda_wk":"Oper. Profit",
                }),
                    dollar_cols=["Insp Labor ST","Insp Labor OT","Lead Labor ST","Lead Labor OT",
                                 "Hourly Labor","Salaried Mgmt","Overhead","Revenue","Oper. Profit"])
            with wt3:
                _fmt_table(_select(wdf, ["week_start","week_end","is_month_end",
                    "revenue_wk","statement_amt","collections","ar_begin","ar_end"]).rename(columns={
                    "week_start":"Week Start","week_end":"Week End","is_month_end":"Month End?",
                    "revenue_wk":"Revenue","statement_amt":"Invoice Amt",
                    "collections":"Collections","ar_begin":"AR Open","ar_end":"AR Close",
                }),
                    dollar_cols=["Revenue","Invoice Amt","Collections","AR Open","AR Close"])
            with wt4:
                _fmt_table(_select(wdf, ["week_start","payroll_cash_out","salaried_wk",
                    "overhead_wk","interest_paid","collections",
                    "cash_begin","loc_draw","loc_repay","cash_end","loc_begin","loc_end"]).rename(columns={
                    "week_start":"Week Start","payroll_cash_out":"Hourly Payroll",
                    "salaried_wk":"Salaried Mgmt","overhead_wk":"Overhead",
                    "interest_paid":"Interest Paid","collections":"Collections",
                    "cash_begin":"Cash Open","loc_draw":"LOC Draw","loc_repay":"LOC Repay",
                    "cash_end":"Cash Close","loc_begin":"LOC Open","loc_end":"LOC Close",
                }),
                    dollar_cols=["Hourly Payroll","Salaried Mgmt","Overhead","Interest Paid",
                                 "Collections","Cash Open","LOC Draw","LOC Repay",
                                 "Cash Close","LOC Open","LOC Close"])
                if all(c in wdf.columns for c in ["check_ar","check_loc","check_cash"]):
                    max_err = wdf[["check_ar","check_loc","check_cash"]].max().max()
                    if max_err < 0.01:
                        st.success("All weekly balances reconcile correctly.")
                    else:
                        st.error(f"Reconciliation error detected: ${max_err:.2f}")

        # ── Charts moved from Brief (Revenue vs Costs, Margins, Payroll Float, Break-Even) ──
        section("Revenue vs. Costs")
        fig_rv = _bar(mo, "period", ["hourly_labor","salaried_cost","overhead"],
                      ["Hourly Labor","Salaried Mgmt","Overhead"], "Monthly Cost Stack")
        fig_rv.add_trace(go.Scatter(x=mo["period"], y=mo["revenue"], name="Revenue",
            mode="lines", line=dict(color=PC[1], width=2)))
        st.plotly_chart(fig_rv, use_container_width=True, config=_CHART_CONFIG)

        _fc1, _fc2 = st.columns(2)
        with _fc1:
            section("Net Income (After Interest)")
            st.plotly_chart(_line(mo, "period",
                ["ebitda", "ebitda_after_interest"],
                ["Operating Profit", "Net Income (after interest)"],
                "Monthly Profit"), use_container_width=True, config=_CHART_CONFIG)
        with _fc2:
            section("Profit Margins")
            st.plotly_chart(_line(mo, "period",
                ["ebitda_margin", "ebitda_ai_margin"],
                ["Operating Margin", "Net Margin"],
                "Monthly Margins", pct_y=True), use_container_width=True, config=_CHART_CONFIG)

        st.divider()

        # ── Payroll Float ─────────────────────────────────────────────────
        section("Payroll Float -- Cash Out vs. Cash In")
        st.caption(
            "Your credit line bridges the gap between weekly payroll and monthly customer payments. "
            f"With Net {int(a['net_days'])} terms, you pay employees weeks before cash arrives."
        )
        if all(c in wdf.columns for c in ["payroll_cash_out", "salaried_wk", "overhead_wk"]):
            wdf_pf = wdf.copy()
            wdf_pf["total_cash_out"] = wdf_pf["payroll_cash_out"] + wdf_pf["salaried_wk"] + wdf_pf["overhead_wk"]
            wdf_pf["week_label"] = wdf_pf["week_num"].apply(lambda n: f"Wk{int(n)+1}")

            fig_pf = go.Figure()
            fig_pf.add_trace(go.Bar(
                x=wdf_pf["week_label"], y=wdf_pf["total_cash_out"],
                name="Cash Out (Payroll + Overhead)", marker_color=PC[3], opacity=0.75
            ))
            fig_pf.add_trace(go.Bar(
                x=wdf_pf["week_label"], y=wdf_pf["collections"],
                name="Cash In (Customer Payments)", marker_color=PC[1], opacity=0.75
            ))
            fig_pf.add_trace(go.Scatter(
                x=wdf_pf["week_label"], y=wdf_pf["loc_end"],
                name="Credit Line Balance", mode="lines",
                line=dict(color=PC[2], width=2, dash="dot"), yaxis="y2"
            ))
            fig_pf.update_layout(
                template=TPL, height=320, barmode="overlay",
                margin=dict(l=10, r=10, t=10, b=60),
                legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
                yaxis=dict(tickformat="$,.0f", title="Weekly Cash ($)"),
                yaxis2=dict(tickformat="$,.0f", title="Credit Line ($)",
                            overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(fig_pf, use_container_width=True, config=_CHART_CONFIG)

        st.divider()

        # ── Cash Flow Waterfall ───────────────────────────────────────────
        section("Cash Flow Summary -- Selected Period")
        wf_start = float(wdf["cash_begin"].iloc[0]) if len(wdf) else float(a.get("initial_cash", 0))
        wf_values = [
            wf_start,
            float(mo["collections"].sum()),
            -float(mo["hourly_labor"].sum()),
            -float(mo["salaried_cost"].sum()),
            -float(mo["overhead"].sum()),
            -float(mo["interest"].sum()),
            float(mo["loc_draw"].sum()),
            -float(mo["loc_repay"].sum()),
            float(mo["cash_end"].iloc[-1]) if len(mo) else 0,
        ]
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","relative","relative","relative","relative","relative","total"],
            x=["Starting Cash","+ Customer Payments","- Hourly Payroll","- Salaried Mgmt",
               "- Overhead","- Interest","+ Credit Draw","- Credit Repay","Ending Cash"],
            y=wf_values,
            connector=dict(line=dict(color="#334155")),
            increasing=dict(marker_color=PC[1]),
            decreasing=dict(marker_color=PC[3]),
            totals=dict(marker_color=PC[0]),
        ))
        fig_wf.update_layout(template=TPL, height=300, margin=dict(l=10,r=10,t=10,b=10),
                             yaxis=dict(tickformat="$,.0f"))
        st.plotly_chart(fig_wf, use_container_width=True, config=_CHART_CONFIG)

        st.divider()

        # ── Break-even Calculator ─────────────────────────────────────────
        section("Break-Even Calculator")
        st.caption("Find the minimum inspectors needed to be profitable in Year 1 at a given payment term.")
        bc1, bc2, bc3 = st.columns(3)
        be_nd = bc1.selectbox(
            "Payment Terms to Test", [30, 45, 60, 90, 120, 150], index=2,
            help="Longer terms require more inspectors to break even -- credit line interest rises."
        )
        if bc2.button("Find Break-Even", use_container_width=True):
            with st.spinner("Calculating..."):
                be = find_breakeven_inspectors(a, be_nd)
            bc3.success(
                f"Need at least **{be} inspectors** staffed every month to be profitable in Year 1 at Net {be_nd}."
            )

    # ── Headcount sub-tab ─────────────────────────────────────────────────
    with d2:
        hc      = st.session_state.headcount_plan
        a_start = st.session_state.assumptions["start_date"]

        section("Quick Fill")
        c1, c2, c3, c4 = st.columns(4)
        fv = c1.number_input("Inspectors", 0, 10_000, 25, step=5, key="fv",
            help="Number of inspectors to staff during this period.")
        ff = c2.number_input("From Month", 1, 120, 1, step=1, key="ff",
            help="First month to fill. Month 1 = your model start date.")
        ft = c3.number_input("To Month",   1, 120, 12, step=1, key="ft",
            help="Last month to fill (inclusive). Month 12 = end of Year 1.")
        if c4.button("Apply", key="hc_apply", use_container_width=True,
                     help="Sets inspector count for the selected month range."):
            for i in range(int(ff) - 1, int(ft)):
                hc[i] = int(fv)
            st.session_state.headcount_plan = hc
            st.rerun()

        month_labels = [f"M{i+1}" for i in range(120)]

        section("Preview")
        lo_hc, hi_hc = st.select_slider("Show months", options=list(range(1, 121)), value=(1, 24), key="hc_rng")
        hc_prev = pd.DataFrame({"period": month_labels, "inspectors": hc}).iloc[lo_hc - 1:hi_hc]
        fig_hc  = px.bar(hc_prev, x="period", y="inspectors", template=TPL,
                         title="Inspectors Staffed per Month", color_discrete_sequence=[PC[0]])
        fig_hc.update_layout(height=240, margin=dict(l=10,r=10,t=36,b=10))
        st.plotly_chart(fig_hc, use_container_width=True, config=_CHART_CONFIG)

        section("Edit -- All 120 Months")
        hc_df  = pd.DataFrame({"Period": month_labels, "Inspectors": hc})
        edited = st.data_editor(hc_df, column_config={
            "Period":     st.column_config.TextColumn(disabled=True),
            "Inspectors": st.column_config.NumberColumn(min_value=0, max_value=10_000, step=1),
        }, use_container_width=True, height=500, num_rows="fixed")
        st.session_state.headcount_plan = edited["Inspectors"].tolist()

    # ── Summary sub-tab ───────────────────────────────────────────────────
    with d3:
        if not results_ready():
            st.markdown('<div class="info-box">Click <b>Run</b> above to generate results.</div>', unsafe_allow_html=True)
            st.stop()

        weekly_df, mo_full, _ = st.session_state.results
        a = st.session_state.assumptions

        mo_s = _apply_range(mo_full)
        lo_s = mo_s["month_idx"].min(); hi_s = mo_s["month_idx"].max()
        wdf_s = weekly_df[(weekly_df["month_idx"] >= lo_s) & (weekly_df["month_idx"] <= hi_s)].copy()

        st.divider()

        section("Income Statement -- Selected Range")
        tot_rev  = mo_s["revenue"].sum()
        tot_hl   = mo_s["hourly_labor"].sum()
        tot_sal  = mo_s["salaried_cost"].sum()
        tot_to   = mo_s["turnover_cost"].sum() if "turnover_cost" in mo_s.columns else 0.0
        tot_ovhd = mo_s["overhead"].sum()
        tot_fovhd = tot_ovhd - tot_to
        tot_exp  = tot_hl + tot_sal + tot_ovhd
        tot_eb   = mo_s["ebitda"].sum()
        tot_int  = mo_s["interest"].sum()
        tot_ni   = mo_s["ebitda_after_interest"].sum()

        def _pct(v): return v / tot_rev if tot_rev else 0.0

        tot_sc_tax  = mo_s["sc_tax"].sum()  if "sc_tax"  in mo_s.columns else 0.0
        tot_fed_tax = mo_s["federal_tax"].sum() if "federal_tax" in mo_s.columns else 0.0
        tot_tax     = tot_sc_tax + tot_fed_tax
        tot_ni_at   = mo_s["net_income_after_tax"].sum() if "net_income_after_tax" in mo_s.columns else tot_ni - tot_tax

        is_df = pd.DataFrame({
            "Line Item": [
                "Revenue", "",
                "  Hourly Labor (Inspectors & Team Leads)",
                "  Salaried Management",
                "  Management Turnover & Replacement",
                "  Fixed Overhead",
                "Total Expenses", "",
                "Operating Profit (EBITDA)",
                "  Credit Line Interest",
                "Net Income (pre-tax)", "",
                f"  SC State Tax ({a.get('sc_state_tax_rate', 0.059):.1%})",
                f"  Federal Tax ({a.get('federal_tax_rate', 0.21):.0%})",
                "Net Income (after tax)",
            ],
            "Amount": [
                tot_rev, None,
                tot_hl, tot_sal, tot_to, tot_fovhd,
                tot_exp, None,
                tot_eb, -tot_int, tot_ni, None,
                -tot_sc_tax, -tot_fed_tax, tot_ni_at,
            ],
            "% of Revenue": [
                1.0, None,
                _pct(tot_hl), _pct(tot_sal), _pct(tot_to), _pct(tot_fovhd),
                _pct(tot_exp), None,
                _pct(tot_eb), -_pct(tot_int), _pct(tot_ni), None,
                -_pct(tot_sc_tax), -_pct(tot_fed_tax), _pct(tot_ni_at),
            ],
        })

        def _is_style(row):
            lbl = row["Line Item"]
            if lbl in ("Revenue", "Net Income (after tax)"):
                return ["font-weight:bold; color:#52D68A"] * len(row)
            if lbl == "Total Expenses":
                return ["font-weight:bold; color:#E05252"] * len(row)
            if lbl in ("Operating Profit (EBITDA)", "Net Income (pre-tax)"):
                return ["font-weight:bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            is_df.style
                 .format({"Amount": "${:,.0f}", "% of Revenue": "{:.1%}"}, na_rep="")
                 .apply(_is_style, axis=1),
            use_container_width=True, height=380,
        )

        st.divider()

        section("Year-by-Year Profit & Loss")
        _to_col = {"turnover_cost": ("turnover_cost", "sum")} if "turnover_cost" in mo_full.columns else {}
        _tax_cols = {}
        if "sc_tax" in mo_full.columns:
            _tax_cols["sc_tax"]               = ("sc_tax",               "sum")
            _tax_cols["federal_tax"]           = ("federal_tax",          "sum")
            _tax_cols["net_income_after_tax"]  = ("net_income_after_tax", "sum")
        annual = mo_full.groupby("year").agg(**{
            "revenue":               ("revenue",               "sum"),
            "hourly_labor":          ("hourly_labor",           "sum"),
            "salaried_cost":         ("salaried_cost",          "sum"),
            "overhead":              ("overhead",               "sum"),
            "ebitda":                ("ebitda",                 "sum"),
            "interest":              ("interest",               "sum"),
            "ebitda_after_interest": ("ebitda_after_interest",  "sum"),
            "peak_loc":              ("loc_end",                "max"),
            "avg_inspectors":        ("inspectors_avg",         "mean"),
            **_to_col,
            **_tax_cols,
        }).reset_index()
        annual["total_expenses"] = annual["hourly_labor"] + annual["salaried_cost"] + annual["overhead"]
        annual["oper_margin"]    = np.where(annual["revenue"] > 0, annual["ebitda"] / annual["revenue"], 0)
        annual["pretax_margin"]  = np.where(annual["revenue"] > 0, annual["ebitda_after_interest"] / annual["revenue"], 0)
        annual["year_label"]     = annual["year"].apply(lambda y: f"Year {y - annual['year'].min() + 1}  ({y})")
        has_tax = "net_income_after_tax" in annual.columns

        if has_tax:
            annual["net_margin_at"] = np.where(annual["revenue"] > 0,
                                                annual["net_income_after_tax"] / annual["revenue"], 0)
            ann_disp = annual[["year_label","avg_inspectors","revenue","total_expenses",
                                "ebitda","oper_margin","interest","ebitda_after_interest",
                                "pretax_margin","net_income_after_tax","net_margin_at","peak_loc"]].rename(columns={
                "year_label": "Year", "avg_inspectors": "Avg Inspectors",
                "revenue": "Revenue", "total_expenses": "Total Expenses",
                "ebitda": "Oper. Profit", "oper_margin": "Oper. %",
                "interest": "Interest", "ebitda_after_interest": "Pre-Tax Income",
                "pretax_margin": "Pre-Tax %", "net_income_after_tax": "Net Income",
                "net_margin_at": "Net %", "peak_loc": "Peak Credit Line",
            })
            _fmt_table(ann_disp,
                dollar_cols=["Revenue","Total Expenses","Oper. Profit","Interest","Pre-Tax Income","Net Income","Peak Credit Line"],
                pct_cols=["Oper. %","Pre-Tax %","Net %"],
                highlight_neg="Net Income")
        else:
            ann_disp = annual[["year_label","avg_inspectors","revenue","total_expenses",
                                "ebitda","oper_margin","interest","ebitda_after_interest","pretax_margin","peak_loc"]].rename(columns={
                "year_label": "Year", "avg_inspectors": "Avg Inspectors",
                "revenue": "Revenue", "total_expenses": "Total Expenses",
                "ebitda": "Oper. Profit", "oper_margin": "Oper. %",
                "interest": "Interest", "ebitda_after_interest": "Net Income",
                "pretax_margin": "Net %", "peak_loc": "Peak Credit Line",
            })
            _fmt_table(ann_disp,
                dollar_cols=["Revenue","Total Expenses","Oper. Profit","Interest","Net Income","Peak Credit Line"],
                pct_cols=["Oper. %","Net %"],
                highlight_neg="Net Income")

        st.divider()

        section("Headcount by Role -- Selected Range")
        gm_avg = float(wdf_s["gm_fte"].mean()) if "gm_fte" in wdf_s.columns else 1.0
        hc_data = {
            "Role":            ["Inspectors","Team Leads","Ops Coordinators","Field Supervisors","Regional Managers","General Manager"],
            "Type":            ["Hourly","Hourly","Salaried","Salaried","Salaried","Salaried"],
            "Billed to Client":["Yes","Yes","No","No","No","No"],
            "Avg per Month":   [mo_s["inspectors_avg"].mean(), mo_s["team_leads_avg"].mean(),
                                mo_s["n_opscoord"].mean(),     mo_s["n_fieldsup"].mean(),
                                mo_s["n_regionalmgr"].mean(),  gm_avg],
            "Peak in Period":  [mo_s["inspectors_avg"].max(), mo_s["team_leads_avg"].max(),
                                mo_s["n_opscoord"].max(),      mo_s["n_fieldsup"].max(),
                                mo_s["n_regionalmgr"].max(),   1.0 if gm_avg > 0 else 0.0],
        }
        st.dataframe(
            pd.DataFrame(hc_data).style.format({"Avg per Month": "{:.1f}", "Peak in Period": "{:.1f}"}),
            use_container_width=True, height=260
        )

        st.divider()

        section("Management Scaling -- When Each Layer Triggers")
        st.caption(
            "As you add inspectors, salaried management layers activate automatically. "
            "Each row shows the inspector count that triggers the **next** salaried hire and the monthly cost step-up."
        )
        _tl_r  = int(a.get("team_lead_ratio", 12))
        _oc_r  = int(a.get("opscoord_span", 75))
        _fs_r  = int(a.get("fieldsup_span", 25))
        _rm_r  = int(a.get("regionalmgr_span", 175))
        _gm_mo   = a.get("gm_loaded_annual", 117_000) / 12
        _oc_mo   = a.get("opscoord_base", 65_000)   * (1 + a.get("mgmt_burden", 0.25)) / 12
        _fs_mo   = a.get("fieldsup_base", 70_000)   * (1 + a.get("mgmt_burden", 0.25)) / 12
        _rm_mo   = a.get("regionalmgr_base", 110_000) * (1 + a.get("mgmt_burden", 0.25)) / 12

        _scale_rows = []
        for _n_insp in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200]:
            if _n_insp == 0:
                _tl  = 0; _oc = 0; _fs = 0; _rm = 0
            else:
                _tl  = ceil(_n_insp / _tl_r)
                _oc  = ceil(_n_insp / _oc_r)
                _fs  = ceil(_n_insp / _fs_r)
                _rm  = ceil(_n_insp / _rm_r)
            _sal_mo = _gm_mo + _oc * _oc_mo + _fs * _fs_mo + _rm * _rm_mo
            _scale_rows.append({
                "Inspectors": _n_insp,
                "Team Leads": _tl,
                "Ops Coords": _oc,
                "Field Sups": _fs,
                "Reg. Mgrs": _rm,
                "Total Salaried Mgmt": int(_oc + _fs + _rm + 1),
                "Salaried Cost / Month": _sal_mo,
            })
        _sc_df = pd.DataFrame(_scale_rows)
        st.dataframe(
            _sc_df.style.format({"Salaried Cost / Month": "${:,.0f}"}),
            use_container_width=True, height=480
        )
        st.caption(
            f"Ratios: 1 Team Lead per {_tl_r} inspectors / "
            f"1 Ops Coord per {_oc_r} / "
            f"1 Field Sup per {_fs_r} / "
            f"1 Reg. Mgr per {_rm_r}. "
            f"GM always active (${_gm_mo:,.0f}/mo loaded)."
        )

        st.divider()

        section("Revenue & Expense Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            _lead_st = mo_s["lead_rev_st"].sum() if "lead_rev_st" in mo_s.columns else 0
            _lead_ot = mo_s["lead_rev_ot"].sum() if "lead_rev_ot" in mo_s.columns else 0
            fig_rp = go.Figure(go.Pie(
                labels=["Inspector Regular Time","Inspector Overtime","Team Lead Regular Time","Team Lead Overtime"],
                values=[mo_s["insp_rev_st"].sum(), mo_s["insp_rev_ot"].sum(), _lead_st, _lead_ot],
                hole=0.42, marker_colors=PC[:4]
            ))
            fig_rp.update_layout(template=TPL, height=280, title="Revenue by Component", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_rp, use_container_width=True, config=_CHART_CONFIG)
        with c2:
            fig_ep = go.Figure(go.Pie(
                labels=["Hourly Labor","Salaried Management","Turnover & Replacement","Fixed Overhead"],
                values=[tot_hl, tot_sal, tot_to, tot_fovhd],
                hole=0.42, marker_colors=[PC[3], PC[2], PC[4], PC[5]]
            ))
            fig_ep.update_layout(template=TPL, height=280, title="Expenses by Component", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_ep, use_container_width=True, config=_CHART_CONFIG)

        section("Monthly Revenue vs. Net Income")
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=mo_s["period"], y=mo_s["revenue"], name="Revenue",
                               marker_color=PC[0], opacity=0.7))
        fig_t.add_trace(go.Scatter(x=mo_s["period"], y=mo_s["ebitda_after_interest"],
                                   name="Net Income", mode="lines+markers",
                                   line=dict(color=PC[1], width=2)))
        fig_t.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
        fig_t.update_layout(template=TPL, height=300, margin=dict(l=10,r=10,t=10,b=60),
                            legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"), yaxis=dict(tickformat="$,.0f"))
        st.plotly_chart(fig_t, use_container_width=True, config=_CHART_CONFIG)

        st.divider()

        section("Key Investor Questions -- Direct Answers")
        st.caption(
            "This section answers the four questions a capital allocator asks before committing to this division."
        )

        _peak2       = float(mo_full["loc_end"].max()) if len(mo_full) else 0
        _peak_mo2    = mo_full.loc[mo_full["loc_end"].idxmax(), "period"] if len(mo_full) else "--"
        _ss_ni2      = float(mo_full.tail(12)["ebitda_after_interest"].mean())
        _tot_int2    = float(mo_full["interest"].sum())
        _yr1_ni2     = float(mo_full[mo_full["month_idx"] < 12]["ebitda_after_interest"].sum())
        _ropc2       = (_ss_ni2 * 12 / _peak2) if _peak2 > 100 else 0
        _be_rows2    = mo_full[mo_full["ebitda_after_interest"] > 0]
        _be_mo2      = _be_rows2.iloc[0]["period"] if len(_be_rows2) else "Not reached"
        _int_cov2    = (_ss_ni2 * 12 + _tot_int2) / max(1, _tot_int2)
        _max_insp    = int(mo_full["inspectors_avg"].max())

        iq1, iq2 = st.columns(2)

        with iq1:
            st.markdown("**Does it work and is it competitive?**")
            if _ss_ni2 > 0:
                st.success(
                    f"Yes -- at steady state the division generates **{fmt_dollar(_ss_ni2)}/month** "
                    f"net after interest ({fmt_dollar(_ss_ni2 * 12)}/year). "
                    f"It reaches profitability in **{_be_mo2}**. "
                    f"Year 1 net income: **{fmt_dollar(_yr1_ni2)}**."
                )
            else:
                st.error(
                    f"Not yet -- steady-state net income is **{fmt_dollar(_ss_ni2)}/month**. "
                    "Adjust bill rate, burden, or headcount to reach profitability."
                )

            st.markdown("**How big can this become inside OpSource?**")
            st.info(
                f"At **{_max_insp} peak inspectors**, this scenario generates "
                f"**{fmt_dollar(mo_full['revenue'].max())}/month** in peak revenue. "
                f"Scaling to 150-200 inspectors is achievable within 24-36 months with "
                f"the right supervisor layering and credit facility. Each additional inspector "
                f"at current rates adds **{fmt_dollar(float(a.get('st_bill_rate', 39)) * (float(a.get('st_hours', 40)) + float(a.get('ot_hours', 10)) * float(a.get('ot_bill_premium', 1.5))) * 52 / 12)}/month** in annualized revenue."
            )

        with iq2:
            st.markdown("**What capital is required and what is the cost of money?**")
            _max_loc3 = float(a.get("max_loc", 1_000_000))
            _apr3 = float(a.get("apr", 0.085))
            if _peak2 <= _max_loc3:
                st.success(
                    f"Peak credit draw: **{fmt_dollar(_peak2)}** in **{_peak_mo2}** -- within your "
                    f"**{fmt_dollar(_max_loc3)}** facility. "
                    f"Total interest cost: **{fmt_dollar(_tot_int2)}** at {_apr3:.1%} APR. "
                    f"EBITDA/Interest coverage at steady state: **{_int_cov2:.1f}x**. "
                    f"Return on peak capital: **{_ropc2:.1f}x** annualized."
                )
            else:
                st.error(
                    f"Peak draw **{fmt_dollar(_peak2)}** exceeds your **{fmt_dollar(_max_loc3)}** facility. "
                    f"Increase the credit line or reduce terms/headcount ramp."
                )

            st.markdown("**Can the team execute without owner involvement?**")
            st.info(
                f"Yes -- the model includes a fully-loaded GM at "
                f"**{fmt_dollar(float(a.get('gm_loaded_annual', 117_000)) / 12)}/month** "
                f"responsible for day-to-day operations, plus automatic supervisor layering "
                f"(1 Field Sup per {int(a.get('fieldsup_span', 25))} inspectors, "
                f"1 Ops Coord per {int(a.get('opscoord_span', 75))}). "
                "Owner role is capital allocation and performance review only."
            )

    # ── Sensitivity sub-tab ───────────────────────────────────────────────
    with d4:
        if not results_ready():
            st.markdown('<div class="info-box">Click <b>Run</b> above first.</div>', unsafe_allow_html=True)
            st.stop()

        weekly_df, mo_full, qdf_full = st.session_state.results
        a  = st.session_state.assumptions
        hc = st.session_state.headcount_plan

        st.caption(
            "Each scenario re-runs the full model changing one variable at a time. "
            "Color coding: green = healthy, yellow = caution, red = danger. "
            "All other assumptions stay at current values."
        )

        def _heatmap(df, row_col, col_col, val_col, title, fmt="$,.0f", reverse=False):
            pivot = df.pivot(index=row_col, columns=col_col, values=val_col)
            vmin, vmax = pivot.values.min(), pivot.values.max()
            colorscale = "RdYlGn" if not reverse else "RdYlGn_r"
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale=colorscale,
                zmin=vmin, zmax=vmax,
                text=[[f"${v:,.0f}" if "$" in fmt else f"{v:.1%}" if "%" in fmt else f"{v:.2f}"
                       for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=11),
                showscale=True,
                colorbar=dict(thickness=14, len=0.8),
            ))
            fig.update_layout(
                template=TPL, title=title, height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(title=col_col, side="bottom"),
                yaxis=dict(title=row_col, autorange="reversed"),
            )
            return fig

        s1, s2, s3, s4, s5, s6 = st.tabs(["Payment Terms","Bill Rate","Payroll Burden","Overtime","Utilization & Wages","Bad Debt & Turnover"])

        with s1:
            section("How payment terms drive your credit line and returns")
            nd_vals = [30, 45, 60, 75, 90, 105, 120, 150]
            with st.spinner("Running sensitivity..."):
                nd_df = run_sensitivity(a, hc, "net_days", nd_vals)
            c1, c2 = st.columns(2)
            with c1:
                fig_nd1 = go.Figure(go.Scatter(x=nd_df["value"], y=nd_df["peak_loc"], mode="lines+markers", line=dict(color=PC[0], width=2), marker=dict(size=7)))
                fig_nd1.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Peak Credit Line vs. Payment Terms", xaxis_title="Net Days", yaxis=dict(tickformat="$,.0f"))
                fig_nd1.add_hline(y=float(a["max_loc"]), line_dash="dot", line_color=PC[3], annotation_text="Your Credit Limit")
                st.plotly_chart(fig_nd1, use_container_width=True, config=_CHART_CONFIG)
            with c2:
                fig_nd2 = go.Figure(go.Scatter(x=nd_df["value"], y=nd_df["ebitda_ai_margin"], mode="lines+markers", line=dict(color=PC[1], width=2), marker=dict(size=7)))
                fig_nd2.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Income Margin vs. Payment Terms", xaxis_title="Net Days", yaxis=dict(tickformat=".1%"))
                fig_nd2.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                st.plotly_chart(fig_nd2, use_container_width=True, config=_CHART_CONFIG)
            _fmt_table(nd_df[["value","peak_loc","annual_interest","ebitda_ai","ebitda_ai_margin"]].rename(columns={"value":"Net Days","peak_loc":"Peak Credit","annual_interest":"Total Interest","ebitda_ai":"Net Income","ebitda_ai_margin":"Net Margin"}), dollar_cols=["Peak Credit","Total Interest","Net Income"], pct_cols=["Net Margin"], highlight_neg="Net Income")

        with s2:
            section("Bill rate impact on margin, credit need, and total income")
            br_vals = [36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0]
            with st.spinner("Running sensitivity..."):
                br_df = run_sensitivity(a, hc, "st_bill_rate", br_vals)
            c1, c2 = st.columns(2)
            with c1:
                fig_br1 = go.Figure(go.Scatter(x=br_df["value"], y=br_df["ebitda_ai_margin"], mode="lines+markers", line=dict(color=PC[1], width=2), marker=dict(size=7)))
                fig_br1.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Margin vs. Bill Rate", xaxis_title="Bill Rate ($/hr)", yaxis=dict(tickformat=".1%"))
                fig_br1.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                st.plotly_chart(fig_br1, use_container_width=True, config=_CHART_CONFIG)
            with c2:
                fig_br2 = go.Figure(go.Scatter(x=br_df["value"], y=br_df["ebitda_ai"], mode="lines+markers", line=dict(color=PC[0], width=2), marker=dict(size=7)))
                fig_br2.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Total Net Income vs. Bill Rate", xaxis_title="Bill Rate ($/hr)", yaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig_br2, use_container_width=True, config=_CHART_CONFIG)
            _fmt_table(br_df[["value","ebitda_margin","ebitda_ai_margin","peak_loc","ebitda_ai"]].rename(columns={"value":"Bill Rate","ebitda_margin":"Oper. Margin","ebitda_ai_margin":"Net Margin","peak_loc":"Peak Credit","ebitda_ai":"Net Income"}), dollar_cols=["Peak Credit","Net Income"], pct_cols=["Oper. Margin","Net Margin"])

        with s3:
            section("How payroll burden compresses margin and grows credit need")
            burd_vals = [0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40, 0.45]
            with st.spinner("Running sensitivity..."):
                burd_df = run_sensitivity(a, hc, "burden", burd_vals)
            burd_df["pct"] = (burd_df["value"] * 100).round(0).astype(int)
            c1, c2 = st.columns(2)
            with c1:
                fig_b1 = go.Figure(go.Scatter(x=burd_df["pct"], y=burd_df["ebitda_ai_margin"], mode="lines+markers", line=dict(color=PC[1], width=2), marker=dict(size=7)))
                fig_b1.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Margin vs. Burden Rate", xaxis_title="Burden (%)", yaxis=dict(tickformat=".1%"))
                fig_b1.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                fig_b1.add_shape(
                    type="line", x0=35, x1=35, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(dash="dot", color="#F59E0B", width=1)
                )
                fig_b1.add_annotation(
                    x=35, y=0.98, xref="x", yref="paper",
                    text="35% caution", font=dict(color="#F59E0B", size=10),
                    showarrow=False, textangle=-90,
                    xanchor="right", yanchor="top"
                )
                st.plotly_chart(fig_b1, use_container_width=True, config=_CHART_CONFIG)
            with c2:
                fig_b2 = go.Figure(go.Scatter(x=burd_df["pct"], y=burd_df["peak_loc"], mode="lines+markers", line=dict(color=PC[3], width=2), marker=dict(size=7)))
                fig_b2.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Peak Credit Line vs. Burden Rate", xaxis_title="Burden (%)", yaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig_b2, use_container_width=True, config=_CHART_CONFIG)
            _fmt_table(burd_df[["pct","ebitda_ai","ebitda_ai_margin","peak_loc"]].rename(columns={"pct":"Burden (%)","ebitda_ai":"Net Income","ebitda_ai_margin":"Net Margin","peak_loc":"Peak Credit"}), dollar_cols=["Net Income","Peak Credit"], pct_cols=["Net Margin"])

        with s4:
            section("How overtime hours affect revenue, margin, and credit need")
            ot_vals = [0, 2, 4, 6, 8, 10, 12, 15, 20]
            with st.spinner("Running sensitivity..."):
                ot_df = run_sensitivity(a, hc, "ot_hours", ot_vals)
            c1, c2 = st.columns(2)
            with c1:
                fig_ot1 = go.Figure(go.Scatter(x=ot_df["value"], y=ot_df["ebitda_ai_margin"], mode="lines+markers", line=dict(color=PC[1], width=2), marker=dict(size=7)))
                fig_ot1.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Margin vs. OT Hours/Week", xaxis_title="OT Hrs/wk", yaxis=dict(tickformat=".1%"))
                st.plotly_chart(fig_ot1, use_container_width=True, config=_CHART_CONFIG)
            with c2:
                fig_ot2 = go.Figure(go.Scatter(x=ot_df["value"], y=ot_df["total_revenue"], mode="lines+markers", line=dict(color=PC[0], width=2), marker=dict(size=7)))
                fig_ot2.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Total Revenue vs. OT Hours/Week", xaxis_title="OT Hrs/wk", yaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig_ot2, use_container_width=True, config=_CHART_CONFIG)
            _fmt_table(ot_df[["value","ebitda_margin","ebitda_ai_margin","total_revenue","ebitda_ai"]].rename(columns={"value":"OT Hrs/wk","ebitda_margin":"Oper. Margin","ebitda_ai_margin":"Net Margin","total_revenue":"Total Revenue","ebitda_ai":"Net Income"}), dollar_cols=["Total Revenue","Net Income"], pct_cols=["Oper. Margin","Net Margin"])

        with s5:
            section("How utilization and wage rates affect margin and credit need")
            st.caption("Utilization = % of scheduled hours actually billed. Wage inflation = inspector base pay increase.")
            sa, sb = st.columns(2)
            with sa:
                section("Inspector Utilization Rate")
                util_vals = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
                with st.spinner("Running sensitivity..."):
                    util_df = run_sensitivity(a, hc, "inspector_utilization", util_vals)
                util_df["pct"] = (util_df["value"] * 100).round(0).astype(int)
                fig_u = go.Figure()
                fig_u.add_trace(go.Scatter(x=util_df["pct"], y=util_df["ebitda_ai_margin"], mode="lines+markers", name="Net Margin", line=dict(color=PC[1], width=2), marker=dict(size=7)))
                fig_u.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                fig_u.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Margin vs. Utilization Rate", xaxis_title="Utilization (%)", yaxis=dict(tickformat=".1%"))
                st.plotly_chart(fig_u, use_container_width=True, config=_CHART_CONFIG)
                _fmt_table(util_df[["pct","ebitda_ai_margin","ebitda_ai","peak_loc"]].rename(columns={"pct":"Utilization (%)","ebitda_ai_margin":"Net Margin","ebitda_ai":"Net Income","peak_loc":"Peak Credit"}), dollar_cols=["Net Income","Peak Credit"], pct_cols=["Net Margin"])
            with sb:
                section("Inspector Wage Inflation")
                base_wage = float(a.get("inspector_wage", 20.0))
                wage_vals = [max(12.0, base_wage - 3), base_wage - 2, base_wage - 1, base_wage, base_wage + 1, base_wage + 2, base_wage + 3, base_wage + 4]
                wage_vals = sorted(set(round(v, 2) for v in wage_vals))
                with st.spinner("Running sensitivity..."):
                    wage_df = run_sensitivity(a, hc, "inspector_wage", wage_vals)
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(x=wage_df["value"], y=wage_df["ebitda_ai_margin"], mode="lines+markers", name="Net Margin", line=dict(color=PC[2], width=2), marker=dict(size=7)))
                fig_w.add_shape(
                    type="line", x0=base_wage, x1=base_wage, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(dash="dot", color="#94A3B8", width=1)
                )
                fig_w.add_annotation(
                    x=base_wage, y=0.98, xref="x", yref="paper",
                    text="Current", font=dict(color="#94A3B8", size=10),
                    showarrow=False, textangle=-90,
                    xanchor="right", yanchor="top"
                )
                fig_w.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                fig_w.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Margin vs. Inspector Wage", xaxis_title="Wage ($/hr)", yaxis=dict(tickformat=".1%"))
                st.plotly_chart(fig_w, use_container_width=True, config=_CHART_CONFIG)
                _fmt_table(wage_df[["value","ebitda_ai_margin","ebitda_ai","peak_loc"]].rename(columns={"value":"Wage ($/hr)","ebitda_ai_margin":"Net Margin","ebitda_ai":"Net Income","peak_loc":"Peak Credit"}), dollar_cols=["Net Income","Peak Credit"], pct_cols=["Net Margin"])

        with s6:
            section("How write-off rate and inspector turnover affect the bottom line")
            sc, sd = st.columns(2)
            with sc:
                section("Bad Debt Write-Off Rate")
                bd_vals = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
                with st.spinner("Running sensitivity..."):
                    bd_df = run_sensitivity(a, hc, "bad_debt_pct", bd_vals)
                bd_df["pct_lbl"] = (bd_df["value"] * 100).round(1)
                fig_bd = go.Figure()
                fig_bd.add_trace(go.Scatter(x=bd_df["pct_lbl"], y=bd_df["ebitda_ai"], mode="lines+markers", name="Net Income", line=dict(color=PC[3], width=2), marker=dict(size=7)))
                fig_bd.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                fig_bd.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Total Net Income vs. Bad Debt Rate", xaxis_title="Write-Off Rate (%)", yaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig_bd, use_container_width=True, config=_CHART_CONFIG)
                _fmt_table(bd_df[["pct_lbl","ebitda_ai_margin","ebitda_ai"]].rename(columns={"pct_lbl":"Bad Debt (%)","ebitda_ai_margin":"Net Margin","ebitda_ai":"Net Income"}), dollar_cols=["Net Income"], pct_cols=["Net Margin"], highlight_neg="Net Income")
            with sd:
                section("Inspector Annual Turnover Rate")
                to_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                with st.spinner("Running sensitivity..."):
                    to_df = run_sensitivity(a, hc, "inspector_turnover_rate", to_vals)
                to_df["pct_lbl"] = (to_df["value"] * 100).round(0).astype(int)
                fig_to = go.Figure()
                fig_to.add_trace(go.Scatter(x=to_df["pct_lbl"], y=to_df["ebitda_ai"], mode="lines+markers", name="Net Income", line=dict(color=PC[4], width=2), marker=dict(size=7)))
                fig_to.add_hline(y=0, line_dash="dot", line_color="#EF4444", line_width=1)
                fig_to.add_shape(
                    type="line", x0=100, x1=100, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(dash="dot", color="#94A3B8", width=1)
                )
                fig_to.add_annotation(
                    x=100, y=0.98, xref="x", yref="paper",
                    text="100% (1x annual)", font=dict(color="#94A3B8", size=10),
                    showarrow=False, textangle=-90,
                    xanchor="right", yanchor="top"
                )
                fig_to.update_layout(template=TPL, height=240, margin=dict(l=10,r=10,t=30,b=10), title="Net Income vs. Inspector Turnover", xaxis_title="Annual Turnover Rate (%)", yaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig_to, use_container_width=True, config=_CHART_CONFIG)
                _fmt_table(to_df[["pct_lbl","ebitda_ai_margin","ebitda_ai"]].rename(columns={"pct_lbl":"Turnover (%)","ebitda_ai_margin":"Net Margin","ebitda_ai":"Net Income"}), dollar_cols=["Net Income"], pct_cols=["Net Margin"], highlight_neg="Net Income")

        st.divider()

        section("Cross-Variable Sensitivity Heatmaps")
        st.caption("Each cell shows the outcome when BOTH variables change simultaneously. Green = healthy, red = danger.")

        hm1, hm2, hm3 = st.tabs(["Bill Rate x Burden -> Net Margin", "Net Days x Inspectors -> Peak LOC", "Bill Rate x Net Days -> Interest Cost"])

        with hm1:
            if st.button("Run Heatmap: Bill Rate x Burden", use_container_width=True):
                bill_rates  = [37.0, 38.0, 39.0, 40.0, 41.0, 42.0]
                burden_pcts = [0.25, 0.28, 0.30, 0.33, 0.35, 0.38]
                rows = []
                prog = st.progress(0)
                total = len(bill_rates) * len(burden_pcts)
                done  = 0
                with st.spinner("Running heatmap..."):
                    for br in bill_rates:
                        for bu in burden_pcts:
                            _a2 = a.copy(); _a2["st_bill_rate"] = br; _a2["burden"] = bu
                            _df = run_sensitivity(_a2, hc, "st_bill_rate", [br])
                            rows.append({"Bill Rate": f"${br:.0f}", "Burden": f"{bu:.0%}", "Net Margin": float(_df["ebitda_ai_margin"].iloc[0])})
                            done += 1
                            prog.progress(done / total)
                prog.empty()
                hm_df = pd.DataFrame(rows)
                fig_hm = _heatmap(hm_df, "Bill Rate", "Burden", "Net Margin", "Net Income Margin -- Bill Rate x Burden", fmt=".1%")
                st.plotly_chart(fig_hm, use_container_width=True, config=_CHART_CONFIG)

        with hm2:
            if st.button("Run Heatmap: Net Days x Inspectors", use_container_width=True):
                net_days_list = [30, 45, 60, 90, 120]
                insp_counts   = [20, 30, 40, 50, 60, 75]
                rows = []
                prog2 = st.progress(0)
                total2 = len(net_days_list) * len(insp_counts)
                done2  = 0
                with st.spinner("Running heatmap..."):
                    for nd in net_days_list:
                        for ic in insp_counts:
                            _a2 = a.copy(); _a2["net_days"] = nd
                            _hc2 = [ic] * 120
                            _df = run_sensitivity(_a2, _hc2, "net_days", [nd])
                            rows.append({"Net Days": str(nd), "Inspectors": str(ic), "Peak LOC": float(_df["peak_loc"].iloc[0])})
                            done2 += 1
                            prog2.progress(done2 / total2)
                prog2.empty()
                hm_df2 = pd.DataFrame(rows)
                fig_hm2 = _heatmap(hm_df2, "Net Days", "Inspectors", "Peak LOC", "Peak Credit Line -- Net Days x Inspector Count", fmt="$,.0f", reverse=True)
                st.plotly_chart(fig_hm2, use_container_width=True, config=_CHART_CONFIG)

        with hm3:
            if st.button("Run Heatmap: Bill Rate x Net Days", use_container_width=True):
                bill_rates3  = [37.0, 38.0, 39.0, 40.0, 41.0, 42.0]
                net_days3    = [30, 45, 60, 90, 120, 150]
                rows3 = []
                prog3 = st.progress(0)
                total3 = len(bill_rates3) * len(net_days3)
                done3  = 0
                with st.spinner("Running heatmap..."):
                    for br3 in bill_rates3:
                        for nd3 in net_days3:
                            _a3 = a.copy(); _a3["st_bill_rate"] = br3; _a3["net_days"] = nd3
                            _df3 = run_sensitivity(_a3, hc, "st_bill_rate", [br3])
                            rows3.append({"Bill Rate": f"${br3:.0f}", "Net Days": str(nd3), "Annual Interest": float(_df3["annual_interest"].iloc[0])})
                            done3 += 1
                            prog3.progress(done3 / total3)
                prog3.empty()
                hm_df3 = pd.DataFrame(rows3)
                fig_hm3 = _heatmap(hm_df3, "Bill Rate", "Net Days", "Annual Interest", "Annual Interest Cost -- Bill Rate x Net Days", fmt="$,.0f", reverse=True)
                st.plotly_chart(fig_hm3, use_container_width=True, config=_CHART_CONFIG)

        st.divider()
        if st.button("Export Full Report + Sensitivity to Excel"):
            with st.spinner("Building..."):
                sens = {
                    "Sens_PayTerms": nd_df, "Sens_BillRate": br_df,
                    "Sens_Burden": burd_df, "Sens_OTHours": ot_df,
                }
                xlsx = build_excel(a, hc, weekly_df, mo_full, qdf_full, sens)
            st.download_button(
                "Download Excel", data=xlsx,
                file_name="containment_division_sensitivity.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
