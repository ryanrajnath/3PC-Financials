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
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Remove Streamlit chrome that overlaps content */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; height: 0; }

/* Layout */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* KPI cards */
.kpi-card {
    background: #1A1D27;
    border: 1px solid #2D3148;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    height: 100%;
}
.kpi-label { color: #8B8FA8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.kpi-value { color: #FAFAFA; font-size: 22px; font-weight: 700; line-height: 1.1; }
.kpi-sub   { color: #4F8BF9; font-size: 11px; margin-top: 3px; }

/* Alerts */
.stale-box { background:#2D2510; border-left:3px solid #F0A843; padding:8px 12px; border-radius:4px; font-size:13px; margin-bottom:8px; }
.warn-box  { background:#2D1F1F; border-left:3px solid #E05252; padding:8px 12px; border-radius:4px; font-size:13px; margin-bottom:6px; }
.info-box  { background:#1A2235; border-left:3px solid #4F8BF9; padding:8px 12px; border-radius:4px; font-size:13px; margin-bottom:6px; }

/* Section headers */
.sec-hdr {
    color: #8B8FA8; font-size: 10px; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 6px 0 4px 0;
    border-bottom: 1px solid #2D3148; padding-bottom: 2px;
}

/* Sidebar width */
[data-testid="stSidebar"] { min-width: 320px !important; max-width: 320px !important; }
[data-testid="stSidebar"] .block-container { padding-top: 0.75rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

PC  = ["#4F8BF9", "#52D68A", "#F0A843", "#E05252", "#A855F7", "#22D3EE"]
TPL = "plotly_dark"


# ── Session state ─────────────────────────────────────────────────────────────
if "assumptions"    not in st.session_state: st.session_state.assumptions    = default_assumptions()
if "headcount_plan" not in st.session_state: st.session_state.headcount_plan = default_headcount()
if "results"        not in st.session_state: st.session_state.results        = None
if "run_hash"       not in st.session_state: st.session_state.run_hash       = None
if "run_ts"         not in st.session_state: st.session_state.run_ts         = None
if "bootstrapped"   not in st.session_state: st.session_state.bootstrapped   = False


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
                      margin=dict(l=10, r=10, t=36, b=10),
                      legend=dict(orientation="h", y=-0.25),
                      yaxis=dict(tickformat=".0%" if pct_y else "$,.0f"))
    fig.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
    return fig

def _bar(df, x, ys, names, title):
    fig = go.Figure()
    for y, nm, c in zip(ys, names, PC):
        if y in df.columns:
            fig.add_trace(go.Bar(x=df[x], y=df[y], name=nm, marker_color=c))
    fig.update_layout(template=TPL, title=title, barmode="stack", height=300,
                      margin=dict(l=10, r=10, t=36, b=10),
                      legend=dict(orientation="h", y=-0.25),
                      yaxis=dict(tickformat="$,.0f"))
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

def _range_slider(key, mo, label="Date range"):
    active = mo[mo["revenue"] > 0]
    last   = int(active["month_idx"].max()) + 1 if not active.empty else 12
    hi_def = min(last + 3, len(mo))
    lo, hi = st.select_slider(
        label, options=list(range(1, len(mo) + 1)),
        value=(1, hi_def), key=key,
        help="Filter charts and tables to this month range."
    )
    return mo[(mo["month_idx"] >= lo - 1) & (mo["month_idx"] <= hi - 1)]


# ── Scenario presets ──────────────────────────────────────────────────────────
def _build_hc(rules):
    hc = [0] * 120
    for start, end, val in rules:
        for i in range(start - 1, min(end, 120)):
            hc[i] = val
    return hc

PRESETS = {
    "Conservative Launch": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate": 38.0, "ot_hours": 5, "burden": 0.33,
            "team_lead_ratio": 15, "lead_wage": 24.0, "lead_ot_hours": 5,
            "net_days": 90, "apr": 0.095, "max_loc": 500_000,
            "software_monthly": 300, "recruiting_monthly": 500,
            "insurance_monthly": 1000, "travel_monthly": 300},
        "headcount": _build_hc([(1,3,10),(4,6,15),(7,12,20),(13,24,25)]),
    },
    "Aggressive Growth": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate": 41.0, "ot_hours": 15, "inspector_wage": 21.0,
            "lead_wage": 26.0, "lead_ot_hours": 15,
            "gm_loaded_annual": 130_000, "opscoord_base": 70_000,
            "fieldsup_base": 75_000, "regionalmgr_base": 120_000,
            "net_days": 120, "max_loc": 2_000_000, "cash_buffer": 50_000,
            "initial_cash": 100_000, "software_monthly": 1000,
            "recruiting_monthly": 3000, "insurance_monthly": 2500,
            "travel_monthly": 1500},
        "headcount": _build_hc([(1,2,25),(3,4,50),(5,6,75),(7,9,100),
                                 (10,12,125),(13,24,150),(25,60,175)]),
    },
    "Steady State": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate": 39.5, "ot_hours": 8, "inspector_wage": 20.5,
            "lead_ot_hours": 8, "gm_loaded_annual": 120_000,
            "opscoord_base": 67_000, "fieldsup_base": 72_000,
            "regionalmgr_base": 115_000, "net_days": 45, "apr": 0.075,
            "max_loc": 750_000, "cash_buffer": 50_000, "initial_cash": 50_000,
            "software_monthly": 500, "recruiting_monthly": 750,
            "insurance_monthly": 1500, "travel_monthly": 500},
        "headcount": [60] * 120,
    },
}

# ── Auto-bootstrap on first load ──────────────────────────────────────────────
if not st.session_state.bootstrapped:
    p = PRESETS["Steady State"]
    st.session_state.assumptions    = p["assumptions"].copy()
    st.session_state.headcount_plan = p["headcount"].copy()
    st.session_state.bootstrapped   = True
    run_and_store()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — All inputs
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    a = st.session_state.assumptions

    # ── Calculate button ─────────────────────────────────────────────────────
    btn_lbl = "▶  Calculate" + (f"  ·  {st.session_state.run_ts}" if st.session_state.run_ts else "")
    if st.button(btn_lbl, type="primary", use_container_width=True):
        run_and_store()
        st.rerun()

    if is_stale():
        st.markdown(
            '<div class="stale-box">Inputs changed — click <b>Calculate</b> to update.</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Preset scenarios ─────────────────────────────────────────────────────
    section("Scenario Presets")
    preset_choice = st.selectbox(
        "Scenario", ["— select —"] + list(PRESETS.keys()),
        label_visibility="collapsed",
        help="Load a pre-built scenario as a starting point. You can adjust any value after loading."
    )
    if preset_choice != "— select —":
        if st.button(f"Load  '{preset_choice}'", use_container_width=True):
            p = PRESETS[preset_choice]
            st.session_state.assumptions    = p["assumptions"].copy()
            st.session_state.headcount_plan = p["headcount"].copy()
            a = st.session_state.assumptions
            st.rerun()

    st.divider()

    # ── Pricing ───────────────────────────────────────────────────────────────
    section("Pricing")
    a["st_bill_rate"] = st.slider(
        "Bill Rate ($/hr, regular time)",
        min_value=28.0, max_value=65.0, value=float(a["st_bill_rate"]),
        step=0.50, format="$%.2f",
        help="Hourly rate charged to the customer per inspector during regular work hours."
    )
    a["ot_hours"] = st.slider(
        "Planned Overtime Hours / Inspector / Week",
        min_value=0, max_value=25, value=int(a["ot_hours"]), step=1,
        help="Average overtime hours per inspector per week. OT is billed at 1.5× the regular rate. Set to 0 if you don't plan overtime."
    )
    a["ot_bill_premium"] = st.number_input(
        "OT Billing Multiplier", min_value=1.0, max_value=3.0,
        value=float(a["ot_bill_premium"]), step=0.25, format="%.2f",
        help=f"Overtime hours are billed at this multiple of the regular rate. At {a['ot_bill_premium']}×, OT bills at ${a['st_bill_rate'] * a['ot_bill_premium']:.2f}/hr."
    )
    a["st_hours"] = st.number_input(
        "Regular Hours / Inspector / Week", min_value=20, max_value=60,
        value=int(a["st_hours"]), step=1, format="%d",
        help="Standard work hours per inspector per week, not counting overtime. Typically 40."
    )

    a["lead_bill_premium"] = st.number_input(
        "Team Lead Bill Rate Multiplier", min_value=1.0, max_value=2.0,
        value=float(a.get("lead_bill_premium", 1.0)), step=0.05, format="%.2f",
        help="Team leads are billed at this multiple of the regular inspector rate. "
             "1.0 = same rate as inspectors (most common). 1.1 = 10% premium for supervisory labor."
    )

    # Live margin preview
    _ot_bill    = a["st_bill_rate"] * a["ot_bill_premium"]
    _util       = float(a.get("inspector_utilization", 1.0))
    _wk_rev     = _util * (a["st_hours"] * a["st_bill_rate"] + a["ot_hours"] * _ot_bill)
    _ob_wk      = float(a.get("inspector_onboarding_cost", 500)) / max(1, int(a.get("inspector_avg_tenure_weeks", 26)))
    _wk_cost    = (a["st_hours"] * a["inspector_wage"] * (1 + a["burden"]) +
                   a["ot_hours"] * a["inspector_wage"] * a["ot_pay_multiplier"] * (1 + a["burden"]) +
                   _ob_wk)
    _margin     = (_wk_rev - _wk_cost) / _wk_rev if _wk_rev else 0
    st.caption(f"Per inspector/week: **${_wk_rev:,.0f} revenue** · **${_wk_cost:,.0f} cost** · **{_margin:.0%} margin**")

    st.divider()

    # ── Inspector Pay ─────────────────────────────────────────────────────────
    section("Inspector Pay")
    a["inspector_wage"] = st.number_input(
        "Hourly Wage ($/hr)", min_value=10.0, max_value=50.0,
        value=float(a["inspector_wage"]), step=0.25, format="%.2f",
        help="Base hourly pay for inspectors before employer taxes and benefits."
    )
    _burden_pct = st.slider(
        "Payroll Burden (%)",
        min_value=15, max_value=55, value=int(round(a["burden"] * 100)), step=1,
        format="%d%%",
        help="Employer cost on top of wages: FICA (~7.65%), state/federal unemployment, workers' comp, and benefits. 30% is a common starting estimate."
    )
    a["burden"] = _burden_pct / 100.0
    a["ot_pay_multiplier"] = st.number_input(
        "OT Pay Multiplier", min_value=1.0, max_value=3.0,
        value=float(a["ot_pay_multiplier"]), step=0.25, format="%.2f",
        help=f"Inspectors earn this multiple of base pay for overtime. At {a['ot_pay_multiplier']}×: ${a['inspector_wage'] * a['ot_pay_multiplier']:.2f}/hr OT pay."
    )
    _util_pct = st.slider(
        "Inspector Utilization Rate (%)", min_value=50, max_value=100,
        value=int(round(float(a.get("inspector_utilization", 1.0)) * 100)), step=1, format="%d%%",
        help="Percentage of scheduled inspector hours that are actually billed to the client. "
             "100% assumes every scheduled hour is productive and billable. In practice, "
             "travel days, site startups, and between-project gaps reduce this — 85–95% is typical. "
             "Note: you still pay inspectors for all scheduled hours; only revenue is reduced."
    )
    a["inspector_utilization"] = _util_pct / 100.0
    a["inspector_onboarding_cost"] = st.number_input(
        "Inspector Onboarding Cost per Hire ($)", min_value=0.0,
        value=float(a.get("inspector_onboarding_cost", 500.0)), step=50., format="%.0f",
        help="One-time cost each time you hire a new inspector: background check (~$150), "
             "drug screen (~$80), PPE, and orientation (~$270). Amortized weekly over expected tenure."
    )
    a["inspector_avg_tenure_weeks"] = st.number_input(
        "Average Inspector Tenure (weeks)", min_value=4, max_value=260,
        value=int(a.get("inspector_avg_tenure_weeks", 26)), step=4, format="%d",
        help="How long the average inspector stays before leaving. 26 weeks (~6 months) is typical "
             "for third-party containment — high-turnover, physically demanding work. "
             "Shorter tenure = higher amortized onboarding cost per week."
    )
    _ob_per_wk = a["inspector_onboarding_cost"] / max(1, a["inspector_avg_tenure_weeks"])
    st.caption(f"Onboarding adds **${_ob_per_wk:.2f}/inspector/week** to labor cost")

    st.divider()

    # ── Team Leads ────────────────────────────────────────────────────────────
    section("Team Leads")
    st.caption("Hourly supervisors billed to the client like inspectors, at a higher wage.")
    a["team_lead_ratio"] = st.slider(
        "Inspectors per Team Lead",
        min_value=5, max_value=25, value=int(a["team_lead_ratio"]), step=1,
        help="One team lead is assigned for every N inspectors (rounded up). At 12: a crew of 25 inspectors has 3 team leads."
    )
    a["lead_wage"] = st.number_input(
        "Team Lead Hourly Wage ($/hr)", min_value=10.0, max_value=60.0,
        value=float(a["lead_wage"]), step=0.25, format="%.2f",
        help="Base hourly pay for team leads before burden. Typically $3–6 more per hour than inspector wage."
    )
    a["lead_ot_hours"] = st.number_input(
        "Team Lead OT Hours / Week", min_value=0, max_value=25,
        value=int(a["lead_ot_hours"]), step=1, format="%d",
        help="Overtime hours per team lead per week. Usually matches inspector OT hours."
    )
    a["lead_st_hours"] = st.number_input(
        "Team Lead Regular Hours / Week", min_value=20, max_value=60,
        value=int(a["lead_st_hours"]), step=1, format="%d",
        help="Standard (non-OT) work hours per team lead per week."
    )

    st.divider()

    # ── Customer Payment Terms ────────────────────────────────────────────────
    section("Customer Payment Terms")
    _billing_options = ["Monthly (standard)", "Weekly (faster cash flow)"]
    _billing_idx = 0 if a.get("billing_frequency", "monthly") == "monthly" else 1
    _billing_sel = st.radio(
        "Invoice Frequency",
        _billing_options,
        index=_billing_idx,
        horizontal=True,
        help=(
            "**Monthly:** One invoice sent at month-end for all work that month. Standard practice but "
            "means your credit line must fund 4–5 weeks of payroll before any cash comes in.\n\n"
            "**Weekly:** Invoice every Friday for that week's work. Collections arrive ~8–9 weeks sooner "
            "on average, reducing peak credit line need by $200K–$400K depending on headcount."
        )
    )
    a["billing_frequency"] = "monthly" if _billing_sel == "Monthly (standard)" else "weekly"
    a["net_days"] = st.slider(
        "Days to Payment After Month-End Invoice",
        min_value=15, max_value=180, value=int(a["net_days"]), step=5,
        help="How long after you send the month-end invoice until customers typically pay. This is the single biggest driver of how large your credit line needs to be. Net 60 is standard in third-party containment."
    )
    a["start_date"] = st.date_input(
        "Model Start Date", value=a["start_date"],
        help="The first day of the model. All dates, months, and years are calculated forward from here."
    )

    st.divider()

    # ── Credit Line ───────────────────────────────────────────────────────────
    section("Credit Line")
    st.caption("Funds payroll while waiting for customers to pay their invoices.")
    a["max_loc"] = st.number_input(
        "Maximum Credit Line ($)", min_value=0.0,
        value=float(a["max_loc"]), step=50_000., format="%.0f",
        help="Your bank's maximum credit limit. The model warns you if cash needs exceed this amount."
    )
    _apr_pct = st.slider(
        "Annual Interest Rate (%)",
        min_value=4.0, max_value=20.0, value=round(float(a["apr"]) * 100, 2),
        step=0.25, format="%.2f%%",
        help="APR charged on the outstanding credit line balance. Interest is calculated monthly on the average weekly balance."
    )
    a["apr"] = _apr_pct / 100.0
    a["initial_cash"] = st.number_input(
        "Starting Cash ($)", min_value=0.0,
        value=float(a["initial_cash"]), step=5_000., format="%.0f",
        help="Cash on hand at the start of the model. Enter 0 if the division launches with no cash reserves."
    )
    a["cash_buffer"] = st.number_input(
        "Minimum Cash Reserve ($)", min_value=0.0,
        value=float(a["cash_buffer"]), step=5_000., format="%.0f",
        help="The model keeps at least this much cash on hand at all times, drawing on the credit line when needed. Acts as a weekly payroll safety net."
    )
    a["auto_paydown"] = st.checkbox(
        "Auto-repay credit line when cash exceeds reserve",
        value=bool(a["auto_paydown"]),
        help="When ON, any cash above the minimum reserve automatically pays down the credit line balance to reduce interest costs."
    )
    _mo_int = (a["apr"] / 12) * a["max_loc"]
    st.caption(f"Interest at full draw: **${_mo_int:,.0f}/month**")

    st.divider()

    # ── Salaried Management ───────────────────────────────────────────────────
    with st.expander("Salaried Management"):
        st.caption("These roles are added automatically as inspector count grows. Each costs the same every week regardless of hours worked.")

        a["gm_loaded_annual"] = st.number_input(
            "General Manager — Total Annual Cost ($)", min_value=0.0,
            value=float(a["gm_loaded_annual"]), step=1_000., format="%.0f",
            help="All-in annual cost for the GM including salary, bonus, and benefits. Enter the fully loaded number — do not add burden on top. Active from Month 1."
        )

        st.markdown("**Operations Coordinator**")
        st.caption("Manages scheduling, dispatch, and client communication from the office.")
        c1, c2 = st.columns(2)
        a["opscoord_base"] = c1.number_input(
            "Base Salary ($)", min_value=0.0, value=float(a["opscoord_base"]),
            step=1_000., format="%.0f", key="oc_sal",
            help="Annual base salary. The management burden rate is added on top to get fully loaded cost."
        )
        a["opscoord_span"] = c2.number_input(
            "Per N inspectors", min_value=10, value=int(a["opscoord_span"]),
            step=5, format="%d", key="oc_sp",
            help="One Ops Coordinator is added per N inspectors. Default: 1 per 75 inspectors."
        )

        st.markdown("**Field Supervisor**")
        st.caption("Travels to job sites and directly manages inspector crews in the field.")
        c1, c2 = st.columns(2)
        a["fieldsup_base"] = c1.number_input(
            "Base Salary ($)", min_value=0.0, value=float(a["fieldsup_base"]),
            step=1_000., format="%.0f", key="fs_sal",
            help="Annual base salary. Management burden is added on top."
        )
        a["fieldsup_span"] = c2.number_input(
            "Per N inspectors", min_value=5, value=int(a["fieldsup_span"]),
            step=5, format="%d", key="fs_sp",
            help="One Field Supervisor per N inspectors. Default: 1 per 25 (each supervisor oversees one crew of ~25 workers across multiple sites)."
        )

        st.markdown("**Regional Manager**")
        st.caption("Oversees Field Supervisors across a geographic area. Typically not needed until 175+ inspectors.")
        c1, c2 = st.columns(2)
        a["regionalmgr_base"] = c1.number_input(
            "Base Salary ($)", min_value=0.0, value=float(a["regionalmgr_base"]),
            step=1_000., format="%.0f", key="rm_sal",
            help="Annual base salary. Management burden is added on top."
        )
        a["regionalmgr_span"] = c2.number_input(
            "Per N inspectors", min_value=50, value=int(a["regionalmgr_span"]),
            step=10, format="%d", key="rm_sp",
            help="One Regional Manager per N inspectors. Default: 1 per 175."
        )

        _mgmt_burden_pct = st.slider(
            "Management Burden Rate (%)", min_value=10, max_value=40,
            value=int(round(a["mgmt_burden"] * 100)), step=1, format="%d%%",
            help="Employer taxes and benefits applied to management base salaries. Does NOT apply to the GM (GM cost is already fully loaded)."
        )
        a["mgmt_burden"] = _mgmt_burden_pct / 100.0

        def _loaded(base): return base * (1 + a["mgmt_burden"])
        st.caption(
            f"Fully loaded: OC **${_loaded(a['opscoord_base']):,.0f}** · "
            f"FS **${_loaded(a['fieldsup_base']):,.0f}** · "
            f"RM **${_loaded(a['regionalmgr_base']):,.0f}**"
        )

    # ── Management Turnover ───────────────────────────────────────────────────
    with st.expander("Management Turnover & Replacement Cost"):
        st.caption(
            "The model estimates the ongoing cost of replacing management roles when they turn over — "
            "recruiting, background screening, and lost productivity during ramp-up. "
            "This cost scales with how many of each role you have active."
        )
        st.markdown("**Industry benchmarks (BLS JOLTS + SHRM 2024 — field operations sector)**")

        _oc_to_pct = st.slider(
            "Ops Coordinator — Annual Turnover (%)",
            min_value=10, max_value=70, value=int(round(a.get("opscoord_turnover", 0.35) * 100)),
            step=1, format="%d%%",
            help="What fraction of your Ops Coordinators leave each year. At 35%, a team of 2 loses roughly 1 person every 18 months. Scheduling roles see high turnover due to stress and limited advancement."
        )
        a["opscoord_turnover"] = _oc_to_pct / 100.0
        a["opscoord_replace_cost"] = st.number_input(
            "Ops Coordinator — Replacement Cost ($)", min_value=0.0,
            value=float(a.get("opscoord_replace_cost", 8_000)), step=500., format="%.0f",
            key="oc_rc",
            help="Total cost to replace one Ops Coordinator: job board posting, background/drug screen, HR time, and ~4 weeks at reduced productivity. Industry average: ~$8,000."
        )

        _fs_to_pct = st.slider(
            "Field Supervisor — Annual Turnover (%)",
            min_value=10, max_value=60, value=int(round(a.get("fieldsup_turnover", 0.25) * 100)),
            step=1, format="%d%%",
            help="Field Supervisors are frequently poached by OEMs and Tier-1 suppliers offering higher base pay. At 25%, you replace roughly 1 in 4 per year."
        )
        a["fieldsup_turnover"] = _fs_to_pct / 100.0
        a["fieldsup_replace_cost"] = st.number_input(
            "Field Supervisor — Replacement Cost ($)", min_value=0.0,
            value=float(a.get("fieldsup_replace_cost", 12_000)), step=500., format="%.0f",
            key="fs_rc",
            help="Higher than Ops Coord because hands-on containment experience is required. May include a recruiter fee. Industry average: ~$12,000."
        )

        _rm_to_pct = st.slider(
            "Regional Manager — Annual Turnover (%)",
            min_value=5, max_value=40, value=int(round(a.get("regionalmgr_turnover", 0.18) * 100)),
            step=1, format="%d%%",
            help="Better compensation and authority improve retention at this level. At 18%, you replace a Regional Manager roughly every 5–6 years."
        )
        a["regionalmgr_turnover"] = _rm_to_pct / 100.0
        a["regionalmgr_replace_cost"] = st.number_input(
            "Regional Manager — Replacement Cost ($)", min_value=0.0,
            value=float(a.get("regionalmgr_replace_cost", 25_000)), step=1_000., format="%.0f",
            key="rm_rc",
            help="Often requires a third-party recruiter (15–20% of base salary) plus ramp time. Industry average: ~$25,000."
        )

    # ── Monthly Fixed Overhead ────────────────────────────────────────────────
    with st.expander("Monthly Fixed Overhead"):
        st.caption("Charged every month whether you have 5 inspectors or 500.")
        a["software_monthly"]   = st.number_input(
            "Software & Technology ($/mo)", min_value=0.0,
            value=float(a["software_monthly"]), step=100., format="%.0f",
            help="Scheduling, time-tracking, field management software, and reporting tools."
        )
        a["recruiting_monthly"] = st.number_input(
            "Inspector Recruiting ($/mo)", min_value=0.0,
            value=float(a["recruiting_monthly"]), step=100., format="%.0f",
            help="Job boards, background checks, drug screening, and onboarding costs for inspectors. Management replacement costs are tracked separately in the Turnover section."
        )
        a["insurance_monthly"]  = st.number_input(
            "Insurance ($/mo)", min_value=0.0,
            value=float(a["insurance_monthly"]), step=100., format="%.0f",
            help="General liability, errors & omissions, commercial auto, and any coverage not included in payroll burden."
        )
        a["travel_monthly"]     = st.number_input(
            "Travel & Field Expenses ($/mo)", min_value=0.0,
            value=float(a["travel_monthly"]), step=100., format="%.0f",
            help="Mileage reimbursement, lodging, and per diem for management and supervisory travel. Inspector mileage is typically billed to clients separately."
        )

        ca_mode = st.radio(
            "Corporate Overhead Allocation",
            ["Fixed monthly amount", "Percentage of revenue"], horizontal=True,
            index=0 if a["corp_alloc_mode"] == "fixed" else 1,
            help="A charge from the parent company for shared services (accounting, HR, IT, etc.). Default $0 — enter only if OpSource charges this division."
        )
        a["corp_alloc_mode"] = "fixed" if ca_mode == "Fixed monthly amount" else "pct_revenue"
        if a["corp_alloc_mode"] == "fixed":
            a["corp_alloc_fixed"] = st.number_input(
                "Corporate Allocation ($/mo)", min_value=0.0,
                value=float(a["corp_alloc_fixed"]), step=500., format="%.0f",
                help="Fixed monthly charge from the parent company. Default $0."
            )
        else:
            _ca_pct_val = st.number_input(
                "Corporate Allocation (% of revenue)", min_value=0.0, max_value=20.0,
                value=float(a["corp_alloc_pct"]) * 100, step=0.5, format="%.1f",
                help="Variable corporate charge as a percentage of gross revenue."
            )
            a["corp_alloc_pct"] = _ca_pct_val / 100.0

        _total_fixed = (a["software_monthly"] + a["recruiting_monthly"] +
                        a["insurance_monthly"] + a["travel_monthly"] +
                        (a["corp_alloc_fixed"] if a["corp_alloc_mode"] == "fixed" else 0))
        st.caption(f"Total fixed overhead: **${_total_fixed:,.0f}/month**")

    # ── Tax Rates (provision only) ────────────────────────────────────────────
    with st.expander("Tax Rates"):
        st.caption(
            "OpSource is a pass-through entity (S-corp or LLC) — income taxes are paid at the "
            "owner level, not the division level. **Pre-tax net income is the correct metric for "
            "evaluating this division's performance.** The tax provision below is shown for "
            "reference only, to estimate the owner's approximate tax obligation on division profits. "
            "Verify the applicable rate with your CPA."
        )
        _sc_pct = st.slider(
            "SC State Tax Rate (%)", min_value=0, max_value=15,
            value=int(round(float(a.get("sc_state_tax_rate", 0.059)) * 100 * 10) / 10),
            step=1, format="%d%%",
            help="South Carolina corporate/pass-through income tax rate. SC's rate has been reducing annually — 5.9% applies to 2026. Verify with your CPA."
        )
        a["sc_state_tax_rate"] = _sc_pct / 100.0
        _fed_pct = st.slider(
            "Federal Corporate Tax Rate (%)", min_value=0, max_value=40,
            value=int(round(float(a.get("federal_tax_rate", 0.21)) * 100)),
            step=1, format="%d%%",
            help="Federal corporate income tax rate. Currently 21% for C-corps. If OpSource is an LLC or S-corp, use the applicable individual rate instead."
        )
        a["federal_tax_rate"] = _fed_pct / 100.0
        _combined = a["sc_state_tax_rate"] + a["federal_tax_rate"]
        st.caption(f"Combined rate: **{_combined:.1%}**  ·  On $100K net income: **${_combined * 100_000:,.0f} in taxes**")

    st.session_state.assumptions = a


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Header + KPIs
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## Containment Division Calculator")
st.caption("OpSource · Weekly cash flow model · 10-year (120-month) horizon")

if is_stale():
    st.markdown(
        '<div class="stale-box">You have changed inputs since the last calculation. '
        'Click <b>Calculate</b> in the sidebar to update results.</div>',
        unsafe_allow_html=True
    )

if results_ready():
    _, mo_h, _ = st.session_state.results
    k1, k2, k3, k4, k5 = st.columns(5)
    peak_mo = mo_h.loc[mo_h["loc_end"].idxmax(), "period"] if mo_h["loc_end"].max() > 0 else "—"
    kpi(k1, "Peak Credit Line",      fmt_dollar(mo_h["loc_end"].max()),                    peak_mo)
    kpi(k2, "10-Year Revenue",        fmt_dollar(mo_h["revenue"].sum()),                    "billed (accrual)")
    kpi(k3, "10-Year Net Income",     fmt_dollar(mo_h["ebitda_after_interest"].sum()),       "after interest")
    kpi(k4, "Total Borrowing Cost",   fmt_dollar(mo_h["interest"].sum()),                   "credit line interest")
    yr1 = mo_h[mo_h["month_idx"] < 12]
    kpi(k5, "Year 1 Net Income",      fmt_dollar(yr1["ebitda_after_interest"].sum()),        "first 12 months")
    st.divider()


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_dash, tab_fin, tab_hc, tab_sum, tab_sens = st.tabs([
    "Dashboard", "Financials", "Headcount Plan", "Scenario Summary", "Sensitivity"
])


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_dash:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Calculate</b> in the sidebar to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a = st.session_state.assumptions

    n_loc  = weekly_df["warn_loc_maxed"].sum()  if "warn_loc_maxed"  in weekly_df.columns else 0
    n_mgmt = weekly_df["warn_mgmt_no_insp"].sum() if "warn_mgmt_no_insp" in weekly_df.columns else 0
    if n_loc:
        st.markdown(
            f'<div class="warn-box">Your credit line was maxed out in {n_loc} week(s). '
            f'Consider a larger credit facility, reducing headcount, or negotiating faster payment terms.</div>',
            unsafe_allow_html=True
        )
    if n_mgmt:
        st.markdown(
            f'<div class="warn-box">You are paying salaried management in {n_mgmt} week(s) '
            f'with no inspectors staffed. Check your headcount plan.</div>',
            unsafe_allow_html=True
        )

    section("Date Range")
    mo = _range_slider("dash_rng", mo_full)
    lo_idx = mo["month_idx"].min(); hi_idx = mo["month_idx"].max()
    qdf = qdf_full[(qdf_full["quarter_idx"] >= lo_idx // 3) & (qdf_full["quarter_idx"] <= hi_idx // 3)].copy()
    wdf = weekly_df[(weekly_df["month_idx"] >= lo_idx) & (weekly_df["month_idx"] <= hi_idx)].copy()

    st.divider()

    # Credit Line / Cash / AR
    section("Credit Line, Cash & Accounts Receivable")
    fig_loc = go.Figure()
    fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["loc_end"],  name="Credit Line Balance",
        fill="tozeroy", line=dict(color=PC[3], width=2)))
    fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["ar_end"],   name="Accounts Receivable",
        line=dict(color=PC[0], width=2)))
    fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["cash_end"], name="Cash on Hand",
        line=dict(color=PC[1], width=2)))
    fig_loc.add_hline(y=float(a["max_loc"]), line_dash="dot", line_color=PC[3],
                      annotation_text="Credit Limit", annotation_font_color=PC[3])
    fig_loc.update_layout(template=TPL, height=330, margin=dict(l=10,r=10,t=10,b=10),
                          legend=dict(orientation="h", y=-0.2), yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_loc, use_container_width=True)

    # Waterfall
    section("Cash Flow Summary — Selected Period")
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
        x=["Starting Cash","+ Customer Payments","− Hourly Payroll","− Salaried Mgmt",
           "− Overhead","− Interest","+ Credit Draw","− Credit Repay","Ending Cash"],
        y=wf_values,
        connector=dict(line=dict(color="#444")),
        increasing=dict(marker_color=PC[1]),
        decreasing=dict(marker_color=PC[3]),
        totals=dict(marker_color=PC[0]),
    ))
    fig_wf.update_layout(template=TPL, height=300, margin=dict(l=10,r=10,t=10,b=10),
                         yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_wf, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Revenue vs. Costs")
        fig_rv = _bar(mo, "period", ["hourly_labor","salaried_cost","overhead"],
                      ["Hourly Labor","Salaried Mgmt","Overhead"], "Monthly Cost Stack")
        fig_rv.add_trace(go.Scatter(x=mo["period"], y=mo["revenue"], name="Revenue",
            mode="lines", line=dict(color=PC[1], width=2)))
        st.plotly_chart(fig_rv, use_container_width=True)
    with c2:
        section("Net Income (After Interest)")
        st.plotly_chart(_line(mo, "period",
            ["ebitda", "ebitda_after_interest"],
            ["Operating Profit (EBITDA)", "Net Income (after interest)"],
            "Monthly Profit"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("Profit Margins")
        st.plotly_chart(_line(mo, "period",
            ["ebitda_margin", "ebitda_ai_margin"],
            ["Operating Margin", "Net Margin"],
            "Monthly Margins", pct_y=True), use_container_width=True)
    with c4:
        section("Staffing by Role")
        st.plotly_chart(_bar(mo, "period",
            ["inspectors_avg","team_leads_avg","n_opscoord","n_fieldsup","n_regionalmgr"],
            ["Inspectors","Team Leads","Ops Coordinators","Field Supervisors","Regional Managers"],
            "Average Monthly Headcount"), use_container_width=True)

    st.divider()

    # Payroll Float / Working Capital Gap
    section("Payroll Float — Cash Out vs. Cash In")
    st.caption(
        "Your credit line bridges the gap between when you pay employees (weekly) and when "
        "customers pay their invoices. The chart below shows weekly cash obligations vs. "
        "cash coming in — the difference is your working capital requirement."
    )
    wf_cols = ["payroll_cash_out", "salaried_wk", "overhead_wk", "collections", "loc_end"]
    if all(c in wdf.columns for c in ["payroll_cash_out", "salaried_wk", "overhead_wk"]):
        wdf_pf = wdf.copy()
        wdf_pf["total_cash_out"] = wdf_pf["payroll_cash_out"] + wdf_pf["salaried_wk"] + wdf_pf["overhead_wk"]

        fig_pf = go.Figure()
        fig_pf.add_trace(go.Bar(
            x=wdf_pf["week_start"].astype(str), y=wdf_pf["total_cash_out"],
            name="Cash Out (Payroll + Overhead)", marker_color=PC[3], opacity=0.75
        ))
        fig_pf.add_trace(go.Bar(
            x=wdf_pf["week_start"].astype(str), y=wdf_pf["collections"],
            name="Cash In (Customer Payments)", marker_color=PC[1], opacity=0.75
        ))
        fig_pf.add_trace(go.Scatter(
            x=wdf_pf["week_start"].astype(str), y=wdf_pf["loc_end"],
            name="Credit Line Balance", mode="lines",
            line=dict(color=PC[2], width=2, dash="dot"), yaxis="y2"
        ))
        fig_pf.update_layout(
            template=TPL, height=320, barmode="overlay",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(tickformat="$,.0f", title="Weekly Cash ($)"),
            yaxis2=dict(tickformat="$,.0f", title="Credit Line ($)",
                        overlaying="y", side="right", showgrid=False),
        )
        fig_pf.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
        st.plotly_chart(fig_pf, use_container_width=True)

        # Working capital metrics
        pf1, pf2, pf3 = st.columns(3)
        peak_weekly_out = wdf_pf["total_cash_out"].max()
        avg_weekly_gap  = (wdf_pf["total_cash_out"] - wdf_pf["collections"]).mean()
        peak_loc_val    = wdf_pf["loc_end"].max()
        kpi(pf1, "Peak Weekly Cash Out",   fmt_dollar(peak_weekly_out), "payroll + overhead")
        kpi(pf2, "Avg Weekly Funding Gap", fmt_dollar(avg_weekly_gap),  "cash out minus cash in")
        kpi(pf3, "Peak Credit Line Draw",  fmt_dollar(peak_loc_val),    "max balance outstanding")

    # Break-even
    section("Break-Even Calculator")
    st.caption("Find the minimum number of inspectors needed to be profitable in Year 1 at a given payment term.")
    bc1, bc2, bc3 = st.columns(3)
    be_nd = bc1.selectbox(
        "Payment Terms to Test", [30, 45, 60, 90, 120, 150], index=2,
        help="The customer payment terms to assume for this break-even test. Longer terms require more inspectors to break even because credit line interest costs increase."
    )
    if bc2.button("Find Break-Even", use_container_width=True):
        with st.spinner("Calculating…"):
            be = find_breakeven_inspectors(a, be_nd)
        bc3.success(
            f"You need at least **{be} inspectors** staffed every month to be profitable in Year 1 at Net {be_nd} payment terms."
        )

    st.divider()

    # Export
    section("Export to Excel")
    if st.button("Build Excel Report"):
        with st.spinner("Building…"):
            xlsx = build_excel(a, st.session_state.headcount_plan, weekly_df, mo_full, qdf_full)
        st.download_button(
            "Download Excel", data=xlsx,
            file_name="containment_division_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ════════════════════════════════════════════════════════════════════════════
# FINANCIALS
# ════════════════════════════════════════════════════════════════════════════
with tab_fin:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Calculate</b> in the sidebar to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a = st.session_state.assumptions

    section("Date Range")
    mo = _range_slider("fin_rng", mo_full)
    lo_idx = mo["month_idx"].min(); hi_idx = mo["month_idx"].max()
    qdf = qdf_full[(qdf_full["quarter_idx"] >= lo_idx // 3) & (qdf_full["quarter_idx"] <= hi_idx // 3)].copy()
    wdf = weekly_df[(weekly_df["month_idx"] >= lo_idx) & (weekly_df["month_idx"] <= hi_idx)].copy()

    st.divider()

    f1, f2, f3 = st.tabs(["Monthly", "Quarterly", "Weekly Detail"])

    with f1:
        dcols = ["period","inspectors_avg","team_leads_avg","revenue",
                 "hourly_labor","salaried_cost","overhead","total_labor",
                 "ebitda","ebitda_margin","interest","ebitda_after_interest","ebitda_ai_margin",
                 "collections","ar_end","loc_end","cash_end"]
        _fmt_table(_select(mo, dcols),
            dollar_cols=["revenue","hourly_labor","salaried_cost","overhead","total_labor",
                         "ebitda","interest","ebitda_after_interest","collections","ar_end","loc_end","cash_end"],
            pct_cols=["ebitda_margin","ebitda_ai_margin"],
            highlight_neg="ebitda_after_interest",
            highlight_loc="loc_end", max_loc_val=float(a["max_loc"]))

    with f2:
        qcols = ["yr_q","revenue","hourly_labor","salaried_cost","overhead","total_labor",
                 "ebitda","ebitda_margin","interest","ebitda_after_interest","ebitda_ai_margin",
                 "ar_end","loc_end","cash_end"]
        _fmt_table(_select(qdf, qcols),
            dollar_cols=["revenue","hourly_labor","salaried_cost","overhead","total_labor",
                         "ebitda","interest","ebitda_after_interest","ar_end","loc_end","cash_end"],
            pct_cols=["ebitda_margin","ebitda_ai_margin"],
            highlight_neg="ebitda_after_interest",
            highlight_loc="loc_end", max_loc_val=float(a["max_loc"]))

    with f3:
        w_neg = int(wdf["warn_neg_ebitda"].sum()) if "warn_neg_ebitda" in wdf.columns else 0
        if w_neg:
            st.markdown(f'<div class="warn-box">Negative operating profit in {w_neg} week(s) in selected range.</div>', unsafe_allow_html=True)
        wt1, wt2, wt3, wt4 = st.tabs(["Headcount & Revenue","Labor & Profit","Invoicing & Payments","Cash & Credit Line"])
        with wt1:
            _fmt_table(_select(wdf, ["week_start","week_end","inspectors","team_leads",
                "n_opscoord","n_fieldsup","n_regionalmgr","insp_st_hrs","insp_ot_hrs",
                "insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"]),
                dollar_cols=["insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"])
        with wt2:
            _fmt_table(_select(wdf, ["week_start","inspectors","team_leads",
                "insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"]),
                dollar_cols=["insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                             "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"])
        with wt3:
            _fmt_table(_select(wdf, ["week_start","week_end","is_month_end",
                "revenue_wk","statement_amt","collections","ar_begin","ar_end"]),
                dollar_cols=["revenue_wk","statement_amt","collections","ar_begin","ar_end"])
        with wt4:
            _fmt_table(_select(wdf, ["week_start","payroll_cash_out","salaried_wk",
                "overhead_wk","interest_paid","collections",
                "cash_begin","loc_draw","loc_repay","cash_end","loc_begin","loc_end"]),
                dollar_cols=["payroll_cash_out","salaried_wk","overhead_wk","interest_paid",
                             "collections","cash_begin","loc_draw","loc_repay",
                             "cash_end","loc_begin","loc_end"])
            if all(c in wdf.columns for c in ["check_ar","check_loc","check_cash"]):
                max_err = wdf[["check_ar","check_loc","check_cash"]].max().max()
                if max_err < 0.01:
                    st.success("All weekly balances reconcile correctly.")
                else:
                    st.error(f"Reconciliation error detected: ${max_err:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# HEADCOUNT PLAN
# ════════════════════════════════════════════════════════════════════════════
with tab_hc:
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
    if c4.button("Apply", use_container_width=True,
                 help="Sets inspector count for the selected month range."):
        for i in range(int(ff) - 1, int(ft)):
            hc[i] = int(fv)
        st.session_state.headcount_plan = hc
        st.rerun()

    month_labels = []
    for i in range(120):
        yr = a_start.year  + (a_start.month - 1 + i) // 12
        mo = (a_start.month - 1 + i) % 12 + 1
        month_labels.append(f"{yr}-{mo:02d}")

    section("Preview")
    lo_hc, hi_hc = st.select_slider("Show months", options=list(range(1, 121)), value=(1, 24), key="hc_rng")
    hc_prev = pd.DataFrame({"period": month_labels, "inspectors": hc}).iloc[lo_hc - 1:hi_hc]
    fig_hc  = px.bar(hc_prev, x="period", y="inspectors", template=TPL,
                     title="Inspectors Staffed per Month", color_discrete_sequence=[PC[0]])
    fig_hc.update_layout(height=240, margin=dict(l=10,r=10,t=36,b=10))
    st.plotly_chart(fig_hc, use_container_width=True)

    section("Edit — All 120 Months")
    hc_df  = pd.DataFrame({"Period": month_labels, "Inspectors": hc})
    edited = st.data_editor(hc_df, column_config={
        "Period":     st.column_config.TextColumn(disabled=True),
        "Inspectors": st.column_config.NumberColumn(min_value=0, max_value=10_000, step=1),
    }, use_container_width=True, height=500, num_rows="fixed")
    st.session_state.headcount_plan = edited["Inspectors"].tolist()


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO SUMMARY
# ════════════════════════════════════════════════════════════════════════════
with tab_sum:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Calculate</b> in the sidebar to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, _ = st.session_state.results
    a = st.session_state.assumptions

    section("Date Range")
    mo_s = _range_slider("sum_rng", mo_full)
    lo_s = mo_s["month_idx"].min(); hi_s = mo_s["month_idx"].max()
    wdf_s = weekly_df[(weekly_df["month_idx"] >= lo_s) & (weekly_df["month_idx"] <= hi_s)].copy()

    st.divider()

    # Income statement
    section("Income Statement — Selected Range")
    tot_rev  = mo_s["revenue"].sum()
    tot_hl   = mo_s["hourly_labor"].sum()
    tot_sal  = mo_s["salaried_cost"].sum()
    tot_to   = mo_s["turnover_cost"].sum() if "turnover_cost" in mo_s.columns else 0.0
    tot_ovhd = mo_s["overhead"].sum()
    tot_fovhd = tot_ovhd - tot_to          # fixed overhead is overhead minus turnover (which is in overhead_wk)
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

    # Annual P&L
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

    # Headcount breakdown
    section("Headcount by Role — Selected Range")
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

    # Pie charts + trend
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
        st.plotly_chart(fig_rp, use_container_width=True)
    with c2:
        fig_ep = go.Figure(go.Pie(
            labels=["Hourly Labor","Salaried Management","Turnover & Replacement","Fixed Overhead"],
            values=[tot_hl, tot_sal, tot_to, tot_fovhd],
            hole=0.42, marker_colors=[PC[3], PC[2], PC[4], PC[5]]
        ))
        fig_ep.update_layout(template=TPL, height=280, title="Expenses by Component", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_ep, use_container_width=True)

    section("Monthly Revenue vs. Net Income")
    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(x=mo_s["period"], y=mo_s["revenue"], name="Revenue",
                           marker_color=PC[0], opacity=0.7))
    fig_t.add_trace(go.Scatter(x=mo_s["period"], y=mo_s["ebitda_after_interest"],
                               name="Net Income", mode="lines+markers",
                               line=dict(color=PC[1], width=2)))
    fig_t.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
    fig_t.update_layout(template=TPL, height=300, margin=dict(l=10,r=10,t=10,b=10),
                        legend=dict(orientation="h", y=-0.2), yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_t, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_sens:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Calculate</b> in the sidebar first.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a  = st.session_state.assumptions
    hc = st.session_state.headcount_plan

    st.caption(
        "Each scenario re-runs the full 10-year model changing one variable at a time. "
        "Everything else stays at your current values."
    )

    def _sc(df, xc, xl, yc, yl, pct=False):
        fig = go.Figure(go.Scatter(
            x=df[xc], y=df[yc], mode="lines+markers",
            line=dict(color=PC[0], width=2), marker=dict(size=7)
        ))
        fig.update_layout(template=TPL, height=250, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title=xl, yaxis_title=yl,
                          yaxis=dict(tickformat=".1%" if pct else "$,.0f"))
        fig.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
        return fig

    s1, s2, s3, s4 = st.tabs(["Payment Terms","Bill Rate","Payroll Burden","Overtime"])

    with s1:
        section("Slower customer payments require a larger credit line")
        nd_vals = [30, 45, 60, 75, 90, 105, 120, 150]
        with st.spinner("Running…"):
            nd_df = run_sensitivity(a, hc, "net_days", nd_vals)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sc(nd_df,"value","Net Days","peak_loc","Peak Credit Line ($)"), use_container_width=True)
        c2.plotly_chart(_sc(nd_df,"value","Net Days","ebitda_ai_margin","Net Income Margin", pct=True), use_container_width=True)
        _fmt_table(nd_df[["value","peak_loc","annual_interest","ebitda_ai","ebitda_ai_margin"]].rename(columns={
            "value":"Net Days","peak_loc":"Peak Credit","annual_interest":"Total Interest",
            "ebitda_ai":"Net Income","ebitda_ai_margin":"Net Margin"}),
            dollar_cols=["Peak Credit","Total Interest","Net Income"],
            pct_cols=["Net Margin"], highlight_neg="Net Income")

    with s2:
        section("Bill rate impact on margin and credit line")
        br_vals = [37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 42.0]
        with st.spinner("Running…"):
            br_df = run_sensitivity(a, hc, "st_bill_rate", br_vals)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sc(br_df,"value","Bill Rate ($/hr)","ebitda_ai_margin","Net Income Margin", pct=True), use_container_width=True)
        c2.plotly_chart(_sc(br_df,"value","Bill Rate ($/hr)","peak_loc","Peak Credit Line ($)"), use_container_width=True)
        _fmt_table(br_df[["value","ebitda_margin","ebitda_ai_margin","peak_loc","ebitda_ai"]].rename(columns={
            "value":"Bill Rate","ebitda_margin":"Oper. Margin","ebitda_ai_margin":"Net Margin",
            "peak_loc":"Peak Credit","ebitda_ai":"Net Income"}),
            dollar_cols=["Peak Credit","Net Income"], pct_cols=["Oper. Margin","Net Margin"])

    with s3:
        section("How employer burden rate affects your profitability")
        burd_vals = [0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
        with st.spinner("Running…"):
            burd_df = run_sensitivity(a, hc, "burden", burd_vals)
        burd_df["pct"] = burd_df["value"] * 100
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sc(burd_df,"pct","Burden (%)","ebitda_ai_margin","Net Income Margin", pct=True), use_container_width=True)
        c2.plotly_chart(_sc(burd_df,"pct","Burden (%)","peak_loc","Peak Credit Line ($)"), use_container_width=True)
        _fmt_table(burd_df[["pct","ebitda_ai","ebitda_ai_margin","peak_loc"]].rename(columns={
            "pct":"Burden (%)","ebitda_ai":"Net Income","ebitda_ai_margin":"Net Margin","peak_loc":"Peak Credit"}),
            dollar_cols=["Net Income","Peak Credit"], pct_cols=["Net Margin"])

    with s4:
        section("How overtime hours affect revenue and margin")
        ot_vals = [0, 2, 4, 6, 8, 10, 12, 15, 20]
        with st.spinner("Running…"):
            ot_df = run_sensitivity(a, hc, "ot_hours", ot_vals)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sc(ot_df,"value","OT Hrs/wk","ebitda_ai_margin","Net Income Margin", pct=True), use_container_width=True)
        c2.plotly_chart(_sc(ot_df,"value","OT Hrs/wk","total_revenue","Total Revenue ($)"), use_container_width=True)
        _fmt_table(ot_df[["value","ebitda_margin","ebitda_ai_margin","total_revenue","ebitda_ai"]].rename(columns={
            "value":"OT Hrs/wk","ebitda_margin":"Oper. Margin","ebitda_ai_margin":"Net Margin",
            "total_revenue":"Total Revenue","ebitda_ai":"Net Income"}),
            dollar_cols=["Total Revenue","Net Income"], pct_cols=["Oper. Margin","Net Margin"])

    st.divider()
    if st.button("Export Full Report + Sensitivity to Excel"):
        with st.spinner("Building…"):
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
