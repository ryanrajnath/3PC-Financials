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

st.set_page_config(page_title="Containment Division Calculator", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 0.75rem; padding-bottom: 1rem; }
.kpi-card {
    background: #1A1D27; border: 1px solid #2D3148;
    border-radius: 10px; padding: 14px 18px; text-align: center;
}
.kpi-label { color: #8B8FA8; font-size: 11px; text-transform: uppercase;
             letter-spacing: 1px; margin-bottom: 3px; }
.kpi-value { color: #FAFAFA; font-size: 24px; font-weight: 700; }
.kpi-sub   { color: #4F8BF9; font-size: 11px; margin-top: 2px; }
.section-hdr {
    color: #8B8FA8; font-size: 10px; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 14px 0 6px 0;
    border-bottom: 1px solid #2D3148; padding-bottom: 3px;
}
.warn-box { background:#2D1F1F; border-left:3px solid #E05252;
            padding:8px 12px; border-radius:4px; font-size:13px; margin-bottom:6px; }
.info-box { background:#1A2235; border-left:3px solid #4F8BF9;
            padding:8px 12px; border-radius:4px; font-size:13px; margin-bottom:6px; }
.stale-box{ background:#2D2510; border-left:3px solid #F0A843;
            padding:8px 12px; border-radius:4px; font-size:13px; margin-bottom:6px; }
.preview-chip {
    background:#1A2235; border:1px solid #2D3148; border-radius:6px;
    padding:6px 10px; font-size:12px; color:#8B8FA8; margin-top:2px;
}
</style>
""", unsafe_allow_html=True)

PC = ["#4F8BF9","#52D68A","#F0A843","#E05252","#A855F7","#22D3EE"]
TPL = "plotly_dark"

# ── Session state ────────────────────────────────────────────────────────────
if "assumptions"    not in st.session_state: st.session_state.assumptions    = default_assumptions()
if "headcount_plan" not in st.session_state: st.session_state.headcount_plan = default_headcount()
if "results"        not in st.session_state: st.session_state.results        = None
if "run_hash"       not in st.session_state: st.session_state.run_hash       = None
if "run_ts"         not in st.session_state: st.session_state.run_ts         = None


# ── Helpers ──────────────────────────────────────────────────────────────────
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

def fmt_pct(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    return f"{v*100:.1f}%"

def kpi(col, label, value, sub=None):
    sub_h = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div>{sub_h}</div>',
                 unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-hdr">{label}</div>', unsafe_allow_html=True)

def preview(col, text):
    col.markdown(f'<div class="preview-chip">{text}</div>', unsafe_allow_html=True)

def run_and_store():
    with st.spinner("Running model…"):
        try:
            w, m, q = run_model(st.session_state.assumptions, st.session_state.headcount_plan)
            st.session_state.results = (w, m, q)
            st.session_state.run_hash = _hash_inputs()
            st.session_state.run_ts   = datetime.now().strftime("%I:%M %p")
        except Exception as e:
            st.error(f"Model error: {e}")
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
    fig.update_layout(template=TPL, title=title, height=320,
                      margin=dict(l=10,r=10,t=36,b=10),
                      legend=dict(orientation="h", y=-0.2),
                      yaxis=dict(tickformat=".0%" if pct_y else "$,.0f"))
    fig.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
    return fig

def _bar(df, x, ys, names, title):
    fig = go.Figure()
    for y, nm, c in zip(ys, names, PC):
        if y in df.columns:
            fig.add_trace(go.Bar(x=df[x], y=df[y], name=nm, marker_color=c))
    fig.update_layout(template=TPL, title=title, barmode="stack", height=320,
                      margin=dict(l=10,r=10,t=36,b=10),
                      legend=dict(orientation="h", y=-0.2),
                      yaxis=dict(tickformat="$,.0f"))
    return fig

def _fmt_table(df, dollar_cols=None, pct_cols=None, highlight_neg=None, highlight_loc=None, max_loc_val=None):
    fmt = {}
    for c in (dollar_cols or []):
        if c in df.columns: fmt[c] = "${:,.0f}"
    for c in (pct_cols or []):
        if c in df.columns: fmt[c] = "{:.1%}"

    styled = df.style.format(fmt)

    def _row_style(row):
        styles = [""] * len(row)
        if highlight_neg and highlight_neg in row.index and row[highlight_neg] < 0:
            i = row.index.get_loc(highlight_neg)
            styles[i] = "color: #E05252; font-weight: bold"
        if highlight_loc and highlight_loc in row.index and max_loc_val:
            i = row.index.get_loc(highlight_loc)
            if row[highlight_loc] > max_loc_val * 0.8:
                styles[i] = "background-color: #3D2D00; color: #F0A843"
        return styles

    if highlight_neg or highlight_loc:
        styled = styled.apply(_row_style, axis=1)

    st.dataframe(styled, use_container_width=True, height=380)

def _range_slider(key, mo, label="Show months"):
    active = mo[mo["revenue"] > 0]
    last   = int(active["month_idx"].max()) + 1 if not active.empty else 12
    hi_def = min(last + 3, len(mo))
    lo, hi = st.select_slider(label, options=list(range(1, len(mo)+1)),
                               value=(1, hi_def), key=key)
    return mo[(mo["month_idx"] >= lo-1) & (mo["month_idx"] <= hi-1)]


# ── Scenario presets ─────────────────────────────────────────────────────────
def _build_hc(rules):
    hc = [0] * 120
    for start, end, val in rules:
        for i in range(start-1, min(end, 120)):
            hc[i] = val
    return hc

PRESETS = {
    "Conservative Launch": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate":38.0,"ot_hours":5,"burden":0.33,"team_lead_ratio":15,
            "lead_wage":24.0,"lead_ot_hours":5,"net_days":90,"apr":0.095,
            "max_loc":500_000,"software_monthly":300,"recruiting_monthly":500,
            "insurance_monthly":1000,"travel_monthly":300},
        "headcount": _build_hc([(1,3,10),(4,6,15),(7,12,20),(13,24,25)]),
    },
    "Aggressive Growth": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate":41.0,"ot_hours":15,"inspector_wage":21.0,"lead_wage":26.0,
            "lead_ot_hours":15,"gm_loaded_annual":130_000,"opscoord_base":70_000,
            "fieldsup_base":75_000,"regionalmgr_base":120_000,
            "net_days":120,"max_loc":2_000_000,"cash_buffer":50_000,
            "initial_cash":100_000,"software_monthly":1000,"recruiting_monthly":3000,
            "insurance_monthly":2500,"travel_monthly":1500},
        "headcount": _build_hc([(1,2,25),(3,4,50),(5,6,75),(7,9,100),
                                 (10,12,125),(13,24,150),(25,60,175)]),
    },
    "Steady State (Mature Division)": {
        "assumptions": {**default_assumptions(),
            "st_bill_rate":39.5,"ot_hours":8,"inspector_wage":20.5,"lead_ot_hours":8,
            "gm_loaded_annual":120_000,"opscoord_base":67_000,"fieldsup_base":72_000,
            "regionalmgr_base":115_000,"net_days":45,"apr":0.075,
            "max_loc":750_000,"cash_buffer":50_000,"initial_cash":50_000,
            "software_monthly":500,"recruiting_monthly":750,
            "insurance_monthly":1500,"travel_monthly":500},
        "headcount": [60] * 120,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# HEADER — persistent, always visible
# ════════════════════════════════════════════════════════════════════════════
h_title, h_btn = st.columns([5, 1])
h_title.markdown("## Containment Division Calculator")
h_title.caption("OpSource · Weekly financial model · 120-month horizon")

btn_label = f"▶  Run Model" + (f"  ·  Last run {st.session_state.run_ts}" if st.session_state.run_ts else "")
if h_btn.button(btn_label, type="primary", use_container_width=True):
    run_and_store()
    st.rerun()

if is_stale():
    st.markdown('<div class="stale-box">Inputs changed since last run — click <b>Run Model</b> to update results.</div>',
                unsafe_allow_html=True)

if results_ready():
    _, mo_h, _ = st.session_state.results
    k1,k2,k3,k4,k5 = st.columns(5)
    peak_mo = mo_h.loc[mo_h["loc_end"].idxmax(),"period"] if mo_h["loc_end"].max()>0 else "—"
    kpi(k1,"Peak LOC",           fmt_dollar(mo_h["loc_end"].max()),             peak_mo)
    kpi(k2,"120-mo Revenue",     fmt_dollar(mo_h["revenue"].sum()),             "accrual")
    kpi(k3,"120-mo EBITDA (AI)", fmt_dollar(mo_h["ebitda_after_interest"].sum()),"after interest")
    kpi(k4,"Total Interest",     fmt_dollar(mo_h["interest"].sum()),            "cost of money")
    yr1 = mo_h[mo_h["month_idx"] < 12]
    kpi(k5,"Year 1 EBITDA (AI)", fmt_dollar(yr1["ebitda_after_interest"].sum()),"first 12 months")
    st.divider()


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_howto, tab_assume, tab_hc, tab_results, tab_summary, tab_sens = st.tabs([
    "How to Use",
    "Assumptions",
    "Headcount Plan",
    "Results",
    "Scenario Summary",
    "Sensitivity Analysis",
])


# ════════════════════════════════════════════════════════════════════════════
# HOW TO USE
# ════════════════════════════════════════════════════════════════════════════
with tab_howto:
    st.markdown("## Welcome to the Containment Division Calculator")
    st.markdown("**Built for OpSource** · Models weekly cash flow and operations over a 120-month (10-year) horizon.")
    st.divider()

    st.markdown("""
### What This Tool Answers
- **How much Line of Credit (LOC)** do you need to cover payroll while waiting for customers to pay?
- **When does the LOC stabilize** — when do collections finally "catch up" to payroll?
- **What is EBITDA** before and after interest on the credit line?
- **How does management cost scale** as you add more inspectors?
- **What's your break-even** inspector count at various payment terms?
""")
    st.divider()

    with st.expander("**Step 1 — Load a Preset or Set Assumptions**", expanded=True):
        st.markdown("""
Go to the **Assumptions** tab. You can either:
- Load a **preset scenario** (Conservative Launch, Aggressive Growth, or Steady State) as a starting point
- Or manually set all parameters — billing rates, wages, burden, management salaries, LOC terms, and overhead

Every field has a **? tooltip** — hover over it for a plain-English explanation.
""")
    with st.expander("**Step 2 — Enter Your Headcount Plan**"):
        st.markdown("""
Go to the **Headcount Plan** tab. Enter the number of **inspectors staffed per month** for up to 120 months.

- Use the **Bulk Fill** tool to quickly fill ranges (e.g., months 1–12 = 25 inspectors)
- Months you leave at 0 have zero hourly labor cost — the model handles gaps cleanly
- Team leads and management headcount are calculated automatically
""")
    with st.expander("**Step 3 — Run the Model**"):
        st.markdown("""
Click **▶ Run Model** at the top right of the page — it's always visible no matter which tab you're on.

The model runs in seconds. The 5 KPI cards at the top update immediately. An orange banner appears if you change inputs without re-running.
""")
    with st.expander("**Step 4 — Read Your Results**"):
        st.markdown("""
Go to the **Results** tab. Use the **month range slider** to zoom into any period.

- **Dashboard** — LOC/cash dynamics, revenue vs cost, EBITDA margins, headcount scaling
- **Monthly** — full financial table with conditional formatting (red = negative EBITDA, orange = LOC near limit)
- **Quarterly** — rolled-up view for reporting
- **Weekly** — detailed week-by-week breakdown including AR schedule and cash roll-forward
""")
    with st.expander("**Step 5 — Review the Scenario Summary**"):
        st.markdown("""
The **Scenario Summary** tab shows a clean income statement view, headcount breakdown by role (avg and peak), revenue/expense pie charts, and an **Annual P&L** table by year — useful for board presentations.
""")
    with st.expander("**Step 6 — Run Sensitivity Analysis**"):
        st.markdown("""
The **Sensitivity Analysis** tab re-runs the full model across a range of values for:
- **Customer Payment Terms** (Net Days) — how much does LOC grow if customers pay slower?
- **Bill Rate** — what's the margin impact of a $1/hr rate change?
- **Payroll Burden** — how sensitive is EBITDA to burden rate assumptions?
- **Overtime Hours** — what's the revenue/margin impact of more or less OT?
""")
    with st.expander("**Step 7 — Export to Excel**"):
        st.markdown("""
Go to **Results → Dashboard** or **Sensitivity Analysis** and click the Export button.

The Excel file includes: Assumptions, Headcount Plan, Weekly, Monthly, Quarterly, and (optionally) all Sensitivity tables.
""")

    st.divider()
    st.markdown("### Key Concepts")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Why do you need a Line of Credit?**
You pay inspectors weekly (with a 1-week lag), but customers receive a monthly statement and pay 60–120 days later. During that gap, you are funding payroll out of pocket — that's what the LOC is for.

**What is Burden?**
Burden is the employer-side cost on top of wages: payroll taxes (FICA ~7.65%), federal and state unemployment tax, workers' comp, and benefits. A 30% burden means a $20/hr inspector costs you $26/hr all-in.

**How does Auto Paydown work?**
When cash on hand exceeds the minimum buffer, excess cash automatically repays the LOC. This minimizes interest cost. Turn it OFF if you want to model holding excess cash rather than paying down debt.
""")
    with c2:
        st.markdown("""
**What is EBITDA vs EBITDA after interest?**
EBITDA (pre-interest) = Revenue − Labor − Overhead. This shows operating profitability before the cost of borrowing.
EBITDA after interest = EBITDA − LOC interest. This is your true bottom line given the financing structure.

**How does management scale?**
The model automatically adds salaried management as your inspector count grows:
- 1 Ops Coordinator per 75 inspectors
- 1 Field Supervisor per 60 inspectors
- 1 Regional Manager per 175 inspectors
- GM is active from Month 1 at full cost

**What is management turnover cost?**
In third-party containment, management roles turn over at above-average rates (35% for Ops Coordinators, 25% for Field Supervisors, 18% for Regional Managers per BLS/SHRM data). The model applies a per-hire replacement cost (recruiting, screening, ramp time) prorated weekly based on active headcount. This is separate from base salary and visible in Assumptions → Management Turnover.

**What are team leads?**
Team leads are hourly workers (not salaried). They supervise inspector crews in the field, are burdened like inspectors, and their hours are billed to the customer at the same rates as inspectors.
""")
    st.info("**Quick Start:** Go to Assumptions → load the 'Steady State' preset → click Run Model → explore the Dashboard.")


# ════════════════════════════════════════════════════════════════════════════
# ASSUMPTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_assume:
    a = st.session_state.assumptions

    # Preset loader
    section("Load a Preset Scenario")
    preset_choice = st.selectbox(
        "Start from a preset (overwrites all assumptions and headcount)",
        ["— Select a preset —"] + list(PRESETS.keys()),
        help="Presets are starting points. You can modify any value after loading.")
    if preset_choice != "— Select a preset —":
        if st.button(f"Load  '{preset_choice}'", type="secondary"):
            p = PRESETS[preset_choice]
            st.session_state.assumptions    = p["assumptions"].copy()
            st.session_state.headcount_plan = p["headcount"].copy()
            a = st.session_state.assumptions
            st.success(f"Loaded '{preset_choice}'. Review values below, then click Run Model.")
            st.rerun()

    st.divider()

    # ── Billing & Pay ────────────────────────────────────────────────
    # Live preview values
    _ot_bill = a["st_bill_rate"] * a["ot_bill_premium"]
    _wk_rev  = (a["st_hours"] * a["st_bill_rate"] + a["ot_hours"] * _ot_bill)
    _insp_cost_wk = (a["st_hours"] * a["inspector_wage"] * (1+a["burden"]) +
                     a["ot_hours"] * a["inspector_wage"] * a["ot_pay_multiplier"] * (1+a["burden"]))

    with st.expander(f"Billing & Inspector Pay  ·  ${a['st_bill_rate']:.2f}/hr bill  ·  ${_wk_rev:,.0f}/inspector/wk revenue", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        a["st_bill_rate"]      = c1.number_input("Regular-Time Bill Rate ($/hr)", value=float(a["st_bill_rate"]), step=0.5, format="%.2f",
            help="Hourly rate charged to the customer for each inspector's regular (non-overtime) hours.")
        a["ot_bill_premium"]   = c2.number_input("Overtime Billing Multiplier", value=float(a["ot_bill_premium"]), step=0.1, format="%.1f",
            help=f"OT hours are billed at this multiple of the regular rate. Currently: OT bills at ${a['st_bill_rate']*a['ot_bill_premium']:.2f}/hr.")
        a["st_hours"]          = c3.number_input("Regular Hours per Inspector/Week", value=int(a["st_hours"]), step=1, format="%d",
            help="Standard work hours per inspector per week, not counting overtime. Typically 40.")
        a["ot_hours"]          = c4.number_input("Overtime Hours per Inspector/Week", value=int(a["ot_hours"]), step=1, format="%d",
            help="Planned overtime hours per inspector per week. These are billed and paid at the OT multiplier. Set 0 for no OT.")

        c5,c6,c7,c8 = st.columns(4)
        a["inspector_wage"]    = c5.number_input("Inspector Hourly Wage ($/hr)", value=float(a["inspector_wage"]), step=0.5, format="%.2f",
            help="Base hourly pay for inspectors before burden (taxes and benefits).")
        a["ot_pay_multiplier"] = c6.number_input("Overtime Pay Multiplier", value=float(a["ot_pay_multiplier"]), step=0.1, format="%.1f",
            help=f"Inspectors earn this multiple of base wage for OT hours. At 1.5× and ${a['inspector_wage']:.2f}/hr, OT pay is ${a['inspector_wage']*a['ot_pay_multiplier']:.2f}/hr.")
        a["burden"]            = c7.number_input("Payroll Burden Rate (e.g. 0.30 = 30%)", value=float(a["burden"]), step=0.01, format="%.2f",
            help="Employer-side cost on top of wages: FICA, unemployment tax, workers' comp, and benefits. Enter as a decimal.")
        a["net_days"]          = c8.number_input("Customer Payment Terms (Days After Month-End)", value=int(a["net_days"]), step=5, format="%d",
            help="Days after the month-end statement that customers typically pay. This is the primary driver of how much LOC you need. Net 60 is common in this industry.")

        # Live previews
        p1,p2,p3 = st.columns(3)
        preview(p1, f"Weekly revenue per inspector: ${(a['st_hours']*a['st_bill_rate'] + a['ot_hours']*a['st_bill_rate']*a['ot_bill_premium']):,.0f}")
        preview(p2, f"Fully loaded inspector cost/wk: ${(a['st_hours']*a['inspector_wage']*(1+a['burden']) + a['ot_hours']*a['inspector_wage']*a['ot_pay_multiplier']*(1+a['burden'])):,.0f}")
        preview(p3, f"Inspector gross margin: {((a['st_hours']*a['st_bill_rate']+a['ot_hours']*a['st_bill_rate']*a['ot_bill_premium']) - (a['st_hours']*a['inspector_wage']*(1+a['burden'])+a['ot_hours']*a['inspector_wage']*a['ot_pay_multiplier']*(1+a['burden'])))/(a['st_hours']*a['st_bill_rate']+a['ot_hours']*a['st_bill_rate']*a['ot_bill_premium'])*100:.1f}%")

    # ── Team Leads ───────────────────────────────────────────────────
    with st.expander(f"Team Leads  ·  1 per {a['team_lead_ratio']} inspectors  ·  ${a['lead_wage']:.2f}/hr"):
        st.caption("Team leads are hourly workers billed to the customer at the same rates as inspectors. They are NOT salaried.")
        c1,c2,c3,c4 = st.columns(4)
        a["team_lead_ratio"]   = c1.number_input("Inspectors per Team Lead", value=int(a["team_lead_ratio"]), step=1, format="%d",
            help="One team lead is added for every N inspectors (rounded up). At 12: 25 inspectors → 3 team leads.")
        a["lead_wage"]         = c2.number_input("Team Lead Hourly Wage ($/hr)", value=float(a["lead_wage"]), step=0.5, format="%.2f",
            help="Base hourly pay for team leads before burden. Typically higher than inspector wage.")
        a["lead_st_hours"]     = c3.number_input("Team Lead Regular Hours/Week", value=int(a["lead_st_hours"]), step=1, format="%d",
            help="Regular (straight-time) hours worked per team lead per week.")
        a["lead_ot_hours"]     = c4.number_input("Team Lead Overtime Hours/Week", value=int(a["lead_ot_hours"]), step=1, format="%d",
            help="Overtime hours per team lead per week. Set 0 if team leads don't work OT.")

    # ── Management ───────────────────────────────────────────────────
    _ops_loaded  = a["opscoord_base"]   * (1 + a["mgmt_burden"])
    _fsup_loaded = a["fieldsup_base"]   * (1 + a["mgmt_burden"])
    _rmgr_loaded = a["regionalmgr_base"]* (1 + a["mgmt_burden"])

    with st.expander(f"Salaried Management  ·  GM ${a['gm_loaded_annual']:,.0f}  ·  1 Field Sup per {a['fieldsup_span']} inspectors"):
        st.caption("These roles are salaried and activate automatically as your inspector count grows. The model adds one for every N inspectors using the thresholds below.")
        c1,c2,c3,c4 = st.columns(4)
        a["gm_loaded_annual"]  = c1.number_input("General Manager — Total Annual Cost ($)", value=float(a["gm_loaded_annual"]), step=1000., format="%.0f",
            help="Fully-loaded annual cost of the GM including salary, bonus, and benefits. Already fully burdened — do NOT add burden again.")
        a["opscoord_base"]     = c2.number_input("Operations Coordinator — Base Annual Salary ($)", value=float(a["opscoord_base"]), step=1000., format="%.0f",
            help="Base salary for an Ops Coordinator. The management burden rate is applied on top. Ops Coordinators handle scheduling, dispatch, and field support.")
        a["fieldsup_base"]     = c3.number_input("Field Supervisor — Base Annual Salary ($)", value=float(a["fieldsup_base"]), step=1000., format="%.0f",
            help="Base salary for a Field Supervisor. Management burden applied on top. Field Supervisors directly oversee inspector crews in the field.")
        a["regionalmgr_base"]  = c4.number_input("Regional Manager — Base Annual Salary ($)", value=float(a["regionalmgr_base"]), step=1000., format="%.0f",
            help="Base salary for a Regional Manager. Management burden applied on top. You likely won't need one until 175+ inspectors.")

        c5,c6,c7,c8 = st.columns(4)
        a["mgmt_burden"]       = c5.number_input("Management Benefit & Tax Rate (e.g. 0.25 = 25%)", value=float(a["mgmt_burden"]), step=0.01, format="%.2f",
            help="Burden rate applied to salaried management base salaries. Covers employer taxes and benefits. NOT applied to GM (GM is already fully loaded).")
        a["opscoord_span"]     = c6.number_input("Inspectors per Operations Coordinator", value=int(a["opscoord_span"]), step=5, format="%d",
            help="The model adds 1 Ops Coordinator for every N inspectors. At 75: 1 hire. At 150: 2 hires. Adjust to match your management structure.")
        a["fieldsup_span"]     = c7.number_input("Inspectors per Field Supervisor", value=int(a["fieldsup_span"]), step=5, format="%d",
            help="One Field Supervisor added per N inspectors. Default 60 — a supervisor covers up to 60 field workers.")
        a["regionalmgr_span"]  = c8.number_input("Inspectors per Regional Manager", value=int(a["regionalmgr_span"]), step=5, format="%d",
            help="One Regional Manager per N inspectors. Default 175 — you likely won't need one for a while.")

        # Live preview
        p1,p2,p3,p4 = st.columns(4)
        preview(p1, f"GM weekly cost: ${a['gm_loaded_annual']/52:,.0f}")
        preview(p2, f"Ops Coord fully loaded: ${_ops_loaded:,.0f}/yr")
        preview(p3, f"Field Sup fully loaded: ${_fsup_loaded:,.0f}/yr")
        preview(p4, f"Reg Mgr fully loaded: ${_rmgr_loaded:,.0f}/yr")

    # ── Management Turnover ──────────────────────────────────────────
    _ops_to_mo   = a.get("opscoord_turnover",    0.35)   * a.get("opscoord_replace_cost",    8_000) / 12
    _fsup_to_mo  = a.get("fieldsup_turnover",    0.25)   * a.get("fieldsup_replace_cost",   12_000) / 12
    _rmgr_to_mo  = a.get("regionalmgr_turnover", 0.18)   * a.get("regionalmgr_replace_cost",25_000) / 12
    _to_total_mo = _ops_to_mo + _fsup_to_mo + _rmgr_to_mo

    with st.expander(f"Management Turnover & Recruiting Cost  ·  Est. ${_to_total_mo:,.0f}/mo per active headcount set"):
        st.caption(
            "Third-party containment and inspection operations have above-average management turnover "
            "due to demanding field conditions, irregular hours, and a competitive labor market for "
            "qualified supervisors. These costs model the recruiting, onboarding, and ramp-up expense "
            "each time a management role is backfilled. The model prorates this monthly — it scales "
            "automatically as you add more roles."
        )
        st.markdown("""
**Industry benchmarks (BLS JOLTS + SHRM 2024 data — staffing/field operations sector):**
- Operations Coordinators: ~30–40% annual turnover (scheduling stress, limited advancement)
- Field Supervisors: ~22–28% (physically demanding, often poached by competitors)
- Regional Managers: ~15–20% (better comp & authority, but still competitive market)

*Replacement cost includes: job board advertising, background/drug screen, recruiter time or fee, and ~4–8 weeks of reduced productivity during ramp-up.*
""")
        st.divider()

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Operations Coordinator**")
            a["opscoord_turnover"]     = st.number_input("Annual Turnover Rate (e.g. 0.35 = 35%)",
                value=float(a.get("opscoord_turnover", 0.35)), step=0.01, format="%.2f",
                key="oc_to",
                help="What fraction of your Ops Coordinator headcount turns over each year. "
                     "At 35%, a team of 2 Ops Coordinators loses ~0.7 people/year on average.")
            a["opscoord_replace_cost"] = st.number_input("Replacement Cost per Hire ($)",
                value=float(a.get("opscoord_replace_cost", 8_000)), step=500., format="%.0f",
                key="oc_rc",
                help="Total cost to replace one Ops Coordinator: job board post (~$500), "
                     "background/drug screen (~$200), HR/manager time (~2 weeks), "
                     "and 4 weeks at partial productivity. Industry avg ~$8,000.")
            preview(st.columns(1)[0], f"Est. ${a['opscoord_turnover']*a.get('opscoord_replace_cost',8000)/12:,.0f}/mo per Ops Coordinator")

        with c2:
            st.markdown("**Field Supervisor**")
            a["fieldsup_turnover"]     = st.number_input("Annual Turnover Rate (e.g. 0.25 = 25%)",
                value=float(a.get("fieldsup_turnover", 0.25)), step=0.01, format="%.2f",
                key="fs_to",
                help="Annual turnover rate for Field Supervisors. At 25%, you replace ~1 in 4 "
                     "supervisors per year. Field roles in containment are physically demanding "
                     "and supervisors are frequently recruited by OEMs or tier-1 suppliers.")
            a["fieldsup_replace_cost"] = st.number_input("Replacement Cost per Hire ($)",
                value=float(a.get("fieldsup_replace_cost", 12_000)), step=500., format="%.0f",
                key="fs_rc",
                help="Total replacement cost per Field Supervisor. Higher than Ops Coord because "
                     "the role requires hands-on technical experience in automotive quality or "
                     "containment. Includes recruiting time, possible agency fee, and ramp. "
                     "Industry avg ~$12,000.")
            preview(st.columns(1)[0], f"Est. ${a['fieldsup_turnover']*a.get('fieldsup_replace_cost',12000)/12:,.0f}/mo per Field Supervisor")

        with c3:
            st.markdown("**Regional Manager**")
            a["regionalmgr_turnover"]     = st.number_input("Annual Turnover Rate (e.g. 0.18 = 18%)",
                value=float(a.get("regionalmgr_turnover", 0.18)), step=0.01, format="%.2f",
                key="rm_to",
                help="Annual turnover rate for Regional Managers. Lower than field roles — "
                     "better compensation and strategic responsibility improve retention. "
                     "At 18%, you replace a Regional Manager roughly every 5–6 years.")
            a["regionalmgr_replace_cost"] = st.number_input("Replacement Cost per Hire ($)",
                value=float(a.get("regionalmgr_replace_cost", 25_000)), step=1000., format="%.0f",
                key="rm_rc",
                help="Replacement cost per Regional Manager. At this level, you often need a "
                     "third-party recruiter (15–20% of base salary = $16–22K) plus ramp time. "
                     "Total all-in cost typically $22–28K. Industry avg ~$25,000.")
            preview(st.columns(1)[0], f"Est. ${a['regionalmgr_turnover']*a.get('regionalmgr_replace_cost',25000)/12:,.0f}/mo per Regional Manager")

    # ── LOC ──────────────────────────────────────────────────────────
    _mo_int_est = (a["apr"]/12) * a["max_loc"]
    with st.expander(f"Line of Credit  ·  {a['apr']*100:.1f}% APR  ·  Max ${a['max_loc']:,.0f}  ·  Net {a['net_days']} days"):
        st.caption("The LOC funds payroll during the gap between when you pay workers and when customers pay you. The model draws and repays it automatically each week.")
        c1,c2,c3,c4 = st.columns(4)
        a["apr"]          = c1.number_input("Annual Interest Rate on LOC (e.g. 0.085 = 8.5%)", value=float(a["apr"]), step=0.005, format="%.3f",
            help="Annual percentage rate on the LOC balance. Enter as decimal. Interest is calculated monthly on the average weekly LOC balance.")
        a["max_loc"]      = c2.number_input("Maximum Credit Line ($)", value=float(a["max_loc"]), step=50000., format="%.0f",
            help="The bank's maximum draw limit. The model warns you if payroll needs exceed this amount.")
        a["initial_cash"] = c3.number_input("Starting Cash Balance ($)", value=float(a["initial_cash"]), step=5000., format="%.0f",
            help="Cash on hand when the model starts. Set 0 if the division launches with no cash reserves.")
        a["cash_buffer"]  = c4.number_input("Minimum Cash Reserve ($)", value=float(a["cash_buffer"]), step=5000., format="%.0f",
            help="The model draws the LOC to keep at least this much cash on hand at all times. Acts as a weekly payroll safety margin.")
        a["auto_paydown"] = st.checkbox("Automatically repay the credit line when cash exceeds the reserve",
            value=bool(a["auto_paydown"]),
            help="ON: excess cash above the buffer sweeps to pay down the LOC, saving interest. OFF: LOC is only drawn when needed, cash is held.")
        a["start_date"]   = st.date_input("Model Start Date", value=a["start_date"],
            help="The first day of the model. All weeks and months are calculated forward from this date.")
        preview(st.columns(1)[0], f"Monthly interest at full draw (${a['max_loc']:,.0f}): ${_mo_int_est:,.0f}/mo")

    # ── Overhead ─────────────────────────────────────────────────────
    _total_ovhd = (a["software_monthly"] + a["recruiting_monthly"] +
                   a["insurance_monthly"] + a["travel_monthly"] +
                   (a["corp_alloc_fixed"] if a["corp_alloc_mode"]=="fixed" else 0))
    with st.expander(f"Fixed Monthly Overhead  ·  ${_total_ovhd:,.0f}/mo total"):
        st.caption("These costs are incurred every month regardless of inspector count.")
        c1,c2,c3,c4 = st.columns(4)
        a["software_monthly"]   = c1.number_input("Software & Technology ($/mo)", value=float(a["software_monthly"]), step=100., format="%.0f",
            help="Monthly cost for scheduling, time tracking, field management, and reporting tools.")
        a["recruiting_monthly"] = c2.number_input("Recruiting & Onboarding ($/mo)", value=float(a["recruiting_monthly"]), step=100., format="%.0f",
            help="Monthly spend on job boards, background checks, drug screening, and new hire onboarding.")
        a["insurance_monthly"]  = c3.number_input("Insurance ($/mo)", value=float(a["insurance_monthly"]), step=100., format="%.0f",
            help="Monthly insurance costs not included in burden — general liability, E&O, commercial auto, etc.")
        a["travel_monthly"]     = c4.number_input("Travel & Field Expenses ($/mo)", value=float(a["travel_monthly"]), step=100., format="%.0f",
            help="Monthly travel, mileage reimbursement, lodging, and per diem for field operations.")

        ca_mode = st.radio("Corporate / Parent Company Overhead Allocation",
                           ["Fixed monthly amount", "Percentage of revenue"], horizontal=True,
                           index=0 if a["corp_alloc_mode"]=="fixed" else 1,
                           help="Charge from the parent company for shared services. Default $0 since this division is inside OpSource.")
        a["corp_alloc_mode"] = "fixed" if ca_mode=="Fixed monthly amount" else "pct_revenue"
        if a["corp_alloc_mode"] == "fixed":
            a["corp_alloc_fixed"] = st.number_input("Corporate Allocation — Fixed ($/mo)", value=float(a["corp_alloc_fixed"]), step=500., format="%.0f",
                help="Fixed monthly charge from corporate. Default $0.")
        else:
            a["corp_alloc_pct"] = st.number_input("Corporate Allocation — % of Revenue (e.g. 0.03 = 3%)", value=float(a["corp_alloc_pct"]), step=0.005, format="%.3f",
                help="Variable corporate charge as a percentage of gross revenue. Enter as decimal.")

    st.session_state.assumptions = a


# ════════════════════════════════════════════════════════════════════════════
# HEADCOUNT PLAN
# ════════════════════════════════════════════════════════════════════════════
with tab_hc:
    hc = st.session_state.headcount_plan
    a_start = st.session_state.assumptions["start_date"]

    section("Bulk Fill")
    c1,c2,c3,c4 = st.columns(4)
    fv = c1.number_input("Number of Inspectors", 0, 10000, 25, step=5, key="fv",
        help="How many inspectors to staff during this range. Team leads and all management "
             "are calculated automatically from this number.")
    ff = c2.number_input("From Month #", 1, 120, 1, step=1, key="ff",
        help="First model month to fill (1 = the first month of your start date).")
    ft = c3.number_input("To Month #",   1, 120, 12, step=1, key="ft",
        help="Last model month to fill (inclusive). Month 12 = end of Year 1, Month 120 = end of Year 10.")
    if c4.button("Apply Fill", use_container_width=True,
                 help="Overwrites inspector counts for the selected month range with the number above."):
        for i in range(int(ff)-1, int(ft)): hc[i] = int(fv)
        st.session_state.headcount_plan = hc
        st.rerun()

    month_labels = []
    for i in range(120):
        yr = a_start.year  + (a_start.month-1+i) // 12
        mo = (a_start.month-1+i) % 12 + 1
        month_labels.append(f"{yr}-{mo:02d}")

    section("Preview")
    lo_hc, hi_hc = st.select_slider("Show months", options=list(range(1,121)), value=(1,24), key="hc_rng")
    hc_prev = pd.DataFrame({"period": month_labels, "inspectors": hc})
    hc_prev = hc_prev.iloc[lo_hc-1:hi_hc]
    fig_hc = px.bar(hc_prev, x="period", y="inspectors", template=TPL,
                    title="Inspectors Staffed per Month",
                    color_discrete_sequence=[PC[0]])
    fig_hc.update_layout(height=260, margin=dict(l=10,r=10,t=36,b=10))
    st.plotly_chart(fig_hc, use_container_width=True)

    section("Edit Headcount — All 120 Months")
    hc_df = pd.DataFrame({"Period": month_labels, "Inspectors": hc})
    edited = st.data_editor(hc_df, column_config={
        "Period":     st.column_config.TextColumn(disabled=True),
        "Inspectors": st.column_config.NumberColumn(min_value=0, max_value=10000, step=1),
    }, use_container_width=True, height=500, num_rows="fixed")
    st.session_state.headcount_plan = edited["Inspectors"].tolist()


# ════════════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Run Model</b> at the top of the page to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a = st.session_state.assumptions

    n_loc  = weekly_df["warn_loc_maxed"].sum()
    n_mgmt = weekly_df["warn_mgmt_no_insp"].sum()
    if n_loc:  st.markdown(f'<div class="warn-box">LOC exceeded the max credit line in {n_loc} week(s). Consider raising the maximum credit line.</div>', unsafe_allow_html=True)
    if n_mgmt: st.markdown(f'<div class="warn-box">Salaried management is active in {n_mgmt} week(s) with 0 inspectors — check your GM start month and headcount plan.</div>', unsafe_allow_html=True)

    section("Date Range Filter")
    mo  = _range_slider("res_rng", mo_full, "Show months")
    lo_idx = mo["month_idx"].min(); hi_idx = mo["month_idx"].max()
    qdf = qdf_full[(qdf_full["quarter_idx"] >= lo_idx//3) & (qdf_full["quarter_idx"] <= hi_idx//3)].copy()
    wdf = weekly_df[(weekly_df["month_idx"] >= lo_idx) & (weekly_df["month_idx"] <= hi_idx)].copy()

    st.divider()
    r1, r2, r3, r4 = st.tabs(["Dashboard", "Monthly", "Quarterly", "Weekly"])

    # ── Dashboard ────────────────────────────────────────────────────
    with r1:
        section("LOC / AR / Cash")
        fig_loc = go.Figure()
        fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["loc_end"],   name="LOC Balance",
            fill="tozeroy", line=dict(color=PC[3], width=2)))
        fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["ar_end"],    name="AR Balance",
            line=dict(color=PC[0], width=2)))
        fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["cash_end"],  name="Cash",
            line=dict(color=PC[1], width=2)))
        fig_loc.add_hline(y=float(a["max_loc"]), line_dash="dot", line_color=PC[3],
                          annotation_text="LOC Limit", annotation_font_color=PC[3])
        fig_loc.update_layout(template=TPL, height=340, margin=dict(l=10,r=10,t=10,b=10),
                              legend=dict(orientation="h",y=-0.15), yaxis=dict(tickformat="$,.0f"))
        st.plotly_chart(fig_loc, use_container_width=True)

        section("Monthly Cash Flow Waterfall")
        mo_wf = mo.copy()
        wf_categories = ["Starting Cash","+ Collections","− Payroll (Hourly)","− Salaried Mgmt","− Overhead","− Interest","+ LOC Draw","− LOC Repay","Ending Cash"]
        wf_values = [
            mo_wf["cash_begin"].iloc[0] if len(mo_wf) else 0,
            mo_wf["collections"].sum(),
            -mo_wf["hourly_labor"].sum(),
            -mo_wf["salaried_cost"].sum(),
            -mo_wf["overhead"].sum(),
            -mo_wf["interest"].sum(),
            mo_wf["loc_draw"].sum(),
            -mo_wf["loc_repay"].sum(),
            mo_wf["cash_end"].iloc[-1] if len(mo_wf) else 0,
        ]
        fig_wf = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute","relative","relative","relative","relative","relative","relative","relative","total"],
            x=wf_categories, y=wf_values,
            connector=dict(line=dict(color="#444")),
            increasing=dict(marker_color=PC[1]),
            decreasing=dict(marker_color=PC[3]),
            totals=dict(marker_color=PC[0]),
        ))
        fig_wf.update_layout(template=TPL, height=320, margin=dict(l=10,r=10,t=10,b=10),
                             yaxis=dict(tickformat="$,.0f"))
        st.plotly_chart(fig_wf, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            section("Revenue vs Cost Stack")
            fig_rv = _bar(mo,"period",["hourly_labor","salaried_cost","overhead"],
                          ["Hourly Labor","Salaried Mgmt","Overhead"],"Monthly Cost Stack")
            fig_rv.add_trace(go.Scatter(x=mo["period"],y=mo["revenue"],name="Revenue",
                mode="lines",line=dict(color=PC[1],width=2)))
            st.plotly_chart(fig_rv, use_container_width=True)
        with c2:
            section("EBITDA")
            st.plotly_chart(_line(mo,"period",["ebitda","ebitda_after_interest"],
                                  ["EBITDA (pre-interest)","EBITDA (after interest)"],"Monthly EBITDA"),
                            use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            section("EBITDA Margins")
            st.plotly_chart(_line(mo,"period",["ebitda_margin","ebitda_ai_margin"],
                                  ["Margin (pre-int)","Margin (after int)"],"EBITDA Margins",pct_y=True),
                            use_container_width=True)
        with c4:
            section("Headcount by Role")
            st.plotly_chart(_bar(mo,"period",
                                 ["inspectors_avg","team_leads_avg","n_opscoord","n_fieldsup","n_regionalmgr"],
                                 ["Inspectors","Team Leads","Ops Coord","Field Sup","Reg Mgr"],
                                 "Average Monthly Headcount"),
                            use_container_width=True)

        st.divider()
        section("Break-Even Calculator")
        bc1,bc2,bc3 = st.columns(3)
        be_nd = bc1.selectbox("Net Days for calculation", [30,60,90,120,150],
                              index=min(1, max(0,[30,60,90,120,150].index(int(a["net_days"]))
                                                if int(a["net_days"]) in [30,60,90,120,150] else 1)),
                              help="Payment terms to assume for this break-even calculation. "
                                   "Longer terms require more inspectors to break even because "
                                   "LOC interest costs increase.")
        if bc2.button("Find Minimum Inspectors", use_container_width=True):
            with st.spinner("Calculating…"):
                be = find_breakeven_inspectors(a, be_nd)
            bc3.success(f"Minimum inspectors for positive EBITDA (after interest): **{be}** at Net {be_nd}")

        st.divider()
        section("Export to Excel")
        if st.button("Build Excel Export"):
            with st.spinner("Building…"):
                xlsx = build_excel(a, st.session_state.headcount_plan, weekly_df, mo_full, qdf_full)
            st.download_button("Download Excel", data=xlsx,
                file_name="containment_division_model.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── Monthly ──────────────────────────────────────────────────────
    with r2:
        dcols = ["period","inspectors_avg","team_leads_avg","revenue","hourly_labor",
                 "salaried_cost","turnover_cost","overhead","total_labor","ebitda","ebitda_margin",
                 "interest","ebitda_after_interest","ebitda_ai_margin",
                 "collections","ar_end","loc_end","cash_end","peak_loc_to_date"]
        _fmt_table(_select(mo,dcols),
                   dollar_cols=["revenue","hourly_labor","salaried_cost","turnover_cost","overhead","total_labor",
                                "ebitda","interest","ebitda_after_interest",
                                "collections","ar_end","loc_end","cash_end","peak_loc_to_date"],
                   pct_cols=["ebitda_margin","ebitda_ai_margin"],
                   highlight_neg="ebitda_after_interest",
                   highlight_loc="loc_end", max_loc_val=float(a["max_loc"]))

    # ── Quarterly ────────────────────────────────────────────────────
    with r3:
        qcols = ["yr_q","revenue","hourly_labor","salaried_cost","overhead","total_labor",
                 "ebitda","ebitda_margin","interest","ebitda_after_interest","ebitda_ai_margin",
                 "ar_end","loc_end","cash_end","peak_loc_to_date"]
        _fmt_table(_select(qdf,qcols),
                   dollar_cols=["revenue","hourly_labor","salaried_cost","overhead","total_labor",
                                "ebitda","interest","ebitda_after_interest",
                                "ar_end","loc_end","cash_end","peak_loc_to_date"],
                   pct_cols=["ebitda_margin","ebitda_ai_margin"],
                   highlight_neg="ebitda_after_interest",
                   highlight_loc="loc_end", max_loc_val=float(a["max_loc"]))

    # ── Weekly ───────────────────────────────────────────────────────
    with r4:
        w_neg = wdf["warn_neg_ebitda"].sum()
        if w_neg: st.markdown(f'<div class="warn-box">Negative EBITDA in {w_neg} week(s) within selected range.</div>', unsafe_allow_html=True)
        wt1,wt2,wt3,wt4 = st.tabs(["Headcount & Revenue","Labor & EBITDA","AR & Collections","Cash & LOC"])
        with wt1:
            _fmt_table(_select(wdf,["week_start","week_end","inspectors","team_leads",
                "n_opscoord","n_fieldsup","n_regionalmgr","insp_st_hrs","insp_ot_hrs",
                "insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"]),
                dollar_cols=["insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"])
        with wt2:
            _fmt_table(_select(wdf,["week_start","inspectors","team_leads",
                "insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"]),
                dollar_cols=["insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                             "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"])
        with wt3:
            _fmt_table(_select(wdf,["week_start","week_end","is_month_end",
                "revenue_wk","statement_amt","collections","ar_begin","ar_end"]),
                dollar_cols=["revenue_wk","statement_amt","collections","ar_begin","ar_end"])
        with wt4:
            _fmt_table(_select(wdf,["week_start","payroll_cash_out","salaried_wk",
                "overhead_wk","interest_paid","collections",
                "cash_begin","loc_draw","loc_repay","cash_end","loc_begin","loc_end"]),
                dollar_cols=["payroll_cash_out","salaried_wk","overhead_wk","interest_paid",
                             "collections","cash_begin","loc_draw","loc_repay","cash_end","loc_begin","loc_end"])
            max_err = wdf[["check_ar","check_loc","check_cash"]].max().max()
            if max_err < 0.01: st.success(f"All reconciliation checks pass — max error ${max_err:.4f}")
            else: st.error(f"Reconciliation error: ${max_err:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO SUMMARY
# ════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Run Model</b> at the top of the page to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, _ = st.session_state.results
    a = st.session_state.assumptions

    section("Date Range Filter")
    mo_s = _range_slider("sum_rng", mo_full, "Show months")
    lo_s = mo_s["month_idx"].min(); hi_s = mo_s["month_idx"].max()
    wdf_s = weekly_df[(weekly_df["month_idx"] >= lo_s) & (weekly_df["month_idx"] <= hi_s)].copy()

    st.divider()

    # ── Income Statement ─────────────────────────────────────────────
    section("Income Statement — Selected Range")
    tot_rev   = mo_s["revenue"].sum()
    tot_hl    = mo_s["hourly_labor"].sum()
    tot_sal   = mo_s["salaried_cost"].sum()
    tot_to    = mo_s["turnover_cost"].sum() if "turnover_cost" in mo_s.columns else 0.0
    tot_ovhd  = mo_s["overhead"].sum()
    tot_exp   = tot_hl + tot_sal + tot_ovhd
    tot_eb    = mo_s["ebitda"].sum()
    tot_int   = mo_s["interest"].sum()
    tot_eb_ai = mo_s["ebitda_after_interest"].sum()

    is_df = pd.DataFrame({
        "Line Item": ["Revenue","",
                      "  Hourly Labor (Inspectors + Team Leads)","  Salaried Management",
                      "  Mgmt Turnover & Recruiting","  Fixed Overhead",
                      "Total Expenses","","EBITDA (before interest)",
                      "  LOC Interest Expense","Net Operating Income (EBITDA after interest)"],
        "Amount":    [tot_rev, None, tot_hl, tot_sal, tot_to,
                      tot_ovhd - tot_to, tot_exp, None, tot_eb, -tot_int, tot_eb_ai],
        "% Revenue": [1.0, None,
                      tot_hl/tot_rev if tot_rev else 0,
                      tot_sal/tot_rev if tot_rev else 0,
                      tot_to/tot_rev if tot_rev else 0,
                      (tot_ovhd-tot_to)/tot_rev if tot_rev else 0,
                      tot_exp/tot_rev if tot_rev else 0,
                      None, tot_eb/tot_rev if tot_rev else 0,
                      -tot_int/tot_rev if tot_rev else 0, tot_eb_ai/tot_rev if tot_rev else 0],
    })
    st.dataframe(
        is_df.style.format({"Amount":"${:,.0f}","% Revenue":"{:.1%}"},na_rep="")
             .apply(lambda row: [
                 "font-weight:bold; color:#52D68A" if row["Line Item"] in ("Revenue","Net Operating Income (EBITDA after interest)") else
                 "font-weight:bold; color:#E05252" if row["Line Item"] == "Total Expenses" else
                 "font-weight:bold" if row["Line Item"] == "EBITDA (before interest)" else ""
                 for _ in row], axis=1),
        use_container_width=True, height=360,
    )

    st.divider()

    # ── Annual P&L ───────────────────────────────────────────────────
    section("Annual P&L Summary (Year by Year)")
    _to_agg = {"turnover_cost": ("turnover_cost", "sum")} if "turnover_cost" in mo_full.columns else {}
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
        **_to_agg,
    }).reset_index()
    annual["total_expenses"]   = annual["hourly_labor"] + annual["salaried_cost"] + annual["overhead"]
    annual["ebitda_margin"]    = np.where(annual["revenue"]>0, annual["ebitda"]/annual["revenue"], 0)
    annual["ebitda_ai_margin"] = np.where(annual["revenue"]>0, annual["ebitda_after_interest"]/annual["revenue"], 0)
    annual["year_label"]       = annual["year"].apply(lambda y: f"Year {y - annual['year'].min() + 1}  ({y})")

    ann_display = annual[["year_label","avg_inspectors","revenue","total_expenses","ebitda","ebitda_margin",
                           "interest","ebitda_after_interest","ebitda_ai_margin","peak_loc"]].rename(columns={
        "year_label":"Year","avg_inspectors":"Avg Inspectors","revenue":"Revenue",
        "total_expenses":"Total Expenses","ebitda":"EBITDA","ebitda_margin":"EBITDA %",
        "interest":"Interest","ebitda_after_interest":"Net Income","ebitda_ai_margin":"Net %","peak_loc":"Peak LOC"})
    _fmt_table(ann_display,
               dollar_cols=["Revenue","Total Expenses","EBITDA","Interest","Net Income","Peak LOC"],
               pct_cols=["EBITDA %","Net %"],
               highlight_neg="Net Income")

    st.divider()

    # ── Headcount ────────────────────────────────────────────────────
    section("Headcount by Role — Selected Range")
    gm_avg = wdf_s["gm_fte"].mean() if "gm_fte" in wdf_s.columns else 0
    hc_data = {
        "Role":            ["Inspectors","Team Leads","Operations Coordinators","Field Supervisors","Regional Managers","General Manager"],
        "Type":            ["Hourly","Hourly","Salaried","Salaried","Salaried","Salaried"],
        "Billed to Client":["Yes","Yes","No","No","No","No"],
        "Avg per Month":   [mo_s["inspectors_avg"].mean(), mo_s["team_leads_avg"].mean(),
                            mo_s["n_opscoord"].mean(), mo_s["n_fieldsup"].mean(),
                            mo_s["n_regionalmgr"].mean(), gm_avg],
        "Peak in Period":  [mo_s["inspectors_avg"].max(), mo_s["team_leads_avg"].max(),
                            mo_s["n_opscoord"].max(), mo_s["n_fieldsup"].max(),
                            mo_s["n_regionalmgr"].max(), 1.0 if gm_avg > 0 else 0],
    }
    st.dataframe(pd.DataFrame(hc_data).style.format({"Avg per Month":"{:.1f}","Peak in Period":"{:.1f}"}),
                 use_container_width=True, height=260)

    st.divider()

    # ── Revenue & Expense pies ────────────────────────────────────────
    section("Revenue & Expense Breakdown")
    c1,c2 = st.columns(2)
    with c1:
        fig_rp = go.Figure(go.Pie(
            labels=["Inspector Regular Time","Inspector Overtime","Team Lead Regular Time","Team Lead Overtime"],
            values=[mo_s["insp_rev_st"].sum(), mo_s["insp_rev_ot"].sum(),
                    mo_s["lead_rev_st"].sum(), mo_s["lead_rev_ot"].sum()],
            hole=0.42, marker_colors=PC[:4]))
        fig_rp.update_layout(template=TPL, height=300, title="Revenue by Component",
                             margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_rp, use_container_width=True)
    with c2:
        fig_ep = go.Figure(go.Pie(
            labels=["Hourly Labor","Salaried Management","Overhead"],
            values=[tot_hl, tot_sal, tot_ovhd],
            hole=0.42, marker_colors=[PC[3],PC[2],PC[4]]))
        fig_ep.update_layout(template=TPL, height=300, title="Expenses by Component",
                             margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_ep, use_container_width=True)

    # ── Revenue vs Net Income trend ───────────────────────────────────
    section("Revenue vs Net Income — Monthly Trend")
    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(x=mo_s["period"],y=mo_s["revenue"],name="Revenue",
                           marker_color=PC[0],opacity=0.7))
    fig_t.add_trace(go.Scatter(x=mo_s["period"],y=mo_s["ebitda_after_interest"],
                               name="Net Income (EBITDA AI)",mode="lines+markers",
                               line=dict(color=PC[1],width=2)))
    fig_t.add_hline(y=0,line_dash="dot",line_color="#444",line_width=1)
    fig_t.update_layout(template=TPL,height=320,margin=dict(l=10,r=10,t=10,b=10),
                        legend=dict(orientation="h",y=-0.15),yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_t, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_sens:
    if not results_ready():
        st.markdown('<div class="info-box">Click <b>Run Model</b> at the top of the page first.</div>', unsafe_allow_html=True)
        st.stop()

    a  = st.session_state.assumptions
    hc = st.session_state.headcount_plan
    st.caption("Each table re-runs the full 120-month model varying one parameter at a time. All other assumptions stay at current values.")

    def _sc(df, xc, xl, yc, yl, pct=False):
        fig = go.Figure(go.Scatter(x=df[xc],y=df[yc],mode="lines+markers",
            line=dict(color=PC[0],width=2),marker=dict(size=7)))
        fig.update_layout(template=TPL,height=260,margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title=xl,yaxis_title=yl,
                          yaxis=dict(tickformat=".1%" if pct else "$,.0f"))
        fig.add_hline(y=0,line_dash="dot",line_color="#444",line_width=1)
        return fig

    s1,s2,s3,s4 = st.tabs(["Customer Payment Terms","Bill Rate","Payroll Burden","Overtime Hours"])

    with s1:
        section("How payment terms affect your credit line requirement")
        nd_vals = [30,45,60,75,90,105,120,150]
        with st.spinner("Running…"):
            nd_df = run_sensitivity(a,hc,"net_days",nd_vals)
        c1,c2 = st.columns(2)
        c1.plotly_chart(_sc(nd_df,"value","Net Days","peak_loc","Peak LOC ($)"), use_container_width=True)
        c2.plotly_chart(_sc(nd_df,"value","Net Days","ebitda_ai_margin","EBITDA Margin (after interest)",pct=True), use_container_width=True)
        _fmt_table(nd_df[["value","peak_loc","annual_interest","ebitda_ai","ebitda_ai_margin"]].rename(columns={
            "value":"Net Days","peak_loc":"Peak LOC","annual_interest":"Total Interest",
            "ebitda_ai":"EBITDA (AI)","ebitda_ai_margin":"Margin (AI)"}),
            dollar_cols=["Peak LOC","Total Interest","EBITDA (AI)"],pct_cols=["Margin (AI)"],
            highlight_neg="EBITDA (AI)")

    with s2:
        section("How bill rate changes affect margin and LOC")
        br_vals = [37.0,37.5,38.0,38.5,39.0,39.5,40.0,40.5,41.0,42.0]
        with st.spinner("Running…"):
            br_df = run_sensitivity(a,hc,"st_bill_rate",br_vals)
        c1,c2 = st.columns(2)
        c1.plotly_chart(_sc(br_df,"value","Bill Rate ($/hr)","ebitda_ai_margin","EBITDA Margin (AI)",pct=True), use_container_width=True)
        c2.plotly_chart(_sc(br_df,"value","Bill Rate ($/hr)","peak_loc","Peak LOC ($)"), use_container_width=True)
        _fmt_table(br_df[["value","ebitda_margin","ebitda_ai_margin","peak_loc","ebitda_ai"]].rename(columns={
            "value":"Bill Rate ($/hr)","ebitda_margin":"EBITDA Margin","ebitda_ai_margin":"Margin (AI)",
            "peak_loc":"Peak LOC","ebitda_ai":"EBITDA (AI)"}),
            dollar_cols=["Peak LOC","EBITDA (AI)"],pct_cols=["EBITDA Margin","Margin (AI)"])

    with s3:
        section("How payroll burden rate affects profitability")
        burd_vals = [0.20,0.25,0.28,0.30,0.33,0.35,0.38,0.40]
        with st.spinner("Running…"):
            burd_df = run_sensitivity(a,hc,"burden",burd_vals)
        burd_df["pct"] = burd_df["value"]*100
        c1,c2 = st.columns(2)
        c1.plotly_chart(_sc(burd_df,"pct","Burden (%)","ebitda_ai_margin","EBITDA Margin (AI)",pct=True), use_container_width=True)
        c2.plotly_chart(_sc(burd_df,"pct","Burden (%)","peak_loc","Peak LOC ($)"), use_container_width=True)
        _fmt_table(burd_df[["pct","ebitda_ai","ebitda_ai_margin","peak_loc"]].rename(columns={
            "pct":"Burden (%)","ebitda_ai":"EBITDA (AI)","ebitda_ai_margin":"Margin (AI)","peak_loc":"Peak LOC"}),
            dollar_cols=["EBITDA (AI)","Peak LOC"],pct_cols=["Margin (AI)"])

    with s4:
        section("How overtime hours affect revenue and margin")
        ot_vals = [0,2,4,6,8,10,12,15,20]
        with st.spinner("Running…"):
            ot_df = run_sensitivity(a,hc,"ot_hours",ot_vals)
        c1,c2 = st.columns(2)
        c1.plotly_chart(_sc(ot_df,"value","OT Hrs/wk","ebitda_ai_margin","EBITDA Margin (AI)",pct=True), use_container_width=True)
        c2.plotly_chart(_sc(ot_df,"value","OT Hrs/wk","total_revenue","Total Revenue ($)"), use_container_width=True)
        _fmt_table(ot_df[["value","ebitda_margin","ebitda_ai_margin","total_revenue","ebitda_ai"]].rename(columns={
            "value":"OT Hrs/wk","ebitda_margin":"EBITDA Margin","ebitda_ai_margin":"Margin (AI)",
            "total_revenue":"Total Revenue","ebitda_ai":"EBITDA (AI)"}),
            dollar_cols=["Total Revenue","EBITDA (AI)"],pct_cols=["EBITDA Margin","Margin (AI)"])

    st.divider()
    if st.button("Export All + Sensitivity to Excel"):
        with st.spinner("Building…"):
            sens = {"Sens_PayTerms":nd_df,"Sens_BillRate":br_df,"Sens_Burden":burd_df,"Sens_OTHours":ot_df}
            xlsx = build_excel(a,hc,weekly_df,mo_full,qdf_full,sens)
        st.download_button("Download Excel",data=xlsx,
            file_name="containment_division_sensitivity.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
