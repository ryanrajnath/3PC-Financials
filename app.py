"""
Containment Division Calculator — OpSource
Dark professional UI with range filtering
"""
from __future__ import annotations
import warnings
from datetime import date
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

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Containment Division Calculator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default hamburger & footer */
#MainMenu, footer { visibility: hidden; }

/* Tighter top padding */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* KPI cards */
.kpi-card {
    background: #1A1D27;
    border: 1px solid #2D3148;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-label { color: #8B8FA8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.kpi-value { color: #FAFAFA; font-size: 26px; font-weight: 700; }
.kpi-sub   { color: #4F8BF9; font-size: 12px; margin-top: 2px; }

/* Section headers */
.section-header {
    color: #8B8FA8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 18px 0 8px 0;
    border-bottom: 1px solid #2D3148;
    padding-bottom: 4px;
}

/* Warning/info banners */
.warn-box {
    background: #2D1F1F;
    border-left: 3px solid #E05252;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-bottom: 8px;
}
.info-box {
    background: #1A2235;
    border-left: 3px solid #4F8BF9;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
PLOT_COLORS   = ["#4F8BF9", "#52D68A", "#F0A843", "#E05252", "#A855F7", "#22D3EE"]

# ── Session state ────────────────────────────────────────────────────────────
if "assumptions"    not in st.session_state:
    st.session_state.assumptions    = default_assumptions()
if "headcount_plan" not in st.session_state:
    st.session_state.headcount_plan = default_headcount()
if "results"        not in st.session_state:
    st.session_state.results        = None

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_dollar(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v*100:.1f}%"

def kpi(col, label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )

def section(label):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def run_and_store():
    with st.spinner("Running model…"):
        try:
            w, m, q = run_model(
                st.session_state.assumptions,
                st.session_state.headcount_plan,
            )
            st.session_state.results = (w, m, q)
            st.success("Model calculated.")
        except Exception as e:
            st.error(f"Model error: {e}")
            st.session_state.results = None

def results_ready():
    return st.session_state.results is not None

def _select(df, cols):
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

def _line(df, x, ys, names, title, pct_y=False):
    fig = go.Figure()
    for y, name, color in zip(ys, names, PLOT_COLORS):
        if y in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y], name=name, mode="lines",
                line=dict(color=color, width=2),
            ))
    fig.update_layout(
        template=PLOT_TEMPLATE, title=title,
        height=320, margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(tickformat=".0%" if pct_y else "$,.0f"),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
    return fig

def _bar(df, x, ys, names, title, stacked=True):
    fig = go.Figure()
    for y, name, color in zip(ys, names, PLOT_COLORS):
        if y in df.columns:
            fig.add_trace(go.Bar(x=df[x], y=df[y], name=name, marker_color=color))
    fig.update_layout(
        template=PLOT_TEMPLATE, title=title,
        barmode="stack" if stacked else "group",
        height=320, margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig

def _range_filter(mo, label="Filter month range"):
    """Return a filtered monthly_df based on a user-selected month range."""
    active = mo[mo["revenue"] > 0]
    if active.empty:
        default_end = min(24, len(mo))
    else:
        last_active = int(active["month_idx"].max()) + 1
        default_end = min(last_active + 3, len(mo))

    lo, hi = st.select_slider(
        label,
        options=list(range(1, len(mo) + 1)),
        value=(1, default_end),
        key=label,
    )
    return mo[(mo["month_idx"] >= lo - 1) & (mo["month_idx"] <= hi - 1)]

def _fmt_table(df, dollar_cols=None, pct_cols=None):
    """Display a styled dark dataframe."""
    fmt = {}
    for c in (dollar_cols or []):
        if c in df.columns:
            fmt[c] = "${:,.0f}"
    for c in (pct_cols or []):
        if c in df.columns:
            fmt[c] = "{:.1%}"
    st.dataframe(
        df.style.format(fmt),
        use_container_width=True,
        height=380,
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER BAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## Containment Division Calculator")
st.caption("OpSource · Weekly financial model · 120-month horizon")

# Always-visible top KPIs (if model has run)
if results_ready():
    _, mo, _ = st.session_state.results
    k1, k2, k3, k4, k5 = st.columns(5)
    peak_mo = mo.loc[mo["loc_end"].idxmax(), "period"] if mo["loc_end"].max() > 0 else "—"
    kpi(k1, "Peak LOC",           fmt_dollar(mo["loc_end"].max()),             peak_mo)
    kpi(k2, "120-mo Revenue",     fmt_dollar(mo["revenue"].sum()),              "accrual")
    kpi(k3, "120-mo EBITDA (AI)", fmt_dollar(mo["ebitda_after_interest"].sum()),"after interest")
    kpi(k4, "Total Interest",     fmt_dollar(mo["interest"].sum()),             "cost of money")
    yr1 = mo[mo["month_idx"] < 12]
    kpi(k5, "Year 1 EBITDA (AI)", fmt_dollar(yr1["ebitda_after_interest"].sum()),"first 12 mo")
    st.divider()

# ════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ════════════════════════════════════════════════════════════════════════════
tab_howto, tab_assume, tab_hc, tab_results, tab_summary, tab_sens = st.tabs([
    "ℹ️  How to Use",
    "⚙️  Assumptions",
    "👥  Headcount Plan",
    "📊  Results",
    "📋  Scenario Summary",
    "🔬  Sensitivity",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB: HOW TO USE
# ════════════════════════════════════════════════════════════════════════════
with tab_howto:
    st.markdown("## Welcome to the Containment Division Calculator")
    st.markdown("**Built for OpSource** · Models weekly operations and cash flow over a 120-month horizon.")
    st.divider()

    st.markdown("""
### How This Tool Works

This calculator answers the key financial questions for running a Containment Division:
- How much **Line of Credit (LOC)** do you need to fund payroll before customers pay?
- When do collections **"catch up"** and LOC stabilizes?
- What is your **EBITDA** before and after interest?
- How does **management headcount** scale with inspector count?
""")

    st.divider()
    st.markdown("### Step-by-Step Guide")

    with st.expander("**Step 1 — Set Your Assumptions** (⚙️ Assumptions tab)", expanded=True):
        st.markdown("""
**Billing & Inspector Pay**
- Set your **ST Bill Rate** (what you charge per hour, default $39/hr)
- Set **OT Bill Premium** — OT is billed at this multiplier × ST rate (default 1.5×)
- Set **ST/OT hours per week** per inspector (default 40 ST + 10 OT = 50 hrs/week)
- Set **Inspector Base Wage** and **Burden %** (payroll taxes + benefits, default 30%)

**Team Leads**
- Team leads are **hourly**, burdened, and billed into revenue (same bill rate as inspectors)
- Default: 1 team lead per 12 inspectors

**Management Layering**
- GM, Ops Coordinators, Field Supervisors, and Regional Managers are **salaried**
- They activate automatically based on inspector count using span-of-control triggers
- GM activates from month 1 by default (configurable)

**Line of Credit (LOC)**
- Set your **Max LOC** (default $1,000,000) and **APR** (default 8.5%)
- **Auto Paydown**: when cash exceeds the buffer, excess automatically repays LOC
- **Net Days**: how long after month-end statement customers take to pay (default 60 days)

**Overhead**
- Fixed monthly costs: Software, Recruiting, Insurance, Travel
- Optional corporate allocation (fixed $ or % of revenue)
""")

    with st.expander("**Step 2 — Enter Your Headcount Plan** (👥 Headcount Plan tab)"):
        st.markdown("""
- Enter the **number of inspectors staffed per month** for up to 120 months (10 years)
- Use **Bulk Fill** to quickly set ranges — e.g., fill months 1–12 with 25 inspectors
- You don't need to fill all 120 months — months with 0 inspectors have no hourly labor cost
- Salaried management (GM, etc.) will still cost money in 0-inspector months if active — watch for this warning
- The **range slider** lets you zoom into any period on the chart
""")

    with st.expander("**Step 3 — Run the Model** (⚙️ Assumptions tab → Run button)"):
        st.markdown("""
- Click **▶ Run / Recalculate** in the Assumptions tab after any change
- The model runs in seconds and updates all tabs automatically
- The **5 KPI cards** at the top always show your current results
""")

    with st.expander("**Step 4 — Read Your Results** (📊 Results tab)"):
        st.markdown("""
**Dashboard sub-tab:**
- LOC / AR / Cash chart — shows the cash cycle visually
- Revenue vs Cost Stack — see when revenue overtakes costs
- EBITDA margins — pre and post interest
- Headcount by role — how management layers in

**Monthly / Quarterly / Weekly sub-tabs:**
- Full financial tables for any time range
- Use the **range slider** at the top to filter to any period
- Months with 0 inspectors show $0 revenue — the model won't break

**Break-Even Calculator:**
- Finds the minimum inspector count for positive EBITDA at a given Net Days
""")

    with st.expander("**Step 5 — Run Sensitivities** (🔬 Sensitivity tab)"):
        st.markdown("""
- Tests how changing **Net Days, Bill Rate, Burden %, or OT Hours** affects your results
- Each sensitivity re-runs the full 120-month model automatically
- Charts and tables show Peak LOC, EBITDA margin, and total interest for each scenario
""")

    with st.expander("**Step 6 — Export to Excel**"):
        st.markdown("""
- Go to **📊 Results → Dashboard** and click **Build Excel Export**
- Or go to **🔬 Sensitivity** and click **Export All + Sensitivity to Excel**
- The Excel file includes: Assumptions, Headcount Plan, Weekly, Monthly, Quarterly, and Sensitivity sheets
""")

    st.divider()
    st.markdown("### Key Concepts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**AR Lag (Net Days)**
Customers are invoiced via a month-end statement.
Collections arrive Net Days later as a lump sum.
During that gap, you draw the LOC to cover payroll.

**LOC Draw Logic**
LOC is drawn automatically when cash would go below
the minimum buffer. With Auto Paydown ON, excess
cash above the buffer automatically repays the LOC.

**Payroll Timing**
Hourly workers (inspectors + team leads) are paid
with a 1-week lag. Salaried management is paid
in the current week.
""")
    with col2:
        st.markdown("""
**EBITDA Pre-Interest vs After Interest**
Pre-interest EBITDA = Revenue − Labor − Overhead
After-interest EBITDA = Pre-interest EBITDA − LOC interest

**Management Scaling**
Roles activate automatically:
- Ops Coordinator: 1 per 75 inspectors
- Field Supervisor: 1 per 60 inspectors
- Regional Manager: 1 per 175 inspectors
- GM: 1 from month 1 (configurable)

**Zero-Inspector Months**
Months with 0 inspectors have $0 hourly labor.
Salaried roles persist unless the division is fully shut down.
""")

    st.divider()
    st.info("**Tip:** Start with the default base case (25 inspectors for 12 months), run the model, and explore the Dashboard tab to understand the cash dynamics before changing assumptions.")


# ════════════════════════════════════════════════════════════════════════════
# TAB: ASSUMPTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_assume:
    a = st.session_state.assumptions

    col_run, col_note = st.columns([1, 3])
    if col_run.button("▶  Run / Recalculate", type="primary", use_container_width=True):
        run_and_store()
    col_note.caption("Change any value below, then click Run to update all results.")

    st.divider()

    # ── Billing & Inspector Pay ──────────────────────────────────────
    section("What You Charge Customers & What You Pay Inspectors")
    c1, c2, c3, c4 = st.columns(4)
    a["st_bill_rate"]      = c1.number_input(
        "Regular-Time Bill Rate ($/hr)",
        value=float(a["st_bill_rate"]), step=0.5, format="%.2f",
        help="The hourly rate you charge the client for each inspector during regular (straight-time) hours.")
    a["ot_bill_premium"]   = c2.number_input(
        "Overtime Bill Multiplier",
        value=float(a["ot_bill_premium"]), step=0.1, format="%.1f",
        help="OT hours are billed at this multiple of the regular rate. Default 1.5× means OT bills at $58.50/hr if regular is $39/hr.")
    a["st_hours"]          = c3.number_input(
        "Regular Hours per Inspector per Week",
        value=int(a["st_hours"]), step=1, format="%d",
        help="Standard work hours per inspector per week, not counting overtime. Typically 40.")
    a["ot_hours"]          = c4.number_input(
        "Overtime Hours per Inspector per Week",
        value=int(a["ot_hours"]), step=1, format="%d",
        help="Overtime hours per inspector per week. These are billed and paid at the OT multiplier. Set to 0 if no planned OT.")

    c5, c6, c7, c8 = st.columns(4)
    a["inspector_wage"]    = c5.number_input(
        "Inspector Hourly Wage ($/hr)",
        value=float(a["inspector_wage"]), step=0.5, format="%.2f",
        help="What you pay each inspector per regular hour, before burden (taxes/benefits).")
    a["ot_pay_multiplier"] = c6.number_input(
        "Overtime Pay Multiplier",
        value=float(a["ot_pay_multiplier"]), step=0.1, format="%.1f",
        help="Inspectors are paid this multiple of their wage for OT hours. Default 1.5× is standard (time-and-a-half).")
    a["burden"]            = c7.number_input(
        "Payroll Burden Rate (e.g. 0.30 = 30%)",
        value=float(a["burden"]), step=0.01, format="%.2f",
        help="The percentage added on top of wages to cover payroll taxes, workers comp, and benefits. Enter as a decimal — 0.30 means 30%.")
    a["net_days"]          = c8.number_input(
        "Customer Payment Terms (Days After Month-End)",
        value=int(a["net_days"]), step=5, format="%d",
        help="How many days after receiving their monthly statement customers typically pay. Net 60 is common in this industry. This drives how long you need to borrow on your line of credit.")

    # ── Team Leads ───────────────────────────────────────────────────
    section("Team Leads — Hourly, Billed to Customer")
    st.caption("Team leads are hourly workers (not salaried). They are counted in your billable hours and billed to the customer at the same rates as inspectors.")
    c1, c2, c3, c4 = st.columns(4)
    a["team_lead_ratio"]   = c1.number_input(
        "Inspectors per Team Lead",
        value=int(a["team_lead_ratio"]), step=1, format="%d",
        help="One team lead is added for every N inspectors. Default is 1 per 12. If you have 25 inspectors, the model adds 3 team leads (⌈25/12⌉).")
    a["lead_wage"]         = c2.number_input(
        "Team Lead Hourly Wage ($/hr)",
        value=float(a["lead_wage"]), step=0.5, format="%.2f",
        help="Base hourly pay for team leads before burden. Typically higher than inspector wage.")
    a["lead_st_hours"]     = c3.number_input(
        "Team Lead Regular Hours/Week",
        value=int(a["lead_st_hours"]), step=1, format="%d",
        help="Regular (straight-time) hours worked per team lead per week.")
    a["lead_ot_hours"]     = c4.number_input(
        "Team Lead Overtime Hours/Week",
        value=int(a["lead_ot_hours"]), step=1, format="%d",
        help="Overtime hours per team lead per week. Set to 0 if team leads don't work OT.")

    # ── Management ───────────────────────────────────────────────────
    section("Salaried Management — Auto-Scaled by Headcount")
    st.caption("These roles are salaried and activate automatically as your inspector count grows. The model adds one role for every N inspectors based on the thresholds below.")
    c1, c2, c3, c4 = st.columns(4)
    a["gm_loaded_annual"]  = c1.number_input(
        "General Manager — Total Annual Cost ($)",
        value=float(a["gm_loaded_annual"]), step=1000., format="%.0f",
        help="The fully-loaded annual cost of the GM including salary, benefits, and taxes. Already fully burdened — do NOT add burden again. Default $117,000.")
    a["opscoord_base"]     = c2.number_input(
        "Operations Coordinator — Base Annual Salary ($)",
        value=float(a["opscoord_base"]), step=1000., format="%.0f",
        help="Base salary for an Ops Coordinator. The management burden % below is applied on top. Ops Coordinators handle scheduling, dispatch, and field support.")
    a["fieldsup_base"]     = c3.number_input(
        "Field Supervisor — Base Annual Salary ($)",
        value=float(a["fieldsup_base"]), step=1000., format="%.0f",
        help="Base salary for a Field Supervisor. Management burden is applied on top. Field Supervisors directly oversee inspector crews in the field.")
    a["regionalmgr_base"]  = c4.number_input(
        "Regional Manager — Base Annual Salary ($)",
        value=float(a["regionalmgr_base"]), step=1000., format="%.0f",
        help="Base salary for a Regional Manager. Management burden is applied on top. Regional Managers oversee multiple field supervisors across a geography.")

    c5, c6, c7, c8 = st.columns(4)
    a["mgmt_burden"]       = c5.number_input(
        "Management Benefit & Tax Rate (e.g. 0.25 = 25%)",
        value=float(a["mgmt_burden"]), step=0.01, format="%.2f",
        help="Burden rate applied to salaried management base salaries (not the GM — GM is already fully loaded). Covers employer taxes and benefits. Enter as decimal.")
    a["opscoord_span"]     = c6.number_input(
        "Inspectors per Operations Coordinator",
        value=int(a["opscoord_span"]), step=5, format="%d",
        help="The model adds 1 Ops Coordinator for every N inspectors. At 75 inspectors you get 1, at 150 you get 2, etc. Adjust based on your management structure.")
    a["fieldsup_span"]     = c7.number_input(
        "Inspectors per Field Supervisor",
        value=int(a["fieldsup_span"]), step=5, format="%d",
        help="The model adds 1 Field Supervisor for every N inspectors. Default is 1 per 60 — meaning a supervisor covers up to 60 workers in the field.")
    a["regionalmgr_span"]  = c8.number_input(
        "Inspectors per Regional Manager",
        value=int(a["regionalmgr_span"]), step=5, format="%d",
        help="The model adds 1 Regional Manager for every N inspectors. Default is 1 per 175. You likely won't need one until you're staffing 175+ inspectors.")

    c9, c10 = st.columns(4)[:2]
    a["gm_start_month"]    = c9.number_input(
        "Month to Hire the GM (model month #)",
        value=int(a["gm_start_month"]), step=1, format="%d",
        help="Which month of the model the GM starts. Month 1 = the very first month. Set to a later month if you plan to delay the GM hire.")
    a["gm_ramp_months"]    = c10.number_input(
        "GM Part-Time Ramp Period (months at half cost)",
        value=int(a["gm_ramp_months"]), step=1, format="%d",
        help="Number of months the GM is part-time (0.5 FTE) before going full-time. Set to 0 if the GM starts full-time immediately.")

    # ── LOC ──────────────────────────────────────────────────────────
    section("Line of Credit (LOC) — Funding the AR Gap")
    st.caption("Because customers pay 60–120 days after month-end, you need a credit line to cover payroll in the meantime. The model draws and repays this automatically.")
    c1, c2, c3, c4 = st.columns(4)
    a["apr"]               = c1.number_input(
        "Annual Interest Rate on LOC (e.g. 0.085 = 8.5%)",
        value=float(a["apr"]), step=0.005, format="%.3f",
        help="The annual interest rate your bank charges on the line of credit balance. Enter as a decimal — 0.085 = 8.5%. Interest is calculated monthly on the average balance.")
    a["max_loc"]           = c2.number_input(
        "Maximum Credit Line Amount ($)",
        value=float(a["max_loc"]), step=50000., format="%.0f",
        help="The maximum amount you can borrow on the line of credit. The model warns you if it needs more than this. Default $1,000,000.")
    a["initial_cash"]      = c3.number_input(
        "Starting Cash Balance ($)",
        value=float(a["initial_cash"]), step=5000., format="%.0f",
        help="Cash on hand at the start of the model. If you're starting from zero, leave at $0.")
    a["cash_buffer"]       = c4.number_input(
        "Minimum Cash Buffer to Keep On Hand ($)",
        value=float(a["cash_buffer"]), step=5000., format="%.0f",
        help="The model maintains at least this much cash at all times by drawing the LOC if needed. Default $25,000 — acts as a safety cushion.")
    a["auto_paydown"]      = st.checkbox(
        "Automatically repay the credit line when cash exceeds the buffer",
        value=bool(a["auto_paydown"]),
        help="When ON: any cash above the buffer is automatically swept to pay down the LOC, reducing interest cost. When OFF: LOC is only drawn when needed, never proactively repaid.")
    a["start_date"]        = st.date_input("Model Start Date", value=a["start_date"],
                                            help="The first day of the model. All weeks and months are calculated forward from this date.")

    # ── Overhead ─────────────────────────────────────────────────────
    section("Fixed Monthly Overhead Costs")
    st.caption("These costs are incurred every month regardless of inspector count.")
    c1, c2, c3, c4 = st.columns(4)
    a["software_monthly"]   = c1.number_input("Software & Tech ($/mo)",
        value=float(a["software_monthly"]), step=100., format="%.0f",
        help="Monthly cost for software tools — scheduling, time tracking, reporting, etc.")
    a["recruiting_monthly"] = c2.number_input("Recruiting & Hiring ($/mo)",
        value=float(a["recruiting_monthly"]), step=100., format="%.0f",
        help="Monthly spend on job boards, recruiters, background checks, and onboarding.")
    a["insurance_monthly"]  = c3.number_input("Insurance ($/mo)",
        value=float(a["insurance_monthly"]), step=100., format="%.0f",
        help="Monthly insurance costs not already included in burden — general liability, E&O, etc.")
    a["travel_monthly"]     = c4.number_input("Travel & Field Expenses ($/mo)",
        value=float(a["travel_monthly"]), step=100., format="%.0f",
        help="Monthly travel, mileage, lodging, and miscellaneous field expenses.")

    ca_mode = st.radio("Corporate / Parent Company Allocation",
                       ["Fixed monthly amount", "Percentage of revenue"],
                       index=0 if a["corp_alloc_mode"] == "fixed" else 1,
                       horizontal=True,
                       help="Whether to charge a fixed overhead allocation from the parent company or a % of revenue. Default is $0 since this division is inside OpSource.")
    a["corp_alloc_mode"] = "fixed" if ca_mode == "Fixed monthly amount" else "pct_revenue"
    if a["corp_alloc_mode"] == "fixed":
        a["corp_alloc_fixed"] = st.number_input("Corporate Allocation ($/mo)",
            value=float(a["corp_alloc_fixed"]), step=500., format="%.0f",
            help="Fixed monthly charge from the parent company. Default $0.")
    else:
        a["corp_alloc_pct"]   = st.number_input("Corporate Allocation (% of Revenue)",
            value=float(a["corp_alloc_pct"]), step=0.005, format="%.3f",
            help="Percentage of revenue charged by the parent company. Enter as decimal — 0.05 = 5%.")

    st.session_state.assumptions = a


# ════════════════════════════════════════════════════════════════════════════
# TAB: HEADCOUNT PLAN
# ════════════════════════════════════════════════════════════════════════════
with tab_hc:
    hc = st.session_state.headcount_plan
    a_start = st.session_state.assumptions["start_date"]

    # Bulk fill
    section("Bulk Fill")
    c1, c2, c3, c4 = st.columns(4)
    fill_val  = c1.number_input("Inspectors",  0, 10000, 25, step=5, key="fv")
    fill_from = c2.number_input("From month",  1, 120,   1,  step=1, key="ff")
    fill_to   = c3.number_input("To month",    1, 120,   12, step=1, key="ft")
    if c4.button("Apply", use_container_width=True):
        for i in range(int(fill_from) - 1, int(fill_to)):
            hc[i] = int(fill_val)
        st.session_state.headcount_plan = hc
        st.rerun()

    # Preview chart (range-filtered)
    section("Preview")
    month_labels = []
    for i in range(120):
        yr = a_start.year  + (a_start.month - 1 + i) // 12
        mo = (a_start.month - 1 + i) % 12 + 1
        month_labels.append(f"{yr}-{mo:02d}")

    hc_preview_df = pd.DataFrame({"month_idx": range(120), "period": month_labels, "inspectors": hc})
    lo_hc, hi_hc = st.select_slider(
        "Show months",
        options=list(range(1, 121)),
        value=(1, 24),
        key="hc_range",
    )
    filtered_hc = hc_preview_df[(hc_preview_df["month_idx"] >= lo_hc - 1) & (hc_preview_df["month_idx"] <= hi_hc - 1)]
    fig_hc = px.bar(filtered_hc, x="period", y="inspectors",
                    template=PLOT_TEMPLATE, title="Inspectors Staffed per Month",
                    color_discrete_sequence=[PLOT_COLORS[0]])
    fig_hc.update_layout(height=280, margin=dict(l=10, r=10, t=36, b=10))
    st.plotly_chart(fig_hc, use_container_width=True)

    # Editable grid
    section("Edit Headcount (all 120 months)")
    hc_df = pd.DataFrame({"Period": month_labels, "Inspectors": hc})
    edited = st.data_editor(
        hc_df,
        column_config={
            "Period":     st.column_config.TextColumn(disabled=True),
            "Inspectors": st.column_config.NumberColumn(min_value=0, max_value=10000, step=1),
        },
        use_container_width=True,
        height=500,
        num_rows="fixed",
    )
    st.session_state.headcount_plan = edited["Inspectors"].tolist()


# ════════════════════════════════════════════════════════════════════════════
# TAB: RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not results_ready():
        st.markdown('<div class="info-box">Go to <b>Assumptions</b> tab and click <b>▶ Run / Recalculate</b> to generate results.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, qdf_full = st.session_state.results
    a = st.session_state.assumptions

    # Warnings
    n_loc  = weekly_df["warn_loc_maxed"].sum()
    n_neg  = weekly_df["warn_neg_ebitda"].sum()
    n_mgmt = weekly_df["warn_mgmt_no_insp"].sum()
    if n_loc:  st.markdown(f'<div class="warn-box">LOC exceeded max line in {n_loc} week(s) — consider raising the LOC limit.</div>', unsafe_allow_html=True)
    if n_mgmt: st.markdown(f'<div class="warn-box">Salaried management persists with 0 inspectors in {n_mgmt} week(s).</div>', unsafe_allow_html=True)

    # ── Global range filter ──────────────────────────────────────────
    section("Date Range Filter")
    active_mo = mo_full[mo_full["revenue"] > 0]
    last_active = int(active_mo["month_idx"].max()) + 1 if not active_mo.empty else 12
    default_end = min(last_active + 3, 120)

    r_lo, r_hi = st.select_slider(
        "Show model months",
        options=list(range(1, 121)),
        value=(1, default_end),
        key="results_range",
    )

    mo  = mo_full[(mo_full["month_idx"] >= r_lo - 1) & (mo_full["month_idx"] <= r_hi - 1)].copy()
    qdf = qdf_full[(qdf_full["quarter_idx"] >= (r_lo - 1) // 3) & (qdf_full["quarter_idx"] <= (r_hi - 1) // 3)].copy()
    wdf = weekly_df[(weekly_df["month_idx"] >= r_lo - 1) & (weekly_df["month_idx"] <= r_hi - 1)].copy()

    st.divider()
    r1, r2, r3, r4 = st.tabs(["📈 Dashboard", "📅 Monthly", "📆 Quarterly", "🗓️ Weekly"])

    # ── DASHBOARD ────────────────────────────────────────────────────
    with r1:
        section("Cash & LOC Dynamics")
        fig_loc = go.Figure()
        fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["loc_end"],
                                     name="LOC Balance", fill="tozeroy",
                                     line=dict(color=PLOT_COLORS[3], width=2)))
        fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["ar_end"],
                                     name="AR Balance", line=dict(color=PLOT_COLORS[0], width=2)))
        fig_loc.add_trace(go.Scatter(x=mo["period"], y=mo["cash_end"],
                                     name="Cash", line=dict(color=PLOT_COLORS[1], width=2)))
        fig_loc.add_hline(y=float(a["max_loc"]), line_dash="dot",
                          line_color="#E05252", annotation_text="LOC Limit",
                          annotation_font_color="#E05252")
        fig_loc.update_layout(template=PLOT_TEMPLATE, height=340,
                              margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=-0.2),
                              yaxis=dict(tickformat="$,.0f"))
        st.plotly_chart(fig_loc, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            section("Revenue vs Cost Stack")
            fig_rv = _bar(mo, "period",
                          ["hourly_labor", "salaried_cost", "overhead"],
                          ["Hourly Labor", "Salaried", "Overhead"],
                          "Monthly Cost Stack")
            fig_rv.add_trace(go.Scatter(x=mo["period"], y=mo["revenue"],
                                        name="Revenue", mode="lines",
                                        line=dict(color=PLOT_COLORS[1], width=2)))
            st.plotly_chart(fig_rv, use_container_width=True)

        with c2:
            section("EBITDA")
            fig_eb = _line(mo, "period",
                           ["ebitda", "ebitda_after_interest"],
                           ["EBITDA (pre-int)", "EBITDA (after int)"],
                           "Monthly EBITDA")
            st.plotly_chart(fig_eb, use_container_width=True)

        section("Margins")
        fig_mg = _line(mo, "period",
                       ["ebitda_margin", "ebitda_ai_margin"],
                       ["EBITDA Margin", "EBITDA Margin (after int)"],
                       "EBITDA Margins", pct_y=True)
        st.plotly_chart(fig_mg, use_container_width=True)

        section("Headcount by Role")
        fig_hc2 = _bar(mo, "period",
                       ["inspectors_avg", "team_leads_avg", "n_opscoord", "n_fieldsup", "n_regionalmgr"],
                       ["Inspectors", "Team Leads", "Ops Coord", "Field Sup", "Reg Mgr"],
                       "Average Monthly Headcount")
        st.plotly_chart(fig_hc2, use_container_width=True)

        # Break-even
        st.divider()
        section("Break-Even Calculator")
        be_c1, be_c2, be_c3 = st.columns(3)
        be_nd = be_c1.selectbox("Net Days", [30, 60, 90, 120, 150],
                                index=1 if int(a["net_days"]) not in [30,60,90,120,150] else
                                [30,60,90,120,150].index(int(a["net_days"])))
        if be_c2.button("Find Break-Even", use_container_width=True):
            with st.spinner("Searching…"):
                be = find_breakeven_inspectors(a, be_nd)
            be_c3.success(f"Min inspectors: **{be}** @ Net {be_nd}")

        # Export
        st.divider()
        section("Export")
        if st.button("Build Excel Export", use_container_width=False):
            with st.spinner("Building…"):
                xlsx = build_excel(a, st.session_state.headcount_plan,
                                   weekly_df, mo_full, qdf_full)
            st.download_button("⬇  Download Excel", data=xlsx,
                               file_name="containment_division_model.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── MONTHLY ──────────────────────────────────────────────────────
    with r2:
        section("Monthly Summary Table")
        display_cols = [
            "period", "inspectors_avg", "team_leads_avg",
            "revenue", "hourly_labor", "salaried_cost", "overhead", "total_labor",
            "ebitda", "ebitda_margin",
            "interest", "ebitda_after_interest", "ebitda_ai_margin",
            "collections", "ar_end", "loc_end", "cash_end", "peak_loc_to_date",
        ]
        _fmt_table(
            _select(mo, display_cols),
            dollar_cols=["revenue","hourly_labor","salaried_cost","overhead","total_labor",
                         "ebitda","interest","ebitda_after_interest",
                         "collections","ar_end","loc_end","cash_end","peak_loc_to_date"],
            pct_cols=["ebitda_margin","ebitda_ai_margin"],
        )

    # ── QUARTERLY ────────────────────────────────────────────────────
    with r3:
        section("Quarterly Summary Table")
        q_display = [
            "yr_q", "revenue", "hourly_labor", "salaried_cost", "overhead", "total_labor",
            "ebitda", "ebitda_margin",
            "interest", "ebitda_after_interest", "ebitda_ai_margin",
            "ar_end", "loc_end", "cash_end", "peak_loc_to_date",
        ]
        _fmt_table(
            _select(qdf, q_display),
            dollar_cols=["revenue","hourly_labor","salaried_cost","overhead","total_labor",
                         "ebitda","interest","ebitda_after_interest",
                         "ar_end","loc_end","cash_end","peak_loc_to_date"],
            pct_cols=["ebitda_margin","ebitda_ai_margin"],
        )

    # ── WEEKLY ───────────────────────────────────────────────────────
    with r4:
        # Warnings
        w_neg = wdf["warn_neg_ebitda"].sum()
        if w_neg:
            st.markdown(f'<div class="warn-box">Negative EBITDA in {w_neg} week(s) within selected range.</div>', unsafe_allow_html=True)

        w1, w2, w3, w4 = st.tabs(["Headcount & Revenue", "Labor & EBITDA", "AR & Collections", "Cash & LOC"])

        with w1:
            cols = ["week_start","week_end","inspectors","team_leads",
                    "n_opscoord","n_fieldsup","n_regionalmgr",
                    "insp_st_hrs","insp_ot_hrs",
                    "insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"]
            _fmt_table(_select(wdf, cols),
                       dollar_cols=["insp_rev_st","insp_rev_ot","lead_rev_st","lead_rev_ot","revenue_wk"])

        with w2:
            cols = ["week_start","inspectors","team_leads",
                    "insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                    "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"]
            _fmt_table(_select(wdf, cols),
                       dollar_cols=["insp_labor_st","insp_labor_ot","lead_labor_st","lead_labor_ot",
                                    "hourly_labor","salaried_wk","overhead_wk","revenue_wk","ebitda_wk"])

        with w3:
            cols = ["week_start","week_end","is_month_end",
                    "revenue_wk","statement_amt","collections","ar_begin","ar_end"]
            _fmt_table(_select(wdf, cols),
                       dollar_cols=["revenue_wk","statement_amt","collections","ar_begin","ar_end"])

        with w4:
            cols = ["week_start","payroll_cash_out","salaried_wk","overhead_wk","interest_paid",
                    "collections","cash_begin","loc_draw","loc_repay","cash_end","loc_begin","loc_end"]
            _fmt_table(_select(wdf, cols),
                       dollar_cols=["payroll_cash_out","salaried_wk","overhead_wk","interest_paid",
                                    "collections","cash_begin","loc_draw","loc_repay",
                                    "cash_end","loc_begin","loc_end"])

            # Reconciliation
            max_err = wdf[["check_ar","check_loc","check_cash"]].max().max()
            if max_err < 0.01:
                st.success(f"Reconciliation checks pass — max error ${max_err:.4f}")
            else:
                st.error(f"Reconciliation error: ${max_err:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# TAB: SCENARIO SUMMARY
# ════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if not results_ready():
        st.markdown('<div class="info-box">Run the model first from the <b>Assumptions</b> tab.</div>', unsafe_allow_html=True)
        st.stop()

    weekly_df, mo_full, _ = st.session_state.results
    a = st.session_state.assumptions

    section("Date Range Filter")
    active_mo = mo_full[mo_full["revenue"] > 0]
    last_active = int(active_mo["month_idx"].max()) + 1 if not active_mo.empty else 12
    default_end = min(last_active + 3, 120)
    sr_lo, sr_hi = st.select_slider(
        "Show months",
        options=list(range(1, 121)),
        value=(1, default_end),
        key="summary_range",
    )
    mo = mo_full[(mo_full["month_idx"] >= sr_lo - 1) & (mo_full["month_idx"] <= sr_hi - 1)].copy()

    st.divider()

    # ── Income Statement ─────────────────────────────────────────────
    section("Income Statement — Selected Range")
    tot_rev    = mo["revenue"].sum()
    tot_hl     = mo["hourly_labor"].sum()
    tot_sal    = mo["salaried_cost"].sum()
    tot_ovhd   = mo["overhead"].sum()
    tot_labor  = tot_hl + tot_sal
    tot_ebitda = mo["ebitda"].sum()
    tot_int    = mo["interest"].sum()
    tot_ebitda_ai = mo["ebitda_after_interest"].sum()

    is_data = {
        "Line Item": [
            "Revenue",
            "  Hourly Labor (Inspectors + TLs)",
            "  Salaried Management",
            "  Overhead",
            "Total Expenses",
            "EBITDA (pre-interest)",
            "  LOC Interest",
            "EBITDA (after interest)",
        ],
        "Amount ($)": [
            tot_rev, tot_hl, tot_sal, tot_ovhd,
            tot_labor + tot_ovhd,
            tot_ebitda, -tot_int, tot_ebitda_ai,
        ],
        "% of Revenue": [
            1.0,
            tot_hl / tot_rev if tot_rev else 0,
            tot_sal / tot_rev if tot_rev else 0,
            tot_ovhd / tot_rev if tot_rev else 0,
            (tot_labor + tot_ovhd) / tot_rev if tot_rev else 0,
            tot_ebitda / tot_rev if tot_rev else 0,
            -tot_int / tot_rev if tot_rev else 0,
            tot_ebitda_ai / tot_rev if tot_rev else 0,
        ],
    }
    is_df = pd.DataFrame(is_data)
    st.dataframe(
        is_df.style
            .format({"Amount ($)": "${:,.0f}", "% of Revenue": "{:.1%}"})
            .apply(lambda row: [
                "font-weight:bold; color:#52D68A" if row["Line Item"] in ("Revenue","EBITDA (after interest)") else
                "font-weight:bold; color:#E05252" if row["Line Item"] == "Total Expenses" else
                "font-weight:bold" if row["Line Item"] == "EBITDA (pre-interest)" else ""
                for _ in row], axis=1),
        use_container_width=True,
        height=330,
    )

    st.divider()

    # ── Headcount Breakdown ──────────────────────────────────────────
    section("Headcount Breakdown — Averages Over Selected Range")
    hc_rows = {
        "Role": ["Inspectors", "Team Leads", "Ops Coordinators", "Field Supervisors", "Regional Managers", "GM"],
        "Avg / Month": [
            mo["inspectors_avg"].mean(),
            mo["team_leads_avg"].mean(),
            mo["n_opscoord"].mean(),
            mo["n_fieldsup"].mean(),
            mo["n_regionalmgr"].mean(),
            (weekly_df[(weekly_df["month_idx"] >= sr_lo-1) & (weekly_df["month_idx"] <= sr_hi-1)]["gm_fte"].mean()),
        ],
        "Peak / Month": [
            mo["inspectors_avg"].max(),
            mo["team_leads_avg"].max(),
            mo["n_opscoord"].max(),
            mo["n_fieldsup"].max(),
            mo["n_regionalmgr"].max(),
            1.0 if weekly_df[weekly_df["month_idx"] <= sr_hi-1]["gm_fte"].max() > 0 else 0,
        ],
        "Type": ["Hourly", "Hourly", "Salaried", "Salaried", "Salaried", "Salaried"],
        "Billed to Revenue": ["Yes", "Yes", "No", "No", "No", "No"],
    }
    hc_df2 = pd.DataFrame(hc_rows)
    st.dataframe(
        hc_df2.style.format({"Avg / Month": "{:.1f}", "Peak / Month": "{:.1f}"}),
        use_container_width=True, height=260,
    )

    st.divider()

    # ── Revenue Breakdown ────────────────────────────────────────────
    section("Revenue Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        rev_pie = go.Figure(go.Pie(
            labels=["Inspector ST", "Inspector OT", "Team Lead ST", "Team Lead OT"],
            values=[mo["insp_rev_st"].sum(), mo["insp_rev_ot"].sum(),
                    mo["lead_rev_st"].sum(), mo["lead_rev_ot"].sum()],
            hole=0.45,
            marker_colors=PLOT_COLORS[:4],
        ))
        rev_pie.update_layout(template=PLOT_TEMPLATE, height=300,
                              title="Revenue by Component",
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(rev_pie, use_container_width=True)

    with c2:
        exp_pie = go.Figure(go.Pie(
            labels=["Hourly Labor", "Salaried Mgmt", "Overhead"],
            values=[tot_hl, tot_sal, tot_ovhd],
            hole=0.45,
            marker_colors=[PLOT_COLORS[3], PLOT_COLORS[2], PLOT_COLORS[4]],
        ))
        exp_pie.update_layout(template=PLOT_TEMPLATE, height=300,
                              title="Expenses by Component",
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(exp_pie, use_container_width=True)

    # ── Monthly trend ────────────────────────────────────────────────
    section("Revenue vs Net Profit — Monthly Trend")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=mo["period"], y=mo["revenue"], name="Revenue",
                               marker_color=PLOT_COLORS[0], opacity=0.7))
    fig_trend.add_trace(go.Scatter(x=mo["period"], y=mo["ebitda_after_interest"],
                                   name="Net Profit (EBITDA AI)", mode="lines+markers",
                                   line=dict(color=PLOT_COLORS[1], width=2)))
    fig_trend.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
    fig_trend.update_layout(template=PLOT_TEMPLATE, height=320,
                            margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", y=-0.2),
                            yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_trend, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB: SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════
with tab_sens:
    if not results_ready():
        st.markdown('<div class="info-box">Run the model first from the <b>Assumptions</b> tab.</div>', unsafe_allow_html=True)
        st.stop()

    a  = st.session_state.assumptions
    hc = st.session_state.headcount_plan

    st.caption("Each table re-runs the full 120-month model varying one parameter at a time.")

    s1, s2, s3, s4 = st.tabs(["Net Days", "Bill Rate", "Burden %", "OT Hours"])

    def _sens_chart(df, x_col, x_label, y_col, y_label, pct=False):
        fig = go.Figure(go.Scatter(
            x=df[x_col], y=df[y_col], mode="lines+markers",
            line=dict(color=PLOT_COLORS[0], width=2),
            marker=dict(size=7),
        ))
        fig.update_layout(
            template=PLOT_TEMPLATE, height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=x_label, yaxis_title=y_label,
            yaxis=dict(tickformat=".1%" if pct else "$,.0f"),
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
        return fig

    with s1:
        section("Net Days vs Peak LOC & Interest")
        nd_vals = [30, 45, 60, 75, 90, 105, 120, 150]
        with st.spinner("Running…"):
            nd_df = run_sensitivity(a, hc, "net_days", nd_vals)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sens_chart(nd_df, "value", "Net Days", "peak_loc", "Peak LOC ($)"), use_container_width=True)
        c2.plotly_chart(_sens_chart(nd_df, "value", "Net Days", "ebitda_ai_margin", "EBITDA Margin (AI)", pct=True), use_container_width=True)
        _fmt_table(
            nd_df[["value","peak_loc","annual_interest","ebitda_ai","ebitda_ai_margin"]].rename(columns={
                "value":"Net Days","peak_loc":"Peak LOC","annual_interest":"Total Interest",
                "ebitda_ai":"EBITDA (AI)","ebitda_ai_margin":"Margin (AI)"}),
            dollar_cols=["Peak LOC","Total Interest","EBITDA (AI)"],
            pct_cols=["Margin (AI)"],
        )

    with s2:
        section("Bill Rate vs EBITDA Margin & Peak LOC")
        br_vals = [37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 42.0]
        with st.spinner("Running…"):
            br_df = run_sensitivity(a, hc, "st_bill_rate", br_vals)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sens_chart(br_df, "value", "Bill Rate ($/hr)", "ebitda_ai_margin", "EBITDA Margin (AI)", pct=True), use_container_width=True)
        c2.plotly_chart(_sens_chart(br_df, "value", "Bill Rate ($/hr)", "peak_loc", "Peak LOC ($)"), use_container_width=True)
        _fmt_table(
            br_df[["value","ebitda_margin","ebitda_ai_margin","peak_loc","ebitda_ai"]].rename(columns={
                "value":"Bill Rate","ebitda_margin":"EBITDA Margin","ebitda_ai_margin":"Margin (AI)",
                "peak_loc":"Peak LOC","ebitda_ai":"EBITDA (AI)"}),
            dollar_cols=["Peak LOC","EBITDA (AI)"],
            pct_cols=["EBITDA Margin","Margin (AI)"],
        )

    with s3:
        section("Burden % vs EBITDA & Peak LOC")
        burd_vals = [0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
        with st.spinner("Running…"):
            burd_df = run_sensitivity(a, hc, "burden", burd_vals)
        burd_df["pct"] = burd_df["value"] * 100
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sens_chart(burd_df, "pct", "Burden (%)", "ebitda_ai_margin", "EBITDA Margin (AI)", pct=True), use_container_width=True)
        c2.plotly_chart(_sens_chart(burd_df, "pct", "Burden (%)", "peak_loc", "Peak LOC ($)"), use_container_width=True)
        _fmt_table(
            burd_df[["pct","ebitda_ai","ebitda_ai_margin","peak_loc"]].rename(columns={
                "pct":"Burden (%)","ebitda_ai":"EBITDA (AI)","ebitda_ai_margin":"Margin (AI)","peak_loc":"Peak LOC"}),
            dollar_cols=["EBITDA (AI)","Peak LOC"],
            pct_cols=["Margin (AI)"],
        )

    with s4:
        section("OT Hours vs EBITDA Margin")
        ot_vals = [0, 2, 4, 6, 8, 10, 12, 15, 20]
        with st.spinner("Running…"):
            ot_df = run_sensitivity(a, hc, "ot_hours", ot_vals)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_sens_chart(ot_df, "value", "OT Hrs/wk", "ebitda_ai_margin", "EBITDA Margin (AI)", pct=True), use_container_width=True)
        c2.plotly_chart(_sens_chart(ot_df, "value", "OT Hrs/wk", "total_revenue", "Total Revenue ($)"), use_container_width=True)
        _fmt_table(
            ot_df[["value","ebitda_margin","ebitda_ai_margin","total_revenue","ebitda_ai"]].rename(columns={
                "value":"OT Hrs/wk","ebitda_margin":"EBITDA Margin","ebitda_ai_margin":"Margin (AI)",
                "total_revenue":"Total Revenue","ebitda_ai":"EBITDA (AI)"}),
            dollar_cols=["Total Revenue","EBITDA (AI)"],
            pct_cols=["EBITDA Margin","Margin (AI)"],
        )

    st.divider()
    if st.button("Export All + Sensitivity to Excel"):
        with st.spinner("Building…"):
            sens = {"Sens_NetDays": nd_df, "Sens_BillRate": br_df,
                    "Sens_Burden": burd_df, "Sens_OTHours": ot_df}
            xlsx = build_excel(a, hc, weekly_df, mo_full, qdf_full, sens)
        st.download_button("⬇  Download Excel", data=xlsx,
                           file_name="containment_division_sensitivity.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
