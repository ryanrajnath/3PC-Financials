"""
app.py — Containment Division Calculator  (OpSource)
Run with:  streamlit run app.py
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

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Containment Division Calculator",
    page_icon="📊",
    layout="wide",
)

# ── Session state init ───────────────────────────────────────────────────────
if "assumptions" not in st.session_state:
    st.session_state.assumptions = default_assumptions()
if "headcount_plan" not in st.session_state:
    st.session_state.headcount_plan = default_headcount()
if "results" not in st.session_state:
    st.session_state.results = None   # (weekly_df, monthly_df, quarterly_df)


# ── Helpers ─────────────────────────────────────────────────────────────────
def fmt_dollar(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"${v:,.0f}"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"

def run_and_store():
    with st.spinner("Running model…"):
        try:
            w, m, q = run_model(
                st.session_state.assumptions,
                st.session_state.headcount_plan,
            )
            st.session_state.results = (w, m, q)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.session_state.results = None


def results_ready() -> bool:
    return st.session_state.results is not None


def _bar_chart(df, x_col, y_cols, labels, title):
    fig = go.Figure()
    for col, label in zip(y_cols, labels):
        fig.add_trace(go.Bar(name=label, x=df[x_col], y=df[col]))
    fig.update_layout(title=title, barmode="stack", height=380,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig

def _line_chart(df, x_col, y_cols, labels, title):
    fig = go.Figure()
    for col, label in zip(y_cols, labels):
        fig.add_trace(go.Scatter(x=df[x_col], y=df[col], name=label, mode="lines"))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def _select(df, cols):
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

def _fmt_sens(df: pd.DataFrame, dollar_cols: list, pct_cols: list):
    styled = df.style
    for c in dollar_cols:
        if c in df.columns:
            styled = styled.format({c: "${:,.0f}"})
    for c in pct_cols:
        if c in df.columns:
            styled = styled.format({c: "{:.1%}"})
    st.dataframe(styled, use_container_width=True)


# ── Sidebar / Navigation ─────────────────────────────────────────────────────
st.sidebar.title("Containment Division")
st.sidebar.caption("OpSource Financial Model")
st.sidebar.divider()

PAGES = [
    "Assumptions",
    "Headcount Plan",
    "Weekly Output",
    "Monthly Output",
    "Quarterly Output",
    "Dashboard",
    "Sensitivity",
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.divider()

if st.sidebar.button("▶  Run / Recalculate Model", type="primary", use_container_width=True):
    run_and_store()

if results_ready():
    _, mo, _ = st.session_state.results
    st.sidebar.success("Model ready")
    st.sidebar.metric("Peak LOC",     fmt_dollar(mo["loc_end"].max()))
    st.sidebar.metric("120-mo Revenue", fmt_dollar(mo["revenue"].sum()))
    st.sidebar.metric("120-mo EBITDA (after int.)",
                      fmt_dollar(mo["ebitda_after_interest"].sum()))
else:
    st.sidebar.info("Click Run to compute results.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Assumptions
# ════════════════════════════════════════════════════════════════════════════
if page == "Assumptions":
    st.title("Assumptions")
    a = st.session_state.assumptions

    # ── Billing & Pay ────────────────────────────────────────────────
    with st.expander("Billing & Inspector Pay", expanded=True):
        c1, c2, c3 = st.columns(3)
        a["st_bill_rate"]     = c1.number_input("ST Bill Rate ($/hr)",     value=float(a["st_bill_rate"]),     step=0.5,  format="%.2f")
        a["ot_bill_premium"]  = c2.number_input("OT Bill Premium (mult.)", value=float(a["ot_bill_premium"]),  step=0.1,  format="%.2f")
        a["st_hours"]         = c3.number_input("ST Hrs/wk per Inspector", value=int(a["st_hours"]),           step=1,    format="%d")
        c4, c5, c6 = st.columns(3)
        a["ot_hours"]         = c4.number_input("OT Hrs/wk per Inspector", value=int(a["ot_hours"]),           step=1,    format="%d")
        a["inspector_wage"]   = c5.number_input("Inspector Base Wage ($/hr)", value=float(a["inspector_wage"]),step=0.5,  format="%.2f")
        a["ot_pay_multiplier"]= c6.number_input("OT Pay Multiplier",        value=float(a["ot_pay_multiplier"]),step=0.1, format="%.2f")
        c7, c8 = st.columns(3)[:2]
        a["burden"]           = c7.number_input("Burden % (decimal)",      value=float(a["burden"]),           step=0.01, format="%.2f",
                                                 help="e.g. 0.30 = 30%")

    # ── Team Leads ───────────────────────────────────────────────────
    with st.expander("Team Leads", expanded=True):
        c1, c2, c3 = st.columns(3)
        a["team_lead_ratio"]  = c1.number_input("Inspectors per Team Lead", value=int(a["team_lead_ratio"]),  step=1,   format="%d")
        a["lead_wage"]        = c2.number_input("Team Lead Base Wage ($/hr)", value=float(a["lead_wage"]),    step=0.5, format="%.2f")
        a["lead_st_hours"]    = c3.number_input("Team Lead ST Hrs/wk",      value=int(a["lead_st_hours"]),    step=1,   format="%d")
        c4, _ = st.columns(3)[:2]
        a["lead_ot_hours"]    = c4.number_input("Team Lead OT Hrs/wk",      value=int(a["lead_ot_hours"]),    step=1,   format="%d")

    # ── Management ───────────────────────────────────────────────────
    with st.expander("Management Layering", expanded=True):
        st.caption("Salaries: GM is fully loaded. Others: base × (1 + Mgmt Burden %).")
        c1, c2, c3, c4 = st.columns(4)
        a["gm_loaded_annual"]  = c1.number_input("GM Fully Loaded Annual $",   value=float(a["gm_loaded_annual"]),  step=1000.0, format="%.0f")
        a["opscoord_base"]     = c2.number_input("Ops Coordinator Base $",      value=float(a["opscoord_base"]),     step=1000.0, format="%.0f")
        a["fieldsup_base"]     = c3.number_input("Field Supervisor Base $",     value=float(a["fieldsup_base"]),     step=1000.0, format="%.0f")
        a["regionalmgr_base"]  = c4.number_input("Regional Manager Base $",     value=float(a["regionalmgr_base"]),  step=1000.0, format="%.0f")

        c5, c6, c7, c8 = st.columns(4)
        a["mgmt_burden"]       = c5.number_input("Mgmt Burden % (decimal)",     value=float(a["mgmt_burden"]),       step=0.01,   format="%.2f")
        a["opscoord_span"]     = c6.number_input("Ops Coord Span (inspectors)", value=int(a["opscoord_span"]),       step=5,      format="%d")
        a["fieldsup_span"]     = c7.number_input("Field Sup Span (inspectors)", value=int(a["fieldsup_span"]),       step=5,      format="%d")
        a["regionalmgr_span"]  = c8.number_input("Reg Mgr Span (inspectors)",   value=int(a["regionalmgr_span"]),    step=5,      format="%d")

        c9, c10 = st.columns(3)[:2]
        a["gm_start_month"]    = c9.number_input("GM Start Month (model month #)", value=int(a["gm_start_month"]), step=1, format="%d",
                                                   help="1 = first month of model")
        a["gm_ramp_months"]    = c10.number_input("GM Ramp Months (0.5 FTE)",    value=int(a["gm_ramp_months"]),   step=1, format="%d")

    # ── AR & Collections ────────────────────────────────────────────
    with st.expander("AR & Collections", expanded=True):
        c1, c2 = st.columns(3)[:2]
        a["net_days"]          = c1.number_input("Net Days (collections lag)", value=int(a["net_days"]), step=5, format="%d",
                                                   help="Days after month-end statement date")
        a["start_date"]        = c2.date_input("Model Start Date", value=a["start_date"])

    # ── LOC ──────────────────────────────────────────────────────────
    with st.expander("Line of Credit (LOC)", expanded=True):
        c1, c2, c3 = st.columns(3)
        a["apr"]               = c1.number_input("Annual Interest Rate (APR, decimal)", value=float(a["apr"]), step=0.005, format="%.3f")
        a["max_loc"]           = c2.number_input("Max LOC Limit ($)",          value=float(a["max_loc"]),    step=50000.0,  format="%.0f")
        a["initial_cash"]      = c3.number_input("Initial Cash Balance ($)",   value=float(a["initial_cash"]), step=5000.0, format="%.0f")
        c4, c5 = st.columns(3)[:2]
        a["auto_paydown"]      = c4.checkbox("Auto Paydown ON", value=bool(a["auto_paydown"]))
        a["cash_buffer"]       = c5.number_input("Minimum Cash Buffer ($)",    value=float(a["cash_buffer"]), step=5000.0, format="%.0f",
                                                   disabled=not a["auto_paydown"])

    # ── Overhead ─────────────────────────────────────────────────────
    with st.expander("Other Overhead (monthly $)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        a["software_monthly"]   = c1.number_input("Software ($/mo)",    value=float(a["software_monthly"]),   step=100.0, format="%.0f")
        a["recruiting_monthly"] = c2.number_input("Recruiting ($/mo)",  value=float(a["recruiting_monthly"]), step=100.0, format="%.0f")
        a["insurance_monthly"]  = c3.number_input("Insurance ($/mo)",   value=float(a["insurance_monthly"]),  step=100.0, format="%.0f")
        a["travel_monthly"]     = c4.number_input("Travel ($/mo)",      value=float(a["travel_monthly"]),     step=100.0, format="%.0f")

        st.caption("Corporate Allocation")
        ca_mode = st.radio("Allocation Type", ["fixed", "pct_revenue"],
                           index=0 if a["corp_alloc_mode"] == "fixed" else 1,
                           horizontal=True)
        a["corp_alloc_mode"] = ca_mode
        if ca_mode == "fixed":
            a["corp_alloc_fixed"] = st.number_input("Corp Alloc Fixed ($/mo)", value=float(a["corp_alloc_fixed"]), step=500.0, format="%.0f")
        else:
            a["corp_alloc_pct"]   = st.number_input("Corp Alloc % of Revenue", value=float(a["corp_alloc_pct"]),   step=0.005, format="%.3f")

    st.session_state.assumptions = a
    st.success("Assumptions saved. Click **Run / Recalculate Model** in the sidebar.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Headcount Plan
# ════════════════════════════════════════════════════════════════════════════
elif page == "Headcount Plan":
    st.title("Headcount Plan — 120 Months")
    st.caption("Enter the number of **inspectors staffed per month**. Team leads and management are calculated automatically.")

    hc = st.session_state.headcount_plan

    # Build display DataFrame by year
    a_start = st.session_state.assumptions["start_date"]
    month_labels = []
    for i in range(120):
        yr  = a_start.year  + (a_start.month - 1 + i) // 12
        mo  = (a_start.month - 1 + i) % 12 + 1
        month_labels.append(f"Y{yr - a_start.year + 1} M{i % 12 + 1:02d}  ({yr}-{mo:02d})")

    hc_df = pd.DataFrame({"Month": month_labels, "Inspectors": hc})

    # Bulk fill options
    with st.expander("Bulk Fill / Paste Helper"):
        col1, col2, col3 = st.columns(3)
        fill_val  = col1.number_input("Fill value (inspectors)", 0, 10000, 25, step=5)
        fill_from = col2.number_input("From month #", 1, 120, 1)
        fill_to   = col3.number_input("To month #",   1, 120, 12)
        if st.button("Apply Fill"):
            for i in range(fill_from - 1, fill_to):
                hc[i] = fill_val
            st.session_state.headcount_plan = hc
            st.rerun()

    edited = st.data_editor(
        hc_df,
        column_config={
            "Month":      st.column_config.TextColumn("Period", disabled=True),
            "Inspectors": st.column_config.NumberColumn("Inspectors", min_value=0, max_value=10000, step=1),
        },
        use_container_width=True,
        height=600,
        num_rows="fixed",
    )

    st.session_state.headcount_plan = edited["Inspectors"].tolist()

    # Preview chart
    fig = px.bar(
        x=month_labels,
        y=st.session_state.headcount_plan,
        labels={"x": "Period", "y": "Inspectors"},
        title="Inspector Headcount Over 120 Months",
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Weekly Output
# ════════════════════════════════════════════════════════════════════════════
elif page == "Weekly Output":
    st.title("Weekly Output")
    if not results_ready():
        st.warning("Run the model first (sidebar button).")
        st.stop()

    weekly_df, _, _ = st.session_state.results

    # Warnings
    n_loc = weekly_df["warn_loc_maxed"].sum()
    n_neg = weekly_df["warn_neg_ebitda"].sum()
    n_mgmt = weekly_df["warn_mgmt_no_insp"].sum()
    if n_loc:   st.error(f"⚠ LOC exceeded max line in {n_loc} week(s).")
    if n_neg:   st.warning(f"⚠ Negative EBITDA in {n_neg} week(s).")
    if n_mgmt:  st.warning(f"⚠ Salaried management cost persists with 0 inspectors in {n_mgmt} week(s).")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Headcount & Revenue", "Labor & EBITDA", "AR Schedule", "Cash & LOC", "Reconciliation"]
    )

    with tab1:
        cols = ["week_start", "week_end", "inspectors", "team_leads",
                "n_opscoord", "n_fieldsup", "n_regionalmgr",
                "insp_st_hrs", "insp_ot_hrs",
                "insp_rev_st", "insp_rev_ot", "lead_rev_st", "lead_rev_ot", "revenue_wk"]
        st.dataframe(_select(weekly_df, cols), use_container_width=True, height=500)

    with tab2:
        cols = ["week_start", "inspectors", "team_leads",
                "insp_labor_st", "insp_labor_ot", "lead_labor_st", "lead_labor_ot",
                "hourly_labor", "salaried_wk", "overhead_wk",
                "revenue_wk", "ebitda_wk"]
        st.dataframe(_select(weekly_df, cols), use_container_width=True, height=500)

    with tab3:
        cols = ["week_start", "week_end", "is_month_end",
                "revenue_wk", "statement_amt", "collections",
                "ar_begin", "ar_end"]
        st.dataframe(_select(weekly_df, cols), use_container_width=True, height=500)

    with tab4:
        cols = ["week_start",
                "payroll_cash_out", "salaried_wk", "overhead_wk", "interest_paid",
                "collections",
                "cash_begin", "loc_draw", "loc_repay", "cash_end",
                "loc_begin", "loc_end",
                "warn_loc_maxed"]
        st.dataframe(_select(weekly_df, cols), use_container_width=True, height=500)

    with tab5:
        cols = ["week_start", "check_ar", "check_loc", "check_cash"]
        st.dataframe(_select(weekly_df, cols), use_container_width=True, height=500)
        max_err = weekly_df[["check_ar","check_loc","check_cash"]].max().max()
        if max_err < 0.01:
            st.success(f"All reconciliation checks pass (max error: ${max_err:.4f})")
        else:
            st.error(f"Reconciliation error detected — max error: ${max_err:.2f}")



# ════════════════════════════════════════════════════════════════════════════
# PAGE: Monthly Output
# ════════════════════════════════════════════════════════════════════════════
elif page == "Monthly Output":
    st.title("Monthly Output")
    if not results_ready():
        st.warning("Run the model first.")
        st.stop()

    _, mo, _ = st.session_state.results

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = _bar_chart(mo, "period",
                         ["hourly_labor", "salaried_cost", "overhead"],
                         ["Hourly Labor", "Salaried", "Overhead"],
                         "Monthly Cost Stack")
        fig.add_trace(go.Scatter(x=mo["period"], y=mo["revenue"], name="Revenue",
                                 mode="lines+markers", line=dict(color="green", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = _line_chart(mo, "period",
                           ["ebitda", "ebitda_after_interest"],
                           ["EBITDA (pre-int)", "EBITDA (after int)"],
                           "Monthly EBITDA")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = _line_chart(mo, "period", ["loc_end", "ar_end"],
                           ["LOC Balance", "AR Balance"], "LOC & AR Balances")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = _line_chart(mo, "period", ["cash_end"], ["Cash"], "Cash Balance")
        fig4.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig4, use_container_width=True)

    # Table
    display_cols = [
        "period", "inspectors_avg", "team_leads_avg",
        "revenue", "total_labor", "overhead",
        "ebitda", "ebitda_margin",
        "interest", "ebitda_after_interest", "ebitda_ai_margin",
        "collections", "ar_end", "loc_end", "cash_end", "peak_loc_to_date",
    ]
    st.dataframe(_select(mo, display_cols), use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Quarterly Output
# ════════════════════════════════════════════════════════════════════════════
elif page == "Quarterly Output":
    st.title("Quarterly Output")
    if not results_ready():
        st.warning("Run the model first.")
        st.stop()

    _, _, qdf = st.session_state.results

    c1, c2 = st.columns(2)
    with c1:
        fig = _bar_chart(qdf, "yr_q",
                         ["hourly_labor", "salaried_cost", "overhead"],
                         ["Hourly Labor", "Salaried", "Overhead"],
                         "Quarterly Cost Stack vs Revenue")
        fig.add_trace(go.Scatter(x=qdf["yr_q"], y=qdf["revenue"],
                                 name="Revenue", mode="lines+markers",
                                 line=dict(color="green", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = _line_chart(qdf, "yr_q",
                           ["ebitda", "ebitda_after_interest"],
                           ["EBITDA (pre-int)", "EBITDA (after int)"],
                           "Quarterly EBITDA")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    display_cols = [
        "yr_q", "revenue", "total_labor", "overhead",
        "ebitda", "ebitda_margin",
        "interest", "ebitda_after_interest", "ebitda_ai_margin",
        "ar_end", "loc_end", "cash_end", "peak_loc_to_date",
    ]
    st.dataframe(_select(qdf, display_cols), use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.title("Dashboard — Key Metrics")
    if not results_ready():
        st.warning("Run the model first.")
        st.stop()

    weekly_df, mo, _ = st.session_state.results
    a = st.session_state.assumptions

    # ── KPI Cards ───────────────────────────────────────────────────
    st.subheader("Summary KPIs")
    k1, k2, k3, k4 = st.columns(4)
    peak_loc = mo["loc_end"].max()
    peak_loc_mo = mo.loc[mo["loc_end"].idxmax(), "period"] if peak_loc > 0 else "—"
    k1.metric("Peak LOC Draw", fmt_dollar(peak_loc), help=f"Reached in {peak_loc_mo}")
    k2.metric("Total LOC Interest (120 mo)", fmt_dollar(mo["interest"].sum()))
    k3.metric("120-mo Revenue", fmt_dollar(mo["revenue"].sum()))
    k4.metric("120-mo EBITDA (after int.)", fmt_dollar(mo["ebitda_after_interest"].sum()))

    k5, k6, k7, k8 = st.columns(4)
    # Steady-state LOC: last 3 months average when headcount is constant
    last3_loc = mo.tail(3)["loc_end"].mean()
    k5.metric("Steady-State LOC (last 3 mo avg)", fmt_dollar(last3_loc))

    # When collections "catch up": first month where LOC starts declining
    loc_declining = mo[mo["loc_end"] < mo["loc_end"].shift(1)]
    if not loc_declining.empty:
        catchup = loc_declining.iloc[0]["period"]
    else:
        catchup = "Never"
    k6.metric("LOC First Decline (catch-up)", catchup)

    # DSCR-like: total EBITDA / total interest
    total_int = mo["interest"].sum()
    total_ebitda = mo["ebitda"].sum()
    dscr = total_ebitda / total_int if total_int > 0 else float("inf")
    k7.metric("EBITDA / Interest (120 mo)", f"{dscr:.1f}x" if total_int > 0 else "N/A")

    # Annualized (Year 1)
    yr1 = mo[mo["month_idx"] < 12]
    k8.metric("Year 1 EBITDA (after int.)", fmt_dollar(yr1["ebitda_after_interest"].sum()))

    st.divider()

    # ── Break-even ───────────────────────────────────────────────────
    st.subheader("Break-Even Analysis")
    be_col1, be_col2 = st.columns(2)
    with be_col1:
        be_nd = st.selectbox("Net Days for Break-Even Calc", [60, 90, 120, 150],
                             index=[60, 90, 120, 150].index(int(a["net_days"]))
                             if int(a["net_days"]) in [60, 90, 120, 150] else 0)
        if st.button("Calculate Break-Even Inspectors"):
            with st.spinner("Searching…"):
                be = find_breakeven_inspectors(a, be_nd)
            st.success(f"Min inspectors for positive EBITDA (after interest): **{be}** @ Net {be_nd}")

    # ── Charts ───────────────────────────────────────────────────────
    st.subheader("Cash & LOC Dynamics")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mo["period"], y=mo["loc_end"],
                             name="LOC Balance", fill="tozeroy",
                             line=dict(color="#e74c3c")))
    fig.add_trace(go.Scatter(x=mo["period"], y=mo["ar_end"],
                             name="AR Balance", line=dict(color="#3498db")))
    fig.add_trace(go.Scatter(x=mo["period"], y=mo["cash_end"],
                             name="Cash", line=dict(color="#2ecc71")))
    fig.add_hline(y=float(a["max_loc"]), line_dash="dot",
                  line_color="red", annotation_text="LOC Limit")
    fig.update_layout(height=400, title="LOC / AR / Cash — Monthly",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.line(mo, x="period", y=["ebitda_margin", "ebitda_ai_margin"],
                       title="EBITDA Margin (pre & post interest)",
                       labels={"value": "Margin", "variable": ""})
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = _bar_chart(mo, "period",
                          ["hourly_labor", "salaried_cost", "overhead"],
                          ["Hourly Labor", "Salaried", "Overhead"],
                          "Cost Components vs Revenue")
        fig3.add_trace(go.Scatter(x=mo["period"], y=mo["revenue"],
                                  name="Revenue", mode="lines",
                                  line=dict(color="green", width=2)))
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Headcount evolution ──────────────────────────────────────────
    st.subheader("Headcount Evolution")
    fig4 = _bar_chart(mo, "period",
                      ["inspectors_avg", "team_leads_avg", "n_opscoord", "n_fieldsup", "n_regionalmgr"],
                      ["Inspectors", "Team Leads", "Ops Coord", "Field Sup", "Reg Mgr"],
                      "Average Monthly Headcount by Role")
    st.plotly_chart(fig4, use_container_width=True)

    # ── Export ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export to Excel")
    if st.button("Generate Excel File"):
        with st.spinner("Building Excel…"):
            weekly_df, monthly_df, quarterly_df = st.session_state.results
            xlsx_bytes = build_excel(
                a,
                st.session_state.headcount_plan,
                weekly_df, monthly_df, quarterly_df,
            )
        st.download_button(
            label="Download Excel",
            data=xlsx_bytes,
            file_name="containment_division_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Sensitivity
# ════════════════════════════════════════════════════════════════════════════
elif page == "Sensitivity":
    st.title("Sensitivity Analysis")
    if not results_ready():
        st.warning("Run the model first.")
        st.stop()

    a  = st.session_state.assumptions
    hc = st.session_state.headcount_plan

    st.info("Each table re-runs the full model with one parameter varied while others stay at base-case values.")

    sens_tabs = st.tabs(["Net Days", "Bill Rate", "Burden %", "OT Hours"])

    # ── Net Days ────────────────────────────────────────────────────
    with sens_tabs[0]:
        st.subheader("Net Days vs Peak LOC & Annual Interest")
        nd_vals = [30, 45, 60, 75, 90, 105, 120, 150]
        with st.spinner("Running…"):
            nd_df = run_sensitivity(a, hc, "net_days", nd_vals)
        nd_display = nd_df[["value", "peak_loc", "annual_interest",
                             "ebitda_ai", "ebitda_ai_margin"]].rename(columns={
            "value": "Net Days",
            "peak_loc": "Peak LOC ($)",
            "annual_interest": "Total Interest ($)",
            "ebitda_ai": "EBITDA after Int ($)",
            "ebitda_ai_margin": "EBITDA Margin (AI)",
        })
        _fmt_sens(nd_display, ["Peak LOC ($)", "Total Interest ($)", "EBITDA after Int ($)"],
                  ["EBITDA Margin (AI)"])

        fig = px.line(nd_df, x="value", y="peak_loc",
                      title="Net Days vs Peak LOC",
                      labels={"value": "Net Days", "peak_loc": "Peak LOC ($)"})
        st.plotly_chart(fig, use_container_width=True)

    # ── Bill Rate ────────────────────────────────────────────────────
    with sens_tabs[1]:
        st.subheader("Bill Rate vs EBITDA Margin & Peak LOC")
        br_vals = [37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 42.0]
        with st.spinner("Running…"):
            br_df = run_sensitivity(a, hc, "st_bill_rate", br_vals)
        br_display = br_df[["value", "ebitda_margin", "ebitda_ai_margin",
                             "peak_loc", "ebitda_ai"]].rename(columns={
            "value": "Bill Rate ($/hr)",
            "ebitda_margin": "EBITDA Margin (pre-int)",
            "ebitda_ai_margin": "EBITDA Margin (AI)",
            "peak_loc": "Peak LOC ($)",
            "ebitda_ai": "EBITDA after Int ($)",
        })
        _fmt_sens(br_display, ["Peak LOC ($)", "EBITDA after Int ($)"],
                  ["EBITDA Margin (pre-int)", "EBITDA Margin (AI)"])
        fig = px.line(br_df, x="value", y=["ebitda_margin", "ebitda_ai_margin"],
                      title="Bill Rate vs EBITDA Margin",
                      labels={"value": "Bill Rate", "variable": ""})
        st.plotly_chart(fig, use_container_width=True)

    # ── Burden ────────────────────────────────────────────────────────
    with sens_tabs[2]:
        st.subheader("Burden % vs EBITDA & Peak LOC")
        burd_vals = [0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
        with st.spinner("Running…"):
            burd_df = run_sensitivity(a, hc, "burden", burd_vals)
        burd_df["value_pct"] = burd_df["value"] * 100
        burd_display = burd_df[["value_pct", "ebitda_ai", "ebitda_ai_margin", "peak_loc"]].rename(columns={
            "value_pct": "Burden (%)",
            "ebitda_ai": "EBITDA after Int ($)",
            "ebitda_ai_margin": "EBITDA Margin (AI)",
            "peak_loc": "Peak LOC ($)",
        })
        _fmt_sens(burd_display, ["EBITDA after Int ($)", "Peak LOC ($)"],
                  ["EBITDA Margin (AI)"])
        fig = px.line(burd_df, x="value_pct", y="ebitda_ai_margin",
                      title="Burden % vs EBITDA Margin (after interest)",
                      labels={"value_pct": "Burden (%)", "ebitda_ai_margin": "Margin"})
        st.plotly_chart(fig, use_container_width=True)

    # ── OT Hours ─────────────────────────────────────────────────────
    with sens_tabs[3]:
        st.subheader("OT Hours per Week vs EBITDA Margin")
        ot_vals = [0, 2, 4, 6, 8, 10, 12, 15, 20]
        with st.spinner("Running…"):
            ot_df = run_sensitivity(a, hc, "ot_hours", ot_vals)
        ot_display = ot_df[["value", "ebitda_margin", "ebitda_ai_margin",
                             "total_revenue", "ebitda_ai"]].rename(columns={
            "value": "OT Hrs/wk",
            "ebitda_margin": "EBITDA Margin (pre-int)",
            "ebitda_ai_margin": "EBITDA Margin (AI)",
            "total_revenue": "Total Revenue ($)",
            "ebitda_ai": "EBITDA after Int ($)",
        })
        _fmt_sens(ot_display, ["Total Revenue ($)", "EBITDA after Int ($)"],
                  ["EBITDA Margin (pre-int)", "EBITDA Margin (AI)"])
        fig = px.line(ot_df, x="value", y=["ebitda_margin", "ebitda_ai_margin"],
                      title="OT Hours vs EBITDA Margin",
                      labels={"value": "OT Hrs/wk", "variable": ""})
        st.plotly_chart(fig, use_container_width=True)

    # ── Export sensitivity ───────────────────────────────────────────
    st.divider()
    if st.button("Export All Sensitivity Tables to Excel"):
        weekly_df, monthly_df, quarterly_df = st.session_state.results
        sens_tables = {
            "Sens_NetDays":  nd_df,
            "Sens_BillRate": br_df,
            "Sens_Burden":   burd_df,
            "Sens_OTHours":  ot_df,
        }
        xlsx = build_excel(a, hc, weekly_df, monthly_df, quarterly_df, sens_tables)
        st.download_button("Download Excel (with Sensitivity)",
                           data=xlsx,
                           file_name="containment_division_sensitivity.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


