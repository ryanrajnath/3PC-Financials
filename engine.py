"""
engine.py — Weekly financial engine for Containment Division Calculator
"""
import calendar
from math import ceil
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    year = d.year + m // 12
    month = m % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _last_day(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


# ---------------------------------------------------------------------------
# Default assumptions
# ---------------------------------------------------------------------------

def default_assumptions() -> dict:
    return {
        "start_date": date(date.today().year + 1, 1, 1),
        # Billing
        "st_bill_rate": 39.0,
        "ot_bill_premium": 1.5,
        "st_hours": 40,
        "ot_hours": 10,
        # Inspector pay
        "inspector_wage": 20.0,
        "ot_pay_multiplier": 1.5,
        "burden": 0.30,
        # Team leads
        "team_lead_ratio": 12,
        "lead_wage": 25.0,
        "lead_st_hours": 40,
        "lead_ot_hours": 10,
        # Management (annual fully-loaded or base)
        "gm_loaded_annual": 117_000.0,
        "opscoord_base": 65_000.0,
        "fieldsup_base": 70_000.0,
        "regionalmgr_base": 110_000.0,
        "mgmt_burden": 0.25,
        # Span of control (inspectors per role)
        "opscoord_span": 75,
        "fieldsup_span": 25,
        "regionalmgr_span": 175,
        # GM ramp
        "gm_start_month": 1,   # 1-based model month when GM activates
        "gm_ramp_months": 0,   # months at 0.5 FTE before going to 1.0
        # AR / collections
        "billing_frequency": "monthly",   # "monthly" | "weekly"
        "net_days": 60,
        # LOC
        "apr": 0.085,
        "max_loc": 1_000_000.0,
        "auto_paydown": True,
        "cash_buffer": 25_000.0,
        "initial_cash": 0.0,
        # Overhead (monthly $)
        "software_monthly": 500.0,
        "recruiting_monthly": 1_000.0,
        "insurance_monthly": 1_500.0,
        "travel_monthly": 500.0,
        # Per-inspector overhead scaling (monthly $ per active inspector)
        "software_per_inspector":    0.0,   # $ per inspector/mo — workforce mgmt, scheduling, QA tools
        "insurance_per_inspector":   0.0,   # $ per inspector/mo — GL/umbrella above burden (WC is in burden%)
        "travel_per_inspector":      0.0,   # $ per inspector/mo — site visits, supervisor travel
        "recruiting_per_inspector":  0.0,   # $ per inspector/mo — ongoing job board, agency fees at scale
        # Corporate allocation
        "corp_alloc_mode": "fixed",   # "fixed" | "pct_revenue"
        "corp_alloc_fixed": 0.0,
        "corp_alloc_pct": 0.0,
        # Management turnover (industry data — third-party containment/inspection staffing)
        # BLS JOLTS + SHRM benchmarks: office/scheduling ops ~35%, field supervision ~25%,
        # regional management ~18% annual voluntary + involuntary combined.
        "opscoord_turnover":    0.35,   # annual rate — scheduling/ops roles turn frequently
        "fieldsup_turnover":    0.25,   # field supervisors — moderate, better pay = better retention
        "regionalmgr_turnover": 0.18,   # regional managers — senior, harder to replace = lower rate
        # Replacement cost per hire: recruiting fee / job board + background check + 4–8 wks ramp
        "opscoord_replace_cost":    8_000.0,   # ~12% of $65K base
        "fieldsup_replace_cost":   12_000.0,   # ~17% of $70K base
        "regionalmgr_replace_cost":25_000.0,   # ~23% of $110K base (may use recruiter)
        # Inspector onboarding cost (amortized into weekly labor cost per the formula:
        # Total Cost = Wage × (1 + Burden%) + Onboarding_Cost_per_week)
        # One-time cost per new inspector hire: background check, drug screen, PPE, orientation
        "inspector_onboarding_cost":  500.0,   # $ per hire — background check ~$150, drug screen ~$80, orientation ~$270
        "inspector_avg_tenure_weeks": 26,       # avg weeks before inspector turns over (~6 months; containment industry high-turnover)
        # Inspector utilization rate — fraction of scheduled hours actually billed to client
        # (accounts for travel days, site startup, between-project gaps)
        "inspector_utilization":  1.0,          # 1.0 = 100% billed; 0.90 = 10% unbillable time
        # Team lead bill rate premium (multiplier on inspector bill rate)
        # 1.0 = same rate as inspectors; 1.1 = 10% premium for supervisory labor
        "lead_bill_premium": 1.0,
        # OT billing methodology
        "ot_bill_mode": "passthrough",   # "markup" = st_bill * ot_prem;  "passthrough" = st_bill + wage_ot_increment
        # Bad debt
        "bad_debt_pct": 0.01,            # 1% of revenue written off — industry standard for B2B industrial staffing
        # Inspector turnover (separate from onboarding amortization)
        "inspector_turnover_rate": 1.0,  # 100% annual — containment inspectors churn at 80-150%/yr
        # Management wind-down lag
        "mgmt_winddown_weeks": 8,        # Weeks after inspectors go to zero before salaried mgmt headcount reduces
        # Borrowing base (AR-based LOC sizing)
        "use_borrowing_base": False,     # If True, LOC draw limited to ar_advance_rate × eligible AR
        "ar_advance_rate": 0.85,         # 85% advance rate on eligible AR (standard for staffing ABL facilities)
        # Tax rates — for after-tax net income reporting (provision only, not cash tax modeling)
        "sc_state_tax_rate":  0.059,  # South Carolina 2026 corporate/pass-through rate
        "federal_tax_rate":   0.21,   # Federal corporate rate
    }


def default_headcount() -> list:
    """25 inspectors for 12 months, then 0."""
    return [25] * 12 + [0] * 108


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_model(assumptions: dict, headcount_plan: list):
    """
    Run the 120-month weekly financial model.

    Returns
    -------
    weekly_df   : pd.DataFrame  — one row per week
    monthly_df  : pd.DataFrame  — one row per model month
    quarterly_df: pd.DataFrame  — one row per quarter
    """
    a = assumptions
    hc = list(headcount_plan)
    # Ensure exactly 120 entries
    hc = (hc + [0] * 120)[:120]

    # --- Unpack -------------------------------------------------------
    start_raw: date = a["start_date"]
    st_bill   = float(a["st_bill_rate"])
    ot_prem   = float(a["ot_bill_premium"])
    st_hrs    = int(a["st_hours"])
    ot_hrs    = int(a["ot_hours"])

    insp_wage = float(a["inspector_wage"])
    ot_mult   = float(a["ot_pay_multiplier"])
    burden    = float(a["burden"])

    tl_ratio       = int(a["team_lead_ratio"])
    lead_wage      = float(a["lead_wage"])
    lead_st        = int(a["lead_st_hours"])
    lead_ot        = int(a["lead_ot_hours"])
    lead_bill_prem = float(a.get("lead_bill_premium", 1.0))

    # Inspector utilization & onboarding cost (fired at point of hire, not amortized)
    utilization  = float(a.get("inspector_utilization", 1.0))
    onboard_cost = float(a.get("inspector_onboarding_cost", 500.0))
    # avg_tenure_wks kept for UI compatibility but no longer used in cost formula

    gm_loaded   = float(a["gm_loaded_annual"])
    ops_base    = float(a["opscoord_base"])
    fsup_base   = float(a["fieldsup_base"])
    rmgr_base   = float(a["regionalmgr_base"])
    mgmt_burd   = float(a["mgmt_burden"])

    ops_loaded  = ops_base  * (1 + mgmt_burd)
    fsup_loaded = fsup_base * (1 + mgmt_burd)
    rmgr_loaded = rmgr_base * (1 + mgmt_burd)

    ops_span  = int(a["opscoord_span"])
    fsup_span = int(a["fieldsup_span"])
    rmgr_span = int(a["regionalmgr_span"])

    # Management turnover cost
    opscoord_to    = float(a.get("opscoord_turnover",    0.35))
    fieldsup_to    = float(a.get("fieldsup_turnover",    0.25))
    regionalmgr_to = float(a.get("regionalmgr_turnover", 0.18))
    opscoord_rc    = float(a.get("opscoord_replace_cost",    8_000.0))
    fieldsup_rc    = float(a.get("fieldsup_replace_cost",   12_000.0))
    regionalmgr_rc = float(a.get("regionalmgr_replace_cost",25_000.0))

    gm_start  = int(a.get("gm_start_month", 1))
    gm_ramp   = int(a.get("gm_ramp_months", 0))

    billing_freq = a.get("billing_frequency", "monthly")   # "monthly" | "weekly"
    net_days     = int(a["net_days"])
    apr          = float(a["apr"])
    max_loc      = float(a["max_loc"])
    auto_pd      = bool(a["auto_paydown"])
    cash_buf     = float(a["cash_buffer"])
    init_cash    = float(a.get("initial_cash", 0.0))

    sw_mo   = float(a.get("software_monthly",   0.0))
    rec_mo  = float(a.get("recruiting_monthly", 0.0))
    ins_mo  = float(a.get("insurance_monthly",  0.0))
    trv_mo  = float(a.get("travel_monthly",     0.0))
    ca_mode = a.get("corp_alloc_mode", "fixed")
    ca_fix  = float(a.get("corp_alloc_fixed", 0.0))
    ca_pct  = float(a.get("corp_alloc_pct",   0.0))

    # --- Generate weeks -----------------------------------------------
    # Align start to Monday
    start = start_raw - timedelta(days=start_raw.weekday())
    model_end = _add_months(start_raw, 120)

    weeks_start = []
    d = start
    while d < model_end:
        weeks_start.append(d)
        d += timedelta(days=7)

    # --- Build base DataFrame -----------------------------------------
    rows = []
    for i, ws in enumerate(weeks_start):
        we = ws + timedelta(days=6)
        yr, mo = ws.year, ws.month

        # Model month index (0-based from start_raw)
        m_idx = (yr - start_raw.year) * 12 + (mo - start_raw.month)
        if m_idx < 0 or m_idx >= 120:
            continue

        insp = hc[m_idx]
        mo_end = _last_day(ws)

        # Last week of month: next Monday crosses into next month
        next_mon = ws + timedelta(days=7)
        is_mo_end = (next_mon.month != mo or next_mon.year != yr)

        # GM FTE
        m1 = m_idx + 1  # 1-based
        if m1 < gm_start:
            gm_fte = 0.0
        else:
            elapsed = m1 - gm_start
            gm_fte = 0.5 if (gm_ramp > 0 and elapsed < gm_ramp) else 1.0

        rows.append({
            "week_num":        i,
            "week_start":      ws,
            "week_end":        we,
            "year":            yr,
            "month":           mo,
            "month_idx":       m_idx,
            "month_end_date":  mo_end,
            "is_month_end":    is_mo_end,
            "inspectors":      insp,
            "gm_fte":          gm_fte,
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("No weeks generated — check start_date and headcount_plan.")

    # --- Headcount derived --------------------------------------------
    df["team_leads"] = df["inspectors"].apply(
        lambda x: ceil(x / tl_ratio) if x > 0 else 0
    )

    # Inspector turnover — compute weekly new hires
    # New hires = incremental headcount growth + replacement of churned workers
    inspector_to_rate = float(a.get("inspector_turnover_rate", 1.0))
    insp_series = df["inspectors"].values
    insp_prev   = np.concatenate([[0], insp_series[:-1]])
    insp_delta  = np.maximum(0, insp_series - insp_prev)   # only ramp-up, not decreases
    insp_churn  = insp_series * (inspector_to_rate / 52.0)  # weekly churn at annual rate
    df["inspector_new_hires"] = insp_delta + insp_churn

    # Management headcount: carries forward through short gaps, but winds down after mgmt_winddown_weeks
    winddown_wks = int(a.get("mgmt_winddown_weeks", 8))
    _mgmt_basis = []
    _last_active = 0
    _inactive_wks = 0
    for _insp in df["inspectors"]:
        if _insp > 0:
            _last_active = int(_insp)
            _inactive_wks = 0
        else:
            _inactive_wks += 1
        if _inactive_wks > winddown_wks:
            _mgmt_basis.append(0)
        else:
            _mgmt_basis.append(_last_active)
    df["mgmt_insp_basis"] = _mgmt_basis

    df["n_opscoord"]    = df["mgmt_insp_basis"].apply(lambda x: ceil(x / ops_span)  if x > 0 else 0)
    df["n_fieldsup"]    = df["mgmt_insp_basis"].apply(lambda x: ceil(x / fsup_span) if x > 0 else 0)
    df["n_regionalmgr"] = df["mgmt_insp_basis"].apply(lambda x: ceil(x / rmgr_span) if x > 0 else 0)

    # --- Revenue (weekly, accrual) ------------------------------------
    # Scheduled hours × utilization rate = billable hours
    df["insp_st_hrs"] = df["inspectors"] * st_hrs * utilization
    df["insp_ot_hrs"] = df["inspectors"] * ot_hrs * utilization
    df["lead_st_hrs"] = df["team_leads"] * lead_st * utilization
    df["lead_ot_hrs"] = df["team_leads"] * lead_ot * utilization

    # OT bill rate — determined by mode after ot_mult and burden are known
    ot_bill_mode = a.get("ot_bill_mode", "passthrough")
    if ot_bill_mode == "passthrough":
        # Industry standard: pass through the wage increment at cost, keep ST margin intact
        # ot_bill = st_bill + (inspector_wage × OT_premium_above_1 × (1 + burden))
        ot_bill = st_bill + (insp_wage * (ot_mult - 1.0) * (1.0 + burden))
        lead_ot_bill_rate = (st_bill * lead_bill_prem) + (lead_wage * (ot_mult - 1.0) * (1.0 + burden))
    else:
        # Markup mode: multiply entire bill rate by premium
        ot_bill = st_bill * ot_prem
        lead_ot_bill_rate = (st_bill * lead_bill_prem) * ot_prem

    # Lead bill rate can carry a premium over inspector bill rate
    lead_bill_st = st_bill * lead_bill_prem
    lead_bill_ot = lead_ot_bill_rate  # already computed above per mode

    df["insp_rev_st"] = df["insp_st_hrs"] * st_bill
    df["insp_rev_ot"] = df["insp_ot_hrs"] * ot_bill
    df["lead_rev_st"] = df["lead_st_hrs"] * lead_bill_st
    df["lead_rev_ot"] = df["lead_ot_hrs"] * lead_bill_ot
    df["revenue_wk"]  = df["insp_rev_st"] + df["insp_rev_ot"] + df["lead_rev_st"] + df["lead_rev_ot"]

    # --- Labor cost (weekly, accrual) ---------------------------------
    # Formula: Total Cost = Wage × hours × (1 + Burden%) + Onboarding_Cost_per_week
    # Inspector hours paid on scheduled hours (not just billable), so no utilization adjustment on cost
    df["insp_labor_st"]   = df["inspectors"] * st_hrs * insp_wage * (1 + burden)
    df["insp_labor_ot"]   = df["inspectors"] * ot_hrs * insp_wage * ot_mult * (1 + burden)
    df["lead_labor_st"]   = df["team_leads"] * lead_st * lead_wage * (1 + burden)
    df["lead_labor_ot"]   = df["team_leads"] * lead_ot * lead_wage * ot_mult * (1 + burden)
    # Onboarding cost fires at the point of hire, not amortized
    df["onboarding_cost_wk"] = df["inspector_new_hires"] * onboard_cost
    df["hourly_labor"]  = (df["insp_labor_st"] + df["insp_labor_ot"] +
                           df["lead_labor_st"] + df["lead_labor_ot"] +
                           df["onboarding_cost_wk"])

    # --- Salaried cost (weekly) ---------------------------------------
    df["gm_cost_wk"]    = df["gm_fte"]        * gm_loaded   / 52
    df["ops_cost_wk"]   = df["n_opscoord"]    * ops_loaded  / 52
    df["fsup_cost_wk"]  = df["n_fieldsup"]    * fsup_loaded / 52
    df["rmgr_cost_wk"]  = df["n_regionalmgr"] * rmgr_loaded / 52
    df["salaried_wk"]   = df["gm_cost_wk"] + df["ops_cost_wk"] + df["fsup_cost_wk"] + df["rmgr_cost_wk"]

    # --- Overhead (weekly) -------------------------------------------
    wks_in_mo = df.groupby("month_idx").size().to_dict()
    df["wks_in_month"] = df["month_idx"].map(wks_in_mo)

    # Per-inspector overhead scaling (monthly $ per active inspector)
    sw_pi  = float(a.get("software_per_inspector",   0.0))
    rec_pi = float(a.get("recruiting_per_inspector", 0.0))
    ins_pi = float(a.get("insurance_per_inspector",  0.0))
    trv_pi = float(a.get("travel_per_inspector",     0.0))

    fixed_mo = sw_mo + rec_mo + ins_mo + trv_mo
    if ca_mode == "fixed":
        fixed_mo += ca_fix
    df["fixed_ovhd_wk"] = fixed_mo / df["wks_in_month"]

    # Per-inspector component: (total monthly per-inspector rate × inspectors) ÷ weeks in month
    pi_mo_total = sw_pi + rec_pi + ins_pi + trv_pi
    df["pi_ovhd_wk"] = (pi_mo_total * df["inspectors"]) / df["wks_in_month"]

    df["corp_alloc_wk"] = df["revenue_wk"] * ca_pct if ca_mode == "pct_revenue" else 0.0

    # Weekly turnover cost: (active headcount × annual turnover rate × replacement cost) / 52 weeks
    df["turnover_cost_wk"] = (
        df["n_opscoord"]     * opscoord_to    * opscoord_rc    / 52 +
        df["n_fieldsup"]     * fieldsup_to    * fieldsup_rc    / 52 +
        df["n_regionalmgr"]  * regionalmgr_to * regionalmgr_rc / 52
    )
    df["overhead_wk"]   = df["fixed_ovhd_wk"] + df["pi_ovhd_wk"] + df["corp_alloc_wk"] + df["turnover_cost_wk"]

    # --- EBITDA (accrual, pre-interest) ------------------------------
    df["ebitda_wk"] = (df["revenue_wk"]
                       - df["hourly_labor"]
                       - df["salaried_wk"]
                       - df["overhead_wk"])

    # --- Statement invoices & collections ----------------------------
    df["statement_amt"] = 0.0
    df["collections"]   = 0.0

    if billing_freq == "weekly":
        # Each week's revenue is invoiced at week-end; collected net_days after week_end.
        for i, row in df.iterrows():
            rev = float(row["revenue_wk"])
            if rev == 0:
                continue
            df.at[i, "statement_amt"] = rev
            collect_date = row["week_end"] + timedelta(days=net_days)
            mask_coll = (df["week_start"] <= collect_date) & (df["week_end"] >= collect_date)
            if mask_coll.any():
                df.loc[mask_coll, "collections"] += rev
    else:
        # Monthly: batch all revenue into a month-end invoice; collected net_days after month-end.
        mo_rev      = df.groupby("month_idx")["revenue_wk"].sum()
        mo_end_date = df.groupby("month_idx")["month_end_date"].first()

        for m_idx, rev in mo_rev.items():
            if rev == 0:
                continue
            # Statement placed in the last week of the month
            mask_stmt = (df["month_idx"] == m_idx) & df["is_month_end"]
            df.loc[mask_stmt, "statement_amt"] = rev

            # Collection: lump sum net_days after month-end date
            stmt_date    = mo_end_date[m_idx]
            collect_date = stmt_date + timedelta(days=net_days)
            mask_coll = (df["week_start"] <= collect_date) & (df["week_end"] >= collect_date)
            if mask_coll.any():
                df.loc[mask_coll, "collections"] += rev

    # Bad debt: reduce collections by write-off rate (revenue accrued but never collected)
    bad_debt_pct = float(a.get("bad_debt_pct", 0.0))
    if bad_debt_pct > 0:
        df["bad_debt_wk"] = df["collections"] * bad_debt_pct
        df["collections"]  = df["collections"] * (1.0 - bad_debt_pct)
    else:
        df["bad_debt_wk"] = 0.0

    # --- AR roll-forward (must precede cash/LOC loop for borrowing base) ----
    ar_beg_arr = np.zeros(n)
    ar_end_arr = np.zeros(n)
    ar = 0.0
    for i in range(n):
        ar_beg_arr[i] = ar
        ar += df.iloc[i]["statement_amt"] - df.iloc[i]["collections"]
        ar_end_arr[i] = ar

    # --- Sequential cash / LOC model ---------------------------------
    cash_beg  = np.zeros(n)
    cash_end  = np.zeros(n)
    loc_beg   = np.zeros(n)
    loc_end   = np.zeros(n)
    loc_draw  = np.zeros(n)
    loc_repay = np.zeros(n)
    interest  = np.zeros(n)
    pay_lag   = np.zeros(n)   # hourly payroll cash-out (1-week lag)

    use_bb       = bool(a.get("use_borrowing_base", False))
    ar_adv_rate  = float(a.get("ar_advance_rate", 0.85))

    # Month -> list of row indices (for avg LOC calculation)
    mo_rows: dict[int, list] = {}
    for i, row in df.iterrows():
        mo_rows.setdefault(row["month_idx"], []).append(i)

    interest_done: set = set()
    cash = init_cash
    loc  = 0.0

    for i in range(n):
        row = df.iloc[i]

        cash_beg[i] = cash
        loc_beg[i]  = loc

        # Cash in: collections this week
        c_in = float(row["collections"])

        # Cash out: hourly payroll (1-week lag) + salaried (current) + overhead
        hp = float(df.iloc[i - 1]["hourly_labor"]) if i > 0 else 0.0
        pay_lag[i] = hp
        sal  = float(row["salaried_wk"])
        ovhd = float(row["overhead_wk"])
        c_out = hp + sal + ovhd

        tentative = cash + c_in - c_out

        # Interest: last week of month only
        intr = 0.0
        if row["is_month_end"]:
            m = row["month_idx"]
            if m not in interest_done:
                prior = [loc_end[j] for j in mo_rows[m] if j < i]
                all_locs = prior + [loc]   # current beginning LOC
                avg_loc  = float(np.mean(all_locs)) if all_locs else 0.0
                intr     = (apr / 12) * avg_loc
                tentative -= intr
                interest[i] = intr
                interest_done.add(m)

        # LOC draw / repay
        target = cash_buf if auto_pd else 0.0
        draw = repay = 0.0

        if tentative < target:
            # Borrowing base: limit draw to advance rate × prior week's eligible AR
            if use_bb and i > 0:
                eligible_ar = ar_end_arr[i - 1]   # prior week closing AR
                borrowing_base_i = eligible_ar * ar_adv_rate
                eff_max = min(borrowing_base_i, max_loc)
            else:
                eff_max = max_loc
            room  = max(0.0, eff_max - loc)
            draw  = min(target - tentative, room)
            loc   += draw
            tentative += draw

        if auto_pd and tentative > cash_buf and loc > 0:
            repay  = min(loc, tentative - cash_buf)
            loc   -= repay
            tentative -= repay

        cash = tentative

        cash_end[i]  = cash
        loc_end[i]   = loc
        loc_draw[i]  = draw
        loc_repay[i] = repay

    df["payroll_cash_out"] = pay_lag
    df["interest_paid"]    = interest
    df["loc_draw"]         = loc_draw
    df["loc_repay"]        = loc_repay
    df["cash_begin"]       = cash_beg
    df["cash_end"]         = cash_end
    df["loc_begin"]        = loc_beg
    df["loc_end"]          = loc_end

    df["ar_begin"] = ar_beg_arr
    df["ar_end"]   = ar_end_arr

    # --- Borrowing base tracking columns ----------------------------
    if use_bb:
        df["borrowing_base"] = np.concatenate([[0], ar_end_arr[:-1]]) * ar_adv_rate
        df["borrowing_base"] = df["borrowing_base"].clip(upper=max_loc)
    else:
        df["borrowing_base"] = max_loc
    df["loc_headroom"] = df["borrowing_base"] - df["loc_end"]

    # --- Validation flags --------------------------------------------
    df["warn_loc_maxed"]   = df["loc_end"] > max_loc + 0.01
    df["warn_neg_ebitda"]  = df["ebitda_wk"] < -0.01
    df["warn_mgmt_no_insp"]= (df["salaried_wk"] > 0) & (df["inspectors"] == 0)

    # --- Reconciliation columns --------------------------------------
    df["check_ar"]  = (df["ar_begin"] + df["statement_amt"] - df["collections"]
                       - df["ar_end"]).abs()
    df["check_loc"] = (df["loc_begin"] + df["loc_draw"] - df["loc_repay"]
                       - df["loc_end"]).abs()
    df["check_cash"]= (df["cash_begin"] + df["collections"] - df["payroll_cash_out"]
                       - df["salaried_wk"] - df["overhead_wk"] - df["interest_paid"]
                       + df["loc_draw"] - df["loc_repay"]
                       - df["cash_end"]).abs()

    # --- Build monthly & quarterly -----------------------------------
    monthly_df   = _build_monthly(df, assumptions)
    quarterly_df = _build_quarterly(monthly_df)

    return df, monthly_df, quarterly_df


# ---------------------------------------------------------------------------
# Monthly rollup
# ---------------------------------------------------------------------------

def _build_monthly(df: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    agg = (df.groupby("month_idx")
             .agg(
                 year            =("year",             "first"),
                 month           =("month",            "first"),
                 month_end_date  =("month_end_date",   "first"),
                 inspectors_avg  =("inspectors",       "mean"),
                 team_leads_avg  =("team_leads",       "mean"),
                 n_opscoord      =("n_opscoord",       "mean"),
                 n_fieldsup      =("n_fieldsup",       "mean"),
                 n_regionalmgr   =("n_regionalmgr",    "mean"),
                 revenue         =("revenue_wk",       "sum"),
                 insp_rev_st     =("insp_rev_st",      "sum"),
                 insp_rev_ot     =("insp_rev_ot",      "sum"),
                 lead_rev_st     =("lead_rev_st",      "sum"),
                 lead_rev_ot     =("lead_rev_ot",      "sum"),
                 hourly_labor    =("hourly_labor",      "sum"),
                 salaried_cost   =("salaried_wk",      "sum"),
                 overhead        =("overhead_wk",      "sum"),
                 turnover_cost   =("turnover_cost_wk", "sum"),
                 ebitda          =("ebitda_wk",        "sum"),
                 statement_amt   =("statement_amt",    "sum"),
                 collections     =("collections",      "sum"),
                 interest        =("interest_paid",    "sum"),
                 loc_draw        =("loc_draw",         "sum"),
                 loc_repay       =("loc_repay",        "sum"),
                 ar_end              =("ar_end",               "last"),
                 loc_end             =("loc_end",              "last"),
                 cash_end            =("cash_end",             "last"),
                 inspector_new_hires =("inspector_new_hires",  "sum"),
                 bad_debt_cash       =("bad_debt_wk",          "sum"),
             )
             .reset_index())

    agg["total_labor"]          = agg["hourly_labor"] + agg["salaried_cost"]
    agg["ebitda_after_interest"]= agg["ebitda"] - agg["interest"]
    agg["ebitda_margin"]        = np.where(agg["revenue"] > 0,
                                           agg["ebitda"] / agg["revenue"], 0.0)
    agg["ebitda_ai_margin"]     = np.where(agg["revenue"] > 0,
                                           agg["ebitda_after_interest"] / agg["revenue"], 0.0)
    # Tax provision (applied only on positive pre-tax income; losses = $0 tax)
    sc_rate  = float(assumptions.get("sc_state_tax_rate",  0.059))
    fed_rate = float(assumptions.get("federal_tax_rate",   0.21))
    pre_tax  = agg["ebitda_after_interest"].clip(lower=0)
    agg["sc_tax"]               = pre_tax * sc_rate
    agg["federal_tax"]          = pre_tax * fed_rate
    agg["total_tax"]            = agg["sc_tax"] + agg["federal_tax"]
    agg["net_income_after_tax"] = agg["ebitda_after_interest"] - agg["total_tax"]
    agg["net_margin_after_tax"] = np.where(agg["revenue"] > 0,
                                           agg["net_income_after_tax"] / agg["revenue"], 0.0)
    agg["peak_loc_to_date"]     = agg["loc_end"].cummax()

    # DSO: accounts receivable days — AR / (monthly revenue / 30)
    agg["dso"] = np.where(
        agg["revenue"] > 0,
        agg["ar_end"] / (agg["revenue"] / 30.0),
        0.0
    )

    # LOC utilization %
    _max_loc_val = float(assumptions.get("max_loc", 1_000_000))
    agg["loc_utilization"] = agg["loc_end"] / _max_loc_val if _max_loc_val > 0 else 0.0

    # FCCR: simplified as EBITDA / interest (interest coverage)
    agg["fccr"] = np.where(
        agg["interest"] > 0,
        agg["ebitda"] / agg["interest"],
        np.where(agg["ebitda"] > 0, 99.0, 0.0)   # 99 = unconstrained; 0 = unprofitable
    )

    # Bad debt expense (monthly)
    _bad_debt_pct = float(assumptions.get("bad_debt_pct", 0.0))
    agg["bad_debt_expense"] = agg["revenue"] * _bad_debt_pct

    agg["period"]               = agg["month_idx"].apply(lambda i: f"M{int(i)+1}")
    return agg


# ---------------------------------------------------------------------------
# Quarterly rollup
# ---------------------------------------------------------------------------

def _build_quarterly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    m = monthly_df.copy()
    m["quarter_idx"] = m["month_idx"] // 3
    m["yr_q"] = m.apply(
        lambda r: f"{r['year']} Q{(r['month'] - 1) // 3 + 1}", axis=1
    )

    qdf = (m.groupby("quarter_idx")
             .agg(
                 yr_q                  =("yr_q",                  "first"),
                 revenue               =("revenue",               "sum"),
                 hourly_labor          =("hourly_labor",           "sum"),
                 salaried_cost         =("salaried_cost",          "sum"),
                 overhead              =("overhead",               "sum"),
                 ebitda                =("ebitda",                 "sum"),
                 interest              =("interest",               "sum"),
                 ebitda_after_interest =("ebitda_after_interest",  "sum"),
                 collections           =("collections",            "sum"),
                 loc_draw              =("loc_draw",               "sum"),
                 loc_repay             =("loc_repay",              "sum"),
                 ar_end                =("ar_end",                 "last"),
                 loc_end               =("loc_end",                "last"),
                 cash_end              =("cash_end",               "last"),
             )
             .reset_index())

    qdf["total_labor"]      = qdf["hourly_labor"] + qdf["salaried_cost"]
    qdf["ebitda_margin"]    = np.where(qdf["revenue"] > 0,
                                       qdf["ebitda"] / qdf["revenue"], 0.0)
    qdf["ebitda_ai_margin"] = np.where(qdf["revenue"] > 0,
                                       qdf["ebitda_after_interest"] / qdf["revenue"], 0.0)
    qdf["peak_loc_to_date"] = qdf["loc_end"].cummax()
    return qdf


# ---------------------------------------------------------------------------
# Sensitivity helper
# ---------------------------------------------------------------------------

def run_sensitivity(base_assumptions: dict, headcount_plan: list,
                    param: str, values: list) -> pd.DataFrame:
    """
    Run the model once per value in `values` for the given `param`.
    Returns a DataFrame with one row per value and summary metrics.
    """
    rows = []
    for v in values:
        a = base_assumptions.copy()
        a[param] = v
        try:
            _, mo, _ = run_model(a, headcount_plan)
            peak_loc    = mo["loc_end"].max()
            total_int   = mo["interest"].sum()
            tot_rev     = mo["revenue"].sum()
            tot_ebitda  = mo["ebitda"].sum()
            tot_ebitda_ai = mo["ebitda_after_interest"].sum()
            margin      = tot_ebitda / tot_rev if tot_rev > 0 else 0.0
            margin_ai   = tot_ebitda_ai / tot_rev if tot_rev > 0 else 0.0
        except Exception:
            peak_loc = total_int = tot_rev = tot_ebitda = tot_ebitda_ai = margin = margin_ai = float("nan")

        rows.append({
            "value":             v,
            "peak_loc":          peak_loc,
            "annual_interest":   total_int,
            "total_revenue":     tot_rev,
            "total_ebitda":      tot_ebitda,
            "ebitda_ai":         tot_ebitda_ai,
            "ebitda_margin":     margin,
            "ebitda_ai_margin":  margin_ai,
        })
    return pd.DataFrame(rows)


def find_breakeven_inspectors(base_assumptions: dict, net_days: int,
                               lo: int = 1, hi: int = 300) -> int:
    """
    Binary search: minimum constant inspectors (12 months) for EBITDA after interest > 0.
    """
    a = base_assumptions.copy()
    a["net_days"] = net_days

    def is_profitable(n_insp):
        hc = [n_insp] * 12 + [0] * 108
        try:
            _, mo, _ = run_model(a, hc)
            return mo["ebitda_after_interest"].sum() > 0
        except Exception:
            return False

    if not is_profitable(hi):
        return hi  # can't break even at upper bound
    if is_profitable(lo):
        return lo

    while lo < hi - 1:
        mid = (lo + hi) // 2
        if is_profitable(mid):
            hi = mid
        else:
            lo = mid
    return hi
