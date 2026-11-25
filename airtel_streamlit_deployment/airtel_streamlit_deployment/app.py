"""
Airtel Telecom Analytics - Domain D: Customer Service dashboard
Two tabs (Overview, Detail), two filters (Connection Type, Region), everything on one screen.

Run from the project root:
    streamlit run dashboard/app.py

Reads the prepared database (customer_features + complaints). The database is found automatically
(data/airtel.db or data/raw/airtel.db, searching upward from this file) or set the AIRTEL_DB
environment variable to a full path.
"""
import os
import sqlite3
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- page + layout constants
st.set_page_config(page_title="Airtel Telecom Analytics - Customer Service", page_icon="📶",
                   layout="wide", initial_sidebar_state="collapsed")

BLUE, DARK, LIGHT, BORDER, RED = "#1f5fd6", "#0b2a8a", "#e8f0fe", "#2d62d9", "#e8453c"
BLUES = ["#1f5fd6", "#4d8af0", "#7fb0f7", "#a9cbfa", "#cfe0fc", "#0b2a8a"]
COMPACT = str(st.query_params.get("compact", "0")) == "1"      # open the app with ?compact=1 on a small screen
CHART_H = 135 if COMPACT else 215        # height (px) of the main charts - lower it if your screen is small
TREND_H = 105 if COMPACT else 175        # height (px) of the trend chart on the Detail tab
MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

st.markdown(f"""
<style>
  /* one-screen layout: remove Streamlit's chrome and tighten spacing */
  header[data-testid="stHeader"], #MainMenu, footer {{display:none !important;}}
  .block-container {{padding:0.7rem 1.2rem 0 1.2rem !important; max-width:100% !important;}}
  div[data-testid="stVerticalBlock"] {{gap:0.45rem;}}
  div[data-testid="stHorizontalBlock"] {{gap:0.6rem;}}
  html, body, [data-testid="stAppViewContainer"] {{background:#fff; overflow-x:hidden;}}

  /* header */
  .hdr {{display:flex; align-items:center; gap:22px; border:2px solid {BORDER}; border-radius:12px;
         padding:6px 20px; height:62px;}}
  .logo {{font-size:30px; font-weight:800; font-style:italic; color:#e40000; letter-spacing:-1px;
          padding-right:22px; border-right:2px solid {BORDER}; line-height:1;}}
  .ttl {{color:{DARK}; font-weight:800; font-size:25px; line-height:1.05;}}
  .sub {{color:{DARK}; font-size:16px; font-weight:500;}}

  /* filter panel + cards (styled through the key= class Streamlit adds to containers) */
  .st-key-filters {{background:{LIGHT}; border:2px solid {BORDER}; border-radius:12px; padding:12px; min-height:{0 if COMPACT else 690}px;}}
  .st-key-filters h3 {{color:{DARK}; margin:0 0 6px 0; font-size:22px;}}
  .st-key-filters label p {{color:{DARK}; font-weight:700; font-size:15px;}}
  [class*="st-key-card"] {{border:2px solid {BORDER}; border-radius:12px; padding:0 !important; overflow:hidden;
                           background:#fff; gap:0 !important;}}
  .ctitle {{background:{LIGHT}; color:{DARK}; font-weight:700; font-size:15px; padding:3px 12px;}}

  /* info banner + KPI cards */
  .info {{background:{LIGHT}; border:2px solid {BORDER}; border-radius:12px; padding:6px 14px; color:{DARK};
          font-weight:600; font-size:14px; display:flex; align-items:center; gap:12px; height:46px;}}
  .info .i {{background:{BLUE}; color:#fff; border-radius:50%; width:28px; height:28px; display:flex;
             align-items:center; justify-content:center; font-weight:800; flex:none;}}
  .kpi {{border:2px solid {BORDER}; border-radius:12px; text-align:center; height:76px; display:flex;
         flex-direction:column; justify-content:center;}}
  .kpi .l {{color:{DARK}; font-size:14px; font-weight:600; line-height:1.1;}}
  .kpi .v {{color:{DARK}; font-size:28px; font-weight:800; line-height:1.15;}}
  .kpi .s {{color:#51608a; font-size:11px; line-height:1;}}

  /* raw-HTML blocks must not be pulled up by Streamlit's negative markdown margin */
  div[data-testid="stMarkdownContainer"]:has(.info, .kpi, .hdr, .ctitle, .lbl) {{margin-bottom:0 !important;}}

  /* tabs as pills (react-aria tabs in current Streamlit, baseweb tabs in older versions) */
  div[role="tablist"] {{gap:10px; border-bottom:0 !important; box-shadow:none !important;}}
  [role="tab"] {{border:2px solid {BORDER} !important; border-radius:10px !important; background:#fff !important;
                 height:38px; padding:0 40px !important;}}
  [role="tab"] p {{font-weight:700; font-size:16px; color:{DARK};}}
  [role="tab"][aria-selected="true"] {{background:{BLUE} !important;}}
  [role="tab"][aria-selected="true"] p {{color:#fff;}}
  .react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{display:none !important;}}
  .lbl {{color:{DARK}; font-weight:700; font-size:17px; padding-top:7px; white-space:nowrap;}}
  div[data-baseweb="select"] > div {{border:2px solid {BORDER}; border-radius:8px;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
def find_db() -> Path:
    """Locate airtel.db or decompress the bundled airtel.db.gz for Streamlit Cloud."""
    env = os.environ.get("AIRTEL_DB")
    if env and Path(env).exists():
        return Path(env)

    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen = set()
    for start in starts:
        for folder in [start, *start.parents]:
            if folder in seen:
                continue
            seen.add(folder)
            for rel in ("data/airtel.db", "data/raw/airtel.db", "airtel.db"):
                p = folder / rel
                if p.exists() and p.stat().st_size > 1_000_000:
                    return p
            for rel in ("data/airtel.db.gz", "data/raw/airtel.db.gz", "airtel.db.gz"):
                gz_path = folder / rel
                if gz_path.exists():
                    extracted = Path("/tmp/airtel.db")
                    if not extracted.exists() or extracted.stat().st_size < 1_000_000:
                        with gzip.open(gz_path, "rb") as source, open(extracted, "wb") as target:
                            shutil.copyfileobj(source, target)
                    return extracted
    raise FileNotFoundError("airtel.db or airtel.db.gz not found")


@st.cache_data(show_spinner=False)
def load(db_path: str, mtime: float):
    """Subscribers (customer_features) and tickets (complaints joined to subscriber attributes)."""
    con = sqlite3.connect(Path(db_path).as_uri() + "?mode=ro", uri=True)
    try:
        subs = pd.read_sql("""SELECT customer_id, connection_type, region, circle_name, is_churned,
                                     complaint_count, unresolved_count
                              FROM customer_features""", con)
        tix = pd.read_sql("""SELECT complaint_id, customer_id, complaint_date, category, is_resolved
                             FROM complaints""", con)
    finally:
        con.close()
    tix = tix.merge(subs[["customer_id", "connection_type", "region", "circle_name", "is_churned"]],
                    on="customer_id", how="left")
    tix["month"] = tix["complaint_date"].str[:7]
    return subs, tix


try:
    DB = find_db()
    SUBS, TIX = load(str(DB), DB.stat().st_mtime)
except FileNotFoundError:
    st.error("Could not find airtel.db. Put it in the project's `data/` folder, or set the AIRTEL_DB "
             "environment variable to its full path, then reload.")
    st.stop()
except Exception as exc:                                        # missing table, wrong file, etc.
    st.error(f"Could not read customer_features / complaints from {DB}: {exc}. "
             "Run the Step 2 pipeline (notebook 04_preparation) on this database first.")
    st.stop()


# --------------------------------------------------------------------------- helpers
def show(fig):
    """Render a Plotly figure without the toolbar, filling the card width."""
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:                                           # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def card(key: str, title: str):
    """A bordered card with a light-blue title strip; use as `with card(...): show(fig)`."""
    box = st.container(key=f"card_{key}")
    box.markdown(f'<div class="ctitle">{title}</div>', unsafe_allow_html=True)
    return box


def kpi(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{sub_html}</div>',
                unsafe_allow_html=True)


def style(fig, height, legend=False):
    fig.update_layout(height=height, margin=dict(l=44, r=10, t=6, b=34), template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=11, color=DARK), showlegend=legend,
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0.5, xanchor="center"))
    if legend:
        fig.update_layout(margin=dict(l=44, r=10, t=30, b=34))
    return fig


def complaint_history(df):
    """Never complained / all tickets closed / at least one open."""
    return np.select([df["complaint_count"] == 0, df["unresolved_count"] == 0],
                     ["Never", "All closed"], default="At least one open")


def chi_square_p(table: np.ndarray):
    """Chi-square test of independence on a contingency table; returns (chi2, df, p)."""
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df))
    except Exception:                                           # scipy missing: exact for df = 2, else None
        p = float(np.exp(-chi2 / 2)) if df == 2 else None
    return chi2, df, p


def fmt_p(p):
    if p is None:
        return "n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# --------------------------------------------------------------------------- header
st.markdown("""
<div class="hdr"><div class="logo">airtel</div>
  <div><div class="ttl">Airtel Telecom Analytics</div><div class="sub">Domain D – Customer Service</div></div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1.05, 7], gap="small")

# --------------------------------------------------------------------------- filters
with left:
    with st.container(key="filters"):
        st.markdown("### Filters")
        conn_opts = ["All"] + sorted(SUBS["connection_type"].unique())
        region_opts = ["All"] + sorted(SUBS["region"].unique())
        conn_sel = st.selectbox("Connection Type", conn_opts,
                                index=conn_opts.index(st.query_params.get("conn")) if st.query_params.get("conn") in conn_opts else 0)
        region_sel = st.selectbox("Region", region_opts,
                                  index=region_opts.index(st.query_params.get("region")) if st.query_params.get("region") in region_opts else 0)

subs = SUBS.copy()
tix = TIX.copy()
if conn_sel != "All":
    subs, tix = subs[subs.connection_type == conn_sel], tix[tix.connection_type == conn_sel]
if region_sel != "All":
    subs, tix = subs[subs.region == region_sel], tix[tix.region == region_sel]

CATEGORIES = sorted(TIX["category"].unique())

# --------------------------------------------------------------------------- tabs
with right:
    tab_overview, tab_detail = st.tabs(["Overview", "Detail"])

    if len(tix) == 0:
        st.warning("No tickets for this combination of filters.")
        st.stop()

    # ======================================================================= OVERVIEW
    with tab_overview:
        st.markdown('<div class="info"><div class="i">i</div><div>This dashboard helps the Head of Customer '
                    'Service prioritise service effort and understand ticket patterns.</div></div>',
                    unsafe_allow_html=True)

        hist = subs.assign(history=complaint_history(subs))
        order = ["Never", "All closed", "At least one open"]
        churn_by_hist = (hist.groupby("history")["is_churned"].agg(["mean", "size"]).reindex(order))
        table = pd.crosstab(hist["history"], hist["is_churned"]).reindex(order).fillna(0).to_numpy()
        chi2, dof, p = chi_square_p(table)
        open_pct = 100 * (tix["is_resolved"] == 0).mean()

        k1, k2, k3 = st.columns(3)
        with k1:
            kpi("Total Tickets", f"{len(tix):,}")
        with k2:
            kpi("T7 Test Result", fmt_p(p), "chi-square: churn vs complaint history")
        with k3:
            kpi("Open Tickets", f"{open_pct:.1f}%")

        # --- churn by complaint history
        with card("hist", "Churn by Complaint History"):
            vals = (100 * churn_by_hist["mean"]).round(2)
            fig = go.Figure(go.Bar(x=["Never", "All closed", "At least one open"], y=vals.fillna(0),
                                   marker_color=["#7fb0f7", "#4d8af0", BLUE],
                                   text=[f"{v:.1f}%" if pd.notna(v) else "" for v in vals],
                                   textposition="outside", cliponaxis=False,
                                   customdata=churn_by_hist["size"],
                                   hovertemplate="%{x}<br>Churn %{y:.2f}%<br>%{customdata:,} subscribers<extra></extra>"))
            fig.update_yaxes(title_text="Churn Rate (%)", range=[0, max(vals.max() * 1.25, 1)], title_font_size=11)
            fig.update_xaxes(title_text="Complaint History", title_font_size=11, title_standoff=2)
            show(style(fig, CHART_H))

        c1, c2 = st.columns(2)
        # --- closure share by category (share of closed tickets)
        with c1:
            with card("closure", "Closure Share by Category"):
                closed = tix[tix.is_resolved == 1].groupby("category").size().reindex(CATEGORIES).fillna(0)
                fig = go.Figure(go.Pie(labels=closed.index, values=closed.values, sort=False, hole=0.0,
                                       domain=dict(x=[0.0, 0.62]),
                                       marker=dict(colors=BLUES, line=dict(color="#fff", width=1)),
                                       textinfo="percent", textfont_size=11,
                                       hovertemplate="%{label}<br>%{value:,} closed tickets (%{percent})<extra></extra>"))
                style(fig, CHART_H, legend=True)
                fig.update_layout(margin=dict(l=10, r=10, t=6, b=6),
                                  legend=dict(orientation="v", x=0.68, y=0.5, font=dict(size=11)))
                show(fig)

        # --- tickets per 100 subscribers: by region, or by circle once a region is chosen
        with c2:
            by_circle = region_sel != "All"
            key = "circle_name" if by_circle else "region"
            with card("per100", "Tickets per 100 Subscribers by " + ("Circle" if by_circle else "Region")):
                n_t = tix.groupby(key).size()
                n_s = subs.groupby(key).size()
                rate = (100 * n_t / n_s).dropna().sort_values(ascending=False)
                fig = go.Figure(go.Bar(x=rate.index, y=rate.values.round(1), marker_color="#4d8af0",
                                       text=[f"{v:.1f}" for v in rate.values], textposition="outside",
                                       cliponaxis=False,
                                       hovertemplate="%{x}<br>%{y:.1f} tickets per 100 subscribers<extra></extra>"))
                fig.update_yaxes(title_text="Tickets per 100 Subscribers", range=[0, rate.max() * 1.25],
                                 title_font_size=11)
                fig.update_xaxes(title_text="Circle" if by_circle else "Region", title_font_size=11,
                                 title_standoff=2, tickangle=0 if len(rate) <= 6 else -35)
                style(fig, CHART_H)
                if len(rate) > 6:
                    fig.update_layout(margin=dict(l=44, r=10, t=6, b=58))
                show(fig)

    # ======================================================================= DETAIL
    with tab_detail:
        st.markdown('<div class="info"><div class="i">i</div><div>Explore a complaint category to understand '
                    'ticket volume, closure and retention association.</div></div>', unsafe_allow_html=True)

        s0, s1, s2 = st.columns([1.15, 1.6, 4])
        with s0:
            st.markdown('<div class="lbl">Complaint Category:</div>', unsafe_allow_html=True)
        with s1:
            cat = st.selectbox("Complaint Category", CATEGORIES, label_visibility="collapsed",
                               index=CATEGORIES.index("Billing") if "Billing" in CATEGORIES else 0)
        sel = tix[tix.category == cat]
        n = len(sel)
        closed_n = int((sel.is_resolved == 1).sum())
        closed_pct = 100 * closed_n / n if n else 0

        k1, k2, k3 = st.columns(3)
        with k1:
            kpi("Tickets in Selected Category", f"{n:,}")
        with k2:
            kpi("Closed Tickets", f"{closed_pct:.0f}%", f"{closed_n:,} tickets")
        with k3:
            kpi("Open Tickets", f"{100 - closed_pct:.0f}%" if n else "0%", f"{n - closed_n:,} tickets")

        # --- ticket volume trend
        with card("trend", f"Ticket Volume Trend - {cat}"):
            monthly = sel.groupby("month").size().reindex(MONTHS).fillna(0)
            fig = go.Figure(go.Scatter(x=MONTH_LABELS, y=monthly.values, mode="lines+markers",
                                       line=dict(color=BLUE, width=2.5), marker=dict(size=8),
                                       hovertemplate="%{x}: %{y:,} tickets<extra></extra>"))
            fig.update_yaxes(title_text="Ticket Count", range=[0, max(float(monthly.max()) * 1.18, 1)], title_font_size=11)
            fig.update_xaxes(title_text="Month", title_font_size=11, title_standoff=2)
            show(style(fig, TREND_H))

        d1, d2 = st.columns([1.7, 1])
        # --- ticket status vs leaving (share of each category's tickets raised by subscribers who left)
        with d1:
            with card("leaving", "Ticket Status vs Leaving"):
                g = tix.groupby("category")["is_churned"].mean().reindex(CATEGORIES) * 100
                fig = go.Figure()
                fig.add_bar(x=CATEGORIES, y=(100 - g).round(1), name="Stayed", marker_color=BLUE,
                            hovertemplate="%{x}<br>Stayed %{y:.1f}%<extra></extra>")
                fig.add_bar(x=CATEGORIES, y=g.round(1), name="Left", marker_color=RED,
                            hovertemplate="%{x}<br>Left %{y:.1f}%<extra></extra>")
                fig.update_layout(barmode="group", bargap=0.25)
                fig.update_yaxes(title_text="Share of Tickets (%)", range=[0, 100], title_font_size=11)
                fig.update_xaxes(title_text="Leaving Status (by complaint category)", title_font_size=11,
                                 title_standoff=2)
                show(style(fig, CHART_H, legend=True))

        # --- selected category status donut
        with d2:
            with card("donut", "Selected Category Status"):
                fig = go.Figure(go.Pie(labels=["Closed", "Open"], values=[closed_n, n - closed_n], hole=0.62,
                                       domain=dict(x=[0.0, 0.66]),
                                       sort=False, marker=dict(colors=[BLUE, RED]), textinfo="none",
                                       hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
                fig.add_annotation(text=f"<b>{closed_pct:.0f}%</b>", x=0.33, y=0.5, showarrow=False,
                                   font=dict(size=22, color=DARK))
                style(fig, CHART_H, legend=True)
                fig.update_layout(margin=dict(l=4, r=4, t=4, b=4),
                                  legend=dict(orientation="v", x=0.72, y=0.5, font=dict(size=11)))
                show(fig)
