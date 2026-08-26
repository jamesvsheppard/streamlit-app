"""
Alert source-overlap visualization app.

Run with:
    streamlit run app.py

Reads data/overlap_results.csv (the rolled-up county/state/source table exported by the
alerts_source_overlap.ipynb notebook) and data/alerts.csv (the un-rolled-up alert rows).
"""
import inspect
import os
from collections import Counter

import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Alert Source Overlap", layout="wide")

# Streamlit >=1.60 added on-by-default "lazy loading" for large st.dataframe grids. That grid
# can fail to hydrate on Streamlit Community Cloud, blanking the whole app (white screen +
# spinner, no Python-side error in the logs). Opt out of it wherever the `lazy` parameter exists
# so rows render eagerly; on older Streamlit (no such param) this is a no-op.
try:
    _DATAFRAME_HAS_LAZY = "lazy" in inspect.signature(st.dataframe).parameters
except (TypeError, ValueError):
    _DATAFRAME_HAS_LAZY = False


def render_dataframe(df, **kwargs):
    if _DATAFRAME_HAS_LAZY:
        kwargs.setdefault("lazy", False)
    st.dataframe(df, **kwargs)


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "overlap_results.csv")
ALERTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alerts.csv")

# the base integer count columns -> friendly labels used in selectors / axis titles
METRICS = {
    "total_alerts": "Total alerts",
    "dupes_between_vh_curate": "Dupes between VH & Curate",
    "dupes_within_source": "Dupes within source",
}

# derived share column (float %): dupes_between_vh_curate / total_alerts
PCT_COL = "pct_dupes_between_vh_curate"
PCT_LABEL = "% dupes between VH & Curate"

# who-was-first columns (integer counts, per cross-source duplicate event)
FIRST_COLS = {
    "dupe_alerts_received_first": "Dupe alerts received first",
    "dupe_alerts_received_same_date": "Dupe alerts received same date",
}

# all integer columns to coerce on load; count columns shown in tooltips
INT_COLS = list(METRICS) + list(FIRST_COLS)
COUNT_LABELS = {**METRICS, **FIRST_COLS}

# metrics a source-stacked bar can show: each splits additively across VH/Curate. (The share %
# is a ratio, and received_same_date is identical on both source rows, so neither stacks meaningfully.)
STACK_METRICS = {
    "total_alerts": "Total alerts",
    "dupes_between_vh_curate": "Dupes between VH & Curate",
    "dupes_within_source": "Dupes within source",
    "dupe_alerts_received_first": "Dupe alerts received first",
}

DEFINITION_ORDER = [
    "Same URL",
    "Same URL + post_date",
    "Same URL + meeting_date",
    "Same meeting_date",
    "Same meeting_date + search_term",
]

# Maps each definition to the raw-alert columns that make two alerts "the same". Duplicate
# detection is always scoped within a county/state, so these are the *additional* key columns.
# Used to highlight duplicate sets in the raw alerts viewer.
DEFINITION_KEYS = {
    "Same URL": ["url"],
    "Same URL + post_date": ["url", "post_date"],
    "Same URL + meeting_date": ["url", "meeting_date"],
    "Same meeting_date": ["meeting_date"],
    "Same meeting_date + search_term": ["meeting_date", "search_term"],
}

# two visually-distinct blues for alternating duplicate sets; dark text keeps contrast in both themes
DUPE_COLORS = ["#cfe3ff", "#8fbce8"]
DUPE_TEXT = "#0a2a43"
DUPE_HIGHLIGHT_CAP = 3000  # skip per-row styling above this many rows to keep the browser snappy

# columns shown (in this order) in the raw alerts viewer
RAW_DISPLAY_COLS = ["source", "county", "state", "post_date", "meeting_date",
                    "milestone_type", "search_term", "url"]

VIEWS = ["Overview", "Data tables", "Bar chart", "Raw Alerts Viewer", "About the metrics"]


@st.cache_data
def load_data(path, mtime):
    # mtime (the file's last-modified time) is hashed into the cache key, so rewriting the CSV
    # invalidates the cache and forces a reload — even when the code itself hasn't changed.
    # (Note: it must NOT be named with a leading underscore, or Streamlit would skip hashing it.)
    df = pd.read_csv(path)
    for col in INT_COLS:
        df[col] = df[col].astype(int)
    # derive the share % if an older export doesn't already include it
    if PCT_COL not in df.columns:
        df[PCT_COL] = (df["dupes_between_vh_curate"] / df["total_alerts"] * 100).round(1).fillna(0)
    return df


@st.cache_data
def load_alerts(path, mtime):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    # align the source column name with the rolled-up table + sidebar filters
    if "milestone_source" in df.columns:
        df = df.rename(columns={"milestone_source": "source"})
    return df


def apply_filters(df, states, counties, sources):
    """Empty selection means 'no filter on this field'."""
    if states:
        df = df[df["state"].isin(states)]
    if counties:
        df = df[df["county"].isin(counties)]
    if sources:
        df = df[df["source"].isin(sources)]
    return df


def dupe_group_ordinals(view, key_cols):
    """Positional list aligned to view's rows: the duplicate-set ordinal (0, 1, 2, … in the order
    sets first appear) for rows that belong to a dupe group, else -1. A dupe group is >=2 rows
    that share (county, state, *key_cols*); rows with a null in any key column are never grouped."""
    grp_cols = ["county", "state"] + [c for c in key_cols if c in view.columns]
    present = view[grp_cols].notna().all(axis=1).tolist()
    keys = [tuple(r) for r in view[grp_cols].to_numpy()]
    counts = Counter(k for k, p in zip(keys, present) if p)
    seen, nxt, out = {}, 0, []
    for k, p in zip(keys, present):
        if p and counts[k] >= 2:
            if k not in seen:
                seen[k] = nxt
                nxt += 1
            out.append(seen[k])
        else:
            out.append(-1)
    return out


def style_alerts(view, key_cols):
    """Return a pandas Styler that shades each duplicate set, alternating between two blues."""
    ordinals = dupe_group_ordinals(view, key_cols)

    def _row_style(row):
        gid = ordinals[row.name]  # row.name is the positional index (view is reset-indexed)
        if gid < 0:
            return [""] * len(row)
        return [f"background-color: {DUPE_COLORS[gid % 2]}; color: {DUPE_TEXT}"] * len(row)

    return view.style.apply(_row_style, axis=1)


def _extract_points(sel):
    """Pull the list of selected point-dicts out of a Streamlit altair selection payload,
    tolerating both dict-like and attribute-style objects and any selection-param name."""
    if not sel:
        return []
    selection = None
    if isinstance(sel, dict):
        selection = sel.get("selection", sel)
    else:
        selection = getattr(sel, "selection", None)
    if isinstance(selection, dict):
        for value in selection.values():
            if isinstance(value, list) and value:
                return value
    return []


def _handle_chart_click():
    """Chart on_select callback: focus the raw viewer on the clicked jurisdiction and switch to it."""
    points = _extract_points(st.session_state.get("chart_select"))
    if not points:
        return
    p = points[0]
    # Each bar is a whole jurisdiction (both sources stacked), so focus on county/state and keep
    # both providers — this also lets the viewer highlight the between-source duplicate pairs.
    st.session_state.raw_focus = {
        "county": p.get("county"), "state": p.get("state"), "source": None,
    }
    # the duplicate definition is shared across views (key "dupe_def"), so the viewer already
    # highlights by whatever was selected on the chart — nothing extra to carry over.
    st.session_state.active_view = "Raw Alerts Viewer"


# ---------------------------------------------------------------- load
if not os.path.exists(DATA_PATH):
    st.error(
        "Could not find `data/overlap_results.csv`.\n\n"
        "Run the **alerts_source_overlap.ipynb** notebook top-to-bottom first — the "
        "export cell writes this file."
    )
    st.stop()

data = load_data(DATA_PATH, os.path.getmtime(DATA_PATH))
# keep only the definitions this app surfaces (drops any others still in the export)
data = data[data["definition"].isin(DEFINITION_ORDER)].copy()
definitions = [d for d in DEFINITION_ORDER if d in data["definition"].unique()]

alerts = load_alerts(ALERTS_PATH, os.path.getmtime(ALERTS_PATH)) if os.path.exists(ALERTS_PATH) else None

# date range of the underlying alerts (min/max post_date), surfaced on the Overview + About tabs
DATE_RANGE_TEXT = None
if alerts is not None and "post_date" in alerts.columns:
    _post_dates = pd.to_datetime(alerts["post_date"], errors="coerce")
    _dmin, _dmax = _post_dates.min(), _post_dates.max()
    if pd.notna(_dmin) and pd.notna(_dmax):
        DATE_RANGE_TEXT = f"{_dmin:%B} {_dmin.day}, {_dmin.year} to {_dmax:%B} {_dmax.day}, {_dmax.year}"

st.title("Alert Source Overlap Explorer")
st.caption(
    "Duplicate alerts between and within Voterheads & Curate, at the county/state/source level, "
    "under five different definitions of a duplicate."
)

# ---------------------------------------------------------------- sidebar filters (apply to every view)
st.sidebar.header("Filters")
st.sidebar.caption("Applied to every tab. Leave a filter empty to include everything.")

state_opts = sorted(data["state"].dropna().unique())
sel_states = st.sidebar.multiselect("State", state_opts)

# county options narrow to the chosen states for easier picking
county_pool = data[data["state"].isin(sel_states)] if sel_states else data
county_opts = sorted(county_pool["county"].dropna().unique())
sel_counties = st.sidebar.multiselect("County", county_opts)

source_opts = sorted(data["source"].dropna().unique())
sel_sources = st.sidebar.multiselect("Source", source_opts)

filtered = apply_filters(data, sel_states, sel_counties, sel_sources)

# Keep the shared duplicate definition alive across views. The Data tables view renders no
# "dupe_def" selectbox, so without re-touching the key each run Streamlit garbage-collects the
# widget value on that run and it reverts to the default. Re-assigning marks it as user state.
if "dupe_def" in st.session_state:
    st.session_state.dupe_def = st.session_state.dupe_def

# ---------------------------------------------------------------- navigation (radio acts as the tab
# bar; unlike st.tabs it can be switched programmatically — e.g. from a bar click)
if "active_view" not in st.session_state:
    st.session_state.active_view = VIEWS[0]
active_view = st.radio("View", VIEWS, key="active_view", horizontal=True, label_visibility="collapsed")
st.divider()

# ================================================================ Overview
if active_view == "Overview":
    if DATE_RANGE_TEXT:
        st.markdown(f"**Date range of data:** {DATE_RANGE_TEXT}")
    st.subheader("At-a-glance overview")
    ov_def = st.selectbox(
        "Duplicate definition", definitions, key="dupe_def",
        help="Duplicate counts depend on the definition — the totals below use this one. "
             "Your choice carries across all tabs.",
    )
    ov = filtered[filtered["definition"] == ov_def]

    dupe_cols = ["dupes_between_vh_curate", "dupes_within_source"]
    total_dupes = int(ov[dupe_cols].sum().sum())

    # county/state pairs with at least one duplicate (across sources or within a source)
    dupes_per_county = (ov.assign(_dupes=ov[dupe_cols].sum(axis=1))
                          .groupby(["county", "state"])["_dupes"].sum())
    counties_with_dupes = int((dupes_per_county > 0).sum())

    vh_first = int(ov.loc[ov["source"] == "Voterheads", "dupe_alerts_received_first"].sum())
    cu_first = int(ov.loc[ov["source"] == "Curate", "dupe_alerts_received_first"].sum())
    vh_within = int(ov.loc[ov["source"] == "Voterheads", "dupes_within_source"].sum())
    cu_within = int(ov.loc[ov["source"] == "Curate", "dupes_within_source"].sum())

    r1a, r1b = st.columns(2)
    r1a.metric(
        "Total duplicate alerts", f"{total_dupes:,}",
        help="dupes_between_vh_curate + dupes_within_source, summed over every row. A duplicated "
             "alert is counted once per row, so it can be counted multiple times.",
    )
    r1b.metric(
        "Counties with duplicates", f"{counties_with_dupes:,}",
        help="Unique county/state pairs with at least one duplicate — across sources "
             "(dupes_between_vh_curate) or within a source (dupes_within_source) — under this definition.",
    )

    st.markdown("**Cross-source alerts received first** — who posted earlier when both providers caught the same event")
    r2a, r2b = st.columns(2)
    r2a.metric("Voterheads received first", f"{vh_first:,}")
    r2b.metric("Curate received first", f"{cu_first:,}")

    st.markdown("**Duplicates within a single source** — repeated alerts from the same provider")
    r3a, r3b = st.columns(2)
    r3a.metric("Dupes within Voterheads", f"{vh_within:,}")
    r3b.metric("Dupes within Curate", f"{cu_within:,}")

# ================================================================ Data tables
elif active_view == "Data tables":
    st.subheader("Base data tables")
    st.caption("One table per duplicate definition. Click a column header to sort. Filters from the sidebar apply.")

    st.caption(
        f"**{PCT_LABEL}** = dupes_between_vh_curate ÷ total_alerts. "
        "Sort by it (descending) to find counties whose alerts are most duplicative across sources."
    )
    display_cols = ["county", "state", "source", "total_alerts", "dupes_between_vh_curate",
                    PCT_COL, "dupe_alerts_received_first", "dupe_alerts_received_same_date",
                    "dupes_within_source"]
    pct_config = {PCT_COL: st.column_config.NumberColumn(PCT_LABEL, format="%.1f%%")}
    for i, defn in enumerate(definitions):
        sub = filtered[filtered["definition"] == defn][display_cols].reset_index(drop=True)
        with st.expander(f"{defn}  —  {len(sub):,} rows", expanded=(i == 0)):
            render_dataframe(sub, width="stretch", hide_index=True, height=360, column_config=pct_config)

# ================================================================ Bar chart
elif active_view == "Bar chart":
    st.subheader("Ranked jurisdictions")

    c1, c2, c3 = st.columns([2, 2, 1])
    sel_def = c1.selectbox("Duplicate definition", definitions, key="dupe_def")
    metric = c2.selectbox(
        "Rank by", list(STACK_METRICS), format_func=lambda c: STACK_METRICS[c], key="chart_metric",
    )
    min_total = c3.number_input(
        "Min. total alerts", min_value=1, value=1, step=1, key="chart_min_total",
        help="Ignore jurisdictions whose combined (Voterheads + Curate) alert count is below this.",
    )

    c4, c5 = st.columns([2, 2])
    order = c4.radio(
        "Order", ["Top (highest first)", "Bottom (lowest first)"],
        horizontal=True, key="chart_order",
    )
    n_rows = c5.slider("Rows to show", min_value=5, max_value=50, value=20, step=5, key="chart_rows")
    ascending = order.startswith("Bottom")

    metric_label = STACK_METRICS[metric]
    chart_df = filtered[filtered["definition"] == sel_def].copy()
    chart_df["jurisdiction"] = (chart_df["county"].fillna("(no county)")
                                + ", " + chart_df["state"].astype(str))

    # roll the two source rows up to the jurisdiction to rank and apply the min-alerts filter
    juris = chart_df.groupby(["county", "state", "jurisdiction"], as_index=False).agg(
        metric_total=(metric, "sum"),
        alerts_total=("total_alerts", "sum"),
    )
    juris = juris[juris["alerts_total"] >= min_total]
    juris = juris.sort_values("metric_total", ascending=ascending).head(n_rows)

    if juris.empty:
        st.info("No jurisdictions match the current filters.")
    else:
        # per-source rows for the selected jurisdictions -> the stacked segments (+ jurisdiction total)
        plot = chart_df.merge(juris[["county", "state", "metric_total"]], on=["county", "state"])
        order_list = juris["jurisdiction"].tolist()  # largest-first (top) or smallest-first (bottom)

        st.markdown(f"**{'Bottom' if ascending else 'Top'} {len(juris)} jurisdictions by {metric_label}** "
                    "— each bar stacks Voterheads + Curate")
        if alerts is not None:
            st.caption("Tip: click a bar to open that jurisdiction's underlying alerts in the Raw Alerts Viewer.")

        click_sel = alt.selection_point(fields=["county", "state"], on="click", name="barsel", empty=False)
        chart = (
            alt.Chart(plot)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric}:Q", title=metric_label, stack="zero"),
                y=alt.Y("jurisdiction:N", sort=order_list, title=None),
                color=alt.Color("source:N", title="Source"),
                tooltip=[
                    alt.Tooltip("jurisdiction:N", title="Jurisdiction"),
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip(f"{metric}:Q", title=metric_label),
                    alt.Tooltip("metric_total:Q", title=f"{metric_label} (total)"),
                ],
            )
            .add_params(click_sel)
            .properties(height=max(300, 30 * len(juris)))
        )
        st.altair_chart(chart, width="stretch", on_select=_handle_chart_click, key="chart_select")

        with st.expander(f"Show these {len(juris)} jurisdictions as a table"):
            pv = (plot.pivot_table(index=["county", "state"], columns="source",
                                   values=metric, fill_value=0, aggfunc="sum").reset_index())
            for s in ("Voterheads", "Curate"):
                if s not in pv.columns:
                    pv[s] = 0
            pv["Total"] = pv["Voterheads"] + pv["Curate"]
            pv = pv.sort_values("Total", ascending=ascending).rename(columns={
                "Voterheads": f"{metric_label} — VH", "Curate": f"{metric_label} — Curate"})
            render_dataframe(pv, width="stretch", hide_index=True)

# ================================================================ Raw Alerts Viewer
elif active_view == "Raw Alerts Viewer":
    st.subheader("Raw alerts viewer")
    if alerts is None:
        st.error("Could not find `data/alerts.csv`.")
    else:
        raw_def = st.selectbox(
            "Highlight duplicates by", definitions, key="dupe_def",
            help="Rows that form a duplicate set under this definition are shaded; adjacent sets "
                 "alternate between two blues so you can tell neighbouring pairs apart. "
                 "Your choice carries across all tabs.",
        )
        key_cols = DEFINITION_KEYS.get(raw_def, [])

        focus = st.session_state.get("raw_focus")
        if focus and any(v is not None for v in focus.values()):
            chip = ", ".join(str(focus[k]) for k in ("county", "state", "source") if focus.get(k) is not None)
            fc1, fc2 = st.columns([5, 1])
            fc1.info(f"Focused on **{chip}** (from a bar click). Sidebar filters are ignored while focused.")
            if fc2.button("Clear focus"):
                st.session_state.raw_focus = None
                st.rerun()
            view = alerts
            for col in ("county", "state", "source"):
                if focus.get(col) is not None:
                    view = view[view[col] == focus[col]]
        else:
            view = apply_filters(alerts, sel_states, sel_counties, sel_sources)

        # default order groups duplicate sets together so pairs sit next to each other
        sort_cols = [c for c in (["county", "state"] + key_cols) if c in view.columns]
        view = view.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        view = view[[c for c in RAW_DISPLAY_COLS if c in view.columns]]

        n_dupes = sum(1 for g in dupe_group_ordinals(view, key_cols) if g >= 0) if len(view) else 0
        st.caption(f"{len(view):,} alerts · {n_dupes:,} in a duplicate set under **{raw_def}** "
                   "(shaded; click a header to sort)")

        if len(view) == 0:
            st.info("No alerts match the current filters.")
        elif len(view) <= DUPE_HIGHLIGHT_CAP:
            st.dataframe(style_alerts(view, key_cols), width="stretch", hide_index=True)
        else:
            st.warning(f"Showing {len(view):,} rows — duplicate highlighting is turned off above "
                       f"{DUPE_HIGHLIGHT_CAP:,} rows to stay responsive. Apply filters, or click a "
                       "bar on the Bar chart tab, to narrow the view and enable highlighting.")
            render_dataframe(view, width="stretch", hide_index=True)

# ================================================================ About
elif active_view == "About the metrics":
    st.subheader("What the definitions and metrics mean")
    if DATE_RANGE_TEXT:
        st.markdown(f"**Date range of data:** {DATE_RANGE_TEXT}")

    st.markdown(
        """
Every table is at the **county / state / source** grain — one row per source
(Voterheads or Curate) per county. All duplicate detection happens *within* a
county/state pair. A row counts as a "duplicate alert" if it shares its dupe-key
with **at least one other row** in the relevant scope, so every member of a
duplicate group is counted.
        """
    )

    st.markdown("#### The five definitions of a duplicate")
    st.markdown(
        """
Each definition changes which columns must match for two alerts to be considered the same:

| Definition | Two alerts are duplicates when they share… |
| --- | --- |
| **Same URL** | the same `url` |
| **Same URL + post_date** | the same `url` **and** `post_date` |
| **Same URL + meeting_date** | the same `url` **and** `meeting_date` |
| **Same meeting_date** | the same `meeting_date` |
| **Same meeting_date + search_term** | the same `meeting_date` **and** `search_term` |

`post_date` is when the alert was published; `meeting_date` is when the actual
event occurs; `search_term` is the keyword that surfaced the alert.

⚠️ **Same meeting_date + search_term** will always show **0 "Dupes between VH & Curate"** (and so
0 received-first / same-date): Voterheads and Curate draw from **disjoint `search_term` vocabularies**,
so a cross-source pair can never share one. Only **Dupes within source** is meaningful for that definition.
        """
    )

    st.markdown("#### The numeric columns")
    st.markdown(
        """
| Column | Meaning (for a given source row) |
| --- | --- |
| **total_alerts** | Every alert received from that source for that county/state (independent of any duplicate logic). |
| **dupes_between_vh_curate** | Alerts from that source whose dupe-key **also appears in the other source** — i.e. the same event was reported by both Voterheads and Curate. |
| **pct_dupes_between_vh_curate** | `dupes_between_vh_curate ÷ total_alerts`, as a **percentage** — the share of a source's alerts that are duplicated across sources. Use it to rank counties by how *proportionally* duplicative their alerts are. |
| **dupe_alerts_received_first** | For each cross-source duplicate event, the two providers' `post_date`s are compared. This counts the events where **this source posted first**. (Voterheads-first events show up on the Voterheads row, Curate-first on the Curate row.) |
| **dupe_alerts_received_same_date** | Cross-source duplicate events where **both providers posted on the same `post_date`**. Counted on *both* the Voterheads and Curate rows, so the value matches across the two. |
| **dupes_within_source** | Alerts whose dupe-key repeats **within this row's own source** — repeated alerts from the same provider. (The grain is county/state/source, so the source is the row itself; on a Voterheads row it counts within-Voterheads repeats, on a Curate row within-Curate.) |
        """
    )

    st.markdown("#### The Raw Alerts Viewer")
    st.markdown(
        """
The **Raw Alerts Viewer** shows the un-rolled-up alert rows behind the metrics. It responds to the
sidebar filters, and **clicking a bar** on the Bar chart tab jumps here focused on that
county/state/source. Rows that form a **duplicate set** under the chosen definition are shaded, with
neighbouring sets alternating between two shades of blue. Highlighting is disabled above
%d rows for responsiveness — narrow the view (via filters or a bar click) to re-enable it.
        """ % DUPE_HIGHLIGHT_CAP
    )

    st.markdown("#### Notes & caveats")
    st.markdown(
        """
- `dupes_between_vh_curate` and `dupes_within_source` are **independent comparison types**, so a
  single alert can be counted in both (e.g. a repeated URL that also appears in the other source).
- **`dupe_alerts_received_first` / `dupe_alerts_received_same_date` are counted per cross-source
  *event*** (one point per duplicated event), whereas `dupes_between_vh_curate` counts individual
  alerts. Cross-source dupes are almost always one Voterheads row to one Curate row, so the two
  line up in practice. When a definition already includes `post_date` in its key (e.g.
  *Same URL + post_date*), every cross-source match is a tie by construction, so
  `received_first` is 0 and everything lands in `received_same_date`.
- **`pct_dupes_between_vh_curate` is sensitive to volume**: a county with 1 alert that
  happens to overlap reads as 100%. On the Bar chart tab, raise **Min. total alerts** to
  focus on counties with enough volume for the share to be meaningful.
- Rows whose dupe-key contains a **null** (e.g. a missing `meeting_date`) are excluded from
  duplicate consideration but still count toward `total_alerts` — an unknown value isn't
  treated as a confident match.
- The dataset is built on the **full county list**, so every county/state appears even with
  **zero alerts** (all counts 0). Alert pairs that only showed up as fuzzy-match artifacts
  (a county/state not in the list) are excluded. The **Overview** tab's metrics and the
  **Bar chart** exclude these zero-alert rows once you set *Min. total alerts* ≥ 1 (the default).
        """
    )
