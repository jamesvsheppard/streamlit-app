"""
Alert source-overlap visualization app.

Run with:
    streamlit run overlap_app.py

Reads sheets/overlap_results.csv, which is exported by the
alerts_source_overlap.ipynb notebook (the "Export ... for the visualization app" cell).
"""
import inspect
import os
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

# everything rankable in the chart, in display order: counts + share %
RANKABLE = {
    "total_alerts": "Total alerts",
    "dupes_between_vh_curate": "Dupes between VH & Curate",
    PCT_COL: PCT_LABEL,
    "dupe_alerts_received_first": "Dupe alerts received first",
    "dupe_alerts_received_same_date": "Dupe alerts received same date",
    "dupes_within_source": "Dupes within source",
}

DEFINITION_ORDER = [
    "Same URL",
    "Same URL + post_date",
    "Same URL + meeting_date",
    "Same meeting_date",
    "Same meeting_date + search_term",
]


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


def apply_filters(df, states, counties, sources):
    """Empty selection means 'no filter on this field'."""
    if states:
        df = df[df["state"].isin(states)]
    if counties:
        df = df[df["county"].isin(counties)]
    if sources:
        df = df[df["source"].isin(sources)]
    return df


# ---------------------------------------------------------------- load
if not os.path.exists(DATA_PATH):
    st.error(
        "Could not find `sheets/overlap_results.csv`.\n\n"
        "Run the **alerts_source_overlap.ipynb** notebook top-to-bottom first — the "
        "export cell writes this file."
    )
    st.stop()

data = load_data(DATA_PATH, os.path.getmtime(DATA_PATH))
# keep only the definitions this app surfaces (drops any others still in the export)
data = data[data["definition"].isin(DEFINITION_ORDER)].copy()
definitions = [d for d in DEFINITION_ORDER if d in data["definition"].unique()]

st.title("Alert Source Overlap Explorer")
st.caption(
    "Duplicate alerts between and within Voterheads & Curate, at the county/state/source level, "
    "under five different definitions of a duplicate."
)

# ---------------------------------------------------------------- sidebar filters (apply to both data tabs)
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

# tiny build indicator — confirms which Streamlit the deploy is actually running (safe to remove)
st.sidebar.caption(f"streamlit {st.__version__} · dataframe lazy-load: "
                   f"{'off' if _DATAFRAME_HAS_LAZY else 'n/a'}")

filtered = apply_filters(data, sel_states, sel_counties, sel_sources)

tab_overview, tab_tables, tab_chart, tab_about = st.tabs(
    ["Overview", "Data tables", "Bar chart", "About the metrics"])

# ================================================================ Tab 0: overview
with tab_overview:
    st.subheader("At-a-glance overview")
    ov_def = st.selectbox(
        "Duplicate definition", definitions, key="ov_def",
        help="Duplicate counts depend on the definition — the totals below use this one.",
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

# ================================================================ Tab 1: data tables
with tab_tables:
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

# ================================================================ Tab 2: bar chart
with tab_chart:
    st.subheader("Ranked counties")

    c1, c2, c3 = st.columns([2, 2, 1])
    sel_def = c1.selectbox("Duplicate definition", definitions, key="chart_def")
    metric = c2.selectbox(
        "Rank by", list(RANKABLE), format_func=lambda c: RANKABLE[c], key="chart_metric",
    )
    min_total = c3.number_input(
        "Min. total alerts", min_value=1, value=1, step=1, key="chart_min_total",
        help="Ignore counties with fewer than this many alerts. Useful when ranking by "
             "share % so low-volume counties sitting at 100% don't dominate the ranking.",
    )

    c4, c5 = st.columns([2, 2])
    order = c4.radio(
        "Order", ["Top (highest first)", "Bottom (lowest first)"],
        horizontal=True, key="chart_order",
    )
    n_rows = c5.slider("Rows to show", min_value=5, max_value=50, value=20, step=5, key="chart_rows")
    ascending = order.startswith("Bottom")

    chart_df = filtered[(filtered["definition"] == sel_def)
                        & (filtered["total_alerts"] >= min_total)].copy()

    if chart_df.empty:
        st.info("No rows match the current filters.")
    else:
        chart_df["label"] = (
            chart_df["county"].fillna("(no county)")
            + ", " + chart_df["state"].astype(str)
            + " (" + chart_df["source"] + ")"
        )
        rows = chart_df.sort_values(metric, ascending=ascending).head(n_rows)
        st.markdown(f"**{'Bottom' if ascending else 'Top'} {len(rows)} by {RANKABLE[metric]}**")

        chart = (
            alt.Chart(rows)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric}:Q", title=RANKABLE[metric]),
                y=alt.Y("label:N", sort=("x" if ascending else "-x"), title=None),
                color=alt.Color("source:N", title="Source"),
                tooltip=[
                    alt.Tooltip("county:N"),
                    alt.Tooltip("state:N"),
                    alt.Tooltip("source:N"),
                    *[alt.Tooltip(f"{c}:Q", title=lbl) for c, lbl in COUNT_LABELS.items()],
                    alt.Tooltip(f"{PCT_COL}:Q", title=PCT_LABEL, format=".1f"),
                ],
            )
            .properties(height=max(300, 26 * len(rows)))
        )
        st.altair_chart(chart, width="stretch")

        with st.expander(f"Show these {len(rows)} rows as a table"):
            render_dataframe(
                rows[["county", "state", "source", "total_alerts", "dupes_between_vh_curate",
                      PCT_COL, "dupe_alerts_received_first", "dupe_alerts_received_same_date",
                      "dupes_within_source"]],
                width="stretch", hide_index=True,
                column_config={PCT_COL: st.column_config.NumberColumn(PCT_LABEL, format="%.1f%%")},
            )

# ================================================================ Tab 3: about
with tab_about:
    st.subheader("What the definitions and metrics mean")

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
