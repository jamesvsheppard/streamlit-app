"""
Alert source-overlap visualization app.

Run with:
    streamlit run overlap_app.py

Reads sheets/overlap_results.csv, which is exported by the
alerts_source_overlap.ipynb notebook (the "Export ... for the visualization app" cell).
"""
import os
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Alert Source Overlap", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sheets", "overlap_results.csv")

# the base integer count columns -> friendly labels used in selectors / axis titles
METRICS = {
    "total_alerts": "Total alerts",
    "dupes_between_vh_curate": "Dupes between VH & Curate",
    "dupes_within_vh": "Dupes within Voterheads",
    "dupes_within_curate": "Dupes within Curate",
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
    "dupes_within_vh": "Dupes within Voterheads",
    "dupes_within_curate": "Dupes within Curate",
}

DEFINITION_ORDER = [
    "Same URL",
    "Same URL + post_date",
    "Same URL + meeting_date",
    "Same post_date + milestone_type",
    "Same meeting_date + milestone_type",
]


@st.cache_data
def load_data(path):
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

data = load_data(DATA_PATH)
definitions = [d for d in DEFINITION_ORDER if d in data["definition"].unique()]

st.title("Alert Source Overlap Explorer")
st.caption(
    "Duplicate alerts between and within Voterheads & Curate, at the county/state/source level, "
    "under five different definitions of a duplicate."
)

# ---------------------------------------------------------------- sidebar filters (apply to both data tabs)
st.sidebar.header("Filters")
st.sidebar.caption("Applied to the Data tables and Bar chart tabs. Leave a filter empty to include everything.")

state_opts = sorted(data["state"].dropna().unique())
sel_states = st.sidebar.multiselect("State", state_opts)

# county options narrow to the chosen states for easier picking
county_pool = data[data["state"].isin(sel_states)] if sel_states else data
county_opts = sorted(county_pool["county"].dropna().unique())
sel_counties = st.sidebar.multiselect("County", county_opts)

source_opts = sorted(data["source"].dropna().unique())
sel_sources = st.sidebar.multiselect("Source", source_opts)

filtered = apply_filters(data, sel_states, sel_counties, sel_sources)

tab_tables, tab_chart, tab_about = st.tabs(["Data tables", "Bar chart", "About the metrics"])

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
                    "dupes_within_vh", "dupes_within_curate"]
    pct_config = {PCT_COL: st.column_config.NumberColumn(PCT_LABEL, format="%.1f%%")}
    for i, defn in enumerate(definitions):
        sub = filtered[filtered["definition"] == defn][display_cols].reset_index(drop=True)
        with st.expander(f"{defn}  —  {len(sub):,} rows", expanded=(i == 0)):
            st.dataframe(sub, width="stretch", hide_index=True, height=360, column_config=pct_config)

# ================================================================ Tab 2: bar chart
with tab_chart:
    st.subheader("Top 20 counties by metric")

    c1, c2, c3 = st.columns([2, 2, 1])
    sel_def = c1.selectbox("Duplicate definition", definitions, key="chart_def")
    metric = c2.selectbox(
        "Rank by (descending)",
        list(RANKABLE),
        format_func=lambda c: RANKABLE[c],
        key="chart_metric",
    )
    min_total = c3.number_input(
        "Min. total alerts", min_value=1, value=1, step=1, key="chart_min_total",
        help="Ignore counties with fewer than this many alerts. Useful when ranking by "
             "share % so low-volume counties sitting at 100% don't dominate the top 20.",
    )

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
        top = chart_df.sort_values(metric, ascending=False).head(20)

        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric}:Q", title=RANKABLE[metric]),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color("source:N", title="Source"),
                tooltip=[
                    alt.Tooltip("county:N"),
                    alt.Tooltip("state:N"),
                    alt.Tooltip("source:N"),
                    *[alt.Tooltip(f"{c}:Q", title=lbl) for c, lbl in COUNT_LABELS.items()],
                    alt.Tooltip(f"{PCT_COL}:Q", title=PCT_LABEL, format=".1f"),
                ],
            )
            .properties(height=max(300, 26 * len(top)))
        )
        st.altair_chart(chart, width="stretch")

        with st.expander("Show the top-20 rows as a table"):
            st.dataframe(
                top[["county", "state", "source", "total_alerts", "dupes_between_vh_curate",
                     PCT_COL, "dupe_alerts_received_first", "dupe_alerts_received_same_date",
                     "dupes_within_vh", "dupes_within_curate"]],
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
| **Same post_date + milestone_type** | the same `post_date` **and** `milestone_type` |
| **Same meeting_date + milestone_type** | the same `meeting_date` **and** `milestone_type` |

`post_date` is when the alert was published; `meeting_date` is when the actual
event occurs.
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
| **dupes_within_vh** | Voterheads alerts whose dupe-key repeats **within Voterheads**. Always 0 on Curate rows. |
| **dupes_within_curate** | Curate alerts whose dupe-key repeats **within Curate**. Always 0 on Voterheads rows. |
        """
    )

    st.markdown("#### Notes & caveats")
    st.markdown(
        """
- The three dupe columns are **independent comparison types**, so a single alert can be
  counted in more than one (e.g. a repeated URL that also appears in the other source).
- **`dupe_alerts_received_first` / `dupe_alerts_received_same_date` are counted per cross-source
  *event*** (one point per duplicated event), whereas `dupes_between_vh_curate` counts individual
  alerts. Cross-source dupes are almost always one Voterheads row to one Curate row, so the two
  line up in practice. When a definition already includes `post_date` in its key (e.g.
  *Same URL + post_date*), every cross-source match is a tie by construction, so
  `received_first` is 0 and everything lands in `received_same_date`.
- **`pct_dupes_between_vh_curate` is sensitive to volume**: a county with 1 alert that
  happens to overlap reads as 100%. On the Bar chart tab, raise **Min. total alerts** to
  focus on counties with enough volume for the share to be meaningful.
- Rows whose dupe-key contains a **null** (e.g. a missing `meeting_date` or
  `milestone_type`) are excluded from duplicate consideration but still count toward
  `total_alerts` — an unknown value isn't treated as a confident match.
- Rows with a **null county** are dropped entirely, since county/state is the grouping key.
        """
    )
