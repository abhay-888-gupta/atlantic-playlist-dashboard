import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Atlantic US Top 50 Playlist Analytics", layout="wide")

# ---------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_playlist_data.csv", parse_dates=["date"])
    songs = pd.read_csv("song_level_features.csv", parse_dates=["first_seen", "last_seen"])
    artists = pd.read_csv("artist_level_features.csv")
    return df, songs, artists

df, songs, artists = load_data()

st.title("🎵 US Top 50 Playlist Performance & Song Popularity Trend Analysis")
st.caption("Atlantic Recording Corporation — Historical Playlist Analytics")

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.header("Filters")

min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

rank_range = st.sidebar.slider("Rank range", 1, 50, (1, 50))

artist_options = sorted(df["artist"].unique())
selected_artists = st.sidebar.multiselect("Artist(s)", artist_options)

song_options = sorted(df["song"].unique())
selected_songs = st.sidebar.multiselect("Song(s)", song_options)

album_type_options = sorted(df["album_type"].unique())
selected_album_types = st.sidebar.multiselect(
    "Album type", album_type_options, default=list(album_type_options)
)

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (df["date"] >= pd.Timestamp(start_date))
    & (df["date"] <= pd.Timestamp(end_date))
    & (df["position"].between(rank_range[0], rank_range[1]))
    & (df["album_type"].isin(selected_album_types))
)
if selected_artists:
    mask &= df["artist"].isin(selected_artists)
if selected_songs:
    mask &= df["song"].isin(selected_songs)

filtered = df[mask]

st.markdown(f"**{filtered.shape[0]:,} rows** match current filters "
            f"({filtered['song'].nunique()} songs, {filtered['artist'].nunique()} artists)")

# ---------------------------------------------------------------
# TABS = 5 CORE MODULES
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Playlist Timeline Explorer",
    "📈 Song Ranking Trends",
    "🏆 Artist Dominance Leaderboard",
    "🔀 Popularity vs Rank",
    "🔞 Explicit vs Non-Explicit",
])

# --- Module 1: Playlist Timeline Explorer ---
with tab1:
    st.subheader("Playlist Timeline Explorer")
    daily_avg = filtered.groupby("date")["position"].mean().reset_index()
    fig = px.line(daily_avg, x="date", y="position",
                  title="Average Chart Position Over Time (lower = more concentrated at top)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Daily snapshot (pick a date to inspect the full Top 50):**")
    snap_date = st.date_input("Snapshot date", value=max_date, min_value=min_date, max_value=max_date)
    snapshot = df[df["date"] == pd.Timestamp(snap_date)].sort_values("position")
    st.dataframe(snapshot[["position", "song", "artist", "popularity", "album_type", "is_explicit"]],
                 use_container_width=True, hide_index=True)

# --- Module 2: Song Ranking Trend Charts ---
with tab2:
    st.subheader("Song Ranking Trend Charts")
    if selected_songs:
        trend_df = filtered[filtered["song"].isin(selected_songs)]
    else:
        top_songs_default = songs.sort_values("days_on_chart", ascending=False).head(5)["song"].tolist()
        trend_df = filtered[filtered["song"].isin(top_songs_default)]
        st.caption("No song selected in sidebar — showing top 5 longest-charting songs by default.")

    fig2 = px.line(trend_df, x="date", y="position", color="song", markers=True,
                    title="Rank Position Over Time")
    fig2.update_yaxes(autorange="reversed", title="Rank Position")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Song-level KPI table**")
    st.dataframe(
        songs.sort_values("days_on_chart", ascending=False)
             [["song", "artist", "days_on_chart", "average_rank", "best_rank",
               "rank_volatility_index", "avg_popularity"]],
        use_container_width=True, hide_index=True
    )

# --- Module 3: Artist Dominance Leaderboard ---
with tab3:
    st.subheader("Artist Dominance Leaderboard")
    top_n = st.slider("Show top N artists", 5, 50, 15)
    top_artists = artists.sort_values("artist_dominance_index", ascending=False).head(top_n)
    fig3 = px.bar(top_artists, x="artist_dominance_index", y="artist", orientation="h",
                  title="Artist Dominance Index", color="artist_dominance_index",
                  color_continuous_scale="Viridis")
    fig3.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        artists[["artist", "unique_songs", "total_days_charted", "average_rank",
                 "best_rank_ever", "avg_popularity", "artist_dominance_index"]],
        use_container_width=True, hide_index=True
    )

# --- Module 4: Popularity vs Rank Scatter ---
with tab4:
    st.subheader("Popularity vs Rank Scatter")
    sample = filtered.sample(min(3000, len(filtered)), random_state=1) if len(filtered) > 0 else filtered
    fig4 = px.scatter(sample, x="position", y="popularity", opacity=0.4,
                       color="album_type", hover_data=["song", "artist"],
                       title="Popularity Score vs Rank Position")
    st.plotly_chart(fig4, use_container_width=True)

    corr = filtered["popularity"].corr(filtered["position"]) if len(filtered) > 1 else np.nan
    st.metric("Correlation (popularity vs rank position)", f"{corr:.2f}" if pd.notna(corr) else "N/A")

    tier_pop = filtered.groupby("rank_tier")["popularity"].mean().reset_index()
    fig4b = px.bar(tier_pop, x="rank_tier", y="popularity", title="Average Popularity by Rank Tier")
    st.plotly_chart(fig4b, use_container_width=True)

# --- Module 5: Explicit vs Non-Explicit Performance Panels ---
with tab5:
    st.subheader("Explicit vs Non-Explicit Performance")
    col1, col2 = st.columns(2)

    explicit_perf = filtered.groupby("is_explicit").agg(
        avg_popularity=("popularity", "mean"),
        avg_position=("position", "mean"),
        count=("song", "count")
    ).reset_index()
    explicit_perf["is_explicit"] = explicit_perf["is_explicit"].map({True: "Explicit", False: "Non-Explicit"})

    with col1:
        fig5 = px.bar(explicit_perf, x="is_explicit", y="avg_popularity",
                      title="Average Popularity", color="is_explicit")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        fig6 = px.bar(explicit_perf, x="is_explicit", y="avg_position",
                      title="Average Rank Position (lower = better)", color="is_explicit")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Single vs Album performance**")
    album_perf = filtered.groupby("album_type").agg(
        avg_popularity=("popularity", "mean"),
        avg_position=("position", "mean"),
        count=("song", "count")
    ).reset_index()
    fig7 = px.bar(album_perf, x="album_type", y="avg_popularity", title="Average Popularity by Album Type")
    st.plotly_chart(fig7, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Atlantic Recording Corporation — Historical Playlist Analytics (not predictive)")