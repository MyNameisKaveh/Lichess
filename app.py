# -*- coding: utf-8 -*-
# =============================================
# Streamlit App for Chess Game Analysis - Lichess API Version
# v4: Single Perf Type selection (selectbox), Removed "All Time".
# =============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests # For Lichess API calls
import json     # For parsing NDJSON response
from datetime import datetime, timedelta, timezone # For time period calculations
import time     # For potential delays and timestamp conversion
import re       # For cleaning names and parsing
import traceback # For printing full tracebacks during debugging

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Lichess Insights",
    page_icon="♟️"
)

# --- Constants & Defaults ---
# Updated Time Periods (Removed "All Time")
TIME_PERIOD_OPTIONS = {
    "Last Month": timedelta(days=30),
    "Last 3 Months": timedelta(days=90),
    "Last Year": timedelta(days=365),
    # "All Time": None # Removed for performance reasons on free tier
}
DEFAULT_TIME_PERIOD = "Last Year"

# Define available performance types for single selection
PERF_TYPE_OPTIONS_SINGLE = ['Bullet', 'Blitz', 'Rapid'] # Limited options
DEFAULT_PERF_TYPE = 'Bullet' # Defaulting to Bullet as requested

DEFAULT_RATED_ONLY = True

# =============================================
# Helper Function: Categorize Time Control (Corrected)
# =============================================
def categorize_time_control(tc_str, speed_info):
    # Prioritize speed info if available and standard
    if isinstance(speed_info, str) and speed_info in ['bullet', 'blitz', 'rapid', 'classical', 'correspondence']:
        return speed_info.capitalize()
    # Fallback parsing (less likely needed if filtering by perf type)
    if not isinstance(tc_str, str) or tc_str in ['-', '?', 'Unknown']: return 'Unknown'
    if tc_str == 'Correspondence': return 'Correspondence'
    if '+' in tc_str:
        try:
            parts = tc_str.split('+'); base = int(parts[0]); increment = int(parts[1]) if len(parts) > 1 else 0
            total = base + 40 * increment
            if total >= 1500: return 'Classical';
            if total >= 480: return 'Rapid';
            if total >= 180: return 'Blitz';
            if total > 0 : return 'Bullet';
            return 'Unknown'
        except (ValueError, IndexError): return 'Unknown'
    else:
        try:
            base = int(tc_str)
            if base >= 1500: return 'Classical';
            if base >= 480: return 'Rapid';
            if base >= 180: return 'Blitz';
            if base > 0 : return 'Bullet';
            return 'Unknown'
        except ValueError: return 'Unknown' # Reduced fallback keywords as speed info is primary

# =============================================
# API Data Loading and Processing Function (Now accepts single perf_type string)
# =============================================
@st.cache_data(ttl=3600) # Cache based on username, time_period, perf_type, rated
def load_from_lichess_api(username: str, time_period_key: str, perf_type: str, rated: bool):
    """ Fetches and processes Lichess games for a specific performance type. """
    if not username: st.warning("Please enter a Lichess username."); return pd.DataFrame()
    if not perf_type: st.warning("Please select a game type."); return pd.DataFrame()

    username_lower = username.lower()
    # Display selected perf type in the info message
    st.info(f"Fetching games for '{username}' ({time_period_key} | Type: {perf_type})...")

    since_timestamp_ms = None
    time_delta = TIME_PERIOD_OPTIONS.get(time_period_key) # Can be None if key removed, handle this later if needed
    if time_delta:
        start_date = datetime.now(timezone.utc) - time_delta
        since_timestamp_ms = int(start_date.timestamp() * 1000)
        st.caption(f"Fetching games since: {start_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    # No "All Time" caption needed anymore

    # --- Prepare API Parameters (using single perf_type) ---
    api_params = {
        "rated": str(rated).lower(),
        "perfType": perf_type.lower(), # Pass the single type directly
        "opening": "true", "moves": "false", "tags": "false", "pgnInJson": "false",
    }
    if since_timestamp_ms: api_params["since"] = since_timestamp_ms

    api_url = f"https://lichess.org/api/games/user/{username}"
    headers = {"Accept": "application/x-ndjson"}
    all_games_data = []
    processed_game_counter = 0
    error_counter = 0
    games_processed_for_log = 0

    try:
        with st.spinner(f"Calling Lichess API for {username} ({perf_type} games)..."):
            response = requests.get(api_url, params=api_params, headers=headers, stream=True)
            response.raise_for_status()

            # st.write("DEBUG: Starting to process game data stream...") # Keep minimal logs now
            for line in response.iter_lines():
                if line:
                    game_data_raw = line.decode('utf-8')
                    games_processed_for_log += 1
                    game_data = None
                    try:
                        game_data = json.loads(game_data_raw)

                        # --- Start Data Extraction (Robust) ---
                        # (Extraction logic remains largely the same as v3)
                        white_player_info = game_data.get('players', {}).get('white', {})
                        black_player_info = game_data.get('players', {}).get('black', {})
                        white_user = white_player_info.get('user', {}) if white_player_info else {}
                        black_user = black_player_info.get('user', {}) if black_player_info else {}
                        opening_info = game_data.get('opening', {})
                        clock_info = game_data.get('clock')

                        game_id = game_data.get('id', 'N/A')
                        created_at_ms = game_data.get('createdAt')
                        game_date = pd.to_datetime(created_at_ms, unit='ms', utc=True, errors='coerce')
                        if pd.isna(game_date): continue

                        variant = game_data.get('variant', 'standard')
                        speed = game_data.get('speed', 'unknown')
                        perf = game_data.get('perf', 'unknown') # Actual perf type from game data
                        status = game_data.get('status', 'unknown')
                        winner = game_data.get('winner')

                        white_name = white_user.get('name', 'Unknown')
                        black_name = black_user.get('name', 'Unknown')
                        white_title = white_user.get('title')
                        black_title = black_user.get('title')
                        white_rating = pd.to_numeric(white_player_info.get('rating'), errors='coerce')
                        black_rating = pd.to_numeric(black_player_info.get('rating'), errors='coerce')

                        player_color, player_elo, opp_name_raw, opp_title_raw, opp_elo = (None, None, 'Unknown', None, None)
                        if username_lower == white_name.lower():
                            player_color, player_elo, opp_name_raw, opp_title_raw, opp_elo = ('White', white_rating, black_name, black_title, black_rating)
                        elif username_lower == black_name.lower():
                            player_color, player_elo, opp_name_raw, opp_title_raw, opp_elo = ('Black', black_rating, white_name, white_title, white_rating)
                        else: continue

                        if player_color is None or pd.isna(player_elo) or pd.isna(opp_elo): continue

                        res_num, res_str = (0.5, "Draw")
                        if status not in ['draw', 'stalemate']:
                           if winner == player_color.lower(): res_num, res_str = (1, "Win")
                           elif winner is not None: res_num, res_str = (0, "Loss")

                        tc_str = "Unknown"
                        if clock_info:
                            init = clock_info.get('initial'); incr = clock_info.get('increment')
                            if init is not None and incr is not None: tc_str = f"{init}+{incr}"
                        elif speed == 'correspondence': tc_str = "Correspondence"

                        eco = opening_info.get('eco', 'Unknown')
                        op_name = opening_info.get('name', 'Unknown Opening').replace('?', '').split(':')[0].strip()

                        term_map = {"mate":"Normal", "resign":"Normal", "stalemate":"Normal", "timeout":"Time forfeit", "draw":"Normal", "outoftime":"Time forfeit", "cheat":"Cheat", "noStart":"Aborted", "unknownFinish":"Unknown", "variantEnd":"Variant End"}
                        term = term_map.get(status, "Unknown")

                        opp_title_final = 'Unknown'
                        if opp_title_raw and opp_title_raw.strip():
                            opp_title_clean = opp_title_raw.replace(' ','').strip().upper()
                            if opp_title_clean and opp_title_clean!='?': opp_title_final = opp_title_clean

                        def clean_name(n): return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM)\s+','',n).strip()
                        opp_name_clean = clean_name(opp_name_raw)
                        # --- End Data Extraction ---

                        # Filter by perf type again just to be sure (API might sometimes include others?)
                        # Or rely solely on API filter which should be sufficient
                        # if perf != perf_type.lower(): continue # Optional strict check

                        game_processed_data = {
                            'Date': game_date, 'Event': perf, 'White': white_name, 'Black': black_name,
                            'Result': "1-0" if winner=='white' else ("0-1" if winner=='black' else "1/2-1/2"),
                            'WhiteElo': int(white_rating) if not pd.isna(white_rating) else 0,
                            'BlackElo': int(black_rating) if not pd.isna(black_rating) else 0,
                            'ECO': eco, 'Opening': op_name, 'TimeControl': tc_str, 'Termination': term,
                            'PlyCount': game_data.get('turns',0), 'LichessID': game_id, 'PlayerID': username,
                            'PlayerColor': player_color, 'PlayerElo': int(player_elo),
                            'OpponentName': opp_name_clean, 'OpponentNameRaw': opp_name_raw,
                            'OpponentElo': int(opp_elo), 'OpponentTitle': opp_title_final,
                            'PlayerResultNumeric': res_num, 'PlayerResultString': res_str,
                            'Variant': variant, 'Speed': speed, 'Status': status, 'PerfType': perf
                        }
                        all_games_data.append(game_processed_data)
                        processed_game_counter += 1

                    except json.JSONDecodeError as e_json: error_counter += 1
                    except Exception as e_proc: error_counter += 1

            # st.write(f"DEBUG: Finished processing stream...") # Minimal logs

    except requests.exceptions.HTTPError as e_http:
        st.error(f"🚨 API Request Failed: {e_http.response.status_code} ({e_http.response.reason}). Check username or Lichess API status.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e_req:
        st.error(f"🚨 API Request Failed (Network Error): {e_req}")
        return pd.DataFrame()
    except Exception as e_outer:
         st.error(f"🚨 An unexpected error occurred: {e_outer}")
         st.text(traceback.format_exc()) # Still log unexpected errors fully
         return pd.DataFrame()

    if error_counter > 0:
         st.warning(f"Skipped {error_counter} entries due to processing errors during stream.")

    if not all_games_data:
        st.warning(f"No games found for '{username}' matching the criteria (Period: {time_period_key}, Type: {perf_type}, Rated: {rated}).")
        return pd.DataFrame()

    df = pd.DataFrame(all_games_data)
    st.success(f"Successfully processed {len(df)} games.")

    # --- Final Feature Engineering ---
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        if df.empty: return df

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['PlayerElo'] = df['PlayerElo'].astype(int)
        df['OpponentElo'] = df['OpponentElo'].astype(int)
        df['EloDiff'] = df['PlayerElo'] - df['OpponentElo']
        df['TimeControl_Category'] = df.apply(lambda row: categorize_time_control(row['TimeControl'], row['Speed']), axis=1)
        df = df.rename(columns={'Opening': 'OpeningName'})
        df = df.sort_values(by='Date').reset_index(drop=True)

    return df
# === End of load_from_lichess_api function ===


# =============================================
# Plotting Functions (Assumed Correct - Same as v3)
# =============================================
# (Insert ALL plotting functions here - plot_win_loss_pie, plot_win_loss_by_color, etc.)
# ... (Code is identical to the previous version) ...
def plot_win_loss_pie(df, display_name):
    if 'PlayerResultString' not in df.columns: return go.Figure()
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values, names=result_counts.index,
                 title=f'Overall Win/Loss/Draw Distribution for {display_name}',
                 color=result_counts.index, color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05 if x == 'Win' else 0 for x in result_counts.index])
    return fig

def plot_win_loss_by_color(df):
    if not all(col in df.columns for col in ['PlayerColor', 'PlayerResultString']): return go.Figure()
    try:
        color_results = df.groupby(['PlayerColor', 'PlayerResultString']).size().unstack(fill_value=0)
        for res in ['Win', 'Draw', 'Loss']:
            if res not in color_results.columns: color_results[res] = 0
        color_results = color_results[['Win', 'Draw', 'Loss']]
        total_per_color = color_results.sum(axis=1)
        color_results_pct = color_results.apply(lambda x: x * 100 / total_per_color[x.name] if total_per_color[x.name] > 0 else 0, axis=1)
        fig = px.bar(color_results_pct, barmode='stack', title='Win/Loss/Draw Percentage by Color',
                     labels={'value': 'Percentage (%)', 'PlayerColor': 'Played As', 'PlayerResultString': 'Result'},
                     color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                     text_auto='.1f', category_orders={"PlayerColor": ["White", "Black"]})
        fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="Color Played")
        fig.update_traces(textangle=0)
        return fig
    except Exception as e:
        st.error(f"Error creating Win/Loss by Color plot: {e}")
        return go.Figure().update_layout(title="Error generating plot")

def plot_rating_trend(df, display_name):
    if not all(col in df.columns for col in ['Date', 'PlayerElo']): return go.Figure()
    df_plot = df.copy()
    df_plot['PlayerElo'] = pd.to_numeric(df_plot['PlayerElo'], errors='coerce')
    df_sorted = df_plot[df_plot['PlayerElo'].notna() & (df_plot['PlayerElo'] > 0)].sort_values('Date')
    if df_sorted.empty: return go.Figure().update_layout(title=f"No valid Elo data for {display_name}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['PlayerElo'], mode='lines+markers', name='Elo Rating',
                        line=dict(color='#1E88E5', width=2), marker=dict(color='#1E88E5', size=5, opacity=0.7), hoverinfo='x+y'))
    fig.update_layout(title=f'{display_name}\'s Rating Trend Over Time', xaxis_title='Date', yaxis_title='Elo Rating',
                      hovermode="x unified", xaxis_rangeslider_visible=True)
    return fig

def plot_performance_vs_opponent_elo(df):
    if not all(col in df.columns for col in ['PlayerResultString', 'EloDiff']): return go.Figure()
    fig = px.box(df, x='PlayerResultString', y='EloDiff', title='Player\'s Elo Advantage vs. Game Result',
                 labels={'PlayerResultString': 'Game Result', 'EloDiff': 'Player Elo - Opponent Elo'},
                 category_orders={"PlayerResultString": ["Win", "Draw", "Loss"]},
                 color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, points='outliers')
    fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Equal Elo", annotation_position="bottom right")
    fig.update_traces(marker=dict(opacity=0.8))
    return fig

def plot_games_per_year(df):
    if 'Year' not in df.columns: return go.Figure()
    games_per_year = df['Year'].value_counts().sort_index()
    fig = px.bar(games_per_year, x=games_per_year.index, y=games_per_year.values,
                 title='Number of Games Played Per Year', labels={'x': 'Year', 'y': 'Number of Games'}, text=games_per_year.values)
    fig.update_traces(marker_color='#2196F3', textposition='outside')
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games", xaxis={'type': 'category'})
    return fig

def plot_win_rate_per_year(df):
    if not all(col in df.columns for col in ['Year', 'PlayerResultNumeric']): return go.Figure()
    wins_per_year = df[df['PlayerResultNumeric'] == 1].groupby('Year').size()
    total_per_year = df.groupby('Year').size()
    win_rate = (wins_per_year.reindex(total_per_year.index, fill_value=0) / total_per_year).fillna(0) * 100
    win_rate.index = win_rate.index.astype(str)
    fig = px.line(win_rate, x=win_rate.index, y=win_rate.values, title='Win Rate (%) Per Year', markers=True,
                  labels={'x': 'Year', 'y': 'Win Rate (%)'})
    fig.update_traces(line_color='#FFC107', line_width=2.5)
    fig.update_layout(yaxis_range=[0, 100])
    return fig

def plot_performance_by_time_control(df):
     if not all(col in df.columns for col in ['TimeControl_Category', 'PlayerResultString']): return go.Figure()
     try:
        tc_results = df.groupby(['TimeControl_Category', 'PlayerResultString']).size().unstack(fill_value=0)
        for res in ['Win', 'Draw', 'Loss']:
            if res not in tc_results.columns: tc_results[res] = 0
        tc_results = tc_results[['Win', 'Draw', 'Loss']]
        total_per_tc = tc_results.sum(axis=1)
        tc_results_pct = tc_results.apply(lambda x: x * 100 / total_per_tc[x.name] if total_per_tc[x.name] > 0 else 0, axis=1)
        found_categories = df['TimeControl_Category'].unique()
        cat_order_preferred = ['Bullet', 'Blitz', 'Rapid', 'Classical', 'Correspondence', 'Unknown']
        cat_order = [cat for cat in cat_order_preferred if cat in found_categories] + \
                    [cat for cat in found_categories if cat not in cat_order_preferred]
        tc_results_pct = tc_results_pct.reindex(index=cat_order).dropna(axis=0, how='all')
        fig = px.bar(tc_results_pct, title='Performance by Time Control Category',
                     labels={'value': 'Percentage (%)', 'TimeControl_Category': 'Time Control', 'PlayerResultString':'Result'},
                     color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                     barmode='group', text_auto='.1f')
        fig.update_layout(xaxis_title="Time Control Category", yaxis_title="Percentage (%)")
        fig.update_traces(textangle=0)
        return fig
     except Exception as e:
        st.error(f"Error creating Performance by Time Control plot: {e}")
        return go.Figure().update_layout(title="Error generating plot")

def plot_opening_frequency(df, top_n=20):
    if 'OpeningName' not in df.columns: return go.Figure()
    opening_counts = df[df['OpeningName'] != 'Unknown Opening']['OpeningName'].value_counts().nlargest(top_n)
    fig = px.bar(opening_counts, y=opening_counts.index, x=opening_counts.values, orientation='h',
                 title=f'Top {top_n} Most Frequent Openings Played', labels={'y': 'Opening Name', 'x': 'Number of Games'}, text=opening_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_traces(marker_color='#673AB7', textposition='outside')
    return fig

def plot_win_rate_by_opening(df, min_games=5, top_n=20):
    if not all(col in df.columns for col in ['OpeningName', 'PlayerResultNumeric']): return go.Figure()
    opening_stats = df.groupby('OpeningName').agg(
        total_games=('PlayerResultNumeric', 'count'), wins=('PlayerResultNumeric', lambda x: (x == 1).sum()))
    opening_stats = opening_stats[(opening_stats['total_games'] >= min_games) & (opening_stats.index != 'Unknown Opening')].copy()
    if opening_stats.empty: return go.Figure().update_layout(title=f"No openings played >= {min_games} times")
    opening_stats['win_rate'] = (opening_stats['wins'] / opening_stats['total_games']) * 100
    opening_stats_plot = opening_stats.nlargest(top_n, 'win_rate')
    fig = px.bar(opening_stats_plot, y=opening_stats_plot.index, x='win_rate', orientation='h',
                 title=f'Top {top_n} Openings by Win Rate (Played >= {min_games} times)',
                 labels={'win_rate': 'Win Rate (%)', 'OpeningName': 'Opening'}, text='win_rate')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside', marker_color='#009688')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Win Rate (%)", yaxis_title="Opening Name")
    return fig

def plot_most_frequent_opponents(df, top_n=20):
    if 'OpponentName' not in df.columns: return go.Figure()
    opp_counts = df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().nlargest(top_n)
    fig = px.bar(opp_counts, y=opp_counts.index, x=opp_counts.values, orientation='h',
                 title=f'Top {top_n} Most Frequent Opponents', labels={'y': 'Opponent Name', 'x': 'Number of Games'}, text=opp_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_traces(marker_color='#FF5722', textposition='outside')
    return fig

# =============================================
# Helper Functions (Assumed Correct)
# =============================================
def filter_and_analyze_gms(df):
    if 'OpponentTitle' not in df.columns: return pd.DataFrame()
    gm_games = df[df['OpponentTitle'] == 'GM'].copy()
    return gm_games

def filter_and_analyze_time_forfeits(df):
    if 'Termination' not in df.columns: return pd.DataFrame(), 0, 0
    tf_games = df[df['Termination'].str.contains("Time forfeit", na=False, case=False)].copy()
    if tf_games.empty: return tf_games, 0, 0
    wins_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 1])
    losses_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 0])
    return tf_games, wins_tf, losses_tf

# =============================================
# Streamlit App Layout - API Version with Single Perf Type Select
# =============================================

st.title("♟️ Lichess Insights")
st.write("Analyze rated game statistics from the Lichess API.")

# --- Input Area ---
col1, col2, col3 = st.columns([2, 1, 1]) # Adjusted column ratios
with col1:
    lichess_username = st.text_input("Lichess Username:", key="username_input", placeholder="e.g., DrNykterstein")
with col2:
    time_period = st.selectbox("Time Period:", options=list(TIME_PERIOD_OPTIONS.keys()), # Use updated options
                               index=list(TIME_PERIOD_OPTIONS.keys()).index(DEFAULT_TIME_PERIOD) if DEFAULT_TIME_PERIOD in TIME_PERIOD_OPTIONS else 0, # Handle if default removed
                               key="time_period_select")
with col3:
    # Use selectbox for single performance type selection
    selected_perf_type = st.selectbox(
        "Game Type:",
        options=PERF_TYPE_OPTIONS_SINGLE, # Use limited options
        index=PERF_TYPE_OPTIONS_SINGLE.index(DEFAULT_PERF_TYPE), # Default to Bullet
        key="perf_type_select"
    )

analyze_button = st.button("Analyze Games", key="analyze_button")

# --- Data Loading and Analysis Area ---
if 'analysis_df' not in st.session_state: st.session_state.analysis_df = None
if 'current_username' not in st.session_state: st.session_state.current_username = ""
if 'current_time_period' not in st.session_state: st.session_state.current_time_period = ""
# Add state for single performance type
if 'current_perf_type' not in st.session_state: st.session_state.current_perf_type = ""

if analyze_button and lichess_username:
    # Check if any input has changed
    if (lichess_username != st.session_state.current_username or
            time_period != st.session_state.current_time_period or
            selected_perf_type != st.session_state.current_perf_type): # Compare single string

        if not selected_perf_type:
             st.warning("Internal error: No game type selected.") # Should not happen with selectbox
        else:
            # st.write("DEBUG: Analyze button clicked. New analysis needed.") # Keep logs minimal now
            st.session_state.analysis_df = None
            if 'selected_section' in st.session_state: del st.session_state['selected_section']

            # Pass the single selected perf type to the API function
            df_loaded = load_from_lichess_api(
                lichess_username,
                time_period,
                selected_perf_type, # Pass the single string
                DEFAULT_RATED_ONLY
            )

            st.session_state.analysis_df = df_loaded
            st.session_state.current_username = lichess_username
            st.session_state.current_time_period = time_period
            st.session_state.current_perf_type = selected_perf_type # Store selected type string
            st.rerun() # Rerun to display results/errors cleanly

    else:
        st.info("Analysis results for these settings are already displayed.")


# --- Display Results ---
if isinstance(st.session_state.analysis_df, pd.DataFrame) and not st.session_state.analysis_df.empty:
    df = st.session_state.analysis_df
    current_display_name = st.session_state.current_username
    current_perf_type = st.session_state.current_perf_type # Get type from state

    st.success(f"Displaying analysis for **{current_display_name}** ({st.session_state.current_time_period})")
    # Update caption to show single selected perf type
    st.caption(f"Game Type: **{current_perf_type.capitalize()}** | Total Rated Games Analyzed: **{len(df):,}**")
    st.markdown("---")

    st.sidebar.title("📊 Analysis Sections")
    analysis_options = [ "Overview", "Time and Date Analysis", "ECO and Opening Analysis",
                         "Opponent Analysis", "Games against GMs", "Time Forfeit Analysis" ]
    if 'selected_section' not in st.session_state: st.session_state.selected_section = "Overview"
    selected_section = st.sidebar.radio( "Choose a section:", analysis_options,
         index=analysis_options.index(st.session_state.selected_section), key="section_radio")
    st.session_state.selected_section = selected_section

    # --- Display Content Based on Selected Section ---
    # (Plotting sections remain the same)
    if selected_section == "Overview":
        st.header("📈 General Overview")
        col_ov1, col_ov2 = st.columns(2);
        with col_ov1: st.plotly_chart(plot_win_loss_pie(df, current_display_name), use_container_width=True)
        with col_ov2: st.plotly_chart(plot_win_loss_by_color(df), use_container_width=True)
        st.plotly_chart(plot_rating_trend(df, current_display_name), use_container_width=True)
        st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)

    elif selected_section == "Time and Date Analysis":
        st.header("📅 Time and Date Analysis")
        tab_time1, tab_time2, tab_time3 = st.tabs(["Games Over Time", "Performance by Year", "Performance by Time Control"])
        with tab_time1: st.plotly_chart(plot_games_per_year(df), use_container_width=True)
        with tab_time2: st.plotly_chart(plot_win_rate_per_year(df), use_container_width=True)
        with tab_time3:
            st.plotly_chart(plot_performance_by_time_control(df), use_container_width=True)
            st.markdown("#### Perf. Data by Time Control")
            try: st.dataframe(df.groupby('TimeControl_Category')['PlayerResultString'].value_counts().unstack(fill_value=0))
            except KeyError: st.warning("Could not generate time control summary table.")

    elif selected_section == "ECO and Opening Analysis":
        st.header("📖 ECO and Opening Analysis")
        tab_eco1, tab_eco2 = st.tabs(["Opening Frequency", "Opening Performance"])
        with tab_eco1:
            n_openings = st.slider("Num top openings:", 5, 50, 20, key="n_openings_freq")
            st.plotly_chart(plot_opening_frequency(df, top_n=n_openings), use_container_width=True)
            st.markdown(f"#### Top {n_openings} Frequencies")
            try: st.dataframe(df[df['OpeningName'] != 'Unknown Opening']['OpeningName'].value_counts().reset_index(name='Count').head(n_openings))
            except KeyError: st.warning("Could not generate opening frequency table.")
        with tab_eco2:
            min_games_opening = st.slider("Min games for perf.:", 1, 25, 5, key="min_games_perf")
            n_openings_perf = st.slider("Num top openings by win rate:", 5, 50, 20, key="n_openings_perf")
            st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_opening, top_n=n_openings_perf), use_container_width=True)

    elif selected_section == "Opponent Analysis":
        st.header("👥 Opponent Analysis")
        tab_opp1, tab_opp2 = st.tabs(["Most Frequent Opponents", "Elo Difference vs Result"])
        with tab_opp1:
            n_opponents = st.slider("Num top opponents:", 5, 50, 20, key="n_opponents_freq")
            st.plotly_chart(plot_most_frequent_opponents(df, top_n=n_opponents), use_container_width=True)
            st.markdown(f"#### Top {n_opponents} Opponents")
            try: st.dataframe(df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().reset_index(name='Games').head(n_opponents))
            except KeyError: st.warning("Could not generate opponent frequency table.")
        with tab_opp2: st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)

    elif selected_section == "Games against GMs":
        st.header("👑 Analysis Against Grandmasters (GMs)")
        gm_games = filter_and_analyze_gms(df)
        if not gm_games.empty:
            st.success(f"Found **{len(gm_games):,}** games vs 'GM'. Analyzing subset...")
            tab_gm1, tab_gm2, tab_gm3, tab_gm4 = st.tabs(["🏆 Summary", "📈 Rating Trend", "📖 Openings", "👥 Opponents"])
            # ... (Code inside GM tabs remains the same) ...
            with tab_gm1:
                 st.plotly_chart(plot_win_loss_pie(gm_games, f"{current_display_name} vs GMs"), use_container_width=True)
                 st.markdown("#### Results vs GMs:"); st.dataframe(gm_games['PlayerResultString'].value_counts().reset_index(name='Count'))
                 st.plotly_chart(plot_win_loss_by_color(gm_games), use_container_width=True)
            with tab_gm2: st.plotly_chart(plot_rating_trend(gm_games, f"{current_display_name} (vs GMs)"), use_container_width=True)
            with tab_gm3:
                 st.plotly_chart(plot_opening_frequency(gm_games, top_n=15), use_container_width=True)
                 min_games_gm = st.slider("Min games (GM opening):", 1, 10, 3, key="gm_open_slider")
                 st.plotly_chart(plot_win_rate_by_opening(gm_games, min_games=min_games_gm, top_n=15), use_container_width=True)
            with tab_gm4:
                st.plotly_chart(plot_most_frequent_opponents(gm_games, top_n=15), use_container_width=True)
                st.markdown("#### Frequent GM Opponents:"); st.dataframe(gm_games['OpponentName'].value_counts().reset_index(name='Games').head(15))

        else: st.warning("ℹ️ No games found vs 'GM' title.")

    elif selected_section == "Time Forfeit Analysis":
        st.header("⏱️ Time Forfeit Analysis")
        tf_games, wins_tf, losses_tf = filter_and_analyze_time_forfeits(df)
        if not tf_games.empty:
            st.success(f"Found **{len(tf_games):,}** games ending due to time forfeit.")
            col_tf1, col_tf2 = st.columns(2); col_tf1.metric("Won on Time", wins_tf); col_tf2.metric("Lost on Time", losses_tf)
            st.markdown("#### Games Ending in Time Forfeit (Recent):")
            st.dataframe(tf_games[['Date','OpponentName','PlayerColor','PlayerResultString','TimeControl','PlyCount','Termination']].sort_values('Date',ascending=False).head(50))
            st.markdown("#### Forfeits by Time Control:"); st.dataframe(tf_games['TimeControl_Category'].value_counts().reset_index(name='Count'))
        else: st.warning("ℹ️ No games found with 'Time forfeit' termination.")


    st.sidebar.markdown("---")
    st.sidebar.info(f"Analysis for {current_display_name}. Using Lichess API.")

elif not analyze_button and st.session_state.analysis_df is None:
     st.info("☝️ Enter Lichess username, select period & game type, then click 'Analyze Games'.")
elif analyze_button and (not isinstance(st.session_state.analysis_df, pd.DataFrame) or st.session_state.analysis_df.empty):
     st.warning("No analysis data generated. Check username or try different settings. Review logs if errors occurred.")

# --- End of App ---
