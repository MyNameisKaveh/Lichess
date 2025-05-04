# -*- coding: utf-8 -*-
# =============================================
# Streamlit App for Chess Game Analysis - Lichess API Version
# Fetches and analyzes games for a given Lichess username and time period.
# =============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests # For Lichess API calls
import json     # For parsing NDJSON response
from datetime import datetime, timedelta, timezone # For time period calculations
import time     # For potential delays and timestamp conversion

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Lichess Insights",
    page_icon="♟️"
)

# --- Constants & Defaults ---
# Default time period options and mapping to timedelta
TIME_PERIOD_OPTIONS = {
    "Last Month": timedelta(days=30),
    "Last 3 Months": timedelta(days=90),
    "Last Year": timedelta(days=365),
    "All Time": None # Special value for no time limit
}
DEFAULT_TIME_PERIOD = "Last Year"
DEFAULT_PERF_TYPES = ['blitz', 'rapid', 'classical'] # Case-insensitive in API? Check docs. Usually lowercase.
DEFAULT_RATED_ONLY = True

# =============================================
# API Data Loading and Processing Function
# =============================================
@st.cache_data(ttl=3600) # Cache API results for 1 hour
def load_from_lichess_api(username: str, time_period_key: str, perf_types: list[str], rated: bool):
    """
    Fetches games for a Lichess user from the API for a specific time period,
    processes the data, and returns a Pandas DataFrame.
    """
    if not username:
        st.warning("Please enter a Lichess username.")
        return pd.DataFrame()

    username_lower = username.lower() # Use lowercase for comparison
    st.info(f"Fetching games for '{username}' ({time_period_key})...")

    # --- Calculate 'since' timestamp ---
    since_timestamp_ms = None
    time_delta = TIME_PERIOD_OPTIONS.get(time_period_key)
    if time_delta:
        start_date = datetime.now(timezone.utc) - time_delta
        # Convert to milliseconds timestamp required by Lichess API
        since_timestamp_ms = int(start_date.timestamp() * 1000)
        st.caption(f"Fetching games since: {start_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        st.caption("Fetching all games (might take a while)...")


    # --- Prepare API Parameters ---
    api_params = {
        "rated": str(rated).lower(), # API expects 'true' or 'false' as strings
        "perfType": ",".join(perf_types).lower(), # Comma-separated list
        "opening": "true",
        "moves": "false", # We don't need moves for this analysis
        "tags": "false", # Basic JSON has most info we need
        "pgnInJson": "false", # Get structured JSON, not PGN text
        # "max": 2000, # Optional: Hard limit for safety, especially for "All Time"
    }
    if since_timestamp_ms:
        api_params["since"] = since_timestamp_ms

    # --- Make API Request ---
    api_url = f"https://lichess.org/api/games/user/{username}"
    headers = {"Accept": "application/x-ndjson"}
    all_games_data = []
    processed_game_counter = 0
    error_counter = 0

    try:
        with st.spinner(f"Calling Lichess API for {username}... (This might take time depending on the number of games)"):
            response = requests.get(api_url, params=api_params, headers=headers, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Process NDJSON Response Stream ---
            for line in response.iter_lines():
                if line: # Filter out keep-alive new lines
                    try:
                        game_data = json.loads(line.decode('utf-8'))

                        # --- Extract and Transform Data ---
                        white_player_info = game_data.get('players', {}).get('white', {})
                        black_player_info = game_data.get('players', {}).get('black', {})
                        white_user = white_player_info.get('user', {})
                        black_user = black_player_info.get('user', {})
                        opening_info = game_data.get('opening', {})
                        clock_info = game_data.get('clock') # Can be None for correspondence

                        # Basic Info
                        game_id = game_data.get('id')
                        created_at_ms = game_data.get('createdAt')
                        game_date = pd.to_datetime(created_at_ms, unit='ms', utc=True) if created_at_ms else pd.NaT
                        variant = game_data.get('variant', 'standard') # Default to standard if missing
                        speed = game_data.get('speed', 'unknown')
                        perf = game_data.get('perf', 'unknown')
                        status = game_data.get('status', 'unknown')
                        winner = game_data.get('winner') # 'white', 'black', or None for draw

                        # Player Info
                        white_name = white_user.get('name', 'Unknown')
                        black_name = black_user.get('name', 'Unknown')
                        white_title = white_user.get('title') # Can be None
                        black_title = black_user.get('title') # Can be None
                        white_rating = white_player_info.get('rating')
                        black_rating = black_player_info.get('rating')

                        # Determine user's color, Elo, opponent, etc.
                        player_color = None
                        player_elo = None
                        opponent_name_raw = 'Unknown'
                        opponent_title_raw = None
                        opponent_elo = None

                        if username_lower == white_name.lower():
                            player_color = 'White'
                            player_elo = white_rating
                            opponent_name_raw = black_name
                            opponent_title_raw = black_title
                            opponent_elo = black_rating
                        elif username_lower == black_name.lower():
                            player_color = 'Black'
                            player_elo = black_rating
                            opponent_name_raw = white_name
                            opponent_title_raw = white_title
                            opponent_elo = white_rating
                        else:
                             # Should not happen if API call is correct, but good to handle
                            continue

                        # Skip if essential player info missing
                        if player_color is None or player_elo is None or opponent_elo is None:
                            continue

                        # --- Determine Result ---
                        player_result_numeric = 0.5 # Default to Draw
                        player_result_string = "Draw"
                        if status not in ['draw', 'stalemate']: # Check if it's not a draw type
                           if winner == player_color.lower():
                               player_result_numeric = 1
                               player_result_string = "Win"
                           elif winner is not None: # If there's a winner, and it's not the player
                               player_result_numeric = 0
                               player_result_string = "Loss"
                           # else: keep draw (e.g., if status is aborted but winner is None)

                        # --- Time Control ---
                        time_control_str = "Unknown"
                        if clock_info:
                            initial = clock_info.get('initial', 0) // 60 # Initial time in minutes
                            increment = clock_info.get('increment', 0)   # Increment in seconds
                            time_control_str = f"{initial * 60}+{increment}" if initial is not None and increment is not None else "Unknown"
                        elif speed == 'correspondence':
                             time_control_str = "Correspondence"


                        # --- Opening Info ---
                        eco = opening_info.get('eco', 'Unknown')
                        opening_name = opening_info.get('name', 'Unknown Opening')
                        # Clean opening name (remove '?', handle variations)
                        opening_name = opening_name.replace('?', '').split(':')[0].strip() # Basic cleaning


                        # --- Termination ---
                        termination_map = {
                            "mate": "Normal", "resign": "Normal", "stalemate": "Normal",
                            "timeout": "Time forfeit", "draw": "Normal", "outoftime": "Time forfeit",
                            "cheat": "Cheat", "noStart": "Aborted", "unknownFinish": "Unknown",
                            "variantEnd": "Variant End" # For variants like KOTH
                        }
                        termination = termination_map.get(status, "Unknown")


                        # --- Clean Opponent Title ---
                        opponent_title_final = 'Unknown'
                        if opponent_title_raw and opponent_title_raw.strip():
                            opponent_title_clean = opponent_title_raw.replace(' ', '').strip().upper()
                            if opponent_title_clean and opponent_title_clean != '?':
                                opponent_title_final = opponent_title_clean

                         # --- Clean opponent names (remove titles like GM, IM etc.) ---
                        def clean_name(name):
                             return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM)\s+', '', name).strip()
                        opponent_name_clean = clean_name(opponent_name_raw)


                        # --- Store data ---
                        game_processed_data = {
                            'Date': game_date,
                            'Event': perf, # Use perf as event type? Or keep 'lichess'?
                            'White': white_name,
                            'Black': black_name,
                            'Result': f"{white_player_info.get('rating', '')}-{black_player_info.get('rating', '')}" if winner else "1/2-1/2", # Approximate PGN result? Or use status?
                            'WhiteElo': white_rating if white_rating else 0,
                            'BlackElo': black_rating if black_rating else 0,
                            'ECO': eco,
                            'Opening': opening_name, # Use name from API
                            'TimeControl': time_control_str,
                            'Termination': termination,
                            'PlyCount': game_data.get('turns', 0),

                            # --- Added/Derived Features ---
                            'LichessID': game_id,
                            'PlayerID': username, # The requested username
                            'PlayerColor': player_color,
                            'PlayerElo': player_elo,
                            'OpponentName': opponent_name_clean,
                            'OpponentNameRaw': opponent_name_raw,
                            'OpponentElo': opponent_elo,
                            'OpponentTitle': opponent_title_final,
                            'PlayerResultNumeric': player_result_numeric,
                            'PlayerResultString': player_result_string,
                            'Variant': variant,
                            'Speed': speed,
                            'Status': status
                        }
                        all_games_data.append(game_processed_data)
                        processed_game_counter += 1

                    except json.JSONDecodeError:
                        error_counter += 1
                        # st.warning(f"Skipping line due to JSON decode error: {line}")
                    except Exception as e:
                        error_counter +=1
                        # st.warning(f"Skipping game due to processing error: {e}")

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 API Request Failed: {e}")
        return pd.DataFrame()
    except Exception as e:
         st.error(f"🚨 An unexpected error occurred during API fetch or processing: {e}")
         return pd.DataFrame()

    if error_counter > 0:
         st.warning(f"Skipped {error_counter} games due to parsing/processing errors.")

    if not all_games_data:
        st.warning("No games matching the criteria were found for this user.")
        return pd.DataFrame()

    # --- Create DataFrame ---
    df = pd.DataFrame(all_games_data)
    st.success(f"Successfully fetched and processed {len(df)} games.")

    # --- Final Feature Engineering (Consistent with PGN version) ---
    if not df.empty:
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']) # Drop games where date couldn't be parsed

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.day_name()

        # Ensure Elos are numeric before calculating difference
        df['PlayerElo'] = pd.to_numeric(df['PlayerElo'], errors='coerce').fillna(0)
        df['OpponentElo'] = pd.to_numeric(df['OpponentElo'], errors='coerce').fillna(0)
        df['EloDiff'] = df['PlayerElo'] - df['OpponentElo']

        # --- Time Control Categorization ---
        # Reuse the function, might need adjustments based on API's TimeControl format
        def categorize_time_control(tc_str, speed_info):
            # Prioritize speed info if available and reliable
            if speed_info in ['bullet', 'blitz', 'rapid', 'classical', 'correspondence']:
                return speed_info.capitalize()

            # Fallback to parsing the time_control_str (e.g., "180+0")
            if not isinstance(tc_str, str) or tc_str == '-' or tc_str == '?' or tc_str == 'Unknown':
                 return 'Unknown'
            if tc_str == 'Correspondence': return 'Correspondence'

            if '+' in tc_str:
                try:
                    parts = tc_str.split('+')
                    base = int(parts[0])
                    increment = int(parts[1]) if len(parts) > 1 else 0
                    # Use Lichess's own categorization logic (approximately)
                    total_time_estimate = base + 40 * increment # Estimate for 40 moves
                    if total_time_estimate >= 1500: return 'Classical' # >= 25 min
                    if total_time_estimate >= 480: return 'Rapid'     # >= 8 min
                    if total_time_estimate >= 180: return 'Blitz'     # >= 3 min
                    if total_time_estimate > 0 : return 'Bullet'      # < 3 min
                    return 'Unknown'
                except (ValueError, IndexError): return 'Unknown'
            else: # Only base time (less common now?)
                try:
                     base = int(tc_str)
                     if base >= 1500: return 'Classical'
                     if base >= 480: return 'Rapid'
                     if base >= 180: return 'Blitz'
                     if base > 0 : return 'Bullet'
                     return 'Unknown'
                except ValueError: return 'Unknown' # Cannot parse

        df['TimeControl_Category'] = df.apply(lambda row: categorize_time_control(row['TimeControl'], row['Speed']), axis=1)

        # Rename 'Opening' to 'OpeningName' for consistency with plotting functions
        df = df.rename(columns={'Opening': 'OpeningName'})

        # Sort by Date
        df = df.sort_values(by='Date').reset_index(drop=True)

    return df


# =============================================
# Plotting Functions (Should work with the new DataFrame structure)
# Copied from previous version - ASSUMED CORRECT
# =============================================
# (Include ALL plot functions: plot_win_loss_pie, plot_win_loss_by_color, plot_rating_trend, etc.)
# ... Make sure column names used here match the final DataFrame ...
# Example: Ensure 'OpeningName' is used instead of 'Opening' if renamed.

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
    # Ensure PlayerElo is numeric
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
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games", xaxis={'type': 'category'}) # Ensure year is treated as category
    return fig

def plot_win_rate_per_year(df):
    if not all(col in df.columns for col in ['Year', 'PlayerResultNumeric']): return go.Figure()
    wins_per_year = df[df['PlayerResultNumeric'] == 1].groupby('Year').size()
    total_per_year = df.groupby('Year').size()
    win_rate = (wins_per_year.reindex(total_per_year.index, fill_value=0) / total_per_year).fillna(0) * 100
    # Ensure index is suitable for plotting
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
        # Use the actual categories found in the data + a preferred order
        found_categories = df['TimeControl_Category'].unique()
        cat_order_preferred = ['Bullet', 'Blitz', 'Rapid', 'Classical', 'Correspondence', 'Unknown']
        # Create the final order based on preferred order + any other found categories
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
    # Uses 'OpeningName' column generated from API data
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
# Helper Functions (Assumed Correct from Previous Version)
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
# Streamlit App Layout - API Version
# =============================================

st.title("♟️ Lichess Insights")
st.write("Analyze chess game statistics directly from Lichess API.")

# --- Input Area ---
col1, col2 = st.columns([2, 1])
with col1:
    lichess_username = st.text_input("Enter Lichess Username:", key="username_input", placeholder="e.g., DrNykterstein")
with col2:
    time_period = st.selectbox("Select Time Period:", options=list(TIME_PERIOD_OPTIONS.keys()), index=list(TIME_PERIOD_OPTIONS.keys()).index(DEFAULT_TIME_PERIOD), key="time_period_select")

# --- Button to Trigger Analysis ---
# We use a button to avoid running the API call on every input change
analyze_button = st.button("Analyze Games", key="analyze_button")

# --- Data Loading and Analysis Area (Triggered by Button) ---
# Initialize session state to store the dataframe
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'current_username' not in st.session_state:
    st.session_state.current_username = ""
if 'current_time_period' not in st.session_state:
    st.session_state.current_time_period = ""

# Only run analysis if button is clicked AND username is provided
if analyze_button and lichess_username:
    # Check if analysis is needed (username or time period changed)
    if lichess_username != st.session_state.current_username or time_period != st.session_state.current_time_period:
        # Run the API call and store the result in session state
        df_loaded = load_from_lichess_api(lichess_username, time_period, DEFAULT_PERF_TYPES, DEFAULT_RATED_ONLY)
        st.session_state.analysis_df = df_loaded
        st.session_state.current_username = lichess_username
        st.session_state.current_time_period = time_period
        # Clear dependent session state if needed (e.g., selected section)
        if 'selected_section' in st.session_state:
             del st.session_state['selected_section']
    else:
        # Data is already loaded for this user/period, just show message
        st.info("Analysis results for this user and time period are already displayed.")


# --- Display Results if DataFrame is available in session state ---
if isinstance(st.session_state.analysis_df, pd.DataFrame) and not st.session_state.analysis_df.empty:
    df = st.session_state.analysis_df # Use the dataframe from session state
    current_display_name = st.session_state.current_username # Use the username as display name for now

    st.success(f"Displaying analysis for **{current_display_name}** ({st.session_state.current_time_period})")
    st.caption(f"Total Rated Games Analyzed ({', '.join(DEFAULT_PERF_TYPES)}): **{len(df):,}**")
    st.markdown("---")

    # --- Sidebar Navigation for Analysis Sections ---
    st.sidebar.title("📊 Analysis Sections")
    analysis_options = [
        "Overview", "Time and Date Analysis", "ECO and Opening Analysis",
        "Opponent Analysis", "Games against GMs", "Time Forfeit Analysis"
    ]
    # Use session state to remember the selected section
    if 'selected_section' not in st.session_state:
         st.session_state.selected_section = "Overview" # Default

    selected_section = st.sidebar.radio(
         "Choose a section:",
         analysis_options,
         index=analysis_options.index(st.session_state.selected_section), # Set index based on state
         key="section_radio"
    )
    st.session_state.selected_section = selected_section # Update state when radio changes


    # =============================================
    # Display Content Based on Selected Section
    # =============================================
    if selected_section == "Overview":
        st.header("📈 General Overview")
        col_ov1, col_ov2 = st.columns(2)
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
            st.markdown("#### Performance Data by Time Control")
            try:
                 tc_summary = df.groupby('TimeControl_Category')['PlayerResultString'].value_counts().unstack(fill_value=0)
                 st.dataframe(tc_summary)
            except KeyError: st.warning("Could not generate time control summary table.")

    elif selected_section == "ECO and Opening Analysis":
        st.header("📖 ECO and Opening Analysis")
        tab_eco1, tab_eco2 = st.tabs(["Opening Frequency", "Opening Performance"])
        with tab_eco1:
            n_openings = st.slider("Number of top openings:", 5, 50, 20, key="n_openings_freq")
            st.plotly_chart(plot_opening_frequency(df, top_n=n_openings), use_container_width=True)
            st.markdown(f"#### Top {n_openings} Opening Frequencies")
            try: st.dataframe(df[df['OpeningName'] != 'Unknown Opening']['OpeningName'].value_counts().reset_index(name='Count').head(n_openings))
            except KeyError: st.warning("Could not generate opening frequency table.")
        with tab_eco2:
            min_games_opening = st.slider("Min games for performance:", 1, 25, 5, key="min_games_perf")
            n_openings_perf = st.slider("Number of top openings by win rate:", 5, 50, 20, key="n_openings_perf")
            st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_opening, top_n=n_openings_perf), use_container_width=True)

    elif selected_section == "Opponent Analysis":
        st.header("👥 Opponent Analysis")
        tab_opp1, tab_opp2 = st.tabs(["Most Frequent Opponents", "Elo Difference vs Result"])
        with tab_opp1:
            n_opponents = st.slider("Number of top opponents:", 5, 50, 20, key="n_opponents_freq")
            st.plotly_chart(plot_most_frequent_opponents(df, top_n=n_opponents), use_container_width=True)
            st.markdown(f"#### Top {n_opponents} Most Frequent Opponents")
            try: st.dataframe(df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().reset_index(name='Games Played').head(n_opponents))
            except KeyError: st.warning("Could not generate opponent frequency table.")
        with tab_opp2: st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)

    elif selected_section == "Games against GMs":
        st.header("👑 Analysis Against Grandmasters (GMs)")
        gm_games = filter_and_analyze_gms(df)
        if not gm_games.empty:
            st.success(f"Found **{len(gm_games):,}** games against 'GM' opponents. Analyzing subset...")
            tab_gm1, tab_gm2, tab_gm3, tab_gm4 = st.tabs(["🏆 GM Perf. Summary", "📈 Rating Trend vs GMs", "📖 Openings vs GMs", "👥 Frequent GM Opponents"])
            with tab_gm1:
                 st.plotly_chart(plot_win_loss_pie(gm_games, f"{current_display_name} vs GMs"), use_container_width=True)
                 st.markdown("#### Results vs GMs Breakdown:")
                 st.dataframe(gm_games['PlayerResultString'].value_counts().reset_index(name='Count'))
                 st.plotly_chart(plot_win_loss_by_color(gm_games), use_container_width=True)
            with tab_gm2: st.plotly_chart(plot_rating_trend(gm_games, f"{current_display_name} (vs GMs)"), use_container_width=True)
            with tab_gm3:
                 st.plotly_chart(plot_opening_frequency(gm_games, top_n=15), use_container_width=True)
                 min_games_gm_opening = st.slider("Min games (GM opening perf):", 1, 10, 3, key="gm_opening_slider")
                 st.plotly_chart(plot_win_rate_by_opening(gm_games, min_games=min_games_gm_opening, top_n=15), use_container_width=True)
            with tab_gm4:
                st.plotly_chart(plot_most_frequent_opponents(gm_games, top_n=15), use_container_width=True)
                st.markdown("#### Most Frequent GM Opponents:")
                st.dataframe(gm_games['OpponentName'].value_counts().reset_index(name='Games Played').head(15))
        else: st.warning("ℹ️ No games found against opponents with 'GM' title.")

    elif selected_section == "Time Forfeit Analysis":
        st.header("⏱️ Time Forfeit Analysis")
        tf_games, wins_tf, losses_tf = filter_and_analyze_time_forfeits(df)
        if not tf_games.empty:
            st.success(f"Found **{len(tf_games):,}** games ending due to time forfeit.")
            col_tf1, col_tf2 = st.columns(2)
            col_tf1.metric("Games Won on Time", wins_tf, help="Opponent lost on time.")
            col_tf2.metric("Games Lost on Time", losses_tf, help="Player lost on time.")
            st.markdown("#### Games Ending in Time Forfeit (Most Recent First):")
            st.dataframe(tf_games[['Date', 'OpponentName', 'PlayerColor', 'PlayerResultString', 'TimeControl', 'PlyCount', 'Termination']].sort_values('Date', ascending=False).head(50))
            st.markdown("#### Time Forfeits by Time Control:")
            st.dataframe(tf_games['TimeControl_Category'].value_counts().reset_index(name='Forfeit Count'))
        else: st.warning("ℹ️ No games found with 'Time forfeit' in 'Termination' tag.")

    # --- Footer in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.info(f"Analysis for {current_display_name}. Using Lichess API.")

# --- Initial State / Prompt Message ---
elif not analyze_button and st.session_state.analysis_df is None:
     st.info("☝️ Enter a Lichess username and select a time period, then click 'Analyze Games'.")

# --- End of App ---
