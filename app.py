# -*- coding: utf-8 -*-
# =============================================
# Streamlit App for Chess Game Analysis
# Based on Kaggle notebook: chess-insights-alireza-firouzja-s-games-analysis
# =============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import chess.pgn
import io
from collections import Counter
import datetime
import re # For cleaning names and parsing

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Chess Insights: Alireza Firouzja",
    page_icon="♟️" # Optional: Add a page icon
)

# --- Constants ---
PLAYER_ID = "alireza2003"                 # User ID for filtering PGN games (from Kaggle analysis)
DISPLAY_NAME = "Alireza Firouzja"         # Full name for display purposes
DEFAULT_PGN_FILE_PATH = "alireza2003.pgn" # Default PGN filename to look for

# =============================================
# Data Loading and Processing Function
# =============================================
@st.cache_data # Cache the result to speed up app interactions
def load_and_process_pgn(pgn_file_content, player_id):
    """
    Loads PGN data from file content, extracts game information,
    cleans the data, and engineers features based on the provided player ID.
    Inspired by the logic in the associated Kaggle notebook.
    """
    st.info(f"Processing PGN data for player ID: {player_id}...")
    all_games_data = []
    pgn = io.StringIO(pgn_file_content) # Read PGN content as a file
    player_id_lower = player_id.lower() # Use lowercase for case-insensitive matching
    game_counter = 0
    processed_game_counter = 0
    errors_parsing = 0

    while True:
        try:
            # Read headers first for efficiency
            game_headers = chess.pgn.read_headers(pgn)
        except Exception as e:
            errors_parsing += 1
            # Optionally log the error or the problematic header section
            # st.warning(f"Minor parsing error reading headers near game {game_counter+1}: {e}")
            # Try to recover or break if it's severe
            # For simplicity, we might just break if errors are too many
            if errors_parsing > 10: # Stop if too many errors
                st.error(f"Encountered too many errors ({errors_parsing}) reading PGN headers. Aborting.")
                return pd.DataFrame() # Return empty on major failure
            continue # Try to skip the problematic part and read the next header

        if game_headers is None:
            break # End of PGN file

        game_counter += 1
        game_data = dict(game_headers) # Convert headers to dictionary

        # --- Extract Basic Info ---
        white_player_raw = game_data.get('White', '').strip()
        black_player_raw = game_data.get('Black', '').strip()
        result = game_data.get('Result', '*')
        date_raw = game_data.get('Date', '????.??.??')

        # --- Determine Player's Color, Opponent, Elos based on PLAYER_ID ---
        player_color = None
        opponent_name_raw = "Unknown"
        opponent_elo = 0
        player_elo = 0
        opponent_title = ''

        # --- Clean opponent names (remove titles like GM, IM etc.) ---
        def clean_name(name):
            return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM)\s+', '', name).strip()

        # --- Match player ID ---
        # Use exact lowercase comparison for the ID
        matched = False
        if player_id_lower == white_player_raw.lower():
            player_color = 'White'
            opponent_name_raw = black_player_raw
            player_elo = int(game_data.get('WhiteElo', 0))
            opponent_elo = int(game_data.get('BlackElo', 0))
            opponent_title = game_data.get('BlackTitle', '') # Get opponent title if available
            matched = True
        elif player_id_lower == black_player_raw.lower():
            player_color = 'Black'
            opponent_name_raw = white_player_raw
            player_elo = int(game_data.get('BlackElo', 0)) # Player is Black, use BlackElo
            opponent_elo = int(game_data.get('WhiteElo', 0)) # Opponent is White, use WhiteElo
            opponent_title = game_data.get('WhiteTitle', '') # Get opponent title if available
            matched = True

        if not matched:
            continue # Skip this game if the player ID doesn't match White or Black

        # --- Determine Player's Result ---
        player_result_numeric = -1 # Default for unknown/ongoing
        player_result_string = "Unknown"
        if player_color == 'White':
            if result == '1-0': player_result_numeric, player_result_string = 1, "Win"
            elif result == '0-1': player_result_numeric, player_result_string = 0, "Loss"
            elif result == '1/2-1/2': player_result_numeric, player_result_string = 0.5, "Draw"
        elif player_color == 'Black':
            if result == '0-1': player_result_numeric, player_result_string = 1, "Win" # Black wins if 0-1
            elif result == '1-0': player_result_numeric, player_result_string = 0, "Loss" # Black loses if 1-0
            elif result == '1/2-1/2': player_result_numeric, player_result_string = 0.5, "Draw"

        # Skip games with unknown results '*' or ongoing games
        if player_result_numeric == -1:
            continue

        # --- Date Handling (Robust) ---
        game_date = pd.NaT # Initialize as Not a Time
        try:
            # Handle PGN's typical date format with potential missing parts '????.??.??'
            clean_date_str = date_raw.replace('.??.??', '-01-01').replace('.??', '-01')
            game_date = pd.to_datetime(clean_date_str, format='%Y.%m.%d', errors='coerce')
        except ValueError:
            pass # game_date remains pd.NaT if format is totally wrong

        # Skip games with invalid dates after trying to parse
        if pd.isna(game_date):
             continue

        # --- Clean opponent name ---
        opponent_name_clean = clean_name(opponent_name_raw)

        # --- Store data for this game ---
        game_info = {
            'Date': game_date,
            'Event': game_data.get('Event', 'Unknown'),
            'White': white_player_raw, # Store original names too
            'Black': black_player_raw,
            'Result': result,
            'WhiteElo': int(game_data.get('WhiteElo', 0)),
            'BlackElo': int(game_data.get('BlackElo', 0)),
            'ECO': game_data.get('ECO', 'Unknown'),
            'Opening': game_data.get('Opening', 'Unknown'), # Use 'Opening' tag if available
            'TimeControl': game_data.get('TimeControl', 'Unknown'),
            'Termination': game_data.get('Termination', 'Unknown'),
            'PlyCount': int(game_data.get('PlyCount', 0)) if game_data.get('PlyCount', '').isdigit() else 0, # Handle non-numeric PlyCount

            # --- Added/Derived Features ---
            'PlayerID': player_id,
            'PlayerColor': player_color,
            'PlayerElo': player_elo,
            'OpponentName': opponent_name_clean, # Store cleaned name
            'OpponentNameRaw': opponent_name_raw, # Store raw name if needed later
            'OpponentElo': opponent_elo,
            'OpponentTitle': opponent_title.replace({' ':'Unknown'}).strip().upper() if opponent_title else 'Unknown', # Normalize title
            'PlayerResultNumeric': player_result_numeric,
            'PlayerResultString': player_result_string,
        }
        all_games_data.append(game_info)
        processed_game_counter += 1 # Increment counter for successfully processed games

    st.info(f"Checked {game_counter} headers. Found and processed {processed_game_counter} valid games for player ID '{player_id}'.")

    if not all_games_data:
        st.warning("No game data collected after processing.")
        return pd.DataFrame() # Return empty DataFrame if no games processed

    df = pd.DataFrame(all_games_data)

    # --- Feature Engineering (Based on Kaggle Notebook Logic) ---
    if not df.empty:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.day_name()

        # Elo Difference (handle potential 0 Elos - treat as NA or keep?)
        # If Elos are 0, maybe difference isn't meaningful. Let's calculate but be aware.
        df['EloDiff'] = df['PlayerElo'].fillna(0) - df['OpponentElo'].fillna(0) # Simple diff, handle NaNs if Elos were objects

        # --- Time Control Categorization (Function from Notebook) ---
        def categorize_time_control(tc):
            if not isinstance(tc, str) or tc == '-' or tc == '?' or tc == 'Unknown':
                return 'Unknown'
            # Standard format like 600+5
            if '+' in tc:
                try:
                    base, increment = map(int, tc.split('+'))
                    # Estimate total time for a 40-move game
                    total_time_estimate = base + 40 * increment
                    if total_time_estimate >= 1500: return 'Classical' # >= 25 min (adjust thresholds as needed)
                    if total_time_estimate >= 600: return 'Rapid'     # >= 10 min
                    if total_time_estimate >= 180: return 'Blitz'     # >= 3 min
                    return 'Bullet'                                    # < 3 min
                except ValueError:
                    return 'Unknown' # Format error
            # Only base time like '300'
            else:
                try:
                    base = int(tc)
                    if base >= 1500: return 'Classical'
                    if base >= 600: return 'Rapid'
                    if base >= 180: return 'Blitz'
                    if base > 0 : return 'Bullet'
                    return 'Unknown'
                except ValueError:
                    # Handle non-numeric or complex cases if needed
                    if 'classical' in tc.lower(): return 'Classical'
                    if 'rapid' in tc.lower(): return 'Rapid'
                    if 'blitz' in tc.lower(): return 'Blitz'
                    if 'bullet' in tc.lower(): return 'Bullet'
                    if 'correspondence' in tc.lower(): return 'Correspondence'
                    return 'Unknown'

        df['TimeControl_Category'] = df['TimeControl'].apply(categorize_time_control)

        # --- ECO / Opening Name (Prefer 'Opening' tag, fallback to ECO) ---
        # Create a more descriptive name if Opening tag is generic or missing
        df['OpeningName'] = df.apply(lambda row: row['Opening'] if row['Opening'] not in ['Unknown', '?', ''] else ('ECO: ' + row['ECO'] if row['ECO'] not in ['Unknown', '?'] else 'Unknown Opening'), axis=1)
        # Further clean opening names if needed (e.g., remove '?' prefixes seen sometimes)
        df['OpeningName'] = df['OpeningName'].str.replace(r'^\?\s*', '', regex=True)

        # --- Sort by Date ---
        df = df.sort_values(by='Date').reset_index(drop=True)

    st.success(f"Data processing complete. Returning DataFrame with {len(df)} rows.")
    return df


# =============================================
# Plotting Functions
# (Directly adapted from Kaggle Notebook's Plotly code where possible)
# =============================================

def plot_win_loss_pie(df, display_name):
    """Generates a pie chart for Win/Loss/Draw distribution."""
    if 'PlayerResultString' not in df.columns: return go.Figure() # Handle missing column
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values,
                 names=result_counts.index,
                 title=f'Overall Win/Loss/Draw Distribution for {display_name}',
                 color=result_counts.index,
                 color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, # Green, Grey, Red
                 hole=0.3) # Optional: make it a donut chart
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05 if x == 'Win' else 0 for x in result_counts.index]) # Pull wins slightly
    return fig

def plot_win_loss_by_color(df):
    """Generates a stacked bar chart for Win/Loss/Draw percentage by color played."""
    if not all(col in df.columns for col in ['PlayerColor', 'PlayerResultString']): return go.Figure()

    try:
        # Group by color and result, count occurrences
        color_results = df.groupby(['PlayerColor', 'PlayerResultString']).size().unstack(fill_value=0)

        # Ensure all result columns exist ('Win', 'Draw', 'Loss')
        for res in ['Win', 'Draw', 'Loss']:
            if res not in color_results.columns:
                color_results[res] = 0
        # Reorder columns for consistent plotting
        color_results = color_results[['Win', 'Draw', 'Loss']]

        # Calculate percentages safely
        total_per_color = color_results.sum(axis=1)
        # Avoid division by zero if a color has 0 games (though unlikely if player played both)
        color_results_pct = color_results.apply(lambda x: x * 100 / total_per_color[x.name] if total_per_color[x.name] > 0 else 0, axis=1)

        fig = px.bar(color_results_pct,
                     barmode='stack', # Stacked bar chart
                     title='Win/Loss/Draw Percentage by Color',
                     labels={'value': 'Percentage (%)', 'PlayerColor': 'Played As', 'PlayerResultString': 'Result'},
                     color='PlayerResultString', # Color bars based on Result
                     color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                     text_auto='.1f', # Show percentage on bars, formatted to 1 decimal
                     category_orders={"PlayerColor": ["White", "Black"]} # Ensure White is typically first
                    )
        fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="Color Played")
        fig.update_traces(textangle=0) # Ensure text is horizontal
        return fig
    except Exception as e:
        st.error(f"Error creating Win/Loss by Color plot: {e}")
        return go.Figure().update_layout(title="Error generating plot") # Return empty figure on error

def plot_rating_trend(df, display_name):
    """Generates a line chart showing Elo rating trend over time."""
    if not all(col in df.columns for col in ['Date', 'PlayerElo']): return go.Figure()

    # Filter out games with missing or zero Elo rating, sort by date
    df_sorted = df[df['PlayerElo'].notna() & (df['PlayerElo'] > 0)].sort_values('Date')
    if df_sorted.empty:
        return go.Figure().update_layout(title=f"No valid Elo data found for {display_name}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['Date'], y=df_sorted['PlayerElo'],
        mode='lines+markers', # Show both lines and points
        name='Elo Rating',
        line=dict(color='#1E88E5', width=2), # Blue line
        marker=dict(color='#1E88E5', size=5, opacity=0.7),
        hoverinfo='x+y' # Show date and Elo on hover
    ))

    fig.update_layout(
        title=f'{display_name}\'s Rating Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Elo Rating',
        hovermode="x unified", # Show info for all traces at a given x-value
        xaxis_rangeslider_visible=True # Add a range slider for zooming
    )
    return fig

def plot_performance_vs_opponent_elo(df):
    """Generates a box plot showing Elo difference distribution by game result."""
    if not all(col in df.columns for col in ['PlayerResultString', 'EloDiff']): return go.Figure()

    fig = px.box(df, x='PlayerResultString', y='EloDiff',
                 title='Player\'s Elo Advantage vs. Game Result',
                 labels={'PlayerResultString': 'Game Result', 'EloDiff': 'Player Elo - Opponent Elo'},
                 # Order boxes logically: Win, Draw, Loss
                 category_orders={"PlayerResultString": ["Win", "Draw", "Loss"]},
                 color='PlayerResultString', # Color boxes by result
                 color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                 points='outliers' # Show outliers
                 )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Equal Elo", annotation_position="bottom right") # Line for zero difference
    fig.update_traces(marker=dict(opacity=0.8)) # Adjust marker opacity
    return fig

def plot_games_per_year(df):
    """Generates a bar chart showing the number of games played per year."""
    if 'Year' not in df.columns: return go.Figure()
    games_per_year = df['Year'].value_counts().sort_index()
    fig = px.bar(games_per_year, x=games_per_year.index, y=games_per_year.values,
                 title='Number of Games Played Per Year',
                 labels={'x': 'Year', 'y': 'Number of Games'},
                 text=games_per_year.values) # Show count on bars
    fig.update_traces(marker_color='#2196F3', textposition='outside') # Blue color, text outside bars
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games")
    return fig

def plot_win_rate_per_year(df):
    """Calculates and plots the win rate per year."""
    if not all(col in df.columns for col in ['Year', 'PlayerResultNumeric']): return go.Figure()

    # Calculate Wins and Total Games per year
    wins_per_year = df[df['PlayerResultNumeric'] == 1].groupby('Year').size()
    total_per_year = df.groupby('Year').size()

    # Calculate win rate, handle years with 0 games (avoid division by zero)
    win_rate = (wins_per_year.reindex(total_per_year.index, fill_value=0) / total_per_year).fillna(0) * 100

    fig = px.line(win_rate, x=win_rate.index, y=win_rate.values,
                  title='Win Rate (%) Per Year', markers=True, # Show points on the line
                  labels={'x': 'Year', 'y': 'Win Rate (%)'})
    fig.update_traces(line_color='#FFC107', line_width=2.5) # Amber/Yellow color
    fig.update_layout(yaxis_range=[0, 100]) # Ensure y-axis is 0-100%
    return fig

def plot_performance_by_time_control(df):
     """Generates a grouped bar chart showing performance by time control category."""
     if not all(col in df.columns for col in ['TimeControl_Category', 'PlayerResultString']): return go.Figure()

     try:
        # Group by time control and result, count occurrences
        tc_results = df.groupby(['TimeControl_Category', 'PlayerResultString']).size().unstack(fill_value=0)

        # Ensure all result columns exist and are ordered
        for res in ['Win', 'Draw', 'Loss']:
            if res not in tc_results.columns:
                tc_results[res] = 0
        tc_results = tc_results[['Win', 'Draw', 'Loss']]

        # Calculate percentages safely
        total_per_tc = tc_results.sum(axis=1)
        tc_results_pct = tc_results.apply(lambda x: x * 100 / total_per_tc[x.name] if total_per_tc[x.name] > 0 else 0, axis=1)

        # Define a logical order for time controls
        cat_order = ['Bullet', 'Blitz', 'Rapid', 'Classical', 'Unknown', 'Correspondence'] # Add others if present
        # Reindex based on the defined order, drop categories not present in the data
        tc_results_pct = tc_results_pct.reindex(index=cat_order).dropna(axis=0, how='all')

        fig = px.bar(tc_results_pct,
                     title='Performance by Time Control Category',
                     labels={'value': 'Percentage (%)', 'TimeControl_Category': 'Time Control', 'PlayerResultString':'Result'},
                     color='PlayerResultString', # Color by result
                     color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                     barmode='group', # Grouped bars (Win/Draw/Loss side-by-side for each TC)
                     text_auto='.1f') # Show percentage values
        fig.update_layout(xaxis_title="Time Control Category", yaxis_title="Percentage (%)")
        fig.update_traces(textangle=0) # Ensure text is horizontal
        return fig
     except Exception as e:
        st.error(f"Error creating Performance by Time Control plot: {e}")
        return go.Figure().update_layout(title="Error generating plot")

def plot_opening_frequency(df, top_n=20):
    """Generates a horizontal bar chart for the most frequent openings."""
    if 'OpeningName' not in df.columns: return go.Figure()
    # Exclude 'Unknown Opening' from top N calculation if desired
    opening_counts = df[df['OpeningName'] != 'Unknown Opening']['OpeningName'].value_counts().nlargest(top_n)
    fig = px.bar(opening_counts, y=opening_counts.index, x=opening_counts.values,
                 orientation='h', # Horizontal bar chart
                 title=f'Top {top_n} Most Frequent Openings Played',
                 labels={'y': 'Opening Name', 'x': 'Number of Games'},
                 text=opening_counts.values) # Show counts on bars
    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars by count
    fig.update_traces(marker_color='#673AB7', textposition='outside') # Deep Purple color
    return fig

def plot_win_rate_by_opening(df, min_games=5, top_n=20):
    """Calculates and plots win rate for openings played at least min_games times."""
    if not all(col in df.columns for col in ['OpeningName', 'PlayerResultNumeric']): return go.Figure()

    # Calculate stats per opening
    opening_stats = df.groupby('OpeningName').agg(
        total_games=('PlayerResultNumeric', 'count'),
        wins=('PlayerResultNumeric', lambda x: (x == 1).sum())
        # Optionally add draws/losses/performance score here if needed
        # draws=('PlayerResultNumeric', lambda x: (x == 0.5).sum()),
        # performance_score = ('PlayerResultNumeric', lambda x: (x.eq(1).sum() + 0.5 * x.eq(0.5).sum()) / x.count() * 100 if x.count() > 0 else 0)
    )
    # Filter openings played enough times and exclude 'Unknown'
    opening_stats = opening_stats[(opening_stats['total_games'] >= min_games) & (opening_stats.index != 'Unknown Opening')].copy() # Ensure it's a copy

    if opening_stats.empty:
        return go.Figure().update_layout(title=f"No openings played >= {min_games} times")

    # Calculate win rate
    opening_stats['win_rate'] = (opening_stats['wins'] / opening_stats['total_games']) * 100

    # Select top N based on win rate for plotting
    opening_stats_plot = opening_stats.nlargest(top_n, 'win_rate')

    fig = px.bar(opening_stats_plot, y=opening_stats_plot.index, x='win_rate', orientation='h',
                 title=f'Top {top_n} Openings by Win Rate (Played >= {min_games} times)',
                 labels={'win_rate': 'Win Rate (%)', 'OpeningName': 'Opening'},
                 text='win_rate') # Show win rate value on bars
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside', marker_color='#009688') # Teal color, text inside
    # Sort bars by win rate
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Win Rate (%)", yaxis_title="Opening Name")
    return fig


def plot_most_frequent_opponents(df, top_n=20):
    """Generates a horizontal bar chart for the most frequent opponents."""
    if 'OpponentName' not in df.columns: return go.Figure()
    # Exclude 'Unknown' opponent if present
    opp_counts = df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().nlargest(top_n)
    fig = px.bar(opp_counts, y=opp_counts.index, x=opp_counts.values, orientation='h',
                 title=f'Top {top_n} Most Frequent Opponents',
                 labels={'y': 'Opponent Name', 'x': 'Number of Games'},
                 text=opp_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
    fig.update_traces(marker_color='#FF5722', textposition='outside') # Deep Orange color
    return fig

# =============================================
# Helper Functions for Specific Analysis
# =============================================

def filter_and_analyze_gms(df):
    """Filters games played against opponents with the 'GM' title."""
    if 'OpponentTitle' not in df.columns:
        st.warning("Column 'OpponentTitle' not found. Cannot filter GM games.")
        return pd.DataFrame()
    # Filter based on the normalized title
    gm_games = df[df['OpponentTitle'] == 'GM'].copy()
    return gm_games

def filter_and_analyze_time_forfeits(df):
    """Filters games ending by time forfeit and returns counts."""
    if 'Termination' not in df.columns:
        st.warning("Column 'Termination' not found. Cannot analyze time forfeits.")
        return pd.DataFrame(), 0, 0
    # Use case-insensitive search for "Time forfeit"
    tf_games = df[df['Termination'].str.contains("Time forfeit", na=False, case=False)].copy()
    if tf_games.empty:
        return tf_games, 0, 0
    # Count wins/losses within the time forfeit games
    wins_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 1])
    losses_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 0])
    return tf_games, wins_tf, losses_tf

# =============================================
# Streamlit App Layout
# =============================================

# --- Sidebar Elements ---
st.sidebar.title("♟️ Chess Analysis Setup")
st.sidebar.markdown(f"Analyzing games for **{DISPLAY_NAME}** (ID: `{PLAYER_ID}`)")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload PGN File (Optional)",
    type="pgn",
    help=f"If no file is uploaded, the app will try to load '{DEFAULT_PGN_FILE_PATH}' from its directory."
)

# --- Data Loading Logic ---
pgn_content = None
data_source_info = ""
df = None # Initialize dataframe
data_loaded = False

if uploaded_file is not None:
    # If a file is uploaded, use its content
    try:
        pgn_content = uploaded_file.getvalue().decode("utf-8")
        data_source_info = f"Using uploaded file: **{uploaded_file.name}**"
        st.sidebar.success("✅ File uploaded successfully!")
    except UnicodeDecodeError:
        st.sidebar.error("🚨 Error decoding file. Please ensure it's a valid UTF-8 encoded PGN.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"🚨 Error reading uploaded file: {e}")
        st.stop()
else:
    # If no file is uploaded, try to load the default file
    try:
        with open(DEFAULT_PGN_FILE_PATH, 'r', encoding='utf-8') as f:
            pgn_content = f.read()
        data_source_info = f"Using default file: **{DEFAULT_PGN_FILE_PATH}**"
        st.sidebar.info(f"ℹ️ No file uploaded. {data_source_info}")
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ Default PGN file '{DEFAULT_PGN_FILE_PATH}' not found. Please upload a file to analyze.")
        # Don't stop here, wait for upload or show message in main area
    except Exception as e:
         st.sidebar.error(f"🚨 Error reading default PGN file: {e}")
         st.stop() # Stop if default file exists but cannot be read

# --- Process Data if Content is Available ---
if pgn_content:
    try:
        # Call the main processing function
        df = load_and_process_pgn(pgn_content, PLAYER_ID)

        if df is None or df.empty:
             st.error(f"🚫 No games found or processed for player ID '{PLAYER_ID}' ({DISPLAY_NAME}) in the provided PGN. Check the PGN content or player ID.")
             st.stop() # Stop execution if processing yielded no data
        else:
             data_loaded = True # Set flag to True only if df is valid

    except Exception as e:
        st.error(f"🚨 An unexpected error occurred during PGN processing:")
        st.exception(e) # Shows detailed traceback for debugging
        data_loaded = False
        st.stop() # Stop execution on processing error

# --- Main Application Area (Display only if data is loaded) ---
if data_loaded and isinstance(df, pd.DataFrame) and not df.empty:

    # --- Main Title and Info ---
    st.title(f"♟️ Chess Insights: {DISPLAY_NAME}")
    st.caption(f"{data_source_info} | Player ID: `{PLAYER_ID}` | Total Games Analyzed: **{len(df):,}**")
    st.markdown("---") # Horizontal line separator

    # --- Sidebar Navigation for Analysis Sections ---
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Analysis Sections")
    analysis_options = [
        "Overview",
        "Time and Date Analysis",
        "ECO and Opening Analysis",
        "Opponent Analysis",
        "Games against GMs",
        # "Famous Opponent Analysis", # Keep commented out until implemented
        "Time Forfeit Analysis"
    ]
    # Use radio buttons in the sidebar for section selection
    selected_section = st.sidebar.radio("Choose a section:", analysis_options, index=0) # Default to Overview

    # =============================================
    # Display Content Based on Selected Section
    # =============================================

    if selected_section == "Overview":
        st.header("📈 General Overview")
        # Use columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_win_loss_pie(df, DISPLAY_NAME), use_container_width=True)
        with col2:
            st.plotly_chart(plot_win_loss_by_color(df), use_container_width=True)

        st.plotly_chart(plot_rating_trend(df, DISPLAY_NAME), use_container_width=True)
        st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)


    elif selected_section == "Time and Date Analysis":
        st.header("📅 Time and Date Analysis")
        # Use tabs within the section
        tab_time1, tab_time2, tab_time3 = st.tabs(["Games Over Time", "Performance by Year", "Performance by Time Control"])
        with tab_time1:
            st.plotly_chart(plot_games_per_year(df), use_container_width=True)
            # Maybe add games per month/day of week here if desired
        with tab_time2:
            st.plotly_chart(plot_win_rate_per_year(df), use_container_width=True)
            # Optionally show table of win rate per year
            # ...
        with tab_time3:
            st.plotly_chart(plot_performance_by_time_control(df), use_container_width=True)
            st.markdown("#### Performance Data by Time Control")
            # Display the underlying data table for the time control plot
            tc_summary = df.groupby('TimeControl_Category')['PlayerResultString'].value_counts().unstack(fill_value=0)
            st.dataframe(tc_summary)


    elif selected_section == "ECO and Opening Analysis":
        st.header("📖 ECO and Opening Analysis")
        tab_eco1, tab_eco2 = st.tabs(["Opening Frequency", "Opening Performance"])
        with tab_eco1:
            n_openings = st.slider("Number of top openings to show:", 5, 50, 20, key="n_openings_freq")
            st.plotly_chart(plot_opening_frequency(df, top_n=n_openings), use_container_width=True)
            # Show table of frequencies as well
            st.markdown(f"#### Top {n_openings} Opening Frequencies")
            st.dataframe(df[df['OpeningName'] != 'Unknown Opening']['OpeningName'].value_counts().reset_index(name='Count').head(n_openings))
        with tab_eco2:
            # Add sliders for interactive filtering
            min_games_opening = st.slider("Minimum games played for performance calculation:", 1, 25, 5, key="min_games_perf")
            n_openings_perf = st.slider("Number of top openings by win rate:", 5, 50, 20, key="n_openings_perf")
            st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_opening, top_n=n_openings_perf), use_container_width=True)


    elif selected_section == "Opponent Analysis":
        st.header("👥 Opponent Analysis")
        tab_opp1, tab_opp2 = st.tabs(["Most Frequent Opponents", "Elo Difference vs Result"])
        with tab_opp1:
            n_opponents = st.slider("Number of top opponents to show:", 5, 50, 20, key="n_opponents_freq")
            st.plotly_chart(plot_most_frequent_opponents(df, top_n=n_opponents), use_container_width=True)
             # Show table of frequencies
            st.markdown(f"#### Top {n_opponents} Most Frequent Opponents")
            st.dataframe(df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().reset_index(name='Games Played').head(n_opponents))
        with tab_opp2:
             # Re-use the Elo difference plot from Overview
             st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)
             # Add more opponent analysis if present in notebook (e.g., performance against specific opponents)


    elif selected_section == "Games against GMs":
        st.header("👑 Analysis Against Grandmasters (GMs)")
        gm_games = filter_and_analyze_gms(df) # Filter the dataframe
        if not gm_games.empty:
            st.success(f"Found **{len(gm_games):,}** games against opponents identified as 'GM'. Analyzing this subset...")

            # Use tabs for different GM analyses
            tab_gm1, tab_gm2, tab_gm3, tab_gm4 = st.tabs([
                "🏆 GM Performance Summary",
                "📈 Rating Trend vs GMs",
                "📖 Openings vs GMs",
                "👥 Frequent GM Opponents"
            ])

            with tab_gm1:
                 # Use the general plot functions on the filtered dataframe
                 st.plotly_chart(plot_win_loss_pie(gm_games, f"{DISPLAY_NAME} vs GMs"), use_container_width=True)
                 st.markdown("#### Results vs GMs Breakdown:")
                 st.dataframe(gm_games['PlayerResultString'].value_counts().reset_index(name='Count'))
                 st.plotly_chart(plot_win_loss_by_color(gm_games), use_container_width=True) # Performance by color vs GMs

            with tab_gm2:
                st.plotly_chart(plot_rating_trend(gm_games, f"{DISPLAY_NAME} (Rating in GM Games)"), use_container_width=True)

            with tab_gm3:
                 st.plotly_chart(plot_opening_frequency(gm_games, top_n=15), use_container_width=True)
                 min_games_gm_opening = st.slider("Min games for GM opening performance:", 1, 10, 3, key="gm_opening_slider")
                 st.plotly_chart(plot_win_rate_by_opening(gm_games, min_games=min_games_gm_opening, top_n=15), use_container_width=True)

            with tab_gm4:
                st.plotly_chart(plot_most_frequent_opponents(gm_games, top_n=15), use_container_width=True)
                st.markdown("#### Most Frequent GM Opponents:")
                st.dataframe(gm_games['OpponentName'].value_counts().reset_index(name='Games Played').head(15))

        else:
            st.warning("ℹ️ No games found against opponents with the title 'GM' in the dataset based on the 'OpponentTitle' column.")


    # elif selected_section == "Famous Opponent Analysis":
        # st.header("🌟 Analysis Against Famous Opponents")
        # TODO: Implement logic to define famous opponents and filter games
        # famous_opponents = ["Carlsen, Magnus", "Nakamura, Hikaru", "Caruana, Fabiano"] # Example list
        # famous_games = df[df['OpponentName'].isin(famous_opponents)]
        # if not famous_games.empty:
        #    st.write(f"Found {len(famous_games)} games against selected famous opponents.")
           # Add plots/tables specific to these games (e.g., head-to-head record)
        # else:
        #    st.warning("No games found against the specified famous opponents.")


    elif selected_section == "Time Forfeit Analysis":
        st.header("⏱️ Time Forfeit Analysis")
        tf_games, wins_tf, losses_tf = filter_and_analyze_time_forfeits(df) # Get filtered data and counts

        if not tf_games.empty:
            st.success(f"Found **{len(tf_games):,}** games ending due to time forfeit (based on 'Termination' tag).")
            # Display metrics in columns
            col_tf1, col_tf2 = st.columns(2)
            col_tf1.metric("Games Won on Time", wins_tf, help="Games where the opponent lost on time.")
            col_tf2.metric("Games Lost on Time", losses_tf, help="Games where the player lost on time.")

            st.markdown("#### Games Ending in Time Forfeit (Most Recent First):")
            # Display relevant columns from the time forfeit games dataframe
            st.dataframe(tf_games[[
                'Date', 'Event', 'OpponentName', 'PlayerColor',
                'PlayerResultString', 'TimeControl', 'PlyCount', 'Termination'
            ]].sort_values('Date', ascending=False).head(50)) # Show top 50 most recent

            # Optional: Analyze further (e.g., forfeits by time control)
            st.markdown("#### Time Forfeits by Time Control:")
            st.dataframe(tf_games['TimeControl_Category'].value_counts().reset_index(name='Forfeit Count'))

        else:
             st.warning("ℹ️ No games found with 'Time forfeit' in the 'Termination' tag.")


    # --- Footer in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.info(f"App based on Kaggle analysis for {DISPLAY_NAME}. Developed using Streamlit.")
    st.sidebar.markdown("Source PGN likely from Lichess Open Database or similar.")


# --- Message if Data Loading Failed ---
elif not pgn_content and not uploaded_file:
    # Only show this if no default file was found AND no file was uploaded
    st.info("👈 Please upload a PGN file using the sidebar to begin analysis.")

# --- End of App ---
