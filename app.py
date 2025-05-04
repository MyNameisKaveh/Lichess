import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Needed for some specific plots like rating trend
import chess.pgn
import io
from collections import Counter
import datetime
import re # For cleaning opponent names

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Chess Insights: Alireza Firouzja")

# --- Constants ---
# This will be the default name used in logic and titles
PLAYER_NAME = "Alireza Firouzja"
# You can change this if your PGN file has a different name
DEFAULT_PGN_FILE_PATH = "firouzja_games.pgn"

# --- Data Loading and Caching ---
@st.cache_data # Crucial for performance!
def load_and_process_pgn(pgn_file_content, player_name):
    """Loads PGN data from content, extracts info, cleans, and engineers features based on the Kaggle notebook."""
    all_games_data = []
    pgn = io.StringIO(pgn_file_content)
    player_name_lower = player_name.lower()

    while True:
        # Use read_game to potentially get move data later if needed, for now just headers
        # For efficiency, only reading headers is faster if moves aren't needed immediately
        game_headers = chess.pgn.read_headers(pgn)
        if game_headers is None:
            break

        game_data = dict(game_headers)

        # --- Basic Info Extraction (similar to notebook) ---
        white_player = game_data.get('White', '').strip()
        black_player = game_data.get('Black', '').strip()
        result = game_data.get('Result', '*')

        # Clean opponent names (remove titles like GM, IM etc.) - Based on notebook logic
        def clean_name(name):
            return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM)\s+', '', name).strip()

        # --- Determine Player's Color, Opponent, Elos ---
        player_color = None
        opponent_name = "Unknown"
        opponent_elo = 0
        player_elo = 0

        # Check based on cleaned names if necessary, but primary check on original
        if player_name_lower in white_player.lower():
            player_color = 'White'
            opponent_name = clean_name(black_player)
            player_elo = int(game_data.get('WhiteElo', 0))
            opponent_elo = int(game_data.get('BlackElo', 0))
            opponent_title = game_data.get('BlackTitle', '')
        elif player_name_lower in black_player.lower():
            player_color = 'Black'
            opponent_name = clean_name(white_player)
            player_elo = int(game_data.get('BlackElo', 0))
            opponent_elo = int(game_data.get('WhiteElo', 0))
            opponent_title = game_data.get('WhiteTitle', '')
        else:
            # Skip game if player not found (robustness)
            continue

        # --- Determine Player's Result ---
        player_result_numeric = -1 # Default for unknown/ongoing
        player_result_string = "Unknown"
        if player_color == 'White':
            if result == '1-0':
                player_result_numeric = 1
                player_result_string = "Win"
            elif result == '0-1':
                player_result_numeric = 0
                player_result_string = "Loss"
            elif result == '1/2-1/2':
                player_result_numeric = 0.5
                player_result_string = "Draw"
        elif player_color == 'Black':
            if result == '0-1':
                player_result_numeric = 1
                player_result_string = "Win"
            elif result == '1-0':
                player_result_numeric = 0
                player_result_string = "Loss"
            elif result == '1/2-1/2':
                player_result_numeric = 0.5
                player_result_string = "Draw"

        # Skip games with unknown results
        if player_result_numeric == -1:
            continue

        # --- Date Handling (from notebook) ---
        raw_date = game_data.get('Date', '????.??.??')
        try:
            # Handle potential "??" for month/day
            clean_date_str = raw_date.replace('.??.??', '-01-01').replace('.??', '-01')
            game_date = pd.to_datetime(clean_date_str, format='%Y.%m.%d', errors='coerce')
        except ValueError:
            game_date = pd.NaT

        # --- Store data ---
        game_info = {
            'Date': game_date,
            'Event': game_data.get('Event', 'Unknown'),
            'White': white_player,
            'Black': black_player,
            'Result': result,
            'WhiteElo': int(game_data.get('WhiteElo', 0)), # Store original Elos too
            'BlackElo': int(game_data.get('BlackElo', 0)),
            'ECO': game_data.get('ECO', 'Unknown'),
            'Opening': game_data.get('Opening', 'Unknown'), # Use 'Opening' tag if available
            'TimeControl': game_data.get('TimeControl', 'Unknown'),
            'Termination': game_data.get('Termination', 'Unknown'),
            'PlyCount': int(game_data.get('PlyCount', 0)), # Get ply count

            # --- Added Features ---
            'PlayerName': player_name,
            'PlayerColor': player_color,
            'PlayerElo': player_elo,
            'OpponentName': opponent_name,
            'OpponentElo': opponent_elo,
            'OpponentTitle': opponent_title,
            'PlayerResultNumeric': player_result_numeric,
            'PlayerResultString': player_result_string,
        }
        all_games_data.append(game_info)

    if not all_games_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_games_data)
    df = df.dropna(subset=['Date']) # Essential step from notebook
    df = df.sort_values(by='Date').reset_index(drop=True) # Sort early

    # --- Feature Engineering (Replicating Notebook Logic) ---
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()

    # Elo Difference
    df['EloDiff'] = df['PlayerElo'] - df['OpponentElo']

    # Time Control Category (Using function from notebook)
    def categorize_time_control(tc):
        if not isinstance(tc, str) or tc == '-' or tc == '?' or tc == 'Unknown':
            return 'Unknown'
        if '+' in tc: # Standard format like 600+5
            try:
                base, increment = map(int, tc.split('+'))
                total_time_estimate = base + 40 * increment # Estimate for 40 moves
                if total_time_estimate >= 1500: return 'Classical' # >= 25 min
                if total_time_estimate >= 600: return 'Rapid'     # >= 10 min
                if total_time_estimate >= 180: return 'Blitz'     # >= 3 min
                return 'Bullet'                                    # < 3 min
            except ValueError:
                return 'Unknown' # Format error
        else: # Only base time like '300' or potentially non-standard
            try:
                base = int(tc)
                if base >= 1500: return 'Classical'
                if base >= 600: return 'Rapid'
                if base >= 180: return 'Blitz'
                if base > 0 : return 'Bullet' # Need a check for 0 or negative
                return 'Unknown'
            except ValueError:
                # Handle complex cases like '300+2 increment (Blitz)' seen in notebook
                if 'classical' in tc.lower(): return 'Classical'
                if 'rapid' in tc.lower(): return 'Rapid'
                if 'blitz' in tc.lower(): return 'Blitz'
                if 'bullet' in tc.lower(): return 'Bullet'
                return 'Unknown' # Could be correspondence, etc.

    df['TimeControl_Category'] = df['TimeControl'].apply(categorize_time_control)

    # ECO / Opening Name (Prefer 'Opening' tag, fallback to ECO)
    # Create a more descriptive name if Opening tag is generic
    df['OpeningName'] = df.apply(lambda row: row['Opening'] if row['Opening'] != 'Unknown' and row['Opening'] != '?' else ('ECO: ' + row['ECO'] if row['ECO'] != 'Unknown' else 'Unknown Opening'), axis=1)

    # Refine Opponent Title (from notebook's observation)
    df['OpponentTitle'] = df['OpponentTitle'].replace({' ':'Unknown'}).fillna('Unknown')


    return df

# --- Plotting Functions (Directly adapted from Kaggle Notebook's Plotly code) ---

def plot_win_loss_pie(df, player_name):
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values,
                 names=result_counts.index,
                 title=f'Overall Win/Loss/Draw Distribution for {player_name}',
                 color=result_counts.index,
                 color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}) # Colors from notebook
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_win_loss_by_color(df):
    # Replicating the stacked bar chart from the notebook
    color_results = df.groupby('PlayerColor')['PlayerResultString'].value_counts().unstack(fill_value=0)
    # Calculate percentages
    color_results_pct = color_results.apply(lambda x: x*100 / sum(x), axis=1)
    fig = px.bar(color_results_pct,
                 barmode='stack',
                 title='Win/Loss/Draw Percentage by Color',
                 labels={'value': 'Percentage (%)', 'PlayerColor': 'Played As', 'PlayerResultString': 'Result'},
                 color='PlayerResultString', # Use the result string for color mapping
                 color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                 text_auto='.1f' # Show percentage on bars
                )
    fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="Color Played")
    return fig


def plot_rating_trend(df, player_name):
    # Replicating the line chart with markers from notebook
    df_sorted = df.dropna(subset=['PlayerElo', 'Date']).sort_values('Date')
    df_sorted = df_sorted[df_sorted['PlayerElo'] > 0] # Filter out potential 0 Elo ratings if placeholders

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['PlayerElo'],
                        mode='lines+markers',
                        name='Elo Rating',
                        line=dict(color='#1E88E5', width=2), # Blue color from notebook
                        marker=dict(color='#1E88E5', size=4)))

    fig.update_layout(title=f'{player_name}\'s Rating Trend Over Time',
                      xaxis_title='Date',
                      yaxis_title='Elo Rating',
                      hovermode="x unified") # Improved hover info
    return fig

def plot_performance_vs_opponent_elo(df):
    # Replicating the box plot from the notebook
    fig = px.box(df, x='PlayerResultString', y='EloDiff',
                 title='Elo Difference Distribution by Game Result',
                 labels={'PlayerResultString': 'Game Result', 'EloDiff': 'Your Elo - Opponent Elo'},
                 category_orders={"PlayerResultString": ["Win", "Draw", "Loss"]}, # Order boxes logically
                 color='PlayerResultString',
                 color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'})
    fig.update_traces(marker=dict(opacity=0.7)) # Slight transparency
    return fig

def plot_games_per_year(df):
    # Replicating the bar chart from the notebook
    games_per_year = df['Year'].value_counts().sort_index()
    fig = px.bar(games_per_year, x=games_per_year.index, y=games_per_year.values,
                 title='Number of Games Played Per Year',
                 labels={'x': 'Year', 'y': 'Number of Games'},
                 text=games_per_year.values) # Show count on bars
    fig.update_traces(marker_color='#2196F3', textposition='outside') # Blue color
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games")
    return fig

def plot_win_rate_per_year(df):
    # Replicating the calculation and line plot from the notebook
    wins_per_year = df[df['PlayerResultNumeric'] == 1].groupby('Year').size()
    total_per_year = df.groupby('Year').size()
    win_rate = (wins_per_year / total_per_year).fillna(0) * 100

    fig = px.line(win_rate, x=win_rate.index, y=win_rate.values,
                  title='Win Rate (%) Per Year', markers=True,
                  labels={'x': 'Year', 'y': 'Win Rate (%)'})
    fig.update_traces(line_color='#FFC107') # Amber color from notebook (example)
    fig.update_layout(yaxis_range=[0, 100]) # Ensure y-axis is 0-100
    return fig

def plot_performance_by_time_control(df):
     # Replicating the grouped bar chart from the notebook
    tc_results = df.groupby('TimeControl_Category')['PlayerResultString'].value_counts(normalize=True).unstack(fill_value=0) * 100
    # Order categories potentially
    cat_order = ['Bullet', 'Blitz', 'Rapid', 'Classical', 'Unknown']
    tc_results = tc_results.reindex(cat_order).dropna(axis=0, how='all') # Reorder and remove if category not present

    fig = px.bar(tc_results,
                 title='Performance by Time Control Category',
                 labels={'value': 'Percentage (%)', 'TimeControl_Category': 'Time Control', 'PlayerResultString':'Result'},
                 color='PlayerResultString',
                 color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                 barmode='group', # Grouped bar as in notebook
                 text_auto='.1f')
    fig.update_layout(xaxis_title="Time Control Category", yaxis_title="Percentage (%)")
    return fig


def plot_opening_frequency(df, top_n=20):
    # Replicating the Opening frequency bar chart from notebook
    opening_counts = df['OpeningName'].value_counts().nlargest(top_n)
    fig = px.bar(opening_counts, y=opening_counts.index, x=opening_counts.values,
                 orientation='h', # Horizontal bar chart
                 title=f'Top {top_n} Most Frequent Openings Played',
                 labels={'y': 'Opening Name', 'x': 'Number of Games'},
                 text=opening_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
    fig.update_traces(marker_color='#673AB7', textposition='auto') # Deep Purple color
    return fig

def plot_win_rate_by_opening(df, min_games=5, top_n=20):
    # Replicating the win rate by opening logic from notebook
    opening_stats = df.groupby('OpeningName').agg(
        total_games=('PlayerResultNumeric', 'count'),
        wins=('PlayerResultNumeric', lambda x: (x == 1).sum()),
        draws=('PlayerResultNumeric', lambda x: (x == 0.5).sum()),
        losses=('PlayerResultNumeric', lambda x: (x == 0).sum())
    )
    opening_stats = opening_stats[opening_stats['total_games'] >= min_games].copy() # Ensure it's a copy
    opening_stats['win_rate'] = (opening_stats['wins'] / opening_stats['total_games']) * 100
    # Calculate performance score: (Wins + 0.5 * Draws) / Total Games
    opening_stats['performance_score'] = ((opening_stats['wins'] + 0.5 * opening_stats['draws']) / opening_stats['total_games']) * 100

    # Select top N based on win rate for plotting
    opening_stats_plot = opening_stats.sort_values('win_rate', ascending=False).head(top_n)

    fig = px.bar(opening_stats_plot, y=opening_stats_plot.index, x='win_rate', orientation='h',
                 title=f'Top {top_n} Openings by Win Rate (Played >= {min_games} times)',
                 labels={'win_rate': 'Win Rate (%)', 'OpeningName': 'Opening'},
                 text='win_rate') # Show win rate value
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto', marker_color='#009688') # Teal color
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Win Rate (%)", yaxis_title="Opening Name")
    return fig


def plot_most_frequent_opponents(df, top_n=20):
    # Replicating opponent frequency plot
    opp_counts = df['OpponentName'].value_counts().nlargest(top_n)
    fig = px.bar(opp_counts, y=opp_counts.index, x=opp_counts.values, orientation='h',
                 title=f'Top {top_n} Most Frequent Opponents',
                 labels={'y': 'Opponent Name', 'x': 'Number of Games'},
                 text=opp_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_traces(marker_color='#FF5722', textposition='auto') # Deep Orange color
    return fig

# --- GM Analysis Helper ---
def filter_and_analyze_gms(df):
    # Logic from notebook to identify GMs and analyze
    gm_games = df[df['OpponentTitle'] == 'GM'].copy()
    return gm_games

# --- Time Forfeit Analysis Helper ---
def filter_and_analyze_time_forfeits(df):
    # Logic from notebook
    tf_games = df[df['Termination'].str.contains("Time forfeit", na=False)].copy()
    wins_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 1])
    losses_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 0])
    return tf_games, wins_tf, losses_tf


# --- Streamlit App Layout ---

# --- Sidebar ---
st.sidebar.title("Chess Analysis")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload PGN File", type="pgn")
pgn_content = None
data_source_info = ""

if uploaded_file is not None:
    try:
        pgn_content = uploaded_file.getvalue().decode("utf-8")
        data_source_info = f"Using uploaded file: {uploaded_file.name}"
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")
        st.stop()
else:
    # Try to load the default file if no upload
    try:
        with open(DEFAULT_PGN_FILE_PATH, 'r', encoding='utf-8') as f:
            pgn_content = f.read()
        data_source_info = f"Using default file: {DEFAULT_PGN_FILE_PATH}"
        st.sidebar.info(data_source_info)
    except FileNotFoundError:
        st.sidebar.warning(f"Default PGN file '{DEFAULT_PGN_FILE_PATH}' not found. Please upload a file.")
        # Don't stop here, wait for upload or show message in main area
    except Exception as e:
         st.sidebar.error(f"Error reading default PGN file: {e}")
         st.stop()

# --- Data Loading ---
df = None # Initialize df
if pgn_content:
    try:
        # Pass PLAYER_NAME to the processing function
        df = load_and_process_pgn(pgn_content, PLAYER_NAME)
        if df.empty:
             st.error(f"No games found or processed for {PLAYER_NAME} in the provided PGN.")
             st.stop()
        data_loaded = True
    except Exception as e:
        st.error(f"An error occurred during PGN processing:")
        st.exception(e) # Shows detailed traceback for debugging
        data_loaded = False
        st.stop()
else:
    st.info("Please upload a PGN file or place 'firouzja_games.pgn' next to the script.")
    data_loaded = False
    st.stop() # Stop if no data could be loaded

# --- Main App Area (Only if data loaded successfully) ---
if data_loaded and df is not None:

    st.title(f"Chess Insights: {PLAYER_NAME}")
    st.caption(data_source_info + f" | Total Games Analyzed: {len(df)}")

    analysis_options = [
        "Overview",
        "Time and Date Analysis",
        "ECO and Opening Analysis",
        "Opponent Analysis",
        "Games against GMs",
        # "Famous Opponent Analysis", # Keep commented out for now
        "Time Forfeit Analysis"
    ]
    st.sidebar.title("Analysis Sections")
    selected_section = st.sidebar.selectbox("Choose a section:", analysis_options)

    # --- Display Selected Section ---

    if selected_section == "Overview":
        st.header("📊 General Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_win_loss_pie(df, PLAYER_NAME), use_container_width=True)
        with col2:
            st.plotly_chart(plot_win_loss_by_color(df), use_container_width=True)

        st.plotly_chart(plot_rating_trend(df, PLAYER_NAME), use_container_width=True)
        st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)


    elif selected_section == "Time and Date Analysis":
        st.header("📅 Time and Date Analysis")
        tab1, tab2, tab3 = st.tabs(["Games Over Time", "Performance by Year", "Performance by Time Control"])
        with tab1:
            st.plotly_chart(plot_games_per_year(df), use_container_width=True)
            # Add more time related plots if available in notebook (e.g., by month/day)
        with tab2:
            st.plotly_chart(plot_win_rate_per_year(df), use_container_width=True)
        with tab3:
            st.plotly_chart(plot_performance_by_time_control(df), use_container_width=True)
            st.dataframe(df.groupby('TimeControl_Category')['PlayerResultString'].value_counts().unstack(fill_value=0))


    elif selected_section == "ECO and Opening Analysis":
        st.header("📖 ECO and Opening Analysis")
        tab1, tab2 = st.tabs(["Opening Frequency", "Opening Performance"])
        with tab1:
            st.plotly_chart(plot_opening_frequency(df, top_n=25), use_container_width=True)
            # Show table too
            st.dataframe(df['OpeningName'].value_counts().reset_index(name='Count').head(25))
        with tab2:
            # Add slider for min_games threshold?
            min_games_opening = st.slider("Minimum games played for performance calculation:", 1, 20, 5)
            st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_opening, top_n=25), use_container_width=True)


    elif selected_section == "Opponent Analysis":
        st.header("👥 Opponent Analysis")
        tab1, tab2 = st.tabs(["Most Frequent Opponents", "Elo Difference vs Result"]) # Reuse Elo diff plot here maybe
        with tab1:
            st.plotly_chart(plot_most_frequent_opponents(df, top_n=25), use_container_width=True)
             # Show table too
            st.dataframe(df['OpponentName'].value_counts().reset_index(name='Games Played').head(25))
        with tab2:
             st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)
             # Add more opponent analysis if present in notebook


    elif selected_section == "Games against GMs":
        st.header("👑 Analysis Against Grandmasters (GMs)")
        gm_games = filter_and_analyze_gms(df)
        if not gm_games.empty:
            st.success(f"Found **{len(gm_games)}** games against opponents with the title 'GM'. Analyzing this subset...")

            tab_gm1, tab_gm2, tab_gm3, tab_gm4 = st.tabs(["GM Performance Summary", "Rating Trend vs GMs", "Openings vs GMs", "Frequent GM Opponents"])

            with tab_gm1:
                 st.plotly_chart(plot_win_loss_pie(gm_games, f"{PLAYER_NAME} vs GMs"), use_container_width=True)
                 st.dataframe(gm_games['PlayerResultString'].value_counts().reset_index(name='Count'))
                 st.plotly_chart(plot_win_loss_by_color(gm_games), use_container_width=True)


            with tab_gm2:
                st.plotly_chart(plot_rating_trend(gm_games, f"{PLAYER_NAME} (vs GMs)"), use_container_width=True)

            with tab_gm3:
                 st.plotly_chart(plot_opening_frequency(gm_games, top_n=15), use_container_width=True)
                 min_games_gm_opening = st.slider("Min games for GM opening performance:", 1, 10, 3, key="gm_slider")
                 st.plotly_chart(plot_win_rate_by_opening(gm_games, min_games=min_games_gm_opening, top_n=15), use_container_width=True)

            with tab_gm4:
                st.plotly_chart(plot_most_frequent_opponents(gm_games, top_n=15), use_container_width=True)
                st.dataframe(gm_games['OpponentName'].value_counts().reset_index(name='Games Played').head(15))

        else:
            st.warning("No games found against opponents with the title 'GM' in the dataset.")


    # elif selected_section == "Famous Opponent Analysis":
        # ... (Code to implement this later if desired) ...


    elif selected_section == "Time Forfeit Analysis":
        st.header("⏱️ Time Forfeit Analysis")
        if 'Termination' in df.columns:
            tf_games, wins_tf, losses_tf = filter_and_analyze_time_forfeits(df)
            if not tf_games.empty:
                st.success(f"Found **{len(tf_games)}** games ending due to time forfeit.")
                col_tf1, col_tf2 = st.columns(2)
                col_tf1.metric("Games Won on Time", wins_tf)
                col_tf2.metric("Games Lost on Time", losses_tf)

                st.write("### Games ending in Time Forfeit:")
                st.dataframe(tf_games[['Date', 'Event', 'OpponentName', 'PlayerColor', 'PlayerResultString', 'TimeControl', 'PlyCount']].sort_values('Date', ascending=False))

                # Optional: Analyze further (e.g., forfeits by time control)
                st.write("### Time Forfeits by Time Control:")
                st.dataframe(tf_games['TimeControl_Category'].value_counts())

            else:
                 st.warning("No games found with 'Time forfeit' termination.")
        else:
            st.warning("Column 'Termination' not found in the data. Cannot analyze time forfeits.")

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info("App based on Kaggle analysis by 'mynameiskaveh'. Uses Streamlit, Pandas, Plotly, python-chess.")

# --- End of App ---
