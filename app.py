import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import chess.pgn
import io
from collections import Counter
import datetime
import re

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Chess Insights: Alireza Firouzja")

# --- Constants ---
PLAYER_ID = "alireza2003" # <--- *** ID کاربری صحیح برای فیلتر کردن ***
DISPLAY_NAME = "Alireza Firouzja" # نام کامل برای نمایش در عناوین و نمودارها
DEFAULT_PGN_FILE_PATH = "alireza2003.pgn" # <--- *** نام فایل PGN صحیح ***

# --- Data Loading and Caching ---
@st.cache_data
def load_and_process_pgn(pgn_file_content, player_id): # تابع حالا player_id می گیرد
    """Loads PGN data, extracts info, cleans, and engineers features using the correct player ID."""
    st.info(f"Processing PGN data for player ID: {player_id}...") # Inform user
    all_games_data = []
    pgn = io.StringIO(pgn_file_content)
    player_id_lower = player_id.lower() # از ID برای جستجو استفاده می کنیم
    game_counter = 0
    processed_game_counter = 0

    while True:
        game_headers = chess.pgn.read_headers(pgn)
        if game_headers is None:
            break
        game_counter += 1
        game_data = dict(game_headers)

        white_player_raw = game_data.get('White', '').strip()
        black_player_raw = game_data.get('Black', '').strip()
        result = game_data.get('Result', '*')
        date_raw = game_data.get('Date', '????.??.??')

        # --- Determine Player's Color, Opponent, Elos based on PLAYER_ID ---
        player_color = None
        opponent_name = "Unknown"
        opponent_elo = 0
        player_elo = 0
        opponent_title = ''

        def clean_name(name):
            # Keep title removal logic
            return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM)\s+', '', name).strip()

        matched = False
        # *** جستجو بر اساس player_id_lower ***
        if player_id_lower == white_player_raw.lower(): # Check for exact match with ID
            player_color = 'White'
            opponent_name = clean_name(black_player_raw)
            player_elo = int(game_data.get('WhiteElo', 0))
            opponent_elo = int(game_data.get('BlackElo', 0))
            opponent_title = game_data.get('BlackTitle', '')
            matched = True
        elif player_id_lower == black_player_raw.lower(): # Check for exact match with ID
            player_color = 'Black'
            opponent_name = clean_name(white_player_raw)
            player_elo = int(game_data.get('BlackElo', 0))
            opponent_elo = int(game_data.get('WhiteElo', 0))
            opponent_title = game_data.get('WhiteTitle', '')
            matched = True

        if not matched:
            continue # Skip game if player ID not found in White or Black field

        # --- Determine Player's Result ---
        player_result_numeric = -1
        player_result_string = "Unknown"
        # ... (result logic remains the same) ...
        if player_color == 'White':
            if result == '1-0': player_result_numeric, player_result_string = 1, "Win"
            elif result == '0-1': player_result_numeric, player_result_string = 0, "Loss"
            elif result == '1/2-1/2': player_result_numeric, player_result_string = 0.5, "Draw"
        elif player_color == 'Black':
            if result == '0-1': player_result_numeric, player_result_string = 1, "Win"
            elif result == '1-0': player_result_numeric, player_result_string = 0, "Loss"
            elif result == '1/2-1/2': player_result_numeric, player_result_string = 0.5, "Draw"


        if player_result_numeric == -1:
            continue # Skip games with unknown results

        # --- Date Handling ---
        game_date = pd.NaT
        try:
            clean_date_str = date_raw.replace('.??.??', '-01-01').replace('.??', '-01')
            game_date = pd.to_datetime(clean_date_str, format='%Y.%m.%d', errors='coerce')
        except ValueError:
            pass

        if pd.isna(game_date):
             continue # Skip if date is invalid

        # --- Store data ---
        game_info = {
            'Date': game_date, 'Event': game_data.get('Event', 'Unknown'),
            'White': white_player_raw, 'Black': black_player_raw, 'Result': result,
            'WhiteElo': int(game_data.get('WhiteElo', 0)), 'BlackElo': int(game_data.get('BlackElo', 0)),
            'ECO': game_data.get('ECO', 'Unknown'), 'Opening': game_data.get('Opening', 'Unknown'),
            'TimeControl': game_data.get('TimeControl', 'Unknown'), 'Termination': game_data.get('Termination', 'Unknown'),
            'PlyCount': int(game_data.get('PlyCount', 0)),
            'PlayerID': player_id, # Store the ID used for filtering
            'PlayerColor': player_color, 'PlayerElo': player_elo,
            'OpponentName': opponent_name, 'OpponentElo': opponent_elo, 'OpponentTitle': opponent_title,
            'PlayerResultNumeric': player_result_numeric, 'PlayerResultString': player_result_string,
        }
        all_games_data.append(game_info)
        processed_game_counter += 1

    st.info(f"Checked {game_counter} headers. Found and processed {processed_game_counter} games for player ID '{player_id}'.")

    if not all_games_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_games_data)

    # --- Feature Engineering (remains mostly the same) ---
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['EloDiff'] = df['PlayerElo'] - df['OpponentElo']
    # ... (TimeControl_Category function) ...
    def categorize_time_control(tc): # Keep the function as defined before
        if not isinstance(tc, str) or tc == '-' or tc == '?' or tc == 'Unknown': return 'Unknown'
        if '+' in tc:
            try:
                base, increment = map(int, tc.split('+'))
                total_time_estimate = base + 40 * increment
                if total_time_estimate >= 1500: return 'Classical'
                if total_time_estimate >= 600: return 'Rapid'
                if total_time_estimate >= 180: return 'Blitz'
                return 'Bullet'
            except ValueError: return 'Unknown'
        else:
            try:
                base = int(tc)
                if base >= 1500: return 'Classical'
                if base >= 600: return 'Rapid'
                if base >= 180: return 'Blitz'
                if base > 0 : return 'Bullet'
                return 'Unknown'
            except ValueError:
                if 'classical' in tc.lower(): return 'Classical'
                if 'rapid' in tc.lower(): return 'Rapid'
                if 'blitz' in tc.lower(): return 'Blitz'
                if 'bullet' in tc.lower(): return 'Bullet'
                return 'Unknown'
    df['TimeControl_Category'] = df['TimeControl'].apply(categorize_time_control)

    df['OpeningName'] = df.apply(lambda row: row['Opening'] if row['Opening'] != 'Unknown' and row['Opening'] != '?' else ('ECO: ' + row['ECO'] if row['ECO'] != 'Unknown' else 'Unknown Opening'), axis=1)
    df['OpponentTitle'] = df['OpponentTitle'].replace({' ':'Unknown'}).fillna('Unknown')
    df = df.sort_values(by='Date').reset_index(drop=True)

    return df

# --- Plotting Functions (Use DISPLAY_NAME for titles) ---

def plot_win_loss_pie(df, display_name): # <--- Takes display_name
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values, names=result_counts.index,
                 title=f'Overall Win/Loss/Draw Distribution for {display_name}', # Use display_name
                 color=result_counts.index, color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# ... (Other plot functions also need to accept display_name if they use it in the title) ...
def plot_rating_trend(df, display_name): # <--- Takes display_name
    df_sorted = df.dropna(subset=['PlayerElo', 'Date']).sort_values('Date')
    df_sorted = df_sorted[df_sorted['PlayerElo'] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['PlayerElo'], mode='lines+markers', name='Elo Rating',
                        line=dict(color='#1E88E5', width=2), marker=dict(color='#1E88E5', size=4)))
    fig.update_layout(title=f'{display_name}\'s Rating Trend Over Time', # Use display_name
                      xaxis_title='Date', yaxis_title='Elo Rating', hovermode="x unified")
    return fig

# --- (Keep other plotting functions as they were, they don't use player name in title) ---
# plot_win_loss_by_color(df)
# plot_performance_vs_opponent_elo(df)
# plot_games_per_year(df)
# plot_win_rate_per_year(df)
# plot_performance_by_time_control(df)
# plot_opening_frequency(df, top_n=20)
# plot_win_rate_by_opening(df, min_games=5, top_n=20)
# plot_most_frequent_opponents(df, top_n=20)
# filter_and_analyze_gms(df)
# filter_and_analyze_time_forfeits(df)
# Make sure these functions are copied correctly from the previous version if needed.


# --- Streamlit App Layout ---

# --- Sidebar ---
st.sidebar.title("Chess Analysis")
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
    try:
        # *** Attempt to read the CORRECT default file ***
        with open(DEFAULT_PGN_FILE_PATH, 'r', encoding='utf-8') as f:
            pgn_content = f.read()
        data_source_info = f"Using default file: {DEFAULT_PGN_FILE_PATH}"
        st.sidebar.info(data_source_info)
    except FileNotFoundError:
        # *** Update warning message for CORRECT filename ***
        st.sidebar.warning(f"Default PGN file '{DEFAULT_PGN_FILE_PATH}' not found. Please upload a file.")
    except Exception as e:
         st.sidebar.error(f"Error reading default PGN file: {e}")
         st.stop()

# --- Data Loading ---
df = None
if pgn_content:
    try:
        # *** Pass the CORRECT player ID to the processing function ***
        df = load_and_process_pgn(pgn_content, PLAYER_ID)
        if df.empty:
             # Use DISPLAY_NAME in user message
             st.error(f"No games found or processed for player ID '{PLAYER_ID}' ({DISPLAY_NAME}) in the provided PGN.")
             st.stop()
        data_loaded = True
    except Exception as e:
        st.error(f"An error occurred during PGN processing:")
        st.exception(e)
        data_loaded = False
        st.stop()
else:
    if not uploaded_file: # Only show if no file was uploaded either
       st.info(f"Please upload a PGN file or ensure '{DEFAULT_PGN_FILE_PATH}' is in the repository.")
    data_loaded = False
    st.stop()

# --- Main App Area ---
if data_loaded and df is not None:

    # *** Use DISPLAY_NAME for the main title ***
    st.title(f"Chess Insights: {DISPLAY_NAME}")
    st.caption(data_source_info + f" | Player ID: {PLAYER_ID} | Total Games Analyzed: {len(df)}")

    analysis_options = [
        "Overview", "Time and Date Analysis", "ECO and Opening Analysis",
        "Opponent Analysis", "Games against GMs", "Time Forfeit Analysis"
    ]
    st.sidebar.title("Analysis Sections")
    selected_section = st.sidebar.selectbox("Choose a section:", analysis_options)

    # --- Display Selected Section ---
    if selected_section == "Overview":
        st.header("📊 General Overview")
        col1, col2 = st.columns(2)
        with col1:
            # *** Pass DISPLAY_NAME to plot function ***
            st.plotly_chart(plot_win_loss_pie(df, DISPLAY_NAME), use_container_width=True)
        with col2:
            st.plotly_chart(plot_win_loss_by_color(df), use_container_width=True) # Doesn't need name

        # *** Pass DISPLAY_NAME to plot function ***
        st.plotly_chart(plot_rating_trend(df, DISPLAY_NAME), use_container_width=True)
        st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True) # Doesn't need name


    elif selected_section == "Games against GMs":
        st.header("👑 Analysis Against Grandmasters (GMs)")
        gm_games = filter_and_analyze_gms(df) # Assuming this function is defined correctly elsewhere
        if not gm_games.empty:
             st.success(f"Found **{len(gm_games)}** games against opponents with the title 'GM'. Analyzing this subset...")
             # ... (rest of the GM tabs) ...
             # Make sure to pass DISPLAY_NAME to plot_win_loss_pie and plot_rating_trend if called within tabs
             with st.tabs(["GM Performance Summary", "Rating Trend vs GMs", "Openings vs GMs", "Frequent GM Opponents"])[0]:
                 st.plotly_chart(plot_win_loss_pie(gm_games, f"{DISPLAY_NAME} vs GMs"), use_container_width=True) # Pass display name here
                 # ... other plots ...
             with st.tabs(...)[1]:
                  st.plotly_chart(plot_rating_trend(gm_games, f"{DISPLAY_NAME} (vs GMs)"), use_container_width=True) # Pass display name here
                  # ... other plots ...


    # ... (Code for other sections like Time, ECO, Opponent, Time Forfeit remains largely the same) ...
    # Make sure plotting functions used inside them don't incorrectly rely on PLAYER_ID for titles.


    # --- Footer ---
    st.sidebar.markdown("---")
    # Use DISPLAY_NAME in the info text
    st.sidebar.info(f"App based on Kaggle analysis for {DISPLAY_NAME}. Uses Streamlit, Pandas, Plotly, python-chess.")

# --- End of App ---
