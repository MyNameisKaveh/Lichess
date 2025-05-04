# -*- coding: utf-8 -*-
# =============================================
# Streamlit App for Chess Game Analysis - Lichess API Version
# v9: FINAL FINAL SyntaxError fix in categorize_time_control.
# =============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta, timezone
import time
import re
import traceback

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Lichess Insights", page_icon="♟️")

# --- Constants & Defaults ---
TIME_PERIOD_OPTIONS = { "Last Month": timedelta(days=30), "Last 3 Months": timedelta(days=90), "Last Year": timedelta(days=365) }
DEFAULT_TIME_PERIOD = "Last Year"
PERF_TYPE_OPTIONS_SINGLE = ['Bullet', 'Blitz', 'Rapid']
DEFAULT_PERF_TYPE = 'Bullet'
DEFAULT_RATED_ONLY = True
FAMOUS_OPPONENTS = [ "DrNykterstein", "MagnusCarlsen", "Hikaru", "AnishGiri", "FabianoCaruana",
                     "lachesisQ", "WesleySo", "GMWSO", "VladislavArtemiev", "Duhless", ]

# =============================================
# Helper Function: Categorize Time Control *** FINAL SYNTAX CORRECTED ***
# =============================================
def categorize_time_control(tc_str, speed_info):
    """Categorizes time control based on speed info or parsed string."""
    # Prioritize speed info if available and standard
    if isinstance(speed_info, str) and speed_info in ['bullet', 'blitz', 'rapid', 'classical', 'correspondence']:
        return speed_info.capitalize()

    # Fallback to parsing the time_control_str (e.g., "180+0")
    if not isinstance(tc_str, str) or tc_str in ['-', '?', 'Unknown']:
        return 'Unknown'
    if tc_str == 'Correspondence':
        return 'Correspondence'

    if '+' in tc_str:
        # This is the block that caused the error repeatedly. Structure MUST be correct now.
        try: # <<< Outer TRY for the whole '+'-logic parsing
            parts = tc_str.split('+')
            if len(parts) == 2:
                # Nested check for integer conversion
                try: # <<< Inner TRY specifically for int conversion
                    base = int(parts[0])
                    increment = int(parts[1])
                except ValueError:
                    return 'Unknown' # Failed converting parts to int

                total = base + 40 * increment # Estimate time for 40 moves
                if total >= 1500: return 'Classical' # >= 25 min
                if total >= 480: return 'Rapid'     # >= 8 min
                if total >= 180: return 'Blitz'     # >= 3 min
                if total > 0 : return 'Bullet'      # < 3 min
                return 'Unknown' # Handle cases like 0+0?
            else:
                return 'Unknown' # Invalid format if not 2 parts
        except IndexError: # Catch potential errors if split doesn't yield enough parts (unlikely with check above but safe)
             return 'Unknown'
        except Exception: # Catch any other unexpected error during parsing
             return 'Unknown'
    else: # Handle cases with only base time (no increment)
        try: # <<< TRY for base time integer conversion
            base = int(tc_str)
            if base >= 1500: return 'Classical'
            if base >= 480: return 'Rapid'
            if base >= 180: return 'Blitz'
            if base > 0 : return 'Bullet'
            return 'Unknown'
        except ValueError: # <<< Corresponding EXCEPT for base time conversion failure
            # If conversion fails, check for keywords as last resort
            tc_lower = tc_str.lower()
            if 'classical' in tc_lower: return 'Classical'
            if 'rapid' in tc_lower: return 'Rapid'
            if 'blitz' in tc_lower: return 'Blitz'
            if 'bullet' in tc_lower: return 'Bullet'
            return 'Unknown'

# =============================================
# API Data Loading and Processing Function (Unchanged from v7)
# =============================================
@st.cache_data(ttl=3600)
def load_from_lichess_api(username: str, time_period_key: str, perf_type: str, rated: bool):
    # ... (Code identical to version 7/8 - it calls the corrected helper now) ...
    if not username: st.warning("Please enter a Lichess username."); return pd.DataFrame()
    if not perf_type: st.warning("Please select a game type."); return pd.DataFrame()
    username_lower = username.lower()
    st.info(f"Fetching games for '{username}' ({time_period_key} | Type: {perf_type})...")
    since_timestamp_ms = None
    time_delta = TIME_PERIOD_OPTIONS.get(time_period_key)
    if time_delta: start_date = datetime.now(timezone.utc) - time_delta; since_timestamp_ms = int(start_date.timestamp() * 1000); st.caption(f"Fetching since: {start_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    api_params = {"rated": str(rated).lower(), "perfType": perf_type.lower(), "opening": "true","moves": "false", "tags": "false", "pgnInJson": "false" }
    if since_timestamp_ms: api_params["since"] = since_timestamp_ms
    api_url = f"https://lichess.org/api/games/user/{username}"; headers = {"Accept": "application/x-ndjson"}
    all_games_data = []; error_counter = 0; games_processed_for_log = 0
    try:
        with st.spinner(f"Calling Lichess API for {username} ({perf_type} games)..."):
            response = requests.get(api_url, params=api_params, headers=headers, stream=True); response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    game_data_raw = line.decode('utf-8'); game_data = None; games_processed_for_log += 1
                    try:
                        game_data = json.loads(game_data_raw)
                        white_info=game_data.get('players',{}).get('white',{}); black_info=game_data.get('players',{}).get('black',{})
                        white_user=white_info.get('user',{}); black_user=black_info.get('user',{})
                        opening_info=game_data.get('opening',{}); clock_info=game_data.get('clock')
                        game_id=game_data.get('id','N/A'); created_at_ms=game_data.get('createdAt')
                        game_date=pd.to_datetime(created_at_ms,unit='ms',utc=True,errors='coerce');
                        if pd.isna(game_date): continue
                        variant=game_data.get('variant','standard'); speed=game_data.get('speed','unknown')
                        perf=game_data.get('perf','unknown'); status=game_data.get('status','unknown'); winner=game_data.get('winner')
                        white_name=white_user.get('name','Unknown'); black_name=black_user.get('name','Unknown')
                        white_title=white_user.get('title'); black_title=black_user.get('title')
                        white_rating=pd.to_numeric(white_info.get('rating'),errors='coerce')
                        black_rating=pd.to_numeric(black_info.get('rating'),errors='coerce')
                        player_color,player_elo,opp_name_raw,opp_title_raw,opp_elo=(None,None,'Unknown',None,None)
                        if username_lower==white_name.lower(): player_color,player_elo,opp_name_raw,opp_title_raw,opp_elo=('White',white_rating,black_name,black_title,black_rating)
                        elif username_lower==black_name.lower(): player_color,player_elo,opp_name_raw,opp_title_raw,opp_elo=('Black',black_rating,white_name,white_title,white_rating)
                        else: continue
                        if player_color is None or pd.isna(player_elo) or pd.isna(opp_elo): continue
                        res_num,res_str=(0.5,"Draw")
                        if status not in ['draw','stalemate']:
                           if winner==player_color.lower(): res_num,res_str=(1,"Win")
                           elif winner is not None: res_num,res_str=(0,"Loss")
                        tc_str="Unknown"
                        if clock_info: init=clock_info.get('initial');incr=clock_info.get('increment');
                        if init is not None and incr is not None: tc_str=f"{init}+{incr}"
                        elif speed=='correspondence': tc_str="Correspondence"
                        eco=opening_info.get('eco','Unknown');op_name=opening_info.get('name','Unknown Opening').replace('?','').split(':')[0].strip()
                        term_map={"mate":"Normal","resign":"Normal","stalemate":"Normal","timeout":"Time forfeit","draw":"Normal","outoftime":"Time forfeit","cheat":"Cheat","noStart":"Aborted","unknownFinish":"Unknown","variantEnd":"Variant End"}
                        term=term_map.get(status,"Unknown")
                        opp_title_final='Unknown'
                        if opp_title_raw and opp_title_raw.strip(): opp_title_clean=opp_title_raw.replace(' ','').strip().upper();
                        if opp_title_clean and opp_title_clean!='?': opp_title_final=opp_title_clean
                        def clean_name(n): return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM)\s+','',n).strip()
                        opp_name_clean=clean_name(opp_name_raw)
                        all_games_data.append({'Date':game_date,'Event':perf,'White':white_name,'Black':black_name,'Result':"1-0" if winner=='white' else ("0-1" if winner=='black' else "1/2-1/2"),'WhiteElo':int(white_rating) if not pd.isna(white_rating) else 0,'BlackElo':int(black_rating) if not pd.isna(black_rating) else 0,'ECO':eco,'Opening':op_name,'TimeControl':tc_str,'Termination':term,'PlyCount':game_data.get('turns',0),'LichessID':game_id,'PlayerID':username,'PlayerColor':player_color,'PlayerElo':int(player_elo),'OpponentName':opp_name_clean,'OpponentNameRaw':opp_name_raw,'OpponentElo':int(opp_elo),'OpponentTitle':opp_title_final,'PlayerResultNumeric':res_num,'PlayerResultString':res_str,'Variant':variant,'Speed':speed,'Status':status,'PerfType':perf})
                    except json.JSONDecodeError: error_counter += 1
                    except Exception: error_counter += 1
    except requests.exceptions.RequestException as e: st.error(f"🚨 API Request Failed: {e}"); return pd.DataFrame()
    except Exception as e: st.error(f"🚨 Unexpected error: {e}"); st.text(traceback.format_exc()); return pd.DataFrame()
    if error_counter > 0: st.warning(f"Skipped {error_counter} entries due to processing errors.")
    if not all_games_data: st.warning(f"No games found for '{username}' matching criteria."); return pd.DataFrame()
    df = pd.DataFrame(all_games_data); st.success(f"Processed {len(df)} games.")
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce'); df = df.dropna(subset=['Date'])
        if df.empty: return df
        df['Year'] = df['Date'].dt.year; df['Month'] = df['Date'].dt.month; df['Hour'] = df['Date'].dt.hour
        df['DayOfWeekNum'] = df['Date'].dt.dayofweek; df['DayOfWeekName'] = df['Date'].dt.day_name()
        df['PlayerElo'] = df['PlayerElo'].astype(int); df['OpponentElo'] = df['OpponentElo'].astype(int)
        df['EloDiff'] = df['PlayerElo'] - df['OpponentElo']
        df['TimeControl_Category'] = df.apply(lambda row: categorize_time_control(row['TimeControl'], row['Speed']), axis=1) # Calls corrected func
        df = df.rename(columns={'Opening': 'OpeningName'}); df = df.sort_values(by='Date').reset_index(drop=True)
    return df

# =============================================
# Plotting Functions (Unchanged - dragmode=False applied)
# =============================================
# (Insert ALL plotting functions here - plot_win_loss_pie, ..., plot_most_frequent_opponents)
# ... (Code identical to previous version v8) ...
def plot_win_loss_pie(df, display_name):
    if 'PlayerResultString' not in df.columns: return go.Figure()
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values, names=result_counts.index, title=f'Overall Results for {display_name}',
                 color=result_counts.index, color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05 if x == 'Win' else 0 for x in result_counts.index])
    fig.update_layout(dragmode=False)
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
        fig = px.bar(color_results_pct, barmode='stack', title='Results by Color',
                     labels={'value': '%', 'PlayerColor': 'Played As', 'PlayerResultString': 'Result'},
                     color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                     text_auto='.1f', category_orders={"PlayerColor": ["White", "Black"]})
        fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="Color Played", dragmode=False)
        fig.update_traces(textangle=0)
        return fig
    except Exception: return go.Figure().update_layout(title="Error")

def plot_rating_trend(df, display_name):
    if not all(col in df.columns for col in ['Date', 'PlayerElo']): return go.Figure()
    df_plot = df.copy(); df_plot['PlayerElo'] = pd.to_numeric(df_plot['PlayerElo'], errors='coerce')
    df_sorted = df_plot[df_plot['PlayerElo'].notna() & (df_plot['PlayerElo'] > 0)].sort_values('Date')
    if df_sorted.empty: return go.Figure().update_layout(title=f"No Elo data for {display_name}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['PlayerElo'], mode='lines+markers', name='Elo',
                        line=dict(color='#1E88E5', width=2), marker=dict(size=5, opacity=0.7)))
    fig.update_layout(title=f'{display_name}\'s Rating Trend', xaxis_title='Date', yaxis_title='Elo Rating',
                      hovermode="x unified", xaxis_rangeslider_visible=True, dragmode=False)
    return fig

def plot_performance_vs_opponent_elo(df):
    if not all(col in df.columns for col in ['PlayerResultString', 'EloDiff']): return go.Figure()
    fig = px.box(df, x='PlayerResultString', y='EloDiff', title='Elo Advantage vs. Result',
                 labels={'PlayerResultString': 'Result', 'EloDiff': 'Your Elo - Opponent Elo'},
                 category_orders={"PlayerResultString": ["Win", "Draw", "Loss"]},
                 color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, points='outliers')
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_traces(marker=dict(opacity=0.8))
    fig.update_layout(dragmode=False)
    return fig

def plot_games_by_dow(df):
    if 'DayOfWeekName' not in df.columns: return go.Figure()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    games_by_dow = df['DayOfWeekName'].value_counts().reindex(dow_order, fill_value=0)
    fig = px.bar(games_by_dow, x=games_by_dow.index, y=games_by_dow.values,
                 title="Games Played per Day of Week", labels={'x': 'Day of Week', 'y': 'Number of Games'},
                 text=games_by_dow.values)
    fig.update_traces(marker_color='#9C27B0', textposition='outside')
    fig.update_layout(dragmode=False)
    return fig

def plot_winrate_by_dow(df):
    if not all(col in df.columns for col in ['DayOfWeekName', 'PlayerResultNumeric']): return go.Figure()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wins_by_dow = df[df['PlayerResultNumeric'] == 1].groupby('DayOfWeekName').size()
    total_by_dow = df.groupby('DayOfWeekName').size()
    win_rate = (wins_by_dow.reindex(total_by_dow.index, fill_value=0) / total_by_dow).fillna(0) * 100
    win_rate = win_rate.reindex(dow_order, fill_value=0)
    fig = px.bar(win_rate, x=win_rate.index, y=win_rate.values,
                 title="Win Rate (%) per Day of Week", labels={'x': 'Day of Week', 'y': 'Win Rate (%)'},
                 text=win_rate.values)
    fig.update_traces(marker_color='#FF9800', texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100], dragmode=False)
    return fig

def plot_games_by_hour(df):
    if 'Hour' not in df.columns: return go.Figure()
    games_by_hour = df['Hour'].value_counts().sort_index()
    games_by_hour = games_by_hour.reindex(range(24), fill_value=0)
    fig = px.bar(games_by_hour, x=games_by_hour.index, y=games_by_hour.values,
                 title="Games Played per Hour of Day (UTC)", labels={'x': 'Hour (UTC)', 'y': 'Number of Games'},
                 text=games_by_hour.values)
    fig.update_traces(marker_color='#03A9F4', textposition='outside')
    fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False)
    return fig

def plot_winrate_by_hour(df):
    if not all(col in df.columns for col in ['Hour', 'PlayerResultNumeric']): return go.Figure()
    wins_by_hour = df[df['PlayerResultNumeric'] == 1].groupby('Hour').size()
    total_by_hour = df.groupby('Hour').size()
    win_rate = (wins_by_hour.reindex(total_by_hour.index, fill_value=0) / total_by_hour).fillna(0) * 100
    win_rate = win_rate.reindex(range(24), fill_value=0)
    fig = px.line(win_rate, x=win_rate.index, y=win_rate.values, markers=True,
                  title="Win Rate (%) per Hour of Day (UTC)", labels={'x': 'Hour (UTC)', 'y': 'Win Rate (%)'})
    fig.update_traces(line_color='#8BC34A')
    fig.update_layout(yaxis_range=[0, 100], xaxis=dict(tickmode='linear'), dragmode=False)
    return fig

def plot_games_per_year(df):
    if 'Year' not in df.columns: return go.Figure()
    games_per_year = df['Year'].value_counts().sort_index()
    fig = px.bar(games_per_year, x=games_per_year.index, y=games_per_year.values,
                 title='Games Played Per Year', labels={'x': 'Year', 'y': 'Games'}, text=games_per_year.values)
    fig.update_traces(marker_color='#2196F3', textposition='outside')
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games", xaxis={'type': 'category'}, dragmode=False)
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
    fig.update_layout(yaxis_range=[0, 100], dragmode=False)
    return fig

def plot_performance_by_time_control(df):
     if not all(col in df.columns for col in ['TimeControl_Category', 'PlayerResultString']): return go.Figure()
     try:
        tc_results = df.groupby(['TimeControl_Category', 'PlayerResultString']).size().unstack(fill_value=0)
        for res in ['Win', 'Draw', 'Loss']:
            if res not in tc_results.columns: tc_results[res] = 0
        tc_results = tc_results[['Win', 'Draw', 'Loss']]
        total_per_tc = tc_results.sum(axis=1)
        tc_results_pct = tc_results.apply(lambda x: x*100/total_per_tc[x.name] if total_per_tc[x.name]>0 else 0, axis=1)
        found_categories = df['TimeControl_Category'].unique()
        cat_order_preferred = ['Bullet', 'Blitz', 'Rapid', 'Classical', 'Correspondence', 'Unknown']
        cat_order = [c for c in cat_order_preferred if c in found_categories] + [c for c in found_categories if c not in cat_order_preferred]
        tc_results_pct = tc_results_pct.reindex(index=cat_order).dropna(axis=0, how='all')
        fig = px.bar(tc_results_pct, title='Performance by Time Control',
                     labels={'value':'%', 'TimeControl_Category':'Category', 'PlayerResultString':'Result'},
                     color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'},
                     barmode='group', text_auto='.1f')
        fig.update_layout(xaxis_title="Time Control Category", yaxis_title="Percentage (%)", dragmode=False)
        fig.update_traces(textangle=0)
        return fig
     except Exception as e: return go.Figure().update_layout(title="Error")

def plot_opening_frequency(df, top_n=20):
    if 'OpeningName' not in df.columns: return go.Figure()
    opening_counts = df[df['OpeningName'] != 'Unknown Opening']['OpeningName'].value_counts().nlargest(top_n)
    fig = px.bar(opening_counts, y=opening_counts.index, x=opening_counts.values, orientation='h',
                 title=f'Top {top_n} Openings', labels={'y': 'Opening', 'x': 'Games'}, text=opening_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, dragmode=False)
    fig.update_traces(marker_color='#673AB7', textposition='outside')
    return fig

def plot_win_rate_by_opening(df, min_games=5, top_n=20):
    if not all(col in df.columns for col in ['OpeningName', 'PlayerResultNumeric']): return go.Figure()
    opening_stats = df.groupby('OpeningName').agg(total_games=('PlayerResultNumeric','count'), wins=('PlayerResultNumeric',lambda x:(x==1).sum()))
    opening_stats = opening_stats[(opening_stats['total_games']>=min_games) & (opening_stats.index!='Unknown Opening')].copy()
    if opening_stats.empty: return go.Figure().update_layout(title=f"No openings played >= {min_games} times")
    opening_stats['win_rate'] = (opening_stats['wins'] / opening_stats['total_games']) * 100
    opening_stats_plot = opening_stats.nlargest(top_n, 'win_rate')
    fig = px.bar(opening_stats_plot, y=opening_stats_plot.index, x='win_rate', orientation='h',
                 title=f'Top {top_n} Openings by Win Rate (Min {min_games} games)',
                 labels={'win_rate':'Win Rate (%)', 'OpeningName':'Opening'}, text='win_rate')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside', marker_color='#009688')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Win Rate (%)", yaxis_title="Opening", dragmode=False)
    return fig

def plot_most_frequent_opponents(df, top_n=20):
    if 'OpponentName' not in df.columns: return go.Figure()
    opp_counts = df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().nlargest(top_n)
    fig = px.bar(opp_counts, y=opp_counts.index, x=opp_counts.values, orientation='h',
                 title=f'Top {top_n} Opponents', labels={'y': 'Opponent', 'x': 'Games'}, text=opp_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, dragmode=False)
    fig.update_traces(marker_color='#FF5722', textposition='outside')
    return fig

# =============================================
# Helper Functions
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
# Streamlit App Layout - v9 (Final Syntax Fix)
# =============================================

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
lichess_username = st.sidebar.text_input("Lichess Username:", key="username_input", placeholder="e.g., DrNykterstein")
time_period = st.sidebar.selectbox("Time Period:", options=list(TIME_PERIOD_OPTIONS.keys()), key="time_period_select")
selected_perf_type = st.sidebar.selectbox("Game Type:", options=PERF_TYPE_OPTIONS_SINGLE, index=PERF_TYPE_OPTIONS_SINGLE.index(DEFAULT_PERF_TYPE), key="perf_type_select")
analyze_button = st.sidebar.button("Analyze Games", key="analyze_button", use_container_width=True)
st.sidebar.markdown("---")

# --- Main Area Title ---
st.title("♟️ Lichess Insights")

# --- Data Loading State Management ---
if 'analysis_df' not in st.session_state: st.session_state.analysis_df = None
if 'current_username' not in st.session_state: st.session_state.current_username = ""
if 'current_time_period' not in st.session_state: st.session_state.current_time_period = ""
if 'current_perf_type' not in st.session_state: st.session_state.current_perf_type = ""

# --- Trigger Analysis ---
if analyze_button and lichess_username:
    if (lichess_username != st.session_state.current_username or
            time_period != st.session_state.current_time_period or
            selected_perf_type != st.session_state.current_perf_type):
        if not selected_perf_type: st.warning("Please select a game type.")
        else:
            st.session_state.analysis_df = None
            if 'selected_section' in st.session_state: del st.session_state['selected_section']
            df_loaded = load_from_lichess_api(lichess_username, time_period, selected_perf_type, DEFAULT_RATED_ONLY)
            st.session_state.analysis_df = df_loaded
            st.session_state.current_username = lichess_username
            st.session_state.current_time_period = time_period
            st.session_state.current_perf_type = selected_perf_type
            st.rerun()
    else:
        st.sidebar.info("Settings haven't changed.")

# --- Display Area ---
if isinstance(st.session_state.analysis_df, pd.DataFrame) and not st.session_state.analysis_df.empty:
    df = st.session_state.analysis_df
    current_display_name = st.session_state.current_username
    current_perf_type = st.session_state.current_perf_type

    st.write(f"Analysis for **{current_display_name}** | Period: **{st.session_state.current_time_period}** | Type: **{current_perf_type.capitalize()}**")
    st.caption(f"Total Rated Games Analyzed: **{len(df):,}**")
    st.markdown("---")

    st.sidebar.title("📊 Analysis Sections")
    analysis_options = [ "Overview", "Time & Date Analysis", "ECO & Opening Analysis", "Opponent Analysis",
                         "Games against GMs", "Famous Opponent Analysis", "Time Forfeit Analysis" ]
    if 'selected_section' not in st.session_state: st.session_state.selected_section = "Overview"
    selected_section = st.sidebar.selectbox( "Choose section:", analysis_options,
         index=analysis_options.index(st.session_state.selected_section), key="section_select")
    st.session_state.selected_section = selected_section

    # --- Display Content Based on Selected Section ---
    # (Code for displaying sections is identical to v7)
    if selected_section == "Overview":
        st.header("📈 General Overview")
        col_ov1, col_ov2 = st.columns(2);
        with col_ov1: st.plotly_chart(plot_win_loss_pie(df, current_display_name), use_container_width=True)
        with col_ov2: st.plotly_chart(plot_win_loss_by_color(df), use_container_width=True)
        st.plotly_chart(plot_rating_trend(df, current_display_name), use_container_width=True)
        st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)

    elif selected_section == "Time & Date Analysis":
        st.header("📅 Time & Date Analysis")
        st.plotly_chart(plot_games_per_year(df), use_container_width=True)
        st.plotly_chart(plot_win_rate_per_year(df), use_container_width=True)
        st.plotly_chart(plot_performance_by_time_control(df), use_container_width=True)
        st.subheader("Performance by Day of Week")
        col_dow1, col_dow2 = st.columns(2)
        with col_dow1: st.plotly_chart(plot_games_by_dow(df), use_container_width=True)
        with col_dow2: st.plotly_chart(plot_winrate_by_dow(df), use_container_width=True)
        st.subheader("Performance by Hour of Day (UTC)")
        col_hod1, col_hod2 = st.columns(2)
        with col_hod1: st.plotly_chart(plot_games_by_hour(df), use_container_width=True)
        with col_hod2: st.plotly_chart(plot_winrate_by_hour(df), use_container_width=True)

    elif selected_section == "ECO & Opening Analysis":
        st.header("📖 ECO & Opening Analysis")
        n_openings = st.slider("Num top openings:", 5, 50, 20, key="n_openings_freq_eco")
        st.plotly_chart(plot_opening_frequency(df, top_n=n_openings), use_container_width=True)
        min_games_opening = st.slider("Min games for perf.:", 1, 25, 5, key="min_games_perf_eco")
        n_openings_perf = st.slider("Num openings by win rate:", 5, 50, 20, key="n_openings_perf_eco")
        st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_opening, top_n=n_openings_perf), use_container_width=True)

    elif selected_section == "Opponent Analysis":
        st.header("👥 Opponent Analysis")
        n_opponents = st.slider("Num top opponents:", 5, 50, 20, key="n_opponents_freq_opp")
        st.plotly_chart(plot_most_frequent_opponents(df, top_n=n_opponents), use_container_width=True)
        st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)
        st.markdown(f"#### Top {n_opponents} Opponents List")
        try: st.dataframe(df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().reset_index(name='Games').head(n_opponents))
        except KeyError: st.warning("Could not generate table.")

    elif selected_section == "Games against GMs":
        st.header("👑 Analysis Against Grandmasters (GMs)")
        gm_games = filter_and_analyze_gms(df)
        if not gm_games.empty:
            st.success(f"Found **{len(gm_games):,}** games vs 'GM'. Analyzing subset...")
            st.plotly_chart(plot_win_loss_pie(gm_games, f"{current_display_name} vs GMs"), use_container_width=True)
            st.plotly_chart(plot_win_loss_by_color(gm_games), use_container_width=True)
            st.plotly_chart(plot_rating_trend(gm_games, f"{current_display_name} (vs GMs)"), use_container_width=True)
            st.plotly_chart(plot_opening_frequency(gm_games, top_n=15), use_container_width=True)
            min_games_gm = st.slider("Min games (GM opening):", 1, 10, 3, key="gm_open_slider")
            st.plotly_chart(plot_win_rate_by_opening(gm_games, min_games=min_games_gm, top_n=15), use_container_width=True)
            st.plotly_chart(plot_most_frequent_opponents(gm_games, top_n=15), use_container_width=True)
            st.markdown("#### Frequent GM Opponents List:")
            st.dataframe(gm_games['OpponentName'].value_counts().reset_index(name='Games').head(15))
        else: st.warning("ℹ️ No games found vs 'GM' title.")

    elif selected_section == "Famous Opponent Analysis":
        st.header("🌟 Analysis Against Famous Opponents")
        st.caption(f"Comparing against: {', '.join(FAMOUS_OPPONENTS)}")
        famous_games = df[df['OpponentNameRaw'].isin(FAMOUS_OPPONENTS)]
        if not famous_games.empty:
            st.success(f"Found **{len(famous_games)}** games against this list.")
            h2h = famous_games.groupby('OpponentNameRaw')['PlayerResultString'].value_counts().unstack(fill_value=0)
            for res in ['Win','Loss','Draw']: h2h[res] = h2h.get(res, 0)
            h2h = h2h[['Win', 'Loss', 'Draw']]; h2h['Total'] = h2h.sum(axis=1); h2h['Score'] = h2h['Win'] + 0.5 * h2h['Draw']
            st.markdown("#### Head-to-Head Results:")
            st.dataframe(h2h.sort_values('Total', ascending=False))
            st.plotly_chart(plot_win_loss_pie(famous_games, f"{current_display_name} vs Famous List"), use_container_width=True)
        else: st.warning(f"ℹ️ No games found against the predefined famous opponents list.")

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
    st.sidebar.info(f"Analysis for {current_display_name}.")

elif not analyze_button and st.session_state.analysis_df is None:
     st.info("☝️ Configure settings in the sidebar and click 'Analyze Games'.")
elif analyze_button and (not isinstance(st.session_state.analysis_df, pd.DataFrame) or st.session_state.analysis_df.empty):
     st.warning("No analysis data generated. Check username or try different settings.")

# --- End of App ---
