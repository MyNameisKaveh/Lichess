# -*- coding: utf-8 -*-
# =============================================
# Streamlit App for Chess Game Analysis - Lichess API Version
# v17: Fixed SyntaxError in display section (multiple 'with' on one line).
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
import calendar

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Lichess Insights", page_icon="♟️")

# --- Constants & Defaults ---
TIME_PERIOD_OPTIONS = { "Last Month": timedelta(days=30), "Last 3 Months": timedelta(days=90), "Last Year": timedelta(days=365), "Last 3 Years": timedelta(days=3*365) }
DEFAULT_TIME_PERIOD = "Last Year"
PERF_TYPE_OPTIONS_SINGLE = ['Bullet', 'Blitz', 'Rapid']
DEFAULT_PERF_TYPE = 'Bullet'
DEFAULT_RATED_ONLY = True
ECO_CSV_PATH = "eco_to_opening.csv"
TITLES_TO_ANALYZE = ['GM', 'IM', 'FM', 'CM', 'WGM', 'WIM', 'WFM', 'WCM', 'NM']
FAMOUS_OPPONENTS = [ "DrNykterstein", "MagnusCarlsen", "Hikaru", "AnishGiri", "FabianoCaruana",
                     "lachesisQ", "WesleySo", "GMWSO", "VladislavArtemiev", "Duhless", ]

# =============================================
# Helper Function: Categorize Time Control (Correct from v16)
# =============================================
def categorize_time_control(tc_str, speed_info):
    # (Code identical to v16)
    if isinstance(speed_info, str) and speed_info in ['bullet', 'blitz', 'rapid', 'classical', 'correspondence']: return speed_info.capitalize()
    if not isinstance(tc_str, str) or tc_str in ['-', '?', 'Unknown','Correspondence']: return 'Unknown' if tc_str!='Correspondence' else 'Correspondence'
    if '+' in tc_str:
        try:
            parts = tc_str.split('+')
            if len(parts) != 2: return 'Unknown'
            base = int(parts[0]); increment = int(parts[1])
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
        except ValueError:
            tc_lower = tc_str.lower()
            if 'classical' in tc_lower: return 'Classical'
            if 'rapid' in tc_lower: return 'Rapid'
            if 'blitz' in tc_lower: return 'Blitz'
            if 'bullet' in tc_lower: return 'Bullet'
            return 'Unknown'

# =============================================
# Helper Function: Load ECO to Opening Mapping (Correct from v16)
# =============================================
@st.cache_data
def load_eco_mapping(csv_path):
    # (Code identical to v16)
    try: df_eco=pd.read_csv(csv_path);
         if "ECO Code" not in df_eco.columns or "Opening Name" not in df_eco.columns: st.error(f"ECO file missing required columns."); return {}
         eco_map=df_eco.drop_duplicates(subset=['ECO Code']).set_index('ECO Code')['Opening Name'].to_dict(); st.sidebar.success(f"Loaded {len(eco_map)} ECO mappings."); return eco_map
    except FileNotFoundError: st.sidebar.error(f"ECO mapping file '{csv_path}' not found."); return {}
    except Exception as e: st.sidebar.error(f"Error loading ECO mapping file: {e}"); return {}

# =============================================
# API Data Loading and Processing Function (Correct from v16)
# =============================================
@st.cache_data(ttl=3600)
def load_from_lichess_api(username: str, time_period_key: str, perf_type: str, rated: bool, eco_map: dict):
    # (Code identical to version 16)
    if not username: st.warning("Please enter a Lichess username."); return pd.DataFrame()
    if not perf_type: st.warning("Please select a game type."); return pd.DataFrame()
    username_lower = username.lower()
    st.info(f"Fetching games for '{username}' ({time_period_key} | Type: {perf_type})...")
    since_timestamp_ms = None; time_delta = TIME_PERIOD_OPTIONS.get(time_period_key)
    if time_delta: start_date = datetime.now(timezone.utc) - time_delta; since_timestamp_ms = int(start_date.timestamp() * 1000); st.caption(f"Fetching since: {start_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else: st.warning("Invalid time period selected.")
    api_params = {"rated":str(rated).lower(), "perfType":perf_type.lower(), "opening":"true", "moves":"false", "tags":"false", "pgnInJson":"false" }
    if since_timestamp_ms: api_params["since"] = since_timestamp_ms
    api_url = f"https://lichess.org/api/games/user/{username}"; headers = {"Accept":"application/x-ndjson"}
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
                        eco=opening_info.get('eco','Unknown'); op_name_api=opening_info.get('name','Unknown Opening').replace('?','').split(':')[0].strip()
                        op_name_custom=eco_map.get(eco, f"ECO: {eco}" if eco!='Unknown' else 'Unknown Opening')
                        term_map={"mate":"Normal","resign":"Normal","stalemate":"Normal","timeout":"Time forfeit","draw":"Normal","outoftime":"Time forfeit","cheat":"Cheat","noStart":"Aborted","unknownFinish":"Unknown","variantEnd":"Variant End"}
                        term=term_map.get(status,"Unknown")
                        opp_title_final = 'Unknown'
                        if opp_title_raw and isinstance(opp_title_raw, str) and opp_title_raw.strip():
                            cleaned_title = opp_title_raw.replace(' ', '').strip().upper()
                            if cleaned_title in TITLES_TO_ANALYZE: opp_title_final = cleaned_title
                        def clean_name(n): return re.sub(r'^(GM|IM|FM|CM|WGM|WIM|WFM|WCM|NM)\s+','',n).strip()
                        opp_name_clean=clean_name(opp_name_raw)
                        all_games_data.append({'Date':game_date,'Event':perf,'White':white_name,'Black':black_name,'Result':"1-0" if winner=='white' else ("0-1" if winner=='black' else "1/2-1/2"),'WhiteElo':int(white_rating) if not pd.isna(white_rating) else 0,'BlackElo':int(black_rating) if not pd.isna(black_rating) else 0,'ECO':eco,'OpeningName_API':op_name_api,'OpeningName_Custom':op_name_custom,'TimeControl':tc_str,'Termination':term,'PlyCount':game_data.get('turns',0),'LichessID':game_id,'PlayerID':username,'PlayerColor':player_color,'PlayerElo':int(player_elo),'OpponentName':opp_name_clean,'OpponentNameRaw':opp_name_raw,'OpponentElo':int(opp_elo),'OpponentTitle':opp_title_final,'PlayerResultNumeric':res_num,'PlayerResultString':res_str,'Variant':variant,'Speed':speed,'Status':status,'PerfType':perf})
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
        df['Year'] = df['Date'].dt.year; df['Month'] = df['Date'].dt.month; df['Day'] = df['Date'].dt.day
        df['Hour'] = df['Date'].dt.hour; df['DayOfWeekNum'] = df['Date'].dt.dayofweek; df['DayOfWeekName'] = df['Date'].dt.day_name()
        df['PlayerElo'] = df['PlayerElo'].astype(int); df['OpponentElo'] = df['OpponentElo'].astype(int)
        df['EloDiff'] = df['PlayerElo'] - df['OpponentElo']
        df['TimeControl_Category'] = df.apply(lambda row: categorize_time_control(row['TimeControl'], row['Speed']), axis=1) # Calls corrected func
        df = df.sort_values(by='Date').reset_index(drop=True)
    return df


# =============================================
# Plotting Functions (Unchanged from v15)
# =============================================
# (Insert ALL plotting functions here - code identical to previous version v15)
# ... plot_win_loss_pie, ..., plot_time_forfeit_by_tc ...
def plot_win_loss_pie(df, display_name):
    if 'PlayerResultString' not in df.columns: return go.Figure()
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values, names=result_counts.index, title=f'Overall Results for {display_name}', color=result_counts.index, color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05 if x == 'Win' else 0 for x in result_counts.index]); fig.update_layout(dragmode=False); return fig
def plot_win_loss_by_color(df):
    if not all(col in df.columns for col in ['PlayerColor', 'PlayerResultString']): return go.Figure()
    try: color_results=df.groupby(['PlayerColor','PlayerResultString']).size().unstack(fill_value=0)
    except KeyError: return go.Figure().update_layout(title="Error: Missing Columns")
    for res in ['Win','Draw','Loss']: color_results[res]=color_results.get(res,0)
    color_results=color_results[['Win','Draw','Loss']]; total=color_results.sum(axis=1); color_results_pct=color_results.apply(lambda x:x*100/total[x.name] if total[x.name]>0 else 0,axis=1)
    fig=px.bar(color_results_pct, barmode='stack', title='Results by Color', labels={'value':'%', 'PlayerColor':'Played As'}, color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, text_auto='.1f', category_orders={"PlayerColor":["White","Black"]})
    fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="Color Played", dragmode=False); fig.update_traces(textangle=0); return fig
def plot_rating_trend(df, display_name):
    if not all(col in df.columns for col in ['Date', 'PlayerElo']): return go.Figure()
    df_plot=df.copy(); df_plot['PlayerElo']=pd.to_numeric(df_plot['PlayerElo'],errors='coerce'); df_sorted=df_plot[df_plot['PlayerElo'].notna() & (df_plot['PlayerElo']>0)].sort_values('Date')
    if df_sorted.empty: return go.Figure().update_layout(title=f"No Elo data")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['PlayerElo'], mode='lines+markers', name='Elo', line=dict(color='#1E88E5',width=2), marker=dict(size=5,opacity=0.7)))
    fig.update_layout(title=f'{display_name}\'s Rating Trend', xaxis_title='Date', yaxis_title='Elo Rating', hovermode="x unified", xaxis_rangeslider_visible=True, dragmode=False); return fig
def plot_performance_vs_opponent_elo(df):
    if not all(col in df.columns for col in ['PlayerResultString', 'EloDiff']): return go.Figure()
    fig=px.box(df, x='PlayerResultString', y='EloDiff', title='Elo Advantage vs. Result', labels={'PlayerResultString':'Result', 'EloDiff':'Your Elo - Opponent Elo'}, category_orders={"PlayerResultString":["Win","Draw","Loss"]}, color='PlayerResultString', color_discrete_map={'Win':'#4CAF50','Draw':'#B0BEC5','Loss':'#F44336'}, points='outliers')
    fig.add_hline(y=0, line_dash="dash", line_color="grey"); fig.update_traces(marker=dict(opacity=0.8)); fig.update_layout(dragmode=False); return fig
def plot_games_by_dow(df):
    if 'DayOfWeekName' not in df.columns: return go.Figure()
    dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    games_by_dow=df['DayOfWeekName'].value_counts().reindex(dow_order, fill_value=0)
    fig=px.bar(games_by_dow, x=games_by_dow.index, y=games_by_dow.values, title="Games by Day of Week", labels={'x':'Day','y':'Games'}, text=games_by_dow.values)
    fig.update_traces(marker_color='#9C27B0', textposition='outside'); fig.update_layout(dragmode=False); return fig
def plot_winrate_by_dow(df):
    if not all(col in df.columns for col in ['DayOfWeekName', 'PlayerResultNumeric']): return go.Figure()
    dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wins_by_dow=df[df['PlayerResultNumeric']==1].groupby('DayOfWeekName').size(); total_by_dow=df.groupby('DayOfWeekName').size()
    win_rate=(wins_by_dow.reindex(total_by_dow.index,fill_value=0)/total_by_dow).fillna(0)*100
    win_rate=win_rate.reindex(dow_order,fill_value=0)
    fig=px.bar(win_rate, x=win_rate.index, y=win_rate.values, title="Win Rate (%) by Day", labels={'x':'Day','y':'Win Rate (%)'}, text=win_rate.values)
    fig.update_traces(marker_color='#FF9800', texttemplate='%{text:.1f}%', textposition='outside'); fig.update_layout(yaxis_range=[0,100], dragmode=False); return fig
def plot_games_by_hour(df):
    if 'Hour' not in df.columns: return go.Figure()
    games_by_hour=df['Hour'].value_counts().sort_index().reindex(range(24),fill_value=0)
    fig=px.bar(games_by_hour, x=games_by_hour.index, y=games_by_hour.values, title="Games by Hour (UTC)", labels={'x':'Hour','y':'Games'}, text=games_by_hour.values)
    fig.update_traces(marker_color='#03A9F4', textposition='outside'); fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False); return fig
def plot_winrate_by_hour(df):
    if not all(col in df.columns for col in ['Hour', 'PlayerResultNumeric']): return go.Figure()
    wins_by_hour=df[df['PlayerResultNumeric']==1].groupby('Hour').size(); total_by_hour=df.groupby('Hour').size()
    win_rate=(wins_by_hour.reindex(total_by_hour.index,fill_value=0)/total_by_hour).fillna(0)*100
    win_rate=win_rate.reindex(range(24),fill_value=0)
    fig=px.line(win_rate, x=win_rate.index, y=win_rate.values, markers=True, title="Win Rate (%) by Hour (UTC)", labels={'x':'Hour','y':'Win Rate (%)'})
    fig.update_traces(line_color='#8BC34A'); fig.update_layout(yaxis_range=[0,100], xaxis=dict(tickmode='linear'), dragmode=False); return fig
def plot_games_per_year(df):
    if 'Year' not in df.columns: return go.Figure()
    games_per_year = df['Year'].value_counts().sort_index()
    fig = px.bar(games_per_year, x=games_per_year.index, y=games_per_year.values, title='Games Per Year', labels={'x':'Year','y':'Games'}, text=games_per_year.values)
    fig.update_traces(marker_color='#2196F3', textposition='outside'); fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games", xaxis={'type':'category'}, dragmode=False); return fig
def plot_win_rate_per_year(df):
    if not all(col in df.columns for col in ['Year', 'PlayerResultNumeric']): return go.Figure()
    wins_per_year=df[df['PlayerResultNumeric']==1].groupby('Year').size(); total_per_year=df.groupby('Year').size()
    win_rate=(wins_per_year.reindex(total_per_year.index,fill_value=0)/total_per_year).fillna(0)*100
    win_rate.index=win_rate.index.astype(str)
    fig=px.line(win_rate, x=win_rate.index, y=win_rate.values, title='Win Rate (%) Per Year', markers=True, labels={'x':'Year','y':'Win Rate (%)'})
    fig.update_traces(line_color='#FFC107', line_width=2.5); fig.update_layout(yaxis_range=[0,100], dragmode=False); return fig
def plot_performance_by_time_control(df):
     if not all(col in df.columns for col in ['TimeControl_Category', 'PlayerResultString']): return go.Figure()
     try: tc_results=df.groupby(['TimeControl_Category','PlayerResultString']).size().unstack(fill_value=0)
     except KeyError: return go.Figure().update_layout(title="Error: Missing Columns")
     for res in ['Win','Draw','Loss']: tc_results[res]=tc_results.get(res,0)
     tc_results=tc_results[['Win','Draw','Loss']]; total=tc_results.sum(axis=1); tc_results_pct=tc_results.apply(lambda x:x*100/total[x.name] if total[x.name]>0 else 0, axis=1)
     found=df['TimeControl_Category'].unique(); pref=['Bullet','Blitz','Rapid','Classical','Correspondence','Unknown']; order=[c for c in pref if c in found]+[c for c in found if c not in pref]
     tc_results_pct=tc_results_pct.reindex(index=order).dropna(axis=0,how='all')
     fig=px.bar(tc_results_pct, title='Performance by Time Control', labels={'value':'%','TimeControl_Category':'Category'}, color='PlayerResultString', color_discrete_map={'Win':'#4CAF50','Draw':'#B0BEC5','Loss':'#F44336'}, barmode='group', text_auto='.1f')
     fig.update_layout(xaxis_title="Time Control Category", yaxis_title="Percentage (%)", dragmode=False); fig.update_traces(textangle=0); return fig
def plot_most_frequent_opponents(df, top_n=20):
    if 'OpponentName' not in df.columns: return go.Figure()
    opp_counts=df[df['OpponentName']!='Unknown']['OpponentName'].value_counts().nlargest(top_n)
    fig=px.bar(opp_counts, y=opp_counts.index, x=opp_counts.values, orientation='h', title=f'Top {top_n} Opponents', labels={'y':'Opponent','x':'Games'}, text=opp_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, dragmode=False); fig.update_traces(marker_color='#FF5722', textposition='outside'); return fig
def plot_time_forfeit_summary(wins_tf, losses_tf):
    data={'Outcome':['Won on Time','Lost on Time'],'Count':[wins_tf,losses_tf]}
    df_tf=pd.DataFrame(data)
    fig=px.bar(df_tf,x='Outcome',y='Count',title="Time Forfeit Summary", color='Outcome', color_discrete_map={'Won on Time':'#4CAF50','Lost on Time':'#F44336'}, text='Count')
    fig.update_layout(showlegend=False, dragmode=False); fig.update_traces(textposition='outside'); return fig
def plot_time_forfeit_by_tc(tf_games_df):
    if 'TimeControl_Category' not in tf_games_df.columns or tf_games_df.empty: return go.Figure().update_layout(title="No TF Data by Category")
    tf_by_tc=tf_games_df['TimeControl_Category'].value_counts()
    fig=px.bar(tf_by_tc,x=tf_by_tc.index,y=tf_by_tc.values, title="Time Forfeits by Time Control", labels={'x':'Category','y':'Forfeits'}, text=tf_by_tc.values)
    fig.update_layout(dragmode=False); fig.update_traces(marker_color='#795548', textposition='outside'); return fig
def plot_games_by_dom(df):
    if 'Day' not in df.columns: return go.Figure()
    games_by_dom=df['Day'].value_counts().sort_index().reindex(range(1,32),fill_value=0)
    fig=px.bar(games_by_dom,x=games_by_dom.index,y=games_by_dom.values, title="Games by Day of Month", labels={'x':'Day','y':'Games'}, text=games_by_dom.values)
    fig.update_traces(marker_color='#E91E63',textposition='outside'); fig.update_layout(xaxis=dict(tickmode='linear'),dragmode=False); return fig
def plot_winrate_by_dom(df):
    if not all(col in df.columns for col in ['Day', 'PlayerResultNumeric']): return go.Figure()
    wins_by_dom=df[df['PlayerResultNumeric']==1].groupby('Day').size(); total_by_dom=df.groupby('Day').size()
    win_rate=(wins_by_dom.reindex(total_by_dom.index,fill_value=0)/total_by_dom).fillna(0)*100
    win_rate=win_rate.reindex(range(1,32),fill_value=0)
    fig=px.line(win_rate,x=win_rate.index,y=win_rate.values, markers=True, title="Win Rate (%) by Day of Month", labels={'x':'Day','y':'Win Rate (%)'})
    fig.update_traces(line_color='#FF5722'); fig.update_layout(yaxis_range=[0,100],xaxis=dict(tickmode='linear'),dragmode=False); return fig
def plot_games_by_month(df):
    if 'Month' not in df.columns: return go.Figure()
    month_map = {i: calendar.month_abbr[i] for i in range(1, 13)}
    games_by_month = df['Month'].map(month_map).value_counts().reindex(calendar.month_abbr[1:], fill_value=0)
    fig = px.bar(games_by_month, x=games_by_month.index, y=games_by_month.values, title="Games Played per Month", labels={'x': 'Month', 'y': 'Number of Games'}, text=games_by_month.values)
    fig.update_traces(marker_color='#3F51B5', textposition='outside'); fig.update_layout(xaxis={'type': 'category'}, dragmode=False); return fig
def plot_winrate_by_month(df):
    if not all(col in df.columns for col in ['Month', 'PlayerResultNumeric']): return go.Figure()
    month_map = {i: calendar.month_abbr[i] for i in range(1, 13)}
    wins_by_month=df[df['PlayerResultNumeric']==1].groupby('Month').size(); total_by_month=df.groupby('Month').size()
    win_rate=(wins_by_month.reindex(total_by_month.index,fill_value=0)/total_by_month).fillna(0)*100
    win_rate.index=win_rate.index.map(month_map); win_rate=win_rate.reindex(calendar.month_abbr[1:],fill_value=0)
    fig=px.bar(win_rate, x=win_rate.index, y=win_rate.values, title="Win Rate (%) per Month", labels={'x': 'Month', 'y': 'Win Rate (%)'}, text=win_rate.values)
    fig.update_traces(marker_color='#00BCD4', texttemplate='%{text:.1f}%', textposition='outside'); fig.update_layout(yaxis_range=[0,100], xaxis={'type': 'category'}, dragmode=False); return fig
def plot_opening_frequency(df, top_n=20, opening_col='OpeningName_API'):
    if opening_col not in df.columns: return go.Figure().update_layout(title=f"Error: Column '{opening_col}' not found")
    source_label = "Lichess API" if opening_col == 'OpeningName_API' else "Custom ECO Map"
    df_plot = df[df[opening_col].apply(lambda x: isinstance(x, str))].copy()
    opening_counts = df_plot[df_plot[opening_col] != 'Unknown Opening'][opening_col].value_counts().nlargest(top_n)
    if opening_counts.empty: return go.Figure().update_layout(title=f"No valid opening data ({source_label})")
    fig = px.bar(opening_counts, y=opening_counts.index, x=opening_counts.values, orientation='h', title=f'Top {top_n} Openings ({source_label})', labels={'y':'Opening','x':'Games'}, text=opening_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, dragmode=False); fig.update_traces(marker_color='#673AB7', textposition='outside'); return fig
def plot_win_rate_by_opening(df, min_games=5, top_n=20, opening_col='OpeningName_API'):
    if not all(col in df.columns for col in [opening_col, 'PlayerResultNumeric']): return go.Figure().update_layout(title=f"Error: Missing columns")
    source_label = "Lichess API" if opening_col == 'OpeningName_API' else "Custom ECO Map"
    df_plot = df[df[opening_col].apply(lambda x: isinstance(x, str))].copy()
    opening_stats = df_plot.groupby(opening_col).agg(total_games=('PlayerResultNumeric','count'), wins=('PlayerResultNumeric',lambda x:(x==1).sum()))
    opening_stats = opening_stats[(opening_stats['total_games']>=min_games)&(opening_stats.index!='Unknown Opening')].copy()
    if opening_stats.empty: return go.Figure().update_layout(title=f"No openings >= {min_games} games ({source_label})")
    opening_stats['win_rate']=(opening_stats['wins']/opening_stats['total_games'])*100
    opening_stats_plot=opening_stats.nlargest(top_n, 'win_rate')
    fig=px.bar(opening_stats_plot, y=opening_stats_plot.index, x='win_rate', orientation='h', title=f'Top {top_n} Openings by Win Rate (Min {min_games} games, {source_label})', labels={'win_rate':'Win Rate (%)',opening_col:'Opening'}, text='win_rate')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside', marker_color='#009688'); fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Win Rate (%)", dragmode=False); return fig


# =============================================
# Helper Functions
# =============================================
# (Identical to v15)
def filter_and_analyze_titled(df, titles):
    if 'OpponentTitle' not in df.columns: st.warning("OpponentTitle column not found."); return pd.DataFrame()
    titles_upper = [str(t).upper() for t in titles]
    titled_games = df[df['OpponentTitle'].isin(titles_upper)].copy(); return titled_games
def filter_and_analyze_time_forfeits(df):
    if 'Termination' not in df.columns: return pd.DataFrame(), 0, 0
    tf_games = df[df['Termination'].str.contains("Time forfeit", na=False, case=False)].copy()
    if tf_games.empty: return tf_games, 0, 0
    wins_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 1]); losses_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 0])
    return tf_games, wins_tf, losses_tf

# =============================================
# Streamlit App Layout - v17 (Final Syntax Fix, Updated Structure)
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

# --- Load ECO Mapping ---
eco_mapping = load_eco_mapping(ECO_CSV_PATH)

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
            st.session_state.analysis_df = None; st.session_state.selected_section = "1. Overview & General Stats"
            df_loaded = load_from_lichess_api(lichess_username, time_period, selected_perf_type, DEFAULT_RATED_ONLY, eco_mapping)
            st.session_state.analysis_df = df_loaded; st.session_state.current_username = lichess_username
            st.session_state.current_time_period = time_period; st.session_state.current_perf_type = selected_perf_type
            st.rerun()
    else: st.sidebar.info("Settings unchanged.")


# --- Display Area ---
if isinstance(st.session_state.analysis_df, pd.DataFrame) and not st.session_state.analysis_df.empty:
    df = st.session_state.analysis_df
    current_display_name = st.session_state.current_username
    current_perf_type = st.session_state.current_perf_type

    st.write(f"Analysis for **{current_display_name}** | Period: **{st.session_state.current_time_period}** | Type: **{current_perf_type.capitalize()}**")
    st.caption(f"Total Rated Games Analyzed: **{len(df):,}**"); st.markdown("---")

    # --- Sidebar Navigation ---
    st.sidebar.title("📊 Analysis Sections")
    analysis_options = [ "1. Overview & General Stats", "2. Performance Over Time", "3. Performance by Color",
                         "4. Time & Date Analysis", "5. ECO & Opening Analysis", "6. Opponent Analysis",
                         "7. Games against Titled Players", "8. Termination Analysis" ]
    if 'selected_section' not in st.session_state or st.session_state.selected_section not in analysis_options: st.session_state.selected_section = analysis_options[0]
    selected_section = st.sidebar.selectbox( "Choose section:", analysis_options, index=analysis_options.index(st.session_state.selected_section), key="section_select")
    st.session_state.selected_section = selected_section

    # --- Display Content Based on Selected Section ---
    st.header(selected_section)

    if selected_section == analysis_options[0]: # Overview
        st.plotly_chart(plot_win_loss_pie(df, current_display_name), use_container_width=True)
        total_games=len(df); wins=len(df[df['PlayerResultNumeric']==1]); losses=len(df[df['PlayerResultNumeric']==0]); draws=len(df[df['PlayerResultNumeric']==0.5])
        win_rate=(wins/total_games*100) if total_games>0 else 0; avg_opp_elo=df['OpponentElo'].mean()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Games", f"{total_games:,}"); col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("W|L|D", f"{wins}|{losses}|{draws}"); col4.metric("Avg Opp Elo", f"{avg_opp_elo:.0f}" if not pd.isna(avg_opp_elo) else "N/A")

    elif selected_section == analysis_options[1]: # Perf Over Time
        st.plotly_chart(plot_rating_trend(df, current_display_name), use_container_width=True)

    elif selected_section == analysis_options[2]: # Perf By Color
         st.plotly_chart(plot_win_loss_by_color(df), use_container_width=True)

    elif selected_section == analysis_options[3]: # Time & Date (Reordered)
        st.subheader("Performance by Year"); col_y1, col_y2 = st.columns(2);
        with col_y1: st.plotly_chart(plot_games_per_year(df), use_container_width=True); with col_y2: st.plotly_chart(plot_win_rate_per_year(df), use_container_width=True)
        st.subheader("Performance by Month"); col_m1, col_m2 = st.columns(2);
        with col_m1: st.plotly_chart(plot_games_by_month(df), use_container_width=True); with col_m2: st.plotly_chart(plot_winrate_by_month(df), use_container_width=True)
        st.subheader("Performance by Day of Week"); col_dow1, col_dow2 = st.columns(2);
        with col_dow1: st.plotly_chart(plot_games_by_dow(df), use_container_width=True); with col_dow2: st.plotly_chart(plot_winrate_by_dow(df), use_container_width=True)
        st.subheader("Performance by Hour of Day (UTC)"); col_hod1, col_hod2 = st.columns(2);
        with col_hod1: st.plotly_chart(plot_games_by_hour(df), use_container_width=True); with col_hod2: st.plotly_chart(plot_winrate_by_hour(df), use_container_width=True)
        st.subheader("Performance by Day of Month"); col_dom1, col_dom2 = st.columns(2);
        with col_dom1: st.plotly_chart(plot_games_by_dom(df), use_container_width=True); with col_dom2: st.plotly_chart(plot_winrate_by_dom(df), use_container_width=True)
        st.subheader("Performance by Time Control Category"); st.plotly_chart(plot_performance_by_time_control(df), use_container_width=True)

    elif selected_section == analysis_options[4]: # ECO & Opening
        st.subheader("Opening Analysis based on Lichess API Names")
        n_openings_api = st.slider("Num top (API):", 5, 50, 15, key="n_openings_freq_api")
        st.plotly_chart(plot_opening_frequency(df, top_n=n_openings_api, opening_col='OpeningName_API'), use_container_width=True)
        min_games_api = st.slider("Min games (API):", 1, 25, 5, key="min_games_perf_api")
        n_perf_api = st.slider("Num by win rate (API):", 5, 50, 15, key="n_openings_perf_api")
        st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_api, top_n=n_perf_api, opening_col='OpeningName_API'), use_container_width=True)
        st.markdown("---"); st.subheader("Opening Analysis based on Custom ECO Mapping")
        if not eco_mapping: st.warning("Custom ECO mapping file not loaded.")
        else:
             n_openings_cust = st.slider("Num top (Custom):", 5, 50, 15, key="n_openings_freq_cust")
             st.plotly_chart(plot_opening_frequency(df, top_n=n_openings_cust, opening_col='OpeningName_Custom'), use_container_width=True)
             min_games_cust = st.slider("Min games (Custom):", 1, 25, 5, key="min_games_perf_cust")
             n_perf_cust = st.slider("Num by win rate (Custom):", 5, 50, 15, key="n_openings_perf_cust")
             st.plotly_chart(plot_win_rate_by_opening(df, min_games=min_games_cust, top_n=n_perf_cust, opening_col='OpeningName_Custom'), use_container_width=True)

    elif selected_section == analysis_options[5]: # Opponent
        st.subheader("Frequent Opponents"); n_opponents_freq = st.slider("Num top opponents:", 5, 50, 20, key="n_opponents_freq_opp")
        st.plotly_chart(plot_most_frequent_opponents(df, top_n=n_opponents_freq), use_container_width=True)
        st.markdown(f"#### Top {n_opponents_freq} Opponents List")
        try: st.dataframe(df[df['OpponentName'] != 'Unknown']['OpponentName'].value_counts().reset_index(name='Games').head(n_opponents_freq))
        except KeyError: st.warning("Could not generate table.")
        st.subheader("Performance vs Opponent Elo"); st.plotly_chart(plot_performance_vs_opponent_elo(df), use_container_width=True)

    elif selected_section == analysis_options[6]: # vs Titled
        st.subheader("Filter by Opponent Title")
        with st.expander("Show Opponent Title Distribution in Data"): st.dataframe(df['OpponentTitle'].value_counts())
        selected_titles = st.multiselect("Select Opponent Titles:", TITLES_TO_ANALYZE, default=['GM','IM'])
        if selected_titles:
            titled_games = filter_and_analyze_titled(df, selected_titles)
            if not titled_games.empty:
                st.success(f"Found **{len(titled_games):,}** games vs {', '.join(selected_titles)}."); st.plotly_chart(plot_win_loss_pie(titled_games, f"{current_display_name} vs {', '.join(selected_titles)}"), use_container_width=True)
                st.plotly_chart(plot_win_loss_by_color(titled_games), use_container_width=True); st.plotly_chart(plot_rating_trend(titled_games, f"{current_display_name} (vs {', '.join(selected_titles)})"), use_container_width=True)
                st.plotly_chart(plot_opening_frequency(titled_games, top_n=15, opening_col='OpeningName_API'), use_container_width=True); st.plotly_chart(plot_most_frequent_opponents(titled_games, top_n=15), use_container_width=True)
            else: st.warning(f"ℹ️ No games found vs {', '.join(selected_titles)}.")
        else: st.info("Select title(s) for analysis.")

    elif selected_section == analysis_options[7]: # Termination
        st.subheader("Time Forfeit Analysis"); tf_games, wins_tf, losses_tf = filter_and_analyze_time_forfeits(df)
        if not tf_games.empty:
            st.plotly_chart(plot_time_forfeit_summary(wins_tf, losses_tf), use_container_width=True); st.plotly_chart(plot_time_forfeit_by_tc(tf_games), use_container_width=True)
            with st.expander("View Recent Time Forfeit Games"): st.dataframe(tf_games[['Date','OpponentName','PlayerColor','PlayerResultString','TimeControl','PlyCount','Termination']].sort_values('Date',ascending=False).head(20))
        else: st.warning("ℹ️ No games found with 'Time forfeit' termination.")
        st.subheader("Overall Termination Types"); termination_counts = df['Termination'].value_counts()
        fig_term = px.bar(termination_counts, x=termination_counts.index, y=termination_counts.values, title="Game Termination Reasons", labels={'x':'Reason','y':'Count'}, text=termination_counts.values)
        fig_term.update_layout(dragmode=False); fig_term.update_traces(textposition='outside'); st.plotly_chart(fig_term, use_container_width=True)

    st.sidebar.markdown("---"); st.sidebar.info(f"Analysis for {current_display_name}.")

elif not analyze_button and st.session_state.analysis_df is None: st.info("☝️ Configure settings and click 'Analyze Games'.")
elif analyze_button and (not isinstance(st.session_state.analysis_df, pd.DataFrame) or st.session_state.analysis_df.empty): st.warning("No analysis data generated. Check settings.")

# --- End of App ---
