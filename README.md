# Lichess Game Analysis Web App

A web application built with Python and Streamlit that allows users to fetch and analyze their chess game data directly from the Lichess API. Gain insights into your performance, openings, opponents, and time management across various time controls.

## 🚀 Live Demo

You can try out the web application live on Streamlit Cloud. Just enter a Lichess username and hit analyze!

[**Lichess Game Analyzer App**](https://lichess-ugoajpjsq3wq52are44rxl.streamlit.app/)

*(Note: Initial loading on Streamlit Cloud might take a few moments.)*

## ✨ Features

This application leverages the Lichess API to provide the following analysis features:

*   **Game Data Fetching:** Retrieve your rated games for a specified time period (Last Month, 3 Months, Year, 3 Years) and game type (Bullet, Blitz, Rapid).
*   **Overall Statistics:** View your total games, win/loss/draw counts, and overall win rate.
*   **Performance by Color:** Breakdown of results when playing as White vs. Black.
*   **Rating Trend:** Visualize your rating progress over time.
*   **Time & Date Analysis:** Analyze game frequency and win rates based on the day of the week, hour of the day (UTC), and day of the month.
*   **Time Control Performance:** See your results categorized by time control (Bullet, Blitz, Rapid, Classical, Correspondence).
*   **Opening Analysis:** Explore your most frequently played openings and their corresponding win rates, using both Lichess API names and a custom ECO mapping.
*   **Opponent Analysis:** Identify your most frequent opponents and analyze your performance against them, including performance relative to Elo difference.
*   **Analysis Against Titled Players:** Filter and analyze your games specifically against titled opponents (GM, IM, FM, etc.).
*   **Termination Analysis:** Understand how your games typically end, with specific insights into wins and losses by time forfeit.
*   **Interactive Visualizations:** All analysis is presented using interactive Plotly charts.
*   **Caching:** Utilizes Streamlit's caching mechanisms for efficient data loading on subsequent analyses with the same settings.

## 🛠️ How it Works

1.  The user enters their Lichess username and selects desired filters (time period, game type) in the sidebar.
2.  Upon clicking "Analyze Games", the application makes a request to the Lichess `/api/games/user/{username}` endpoint.
3.  Game data is streamed, parsed (NDJSON format), and relevant information for each game is extracted.
4.  Data is structured into a pandas DataFrame.
5.  Various functions process the DataFrame to calculate statistics and prepare data for visualization.
6.  Plotly is used to generate interactive charts based on the processed data.
7.  Streamlit renders the user interface, handles user input, and displays the charts and metrics.

## 🔧 Technologies Used

*   **Python**
*   **Streamlit:** For building the web application UI.
*   **Pandas:** For data manipulation and analysis.
*   **Plotly:** For interactive data visualizations.
*   **Requests:** For making HTTP calls to the Lichess API.
*   **Lichess API:** The data source for chess games.

Kaveh 
