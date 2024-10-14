import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import ast
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("IPL Matches 2008-2020.csv")
matches = pd.read_csv("IPL_Matches_2008_2022.csv")
deliveries = pd.read_csv("IPL_Ball_by_Ball_2008_2022.csv")
matches.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
matches.replace('Kings XI Punjab','Punjab Kings',inplace=True)
matches.replace('Delhi Daredevils','Delhi Capitals',inplace=True)
deliveries.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
deliveries.replace('Kings XI Punjab','Punjab Kings',inplace=True)
deliveries.replace('Delhi Daredevils','Delhi Capitals',inplace=True)
merged = pd.merge(deliveries,matches,on='ID',how='left')
merged['BowlingTeam'] = merged.apply(lambda row : row['Team1'] if row['Team1'] != row['BattingTeam'] else row['Team2'], axis=1)
merged['isFour'] = merged['batsman_run'].apply(lambda x: 1 if x == 4 else 0)
merged['isSix'] = merged['batsman_run'].apply(lambda x: 1 if x == 6 else 0)
merged['isLegal'] = merged['extra_type'].apply(lambda x: 1 if x in ['legbyes', 'byes'] or pd.isna(x) else 0)
replace_years = {'2007/08':'2008','2009/10':'2010','2020/21':'2020'}
merged['Season'] = merged['Season'].replace(replace_years)

stadiums = {'Arun Jaitley Stadium':'Arun Jaitley Stadium, Delhi',
            'Dr DY Patil Sports Academy':'Dr DY Patil Sports Academy, Mumbai',
            'Eden Gardens':'Eden Gardens, Kolkata',
            'M.Chinnaswamy Stadium':'M Chinnaswamy Stadium',
            'MA Chidambaram Stadium':'MA Chidambaram Stadium, Chepauk, Chennai',
            'MA Chidambaram Stadium, Chepauk':'MA Chidambaram Stadium, Chepauk, Chennai',
            'Maharashtra Cricket Association Stadium':'Maharashtra Cricket Association Stadium, Pune',
            'Punjab Cricket Association IS Bindra Stadium':'Punjab Cricket Association Stadium, Mohali',
            'Punjab Cricket Association IS Bindra Stadium, Mohali':'Punjab Cricket Association Stadium, Mohali',
            'Rajiv Gandhi International Stadium':'Rajiv Gandhi International Stadium, Uppal','Wankhede Stadium':'Wankhede Stadium, Mumbai'}
matches['Venue']=matches['Venue'].map(stadiums).fillna(matches.Venue)

team_list = df.team1.unique()
team_colors = {
    'Rajasthan Royals': '#FFC0CB',  # pink
    'Royal Challengers Bangalore': '#FF0000',  # red
    'Sunrisers Hyderabad': '#FFA500',  # Orange
    'Delhi Capitals': '#FF0000',  # red
    'Chennai Super Kings': '#FFFF00',  # Yellow
    'Gujarat Titans': '#800080',  # Purple
    'Lucknow Super Giants': '#FFC0CB',  # Pink
    'Kolkata Knight Riders': '#800080',  # Purple
    'Punjab Kings': '#C0C0C0',  # Silver
    'Mumbai Indians': '#0000FF',  # Blue
    'Rising Pune Supergiants': '#FF1493',  # DeepPink
    'Gujarat Lions': '#FF4500',  # OrangeRed
    'Pune Warriors': '#8A2BE2',  # BlueViolet
    'Deccan Chargers': '#000080',  # Navy
    'Kochi Tuskers Kerala': '#800000'  # Maroon (same as Kolkata Knight Riders for demonstration)
}

team_totals = merged.groupby(['ID','innings'])[['total_run']].sum().reset_index()
def set_target(row):
    if row['innings'] == 1:
        return 0  
    else:
        first_innings_total = team_totals.loc[(team_totals['ID'] == row['ID']) & (team_totals['innings'] == 1), 'total_run'].values
        return first_innings_total[0] if len(first_innings_total) > 0 else None
team_totals['target'] = team_totals.apply(set_target,axis=1)
merged_df = pd.merge(merged,team_totals,on=['ID','innings'])
second_innings = merged_df[merged_df.innings==2]
second_innings['Wickets_left'] = 10 - second_innings.groupby(['ID'])[['isWicketDelivery']].cumsum()
second_innings['current_score'] = second_innings.groupby(['ID'])[['total_run_x']].cumsum()
second_innings['balls_left'] = np.where(120 - second_innings['overs']*6 - second_innings['ballnumber']>=0,120 - second_innings['overs']*6 - second_innings['ballnumber'], 0)
second_innings['runs_left'] = np.where(second_innings['target']-second_innings['current_score']>=0, second_innings['target']-second_innings['current_score'], 0)
second_innings['current_run_rate'] = (second_innings['current_score']*6)/(120-second_innings['balls_left'])
second_innings['required_run_rate'] = np.where(second_innings['balls_left']>0, second_innings['runs_left']*6/second_innings['balls_left'], 0)
second_innings['result'] = second_innings.apply(lambda row: 1 if row['BattingTeam']==row['WinningTeam'] else 0,axis=1)
final =  second_innings[['BattingTeam','BowlingTeam','Venue','balls_left','runs_left','Wickets_left','current_run_rate','required_run_rate','current_score','target','result']]



with st.sidebar:
    options = option_menu("Main Menu", ["About", "Team vs Team",'Player Statistics','Team Records','Batsman vs Bowler','Win Predictor'], 
    icons=['house', 'cast','person-circle', 'bar-chart','trophy','currency-dollar'], menu_icon="gear", default_index=0)


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

if options == "About":
    text_file_path = "About.txt"  
    image_file_path = "Images/IPL.png" 
    text_content = load_text_file(text_file_path)

    st.image(image_file_path, use_column_width=True)
    st.text_area("About IPL", text_content, height=300)

import Team_vs_Team

if options == 'Team vs Team':
    Team_vs_Team.team_vs_team(matches, merged, team_colors)
         
import Player_Statistics

if options == 'Player Statistics':
   Player_Statistics.player_statistics(merged,deliveries)

import Team_Records

if options=='Team Records':
    Team_Records.team_records(merged,matches,team_colors)

import Batsman_vs_Bowler

if options=='Batsman vs Bowler':
    Batsman_vs_Bowler.batsman_vs_bowler(merged)
    
          

#import WinPredictor

#if options == 'Win Predictor':
   # WinPredictor.WinPredictor(final,matches,team_colors)
    
    
