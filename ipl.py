import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import ast
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\dines\Downloads\archive (3)\IPL Matches 2008-2020.csv")
matches = pd.read_csv(r"C:\Users\dines\Dinesh\Machine learning\IPL\IPL_Matches_2008_2022.csv")
deliveries = pd.read_csv(r"C:\Users\dines\Dinesh\Machine learning\IPL\IPL_Ball_by_Ball_2008_2022.csv")
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

trf = ColumnTransformer([('trf', OneHotEncoder(sparse_output=False,drop='first'),['BattingTeam','BowlingTeam','Venue'])],remainder = 'passthrough')

X = final.drop('result', axis=1)
y = final['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

pipe = Pipeline(steps=[('step1',trf),('step2',RandomForestClassifier())])
pipe.fit(X_train, y_train)

with st.sidebar:
    options = option_menu("Main Menu", ["About", "Team vs Team",'Player Statistics','Team Records','Batsman vs Bowler','Auction Analysis','Venue Analysis','Win Predictor'], 
    icons=['house', 'cast','person-circle', 'bar-chart','trophy','currency-dollar'], menu_icon="gear", default_index=0)

if options == 'Team vs Team':
    c1,c2=st.columns(2)
    with c1:
        team_one =  st.sidebar.selectbox('Team 1',matches.Team1.unique())
    with c2:
        team_two =  st.sidebar.selectbox('Team 2',matches.Team1.unique())
    t1_t2 = merged[(merged.Team1==team_one) & (merged.Team2==team_two) | (merged.Team1==team_two) & (merged.Team2==team_one)]
    t1_t2['isBowlerWicket'] = (~t1_t2['kind'].isin(['run out', 'retired out', 'retired hurt', 'obstructing the field',np.nan])).astype(int)
    def h2h(t1,t2):
        temp = matches[(matches.Team1==t1) & (matches.Team2==t2) | (matches.Team1==t2) & (matches.Team2==t1)]
        matches_played = len(temp)
        t1_win = len(temp[temp.WinningTeam==t1])
        t2_win = len(temp[temp.WinningTeam==t2])
        NR = len(temp[temp.WinningTeam.isna()])
        t1_toss_win = len(temp[temp.TossWinner == t1])
        t2_toss_win = len(temp[temp.TossWinner == t2])
        return matches_played,t1_win,t2_win,NR,t1_toss_win,t2_toss_win
    def bat_stats(team):
        bat = merged[merged.BattingTeam == team]
        bat = bat.groupby(['Season'])[['total_run']].sum().reset_index()
        return bat
    def bowl_stats(team):
        bowl = merged[merged.BowlingTeam == team]
        bowl = bowl.groupby(['Season'])[['isWicketDelivery']].sum().reset_index()
        return bowl
    def powerplay_runrate(team):
        bat = merged[merged.BattingTeam == team]
        powerplay = bat[bat.overs.isin([0, 1, 2,3,4,5])]
        powerplay = powerplay.groupby(['Season','ID'])[['total_run']].sum()
        powerplay['Run Rate'] = powerplay['total_run'].apply(lambda x: round(x/6,2))
        return powerplay.groupby(['Season'])[['Run Rate']].mean().round(2).reset_index()
    def deathover_runrate(team):
        bat = merged[merged.BattingTeam == team]
        deathover = bat[bat.overs.isin([15,16,17,18,19])]
        deathover = deathover.groupby(['Season','ID'])[['total_run']].sum()
        deathover['Run Rate'] = deathover['total_run'].apply(lambda x: round(x/6,2))
        return deathover.groupby(['Season'])[['Run Rate']].mean().round(2).reset_index()
    batting_chart = pd.merge(bat_stats(team_one),bat_stats(team_two),on='Season')
    batting_chart.columns = ['Season',team_one,team_two]
    bowling_chart = pd.merge(bowl_stats(team_one),bowl_stats(team_two),on='Season')
    bowling_chart.columns = ['Season',team_one,team_two]
    powerplay_chart = pd.merge(powerplay_runrate(team_one),powerplay_runrate(team_two),on='Season')
    powerplay_chart.columns = ['Season',team_one,team_two]
    deathover_chart = pd.merge(deathover_runrate(team_one),deathover_runrate(team_two),on='Season')
    deathover_chart.columns =['Season',team_one,team_two]
    
    val = t1_t2.groupby(['batter'])[['batsman_run']].sum()
    val1= t1_t2.groupby(['bowler'])[['isBowlerWicket']].sum()
    val2 = t1_t2.groupby(['batter'])[['isSix']].sum()
    val3 = t1_t2.groupby(['batter'])[['isFour']].sum()
    team_one_last5 = matches[(matches.Team1 == team_one) | (matches.Team2 == team_one)].sort_values(by='Date',ascending=False)[:8].reset_index(drop=True)
    team_two_last5 = matches[(matches.Team1 == team_two) | (matches.Team2 == team_two)].sort_values(by='Date',ascending=False)[:8].reset_index(drop=True)
    recent_team1 = ""
    for i in range(8):
        single_match = team_one_last5.iloc[i,:]
        res1 = "✅" if single_match['WinningTeam'] == team_one else "❌"
        recent_team1 += res1
    recent_team2 = ""
    for i in range(8):
        single_match = team_two_last5.iloc[i,:]
        res2 = "✅" if single_match['WinningTeam'] == team_two else "❌"
        recent_team2 += res2
    c1,c2 = st.columns(2)
    if st.sidebar.button('submit'):
        with c1:
            with st.container(border=True): 
                path = f'{team_one}.jpg'
                st.image(path,use_column_width=True) 
        with c2:
            with st.container(border=True):
                path = f'{team_two}.jpg'
                st.image(path,use_column_width=True)

        result = h2h(team_one,team_two)
        
        col1,col2,col3=st.columns(3)
        col4,col5,col6=st.columns(3)
        col7,col8,col9=st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">Matches Played</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{result[0]}</h4>', unsafe_allow_html=True)
        with col2:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">Team 1 Won</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{result[1]}</h4>', unsafe_allow_html=True)
        with col3:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">Team 2 Won</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{result[2]}</h4>', unsafe_allow_html=True)
        with col4:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">{"No result"}</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{result[3]}</h4>', unsafe_allow_html=True)
        with col5:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">{"Most Runs"}</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{val.idxmax()[0]}({val.max()[0]})</h4>', unsafe_allow_html=True)
        with col6:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">{"Most Wickets"}</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{val1.idxmax()[0]}({val1.max()[0]})</h4>', unsafe_allow_html=True)
        with col7:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center;">{"Most Sixes"}</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{val2.idxmax()[0]}({val2.max()[0]})</h4>', unsafe_allow_html=True)
        with col8:
            with st.container(border=True):
                st.markdown(f'<div style="text-align: center; margin-top: 0;">{"Most Fours"}</div>', unsafe_allow_html=True)
                st.markdown(f'<h4 style="text-align: center;">{val3.idxmax()[0]}({val3.max()[0]})</h4>', unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                trace = go.Pie(labels=[team_one,team_two], values=[result[1],result[2]], hole=0.5,marker=dict(colors=[team_colors[team_one],team_colors[team_two]]))
                fig = go.Figure(data=trace)
                st.subheader('Win Percentage')
                st.plotly_chart(fig,use_container_width=True)
        with col2:
            with st.container(border=True):
                trace2 = go.Pie(labels=[team_one,team_two], values=[result[4],result[5]], hole=0.5,marker=dict(colors=[team_colors[team_one],team_colors[team_two]]))
                fig0 = go.Figure(data=trace2)
                st.subheader('Toss Win Percentage')
                st.plotly_chart(fig0,use_container_width=True)    
        with col3:
            with st.container(border=True):
                st.subheader('PowerPlay RunRate Comparision')
                st.line_chart(powerplay_chart,x='Season',y=[team_one,team_two],color=[team_colors[team_one],team_colors[team_two]])
        with col4:
            with st.container(border=True):
                st.subheader('Deathover Runrate Comparision')
                st.line_chart(deathover_chart,x='Season',y=[team_one,team_two],color=[team_colors[team_one],team_colors[team_two]])
        with st.container(border=True):
            fig1 = go.Figure(data=[go.Bar(name=team_one, x=batting_chart['Season'], y=batting_chart[team_one],marker_color=team_colors[team_one]),go.Bar(name=team_two, x=batting_chart['Season'], y=batting_chart[team_two],marker_color=team_colors[team_two])])
            fig1.update_layout(barmode='group', xaxis_title='Season', yaxis_title='Runs')
            fig1.update_xaxes(tickvals=batting_chart['Season'])
            st.subheader('Runs over the Seasons')
            st.plotly_chart(fig1, use_container_width=True)
        with st.container(border=True):
            fig2 = go.Figure(data=[go.Bar(name=team_one, x=bowling_chart['Season'], y=bowling_chart[team_one],marker_color=team_colors[team_one]),go.Bar(name=team_two, x=bowling_chart['Season'], y=bowling_chart[team_two],marker_color=team_colors[team_two])])
            fig2.update_layout(barmode='group', xaxis_title='Season', yaxis_title='Wickets')
            fig2.update_xaxes(tickvals=bowling_chart['Season'])
            st.subheader('Wickets over the Seasons')
            st.plotly_chart(fig2, use_container_width=True)

        with st.container(border=True):
            st.header('Recent Form :')
            st.subheader(f"{team_one}: {recent_team1}")
            st.subheader(f"{team_two}: {recent_team2}")
        

        
        

if options == 'Player Statistics': #Individual Player Stats
    def player_info(name):
        player_stats = deliveries.groupby(['ID','batter'])[['batsman_run']].sum()
        scores = player_stats.loc[(slice(None),name), 'batsman_run'].values
        matches_played = len(scores)
        runs_scored = scores.sum()
        fifties = len([i for i in scores if i>=50 and i<100])
        hundreads = len([i for i in scores if i>=100])
        highest_score = scores.max()
        return matches_played,runs_scored,fifties,hundreads,highest_score
    def batting_stats(name):
        batsman_data = merged[merged['batter']==name]
        one = batsman_data.groupby(['Season'])[['ID','batsman_run','isFour','isSix','isLegal']].agg({'ID':'nunique','batsman_run':'sum','isFour':'sum','isSix':'sum','isLegal':'sum'})
        bat_runs = batsman_data.groupby(['Season','ID'])[['batsman_run']].sum()
        bat_runs['isFifty'] = bat_runs['batsman_run'].apply(lambda x: 1 if x>=50 and x<100 else 0)
        bat_runs['isHundread'] = bat_runs['batsman_run'].apply(lambda x: 1 if x>=100 else 0)
        bat_runs['isDuck'] = bat_runs['batsman_run'].apply(lambda x: 1 if x==0 else 0)
        two = bat_runs.groupby('Season')[['batsman_run','isFifty','isHundread','isDuck']].agg({'batsman_run':'max','isFifty':'sum','isHundread':'sum','isDuck':'sum'})
        res = pd.concat([one,two],axis=1)
        res.columns = ['Matches','Runs',"Fours","Sixes",'Balls','High Score',"50's","100's","Duckout's"]
        res['Strike Rate'] = round((res['Runs']/res['Balls']) * 100 , 2)
        res = res[['Matches','Runs','Balls','Strike Rate','High Score',"100's","50's","Fours","Sixes","Duckout's"]]
        res = res.reset_index()
        return res
    def bowling_stats(name):
        bowler_data = merged[merged.bowler == name]
        matches_overs = bowler_data.groupby(['Season'])[['ID','isLegal']].agg({'ID':'nunique','isLegal':'sum'})
        runs_conceded = bowler_data[(bowler_data.extra_type!='legbyes') & (bowler_data.extra_type!='byes')].groupby(['Season'])[['total_run']].sum()
        wickets_taken = bowler_data[~(bowler_data['kind'].isin(['run out', 'retired out', 'retired hurt', 'obstructing the field']))].groupby(['Season'])[['isWicketDelivery']].sum()
        wickets_taken_eachmatch = bowler_data[~(bowler_data['kind'].isin(['run out', 'retired out', 'retired hurt', 'obstructing the field']))].groupby(['Season', 'ID'])[['isWicketDelivery']].sum()
        runs_conceded_eachmatch = bowler_data[(bowler_data.extra_type!='legbyes') & (bowler_data.extra_type!='byes')].groupby(['Season','ID'])[['total_run']].sum()
        multiwckts = pd.concat([wickets_taken_eachmatch,runs_conceded_eachmatch],axis=1)
        multiwckts["3W"] = multiwckts['isWicketDelivery'].apply(lambda x: 1 if x==3 else 0)
        multiwckts["4W"] = multiwckts['isWicketDelivery'].apply(lambda x: 1 if x==4 else 0)
        multiwckts["5W"] = multiwckts['isWicketDelivery'].apply(lambda x: 1 if x==5 else 0)
        multiwckts["6W"] = multiwckts['isWicketDelivery'].apply(lambda x: 1 if x==6 else 0)
        multiwckts = multiwckts.reset_index().groupby('Season')[['3W','4W','5W','6W']].sum()
        dd = pd.concat([wickets_taken_eachmatch,runs_conceded_eachmatch],axis=1).sort_values(by=['isWicketDelivery', 'total_run'], ascending=[False, True])
        dd = dd.reset_index()
        dd.drop_duplicates(subset='Season', keep='first',inplace=True)
        dd['best'] = dd.apply(lambda row: f"{row['isWicketDelivery']}/{row['total_run']}", axis=1)
        dd = dd[['Season','best']]
        dd.set_index(['Season'],inplace=True)
        final = pd.concat([matches_overs,runs_conceded,wickets_taken,dd,multiwckts],axis=1)
        final['isLegal'] = final['isLegal'].apply(lambda x: (x//6)+(x%6)/10)
        final['Economy Rate'] = round(final['total_run']/final['isLegal'],2)
        final.columns = ['Matches','Overs',"Runs",'Wickets','Best',"3W's","4W's","5W's","6W's","Economy Rate"]
        final = final[['Matches','Overs','Runs','Wickets','Economy Rate','Best',"3W's","4W's","5W's","6W's"]]
        return final
    def fielding_stats(name):
        matches = merged[(merged.batter == name) | (merged.bowler == name)].groupby(['Season'])[['ID']].nunique()
        catches = merged[(merged.fielders_involved == name) & (merged.kind == 'caught')].groupby(['Season'])[['fielders_involved']].count()
        runouts = merged[(merged.fielders_involved == name) & (merged.kind == 'run out')].groupby(['Season'])[['fielders_involved']].count()
        stumps = merged[(merged.fielders_involved == name) & (merged.kind == 'stumped')].groupby(['Season'])[['fielders_involved']].count()
        res = pd.concat([matches,catches,runouts,stumps],axis=1)
        res.columns = ['Matches','Catches','Runouts','Stumps']
        res.fillna(0,inplace=True)
        res.Runouts = res.Runouts.astype('int')
        res.Stumps = res.Stumps.astype('int')
        return res
            
    names = list(deliveries.batter.unique())
    player_name = st.sidebar.selectbox('Player Name',names)
    res = player_info(player_name)
    with st.container(border=True):
        col1,col2 = st.columns(2)
        with col2:
            st.write(f'Name : {player_name}')
            st.write(f'Matches Played : {res[0]}')
            st.write(f'Runs Scores : {res[1]}')
            st.write(f'Fifties : {res[2]}')
            st.write(f'Hundreads : {res[3]}')
            st.write(f'highest_score : {res[4]}')
        with col1:
            st.image('Unknown.jpg',width=200)
    st.divider()

    tab1,tab2,tab3,tab4,tab5 = st.tabs(['Batting Stats:cricket_bat_and_ball:','Bowling Stats🥎','Fielding Stats',"Charts 📊",'Summary Stats'])
    with tab1:
        st.write(batting_stats(player_name))
    with tab2:
        st.write(bowling_stats(player_name))
    with tab3:
        st.write(fielding_stats(player_name))
    with tab4:
        runs_chart = batting_stats(player_name)
        runs_chart = runs_chart.reset_index()
        wickets_chart = bowling_stats(player_name)
        wickets_chart = wickets_chart.reset_index()
        p1,p2 = st.columns(2)
        with p1:
            st.subheader('Runs in Each Season')
            st.bar_chart(data=runs_chart,x='Season',y='Runs')
        with p2:
            st.subheader('Wickets in Each Season')
            st.bar_chart(data=wickets_chart,x='Season',y='Wickets',color='#FF0000')
        pie_chart = merged[merged['batter']==player_name].groupby(['kind'])['kind'].count()
        hole = go.Pie(labels=pie_chart.index, values=pie_chart.values, hole=0.7)
        st.divider()
        st.subheader('Chances of Getting Out')
        st.plotly_chart(go.Figure(data=hole))
        st.divider()
        p3,p4 = st.columns(2)
        with p3:
            st.subheader('Sixes in Each Season')
            st.bar_chart(data=runs_chart,x='Season',y='Sixes',color="#00FF00")
        with p4:
            st.subheader('Fours in Each Season')
            st.bar_chart(data=runs_chart,x='Season',y='Fours',color="#ffaa00")
        
    
    with tab5:
        sixes = go.Figure(go.Indicator(mode = "gauge+number",value = batting_stats(player_name)["Sixes"].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Six'O Meter"}))
        fours = go.Figure(go.Indicator(mode = "gauge+number",value = batting_stats(player_name)["Fours"].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Four'O Meter"}))
        mtchs = go.Figure(go.Indicator(mode = "gauge+number",value = batting_stats(player_name)["Matches"].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Matches Played"}))
        Runs_scored = go.Figure(go.Indicator(mode = "gauge+number",value = batting_stats(player_name)["Runs"].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Runs Scored"}))
        Fifties = go.Figure(go.Indicator(mode = "gauge+number",value = batting_stats(player_name)["50's"].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "50's"}))
        Hundreds = go.Figure(go.Indicator(mode = "gauge+number",value = batting_stats(player_name)["100's"].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "100's"}))
        SR = go.Figure(go.Indicator(mode = "gauge+number",value = round((batting_stats(player_name)["Runs"].sum()/batting_stats(player_name)["Balls"].sum())*100,1),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Strike Rate"}))
        col1,col2,col3 = st.columns(3)
        col4,col5,col6 = st.columns(3)
        with col3:
            with st.container(border=True):
                st.plotly_chart(fours,use_container_width=True)
        with col4:
            with st.container(border=True):
                st.plotly_chart(sixes,use_container_width=True)
        with col1:
            with st.container(border=True):
                st.plotly_chart(mtchs,use_container_width=True)
        with col2:
            with st.container(border=True):
                st.plotly_chart(Runs_scored,use_container_width=True)
        with col5:
            with st.container(border=True):
                st.plotly_chart(Fifties,use_container_width=True)
        with col6:
            with st.container(border=True):
                st.plotly_chart(Hundreds,use_container_width=True)
        st.plotly_chart(SR)



if options=='Team Records':
    team_name = st.sidebar.selectbox('Select Team',matches.Team1.unique())
    if st.sidebar.button('Submit'):
        matches_played = len(matches[matches.Team1==team_name]) + len(matches[matches.Team2==team_name])
        matches_won = len(matches[matches.WinningTeam == team_name])
        Win_percent = round(matches_won / matches_played * 100 , 2)
        toss_won = len(matches[matches.TossWinner == team_name])
        Won_Chasing = len(matches[(matches.WinningTeam == team_name) & (matches.WonBy == 'Wickets')])
        Won_Defending = len(matches[(matches.WinningTeam == team_name) & (matches.WonBy == 'Runs')])
        Won_superover = matches_won - (Won_Chasing+Won_Defending)
        def player_names(team):
            part_one = merged[merged.Team1 == team]['Team1Players'].explode().unique()
            part_two = merged[merged.Team2 == team]['Team2Players'].explode().unique()
            all_players = set(part_one) | set(part_two)
            l = []
            for i in all_players:
                l.append(ast.literal_eval(i))
            all_names = [name for sublist in l for name in sublist]
            all_names = list(set(all_names))
            return all_names
        def no_of_matches(team,player):
            cnt = 0
            mm = matches[(matches.Team1==team) | (matches.Team2==team)]
            for i in mm.Team1Players:
                if player in i:
                    cnt+=1
            for j in mm.Team2Players:
                if player in j:
                    cnt+=1
            return player,cnt
        def Most_runs(team):
            all_data = []
            matches_played = []
            names = player_names(team)
            for i in names:
                matches_played.append(list(no_of_matches(team,i)))
                player_data = merged[(merged.batter==i) & (merged.BattingTeam==team)]
                if len(player_data)!=0:
                    new = player_data.groupby(['ID','innings','batter'])[['batsman_run','innings','isLegal','isFour','isSix']].agg({'batsman_run':'sum','innings':'count','isLegal':'sum','isFour':'sum','isSix':'sum'}).reset_index(['ID','batter'])
                    new['isFifty'] = new['batsman_run'].apply(lambda x: 1 if x>=50 and x<100 else 0)
                    new['isHundread'] = new['batsman_run'].apply(lambda x: 1 if x>=100 else 0)
                    new['Duckout'] = new['batsman_run'].apply(lambda x: 1 if x==0 else 0)
                    ar = new.groupby(['batter'])[['batsman_run','innings','isLegal','isFour','isSix','isFifty','isHundread','Duckout']].agg({'batsman_run':'sum','innings':'count','isLegal':'sum','isFour':'sum','isSix':'sum','isFifty':'sum','isHundread':'sum','Duckout':'sum'}).reset_index().values
                    all_data.append(ar[0])
                else:
                    all_data.append([i,0,0,0,0,0,0,0,0])
            df= pd.DataFrame(all_data,columns=['Player','Runs','Innings','Balls',"4's","6's","50's","100's","Duck's"])
            df = df.sort_values(by='Runs',ascending=False)
            df1 = pd.DataFrame(matches_played,columns=['Player','Matches'])
            final = pd.merge(df,df1,on='Player')
            final['S/R'] = round((final['Runs']/final['Balls']) * 100 , 2)
            final = final[['Player','Matches','Innings','Runs','Balls','S/R',"4's","6's","50's","100's","Duck's"]]
            final.fillna(0,inplace=True)
            return final
        def Most_Wickets(team):
            all_data = []
            matches_played = []
            names = player_names(team)
            for i in names:
                matches_played.append(list(no_of_matches(team,i)))
                bowler_data = merged[(merged.bowler==i) & (merged.BowlingTeam==team)]
                if len(bowler_data)!=0:
                    bowler_data['isBowlerWicket'] = (~bowler_data['kind'].isin(['run out', 'retired out', 'retired hurt', 'obstructing the field',np.nan])).astype(int)
                    bowler_data['runs_conceded'] = bowler_data.apply(lambda x : x['total_run'] if (x['extra_type'] != 'byes') & (x['extra_type'] != 'legbyes') else x['total_run']-x['extras_run'],axis=1)
                    best = bowler_data.groupby(['ID','innings'])[['isWicketDelivery','runs_conceded']].sum().sort_values(by=['isWicketDelivery','runs_conceded'],ascending=[False,True]).iloc[:1,:].values.flatten()
                    res = bowler_data.groupby(['bowler','ID','innings'])[['isBowlerWicket','ballnumber','runs_conceded']].agg({'isBowlerWicket':'sum','ballnumber':'count','runs_conceded':'sum'}).reset_index().groupby(['bowler'])[['innings','isBowlerWicket','ballnumber','runs_conceded']].agg({'innings':'count','isBowlerWicket':'sum','ballnumber':'sum','runs_conceded':'sum'}).reset_index().values[0]
                    all_data.append(res)
            df1 = pd.DataFrame(matches_played,columns=['Player','Matches'])
            df2 = pd.DataFrame(all_data,columns=['Player','Innings','Wickets','Balls','Runs'])
            df = pd.merge(df1,df2,on='Player')
            df['Overs'] = (df['Balls']//6) + (df['Balls']%6)/10
            df['Economy_Rate'] = round(df['Runs']/df['Overs'],2)
            df['Average'] = df.apply(lambda x: round(x['Runs']/x['Wickets'], 2) if x['Wickets'] != 0 else 0, axis=1)
            df['Strike Rate'] = df.apply(lambda x: round(x['Balls']/x['Wickets'], 2) if x['Wickets'] != 0 else 0, axis=1)
            df = df.sort_values(by='Wickets',ascending=False).reset_index(drop=True)
            return df
        with st.container(border=True):
            col1,col2 = st.columns(2)
            with col1:
                path = f'{team_name}.jpg'
                st.image(path,use_column_width=True)
            with col2:
                st.write('Matches Played:',matches_played)
                st.write('Matches Won:',matches_won)
                st.write('Win%:',Win_percent)
                st.write('Tosses Won:',toss_won)
                st.write('Won Chasing:',Won_Chasing)
                st.write('Won Defending:',Won_Defending)
                st.write('Won By SuperOver:',Won_superover)
        
        tab1,tab2,tab3,tab4 = st.tabs(['Most Runs','Most Wickets','Bar Charts','Last 5 Matches'])
        with tab1:
            st.write(Most_runs(team_name))
        with tab2:
            st.write(Most_Wickets(team_name))
        with tab3:
            team_batting = merged[merged.BattingTeam == team_name]
            team_bowling = merged[merged.BowlingTeam == team_name]
            runs_each_season = team_batting.groupby(['Season'])[['total_run']].sum().reset_index()
            wckts_each_season = team_bowling.groupby(['Season'])[['isWicketDelivery']].sum().reset_index()
            sixes_each_season = team_batting.groupby(['Season'])[['isSix']].sum().reset_index()
            fours_each_season = team_batting.groupby(['Season'])[['isFour']].sum().reset_index()
            
            col1,col2 = st.columns(2)
            col3,col4 = st.columns(2)
            with col1:
                st.subheader('Runs Scored Each Season')
                st.bar_chart(data=runs_each_season,x='Season',y='total_run',color=team_colors[team_name])
            with col2:
                st.subheader('Wickets Taken Each Season')
                st.bar_chart(data=wckts_each_season,x='Season',y='isWicketDelivery',color=team_colors[team_name])
            with col3:
                st.subheader('Sixes Hit Each Season')
                st.bar_chart(data=sixes_each_season,x='Season',y='isSix',color=team_colors[team_name])
            with col4:
                st.subheader('Fours Hit Each Season')
                st.bar_chart(data=fours_each_season,x='Season',y='isFour',color=team_colors[team_name])
        with tab4:
            last5 = matches[(matches.Team1 == team_name) | (matches.Team2 == team_name)].sort_values(by='Date',ascending=False)[:5]
            for i in range(5):
                single_match = last5.iloc[i,:]
                opp_team = single_match['Team2'] if single_match['Team1'] == team_name else single_match['Team1']
                result = "Won✅" if single_match['WinningTeam'] == team_name else "Lost❌"
                margin = single_match['Margin'],single_match['WonBy']
                mom = single_match['Player_of_Match']
                with st.container(border=True):
                    st.write("Opponent :",opp_team)
                    st.write("Result :",result)
                    st.write("By:",margin)
                    st.write("Man of the Match:",mom)


if options=='Batsman vs Bowler':
    batsmans = list(merged.batter.unique())
    bowlers = list(merged.bowler.unique())
    col1,col2=st.columns(2)
    with col1:
        with st.container(border=True):
            #st.image('batsman.jpg',use_column_width=True)
            batter =  st.sidebar.selectbox('Select Batsman',batsmans)
                
    with col2:
        with st.container(border=True):
            #st.image('bowler.jpg',use_column_width=True)
            bowler =  st.sidebar.selectbox('Select Bowler',bowlers)
    if st.sidebar.button('Submit'):
        vs = merged[(merged.batter == batter) & (merged.bowler == bowler)]
        vs['isBowlerWicket'] = (~vs['kind'].isin(['run out', 'retired out', 'retired hurt', 'obstructing the field',np.nan])).astype(int)
        vs['isDotBall'] = vs['total_run'].apply(lambda x: 1 if x==0 else 0)
        sr = round(vs['batsman_run'].sum()/vs['isLegal'].sum(),2) * 100
        with st.container(border=True):
            fig1 = go.Figure(go.Indicator(mode = "gauge+number",value = vs.groupby(['ID'])['innings'].nunique().sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Innings"},gauge={'bar': {'color': "darkblue"}}))
            fig2 = go.Figure(go.Indicator(mode = "gauge+number",value = vs['batsman_run'].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Runs Scored"}))
            fig3 = go.Figure(go.Indicator(mode = "gauge+number",value = vs['isLegal'].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Balls"}))
            fig4 = go.Figure(go.Indicator(mode = "gauge+number",value = vs['isBowlerWicket'].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Dismissals"}))
            fig5 = go.Figure(go.Indicator(mode = "gauge+number",value = vs['isFour'].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Four'O Meter"}))
            fig6 = go.Figure(go.Indicator(mode = "gauge+number",value = vs['isSix'].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Six'O Meter"}))
            fig7 = go.Figure(go.Indicator(mode = "gauge+number",value = sr,domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Strike Rate"}))
            fig8 = go.Figure(go.Indicator(mode = "gauge+number",value = vs['isDotBall'].sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Dot Balls"}))

            fig1.update_layout(width=400, height=300)
            fig2.update_layout(width=400, height=300)
            fig3.update_layout(width=400, height=300)
            fig4.update_layout(width=400, height=300)
            fig5.update_layout(width=400, height=300)
            fig6.update_layout(width=400, height=300)
            fig7.update_layout(width=400, height=300)
            fig8.update_layout(width=400, height=300)
        col1,col2,col3 = st.columns(3)
        col4,col5,col6 = st.columns(3)
        col7,col8,col9 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.plotly_chart(fig1,use_container_width=True)
        with col5:
            with st.container(border=True):
                st.plotly_chart(fig2,use_container_width=True)
        with col2:
            with st.container(border=True):
                st.plotly_chart(fig3,use_container_width=True)
        with col4:
            with st.container(border=True):
                st.plotly_chart(fig4,use_container_width=True)
        with col8:
            with st.container(border=True):
                st.plotly_chart(fig5,use_container_width=True)
        with col7:
            with st.container(border=True):
                st.plotly_chart(fig6,use_container_width=True)
        with col6:
            with st.container(border=True):
                st.plotly_chart(fig7,use_container_width=True)
        with col3:
            with st.container(border=True):
                st.plotly_chart(fig8,use_container_width=True)
            
if options == "Auction Analysis":
    tab1,tab2 = st.tabs(['Year-Wise Analysis','Team-Wise Analysis'])
    with tab1:
        auction = pd.read_csv(r"C:\Users\dines\Dinesh\Machine learning\IPL\auction.csv")
        auction.drop(columns = 'Unnamed: 0',inplace=True)
        auction.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
        auction.replace('Kings XI Punjab','Punjab Kings',inplace=True)
        auction.replace('Delhi Daredevils','Delhi Capitals',inplace=True)  
        auction.replace(' India','India',inplace=True)
        auction['Winning bid'] = auction['Winning bid'].str.replace(",","").astype('float')
        data = auction.groupby('Year')['Winning bid'].sum()
        fig = px.line(data, x = data.index, y = 'Winning bid', title = "Overall Expense Each year", text=data.index)
        fig.update_traces(textposition="top right")
        st.plotly_chart(fig)
        data = data.div(10000000)
        fig = px.bar(data, x = data.index, y = 'Winning bid', title = "Overall Expense Each year (in Crores)",text= 'Winning bid',color=data.index)
        st.plotly_chart(fig)

        data = auction.groupby(['Team', 'Year'])[['Winning bid']].sum().reset_index()
        data =data.sort_values(by='Winning bid', ascending=False,key=lambda x: data['Year'].groupby(x).transform('sum'))
        fig = px.bar(data, x='Year', y='Winning bid', color='Team')
        fig.update_layout(title_text = "Overall comparision of different team spends")
        st.plotly_chart(fig)
    with tab2:
        team = st.selectbox('Select Team',auction.Team.unique())
        df = auction[auction.Team==team]
        st.write(df)
        data = df.groupby(['Year'])[['Winning bid']].sum()
        fig = px.bar(data, x = data.index.values, y = 'Winning bid', title = "Overall Expense Each year (in Crores)",text= 'Winning bid',color=data.index.values)
        st.plotly_chart(fig)
        st.subheader('Top Picks')
        st.write(df.sort_values(by=['Winning bid'],ascending=False)[:5])

if options == 'Venue Analysis':
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
    data = pd.DataFrame(matches.Venue.value_counts()).reset_index()
    data.columns = ['Venue','Count']
    st.subheader('Matches Played at Each Venue')
    st.bar_chart(data=data,y='Count',x='Venue')
    venue_name = st.selectbox('Select a Venue',data.Venue.unique())
    with st.container(border=True):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div style="text-align: center;">Matches Played</div>', unsafe_allow_html=True)
            st.markdown(f'<h4 style="text-align: center;">{len(matches[matches.Venue == venue_name])}</h4>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="text-align: center;">Won Batting First</div>', unsafe_allow_html=True)
            st.markdown(f'<h4 style="text-align: center;">{len(matches[(matches.Venue == venue_name) & (matches.WonBy == "Runs")])}</h4>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="text-align: center;">Won Bowling First</div>', unsafe_allow_html=True)
            st.markdown(f'<h4 style="text-align: center;">{len(matches[(matches.Venue == venue_name) & (matches.WonBy == "Wickets")])}</h4>', unsafe_allow_html=True)

if options == 'Win Predictor':
    
    batting_team = st.selectbox('Batting Team',matches.Team1.unique())
    bowling_team = st.selectbox('Bowling Team',matches.Team1.unique())
    venue = st.selectbox('Venue',matches.Venue.unique())
    balls_left = st.select_slider('Balls Left',options=range(1,121))
    runs_left = st.select_slider('Runs Left',options=range(1,301))
    wickets_left = st.select_slider('Wickets Left',options=range(1,11))
    target = st.number_input("Enter Target")
    
    if st.button('Submit'):
        current_score = target - runs_left
        current_runrate = current_score / ((120-balls_left)/6)
        required_runrate = runs_left / (balls_left/6)
        win_predict = pipe.predict_proba(pd.DataFrame([[batting_team,bowling_team,venue,balls_left,runs_left,wickets_left,current_runrate,required_runrate,current_score,target]],columns=['BattingTeam', 'BowlingTeam', 'Venue', 'balls_left', 'runs_left','Wickets_left', 'current_run_rate', 'required_run_rate','current_score', 'target']))
        trace = go.Pie(labels=[batting_team,bowling_team], values=[win_predict[0][1],win_predict[0][0]], hole=0.5,marker=dict(colors=[team_colors[batting_team],team_colors[bowling_team]]))
        fig = go.Figure(data=trace)
        st.subheader('Win Percentage')
        st.plotly_chart(fig,use_container_width=True)
