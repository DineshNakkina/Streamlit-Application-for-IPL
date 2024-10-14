import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def team_vs_team(matches, merged, team_colors):
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
                path = f'Images/{team_one}.jpg'
                st.image(path,use_column_width=True) 
        with c2:
            with st.container(border=True):
                path = f'Images/{team_two}.jpg'
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
                st.plotly_chart(fig,use_container_width=True, key="win_percentage_chart")
        with col2:
            with st.container(border=True):
                trace2 = go.Pie(labels=[team_one,team_two], values=[result[4],result[5]], hole=0.5,marker=dict(colors=[team_colors[team_one],team_colors[team_two]]))
                fig0 = go.Figure(data=trace2)
                st.subheader('Toss Win Percentage')
                st.plotly_chart(fig0,use_container_width=True, key="toss_win_chart")    
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
            st.plotly_chart(fig1, use_container_width=True, key="batting_chart")
        with st.container(border=True):
            fig2 = go.Figure(data=[go.Bar(name=team_one, x=bowling_chart['Season'], y=bowling_chart[team_one],marker_color=team_colors[team_one]),go.Bar(name=team_two, x=bowling_chart['Season'], y=bowling_chart[team_two],marker_color=team_colors[team_two])])
            fig2.update_layout(barmode='group', xaxis_title='Season', yaxis_title='Wickets')
            fig2.update_xaxes(tickvals=bowling_chart['Season'])
            st.subheader('Wickets over the Seasons')
            st.plotly_chart(fig2, use_container_width=True, key="bowling_chart")

        with st.container(border=True):
            st.header('Recent Form :')
            st.subheader(f"{team_one}: {recent_team1}")
            st.subheader(f"{team_two}: {recent_team2}")
