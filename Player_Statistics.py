import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def player_statistics(merged,deliveries):
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
            st.image('Images/Unknown.jpg',width=200)
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
