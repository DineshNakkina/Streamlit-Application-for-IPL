import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast

def team_records(merged,matches,team_colors):
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
                path = f'Images/{team_name}.jpg'
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
                margin = int(single_match['Margin']),single_match['WonBy']
                mom = single_match['Player_of_Match']
                with st.container(border=True):
                    st.write("Opponent :",opp_team)
                    st.write("Result :",result)
                    st.write("By:",margin)
                    st.write("Man of the Match:",mom)
