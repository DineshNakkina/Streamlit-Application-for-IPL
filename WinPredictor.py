import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import streamlit as st


def WinPredictor(final,matches,team_colors):

    trf = ColumnTransformer([('trf', OneHotEncoder(sparse_output=False,drop='first'),['BattingTeam','BowlingTeam','Venue'])],remainder = 'passthrough')

    X = final.drop('result', axis=1)
    y = final['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    pipe = Pipeline(steps=[('step1',trf),('step2',RandomForestClassifier())])
    pipe.fit(X_train, y_train)

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
        trace = go.Pie(labels=[batting_team,bowling_team], values=[win_predict[0][0],win_predict[0][1]], hole=0.5,marker=dict(colors=[team_colors[batting_team],team_colors[bowling_team]]))
        fig = go.Figure(data=trace)
        st.subheader('Win Percentage')
        st.plotly_chart(fig,use_container_width=True)