import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def batsman_vs_bowler(merged):
    batsmans = list(merged.batter.unique())
    bowlers = list(merged.bowler.unique())
    col1,col2=st.columns(2)
    with col1:
        with st.container(border=True):
            batter =  st.sidebar.selectbox('Select Batsman',batsmans)
                
    with col2:
        with st.container(border=True):
            bowler =  st.sidebar.selectbox('Select Bowler',bowlers)
    if st.sidebar.button('Submit'):
        vs = merged[(merged.batter == batter) & (merged.bowler == bowler)]
        vs['isBowlerWicket'] = (~vs['kind'].isin(['run out', 'retired out', 'retired hurt', 'obstructing the field',np.nan])).astype(int)
        vs['isDotBall'] = vs['total_run'].apply(lambda x: 1 if x==0 else 0)
        sr = round(vs['batsman_run'].sum()/vs['isLegal'].sum(),2) * 100
        with st.container(border=True):
            fig1 = go.Figure(go.Indicator(mode = "gauge+number",value = vs.groupby(['ID'])['innings'].nunique().sum(),domain = {'x': [0, 1], 'y': [0, 1]},title = {'text': "Innings"}))
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