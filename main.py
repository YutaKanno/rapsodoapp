import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import numpy as np
from matplotlib.patches import Ellipse
import io
from datetime import datetime  
import functions



PASSWORD = 'bbc298'
USERNAME = 'itftkb'


# パスワードなどをセッションで保存
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ログイン画面
def login():
    st.title("ログイン画面")

    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type="password")
    login_button = st.button("ログイン")

    if login_button:
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("ログイン成功！")
            st.write('ボタンをもう一度押してください')
        else:
            st.error("ユーザー名またはパスワードが間違っています")

# メイン画面
def main_page():
    rapdata = pd.read_csv('rapdata.csv')
    rapdata['Date'] = pd.to_datetime(rapdata['Date'])
    # Set the page layout to wide mode
    st.set_page_config(layout="wide")

    # Add a sidebar with a selectbox for the Name column
    st.sidebar.image('./Rapsodo-Logo.png')
    st.sidebar.header("Filter by Name")
    
    selected_name = st.sidebar.selectbox("Select a Name", rapdata['Name'].unique())
    filt_data = rapdata[rapdata['Name'] == selected_name]
    # Add a sidebar for date filtering
    st.sidebar.header("Filter by Date")
    start_date = st.sidebar.date_input("Start Date", value=rapdata['Date'].min().date())
    end_date = st.sidebar.date_input("End Date", value=rapdata['Date'].max())
    filt_data2 = filt_data[(filt_data['Date'] >= pd.to_datetime(start_date)) & 
                            (filt_data['Date'] <= pd.to_datetime(end_date))]
    # Add a sidebar for Pitch_Type filtering
    st.sidebar.header("Filter by Pitch Type")
    selected_pitch_types = st.sidebar.multiselect(
        "Select Pitch Type(s)", 
        options=filt_data2['Pitch_Type'].unique(), 
        default=filt_data2['Pitch_Type'].unique()
    )
    filtered_data = filt_data2[filt_data2['Pitch_Type'].isin(selected_pitch_types)]
    
    
    
    
    st.image('header.png')
    st.write('<<選手選択,その他フィルターは左のサイドバーから>>')
    st.header(selected_name)
    st.subheader(f'{start_date}~{end_date}')
    st.header("Summary")


    

    

    table = functions.make_rap_table(filtered_data)
    st.dataframe(table, use_container_width=True)
        
        
    st.header("Movement")

    col1_1, col1_2 = st.columns(2)

    with col1_1:
        st.subheader("Trajectory Data")
        mov_plot = functions.mov_plot(filtered_data, 'trajectory')
        st.pyplot(mov_plot)

    with col1_2:
        st.subheader("Spin Based Data")
        mov_plot = functions.mov_plot(filtered_data, 'spin')
        st.pyplot(mov_plot)
        
    st.header('Density Plot')
    plot_kind = st.selectbox('', ['球速', '回転数', '回転効率'], key='density_plot_kind')
    speed_plot = functions.make_density_plot(filtered_data, plot_kind)
    st.pyplot(speed_plot)


    col1, col2 = st.columns(2)
    with col1:
        st.header('Release')
        release = functions.release_plot(filtered_data)
        st.pyplot(release)
        
    with col2:
        st.header('Tilt')
        axis = functions.plot_axis(filtered_data)
        st.plotly_chart(axis, use_container_width=True)
        
    col1, col2 = st.columns(2)
    with col1:
        st.header('from Catcher')
        spin_rate = functions.zone_plot(filtered_data, 'c')
        st.pyplot(spin_rate)
    with col2:
        st.header('from Pitcher')
        spin_eff = functions.zone_plot(filtered_data, 'p')
        st.pyplot(spin_eff)
        

    st.header('Trend Plot')
    plot_kind2 = st.selectbox('', ['球速', '回転数', '回転効率'], key='trend_plot_kind')
    trend_plot = functions.trend_plot(filtered_data, plot_kind2)
    st.pyplot(trend_plot)


    st.header('By Date Summary')
    summary2 = functions.make_rap_table_by_date(filtered_data)
    st.dataframe(summary2)


    st.header('Dataset')
    st.dataframe(filtered_data)
    



# 表示切り替え
if st.session_state.logged_in:
    main_page()
else:
    login()