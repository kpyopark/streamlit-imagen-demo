import streamlit as st
import generator

st.set_page_config(layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["이미지 생성", "업스케일링", "마스크수정", "전체수정"])

with tab1:
    generator.main()  # app2.py의 run_app2 함수 호출
