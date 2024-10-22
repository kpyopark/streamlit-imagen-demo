import streamlit as st
import generator
import edit
import controlled_editing

st.set_page_config(layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["이미지 생성", "업스케일링", "마스크수정", "Sketch to Image"])

with tab1:
    generator.main()  # app2.py의 run_app2 함수 호출
with tab3:
    edit.edit_image_app()
with tab4:
    controlled_editing.controlled_editing_tab()
