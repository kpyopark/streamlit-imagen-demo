import streamlit as st
import generator
import edit
import controlled_editing
import product_editing

st.set_page_config(layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["이미지 생성", "Product Editing", "마스크수정", "Sketch to Image"])

with tab1:
    generator.main()  # app2.py의 run_app2 함수 호출
with tab2:
    product_editing.main()
with tab3:
    edit.main()
with tab4:
    controlled_editing.main()
