import streamlit as st
import os
from vertexai.preview.vision_models import Image
import sketchToImage

def controlled_editing_tab():
    # 좌측 컬럼 생성
    controlled_edited_col1, controlled_edited_col2 = st.columns([1, 1])
    
    with controlled_edited_col1:
        # 이미지 파일 업로더
        controlled_edited_uploaded_files = st.file_uploader(
            "이미지 파일을 선택하세요",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="controlled_edited_uploader"
        )
        
        # 업로드된 이미지 경로 저장
        controlled_edited_image_paths = []
        if controlled_edited_uploaded_files:
            for uploaded_file in controlled_edited_uploaded_files:
                # 임시 파일로 저장
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                controlled_edited_image_paths.append(temp_path)
                
                # 이미지 미리보기 표시
                st.image(temp_path, caption=uploaded_file.name, width=200)
        
        # 프롬프트 입력
        controlled_edited_prompt = st.text_area(
            "이미지 변경용 프롬프트를 입력하세요",
            key="controlled_edited_prompt"
        )
        
        # 네거티브 프롬프트 입력
        controlled_edited_negative_prompt = st.text_area(
            "네거티브 프롬프트를 입력하세요",
            key="controlled_edited_negative_prompt"
        )
        
        # 컨트롤 타입 선택
        controlled_edited_control_type = st.selectbox(
            "컨트롤 타입 선택",
            ["CONTROL_TYPE_CANNY", "CONTROL_TYPE_SCRIBBLE"],
            key="controlled_edited_control_type"
        )
        
        # 수정 버튼
        if st.button("이미지 수정", key="controlled_edited_modify_button"):
            if controlled_edited_image_paths:
                try:
                    # 이미지 수정 함수 호출
                    controlled_edited_results = sketchToImage.controlled_editing(
                        controlled_edited_prompt,
                        controlled_edited_negative_prompt,
                        controlled_edited_image_paths,
                        controlled_edited_control_type
                    )
                    
                    # 결과 이미지 저장 및 표시
                    for idx, result_image in enumerate(controlled_edited_results):
                        # 결과 이미지 저장
                        output_path = f"controlled_edited_result_{idx}.png"
                        result_image.save(output_path)
                        
                        # 저장된 이미지를 우측 컬럼에 표시
                        with controlled_edited_col2:
                            st.image(output_path, caption=f"결과 이미지 {idx+1}")
                            
                except Exception as e:
                    st.error(f"이미지 수정 중 오류가 발생했습니다: {str(e)}")
            else:
                st.warning("이미지를 먼저 업로드해주세요.")
        
        # 임시 파일 정리
        for path in controlled_edited_image_paths:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    controlled_editing_tab()