import streamlit as st
import os
from vertexai.preview.vision_models import Image
import sketchToImage

def initialize_session_state():
    """Session state 초기화 함수"""
    if 'extracted_params' not in st.session_state:
        st.session_state.extracted_params = {
            'org_description': '',
            'main_object': '',
            'subject_type': 'SUBJECT_TYPE_DEFAULT',
            'prompt': '',
            'negative_prompt': '',
            'control_type': 'CONTROL_TYPE_CANNY',
            'mask_mode': 'NONE',
            'edit_mode': 'NONE',
            'edit_type': 'RAW_EDITING',
        }

def main():
    # Session state 초기화
    initialize_session_state()
    
    # 좌측 컬럼 생성
    controlled_edited_col1, controlled_edited_col2 = st.columns([1, 1])
    
    with controlled_edited_col1:
        # 이미지 파일 업로더
        controlled_edited_uploaded_files = st.file_uploader(
            "이미지 파일을 선택하세요",
            type=['png'],
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
                st.image(temp_path, caption=uploaded_file.name, use_column_width=True)
        
        controlled_edited_editing_goal = st.text_area(
            "원하는 수정 사항을 적어주세요. ",
            key="controlled_edited_editing_goal"
        )
        
        # 자동 파라미터 추출 버튼
        if st.button("자동 파라미터 추출", key="controlled_edited_extract_button"):
            if controlled_edited_image_paths:
                try:
                    # 자동 파라미터 추출 함수 호출
                    result = sketchToImage.call_gemini_for_editing(
                        controlled_edited_image_paths[0],
                        controlled_edited_editing_goal
                    )

                    print(result)
                    
                    # 결과를 session state에 저장
                    st.session_state.extracted_params = {
                        'org_description': result.get('org_image_description', ''),
                        'main_object': result.get('main_object_description', ''),
                        'edit_type': result.get('edit_type', 'RAW_EDITING'),
                        'edit_mode': result.get('edit_mode', 'NONE'),
                        'mask_mode': result.get('mask_mode', 'NONE'),
                        'subject_type': result.get('subject_type', 'SUBJECT_TYPE_DEFAULT'),
                        'prompt': result.get('positive_prompt', ''),
                        'negative_prompt': result.get('negative_prompt', ''),
                        'control_type': result.get('control_type', 'CONTROL_TYPE_SCRIBBLE'),
                    }
                    
                    st.success("파라미터가 성공적으로 추출되었습니다!")
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"이미지 분석 중 오류가 발생했습니다: {str(e)}")
            else:
                st.warning("이미지를 먼저 업로드해주세요.")

        # 에디트 모드 선택 (extracted_params의 값을 기본값으로 사용)
        controlled_edited_edit_type = st.selectbox(
            "Edit type 선택",
            ["CONTROLLED_EDITING", "SUBJECT_EDITING", "RAW_EDITING", "STYLE_EDITING"],
            index=["CONTROLLED_EDITING", "SUBJECT_EDITING", "RAW_EDITING", "STYLE_EDITING"].index(
                st.session_state.extracted_params.get('edit_type', "RAW_EDITING")
            ),
            key="controlled_edited_edit_type"
        )

        controlled_edited_edit_mode = st.selectbox(
            "Edit mode 선택",
            ["EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_OUTPAINT", "NONE"],
            index=["EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_OUTPAINT", "NONE"].index(
                st.session_state.extracted_params.get('edit_mode', "NONE")
            ),
            key="controlled_edited_edit_mode"
        )

        # 원본 이미지 설명
        controlled_edited_org_description = st.text_area(
            "원본 이미지의 설명을 입력하세요",
            value=st.session_state.extracted_params.get('org_description', ''),
            key="controlled_edited_org_description"
        )

        # 주요 객체 설명
        controlled_edited_main_object = st.text_area(
            "주요 중심 객체에 대한 설명을 입력하세요",
            value=st.session_state.extracted_params.get('main_object', ''),
            key="controlled_edited_main_object"
        )

        # 주요 객체 타입
        subject_types = ["SUBJECT_TYPE_PERSON", "SUBJECT_TYPE_ANIMAL", "SUBJECT_TYPE_PRODUCT", "SUBJECT_TYPE_DEFAULT"]
        controlled_edited_subject_type = st.selectbox(
            "주요 오브젝트의 타입을 선택하세요",
            subject_types,
            index=subject_types.index(st.session_state.extracted_params.get('subject_type', "SUBJECT_TYPE_DEFAULT")),
            key="controlled_edited_subject_type"
        )

        # 프롬프트 입력
        controlled_edited_prompt = st.text_area(
            "이미지 변경용 프롬프트를 입력하세요",
            value=st.session_state.extracted_params.get('prompt', ''),
            key="controlled_edited_prompt"
        )
        
        # 네거티브 프롬프트
        controlled_edited_negative_prompt = st.text_area(
            "네거티브 프롬프트를 입력하세요",
            value=st.session_state.extracted_params.get('negative_prompt', ''),
            key="controlled_edited_negative_prompt"
        )
        
        # 컨트롤 타입
        control_types = ["CONTROL_TYPE_CANNY", "CONTROL_TYPE_SCRIBBLE"]
        controlled_edited_control_type = st.selectbox(
            "컨트롤 타입 선택",
            control_types,
            index=control_types.index(st.session_state.extracted_params.get('control_type', "CONTROL_TYPE_CANNY")),
            key="controlled_edited_control_type"
        )

        # 마스크 모드
        mask_modes = ["MASK_MODE_BACKGROUND", "MASK_MODE_FOREGROUND", "NONE"]
        controlled_edited_mask_mode = st.selectbox(
            "마스크 모드 선택",
            mask_modes,
            index=mask_modes.index(st.session_state.extracted_params.get('mask_mode', "NONE")),
            key="controlled_edited_mask_mode"
        )

        seed = st.text_input(label='seed value')
        
        controlled_edited_dilation = st.slider("Mask Dilation", min_value=0.0, max_value=1.0, value=0.005, step=0.001)
        st.write('value : {:.3f}'.format(controlled_edited_dilation))

        # 이미지 수정 버튼
        if st.button("이미지 수정", key="controlled_edited_modify_button"):
            if controlled_edited_image_paths:
                try:
                    if controlled_edited_edit_type == "SUBJECT_EDITING":
                        controlled_edited_results = sketchToImage.subject_editing(
                            controlled_edited_prompt,
                            controlled_edited_negative_prompt,
                            controlled_edited_org_description,
                            controlled_edited_image_paths
                        )
                    elif controlled_edited_edit_type == "RAW_EDITING":
                        controlled_edited_results = sketchToImage.raw_editing(
                            controlled_edited_prompt,
                            controlled_edited_negative_prompt,
                            controlled_edited_edit_mode,
                            controlled_edited_mask_mode,
                            controlled_edited_dilation,
                            controlled_edited_image_paths, 
                            seed=seed
                        )
                    elif controlled_edited_edit_type == "STYLE_EDITING":
                        controlled_edited_results = sketchToImage.style_editing(
                            controlled_edited_prompt,
                            controlled_edited_negative_prompt,
                            controlled_edited_image_paths
                        )
                    elif controlled_edited_edit_type == "CONTROLLED_EDITING":
                        controlled_edited_results = sketchToImage.controlled_editing(
                            controlled_edited_prompt,
                            controlled_edited_negative_prompt,
                            controlled_edited_image_paths,
                            controlled_edited_control_type
                        )
                    else:
                        print("Wrong Editing Mode.")
                    
                    # 결과 이미지 저장 및 표시
                    for idx, result_image in enumerate(controlled_edited_results):
                        output_path = f"controlled_edited_result_{idx}.png"
                        result_image.save(output_path)
                        
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
    main()