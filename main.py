import streamlit as st
import json
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai

from dotenv import load_dotenv
import os
from vertexai.preview.vision_models import ImageGenerationModel

load_dotenv()

PROJECT_ID=os.environ.get("PROJECT_ID")
LOCATION=os.environ.get("LOCATION")

# Vertex AI 초기화 (프로젝트 ID와 위치 설정 필요)
vertexai.init(project=PROJECT_ID, location=LOCATION)

imagen2_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
imagen3_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-preview-0611")
#imagen3_model = ImageGenerationModel.from_pretrained("imagegeneration@007")

def generate_images_via_imagen3(positive_prompt, negative_prompt):
  response = imagen3_model.generate_images(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    number_of_images=2,
  )
  return response

def extract_json_value(json_str):
    start_index = json_str.find('```json') + 7
    end_index = json_str.find('```', start_index)
    json_str = json_str[start_index:end_index].strip()
    print(json_str)
    return json.loads(json_str)

def call_gemini(prompt, instruction):
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [prompt, instruction],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        }
    )
    return response.text

def main():
    st.set_page_config(layout="wide")
    
    # 화면을 두 개의 컬럼으로 나눕니다
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.title("프롬프트 분석")

        # 사용자 프롬프트 입력
        user_prompt = st.text_input("이미지를 생성할 프롬프트를 입력하세요:")

        # 프롬프트 재해석 리스트
        prompt_options = [
            "사용자가 입력한 프롬프트를 Imagen에서 그릴 수 있게 좋은 프롬프트로 변경해줘. 출력은 JSON으로 positive, negative 두개의 키가 있어야함",
            "프롬프트를 더 상세하게 설명해줘.  출력은 JSON으로 positive, negative 두개의 키가 있어야함",
            "프롬프트에서 주요 키워드를 추출해줘.  출력은 JSON으로 positive, negative 두개의 키가 있어야함"
        ]
        selected_prompt = st.selectbox("프롬프트 재해석 옵션", prompt_options)

        if st.button("분석"):
            if user_prompt and selected_prompt:
                # Gemini 호출
                result = call_gemini(user_prompt, selected_prompt)
                
                # JSON 결과 표시
                st.subheader("분석 결과")
                try:
                    json_result = extract_json_value(result)
                    st.json(json_result)

                    # 이미지 생성
                    imagen3_response = generate_images_via_imagen3(json_result['positive'], json_result['negative'])

                    # Imagen3 결과 업데이트
                    with right_column.container():
                        st.subheader("Imagen 3 결과")
                        imagen3_response.images[0].save('imagen3_image1.jpg')
                        imagen3_response.images[1].save('imagen3_image2.jpg')
                        img3 = st.image("imagen3_image1.jpg", use_column_width=True)
                        img4 = st.image("imagen3_image2.jpg", use_column_width=True)
                except json.JSONDecodeError:
                    st.text(result)

    with right_column:
        st.title("생성된 이미지")

        # 이미지 표시 (실제 Imagen 호출 대신 예시 이미지 사용)
        imagen2_col, imagen3_col = st.columns(2)
        
        with imagen2_col:
            st.subheader("Imagen 2 결과")
            img1 = st.image("https://via.placeholder.com/300x300.png?text=Imagen2+Image+1", use_column_width=True)
            img2 = st.image("https://via.placeholder.com/300x300.png?text=Imagen2+Image+2", use_column_width=True)

        with imagen3_col:
            st.subheader("Imagen 3 결과")
            img3 = st.image("https://via.placeholder.com/300x300.png?text=Imagen3+Image+1", use_column_width=True)
            img4 = st.image("https://via.placeholder.com/300x300.png?text=Imagen3+Image+2", use_column_width=True)

        # 이미지 클릭 시 확대 기능
        st.subheader("이미지 확대")
        selected_image = st.selectbox("확대할 이미지 선택", ["Imagen2 Image 1", "Imagen2 Image 2", "Imagen3 Image 1", "Imagen3 Image 2"])
        
        if selected_image:
            st.image(f"https://via.placeholder.com/800x800.png?text={selected_image.replace(' ', '+')}", use_column_width=True)

if __name__ == "__main__":
    main()