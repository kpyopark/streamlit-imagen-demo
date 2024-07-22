import streamlit as st
import json
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
from dotenv import load_dotenv
import os
from vertexai.preview.vision_models import ImageGenerationModel
import asyncio
import concurrent.futures

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")

vertexai.init(project=PROJECT_ID, location=LOCATION)

imagen2_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
imagen3_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-preview-0611")

async def generate_images(model, positive_prompt, negative_prompt):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(
            pool,
            lambda: model.generate_images(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                number_of_images=2,
            )
        )
    return response

def extract_json_value(json_str):
    start_index = json_str.find('```json') + 7
    end_index = json_str.find('```', start_index)
    json_str = json_str[start_index:end_index].strip()
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

async def generate_all_images(positive_prompt, negative_prompt):
    tasks = [
        generate_images(imagen2_model, positive_prompt, negative_prompt),
        generate_images(imagen3_model, positive_prompt, negative_prompt)
    ]
    return await asyncio.gather(*tasks)

def main():
    st.set_page_config(layout="wide")
    
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.title("프롬프트 분석")

        user_prompt = st.text_input("이미지를 생성할 프롬프트를 입력하세요:")

        prompt_options = [
            "사용자가 입력한 프롬프트를 Imagen에서 그릴 수 있게 좋은 프롬프트로 변경해줘. 출력은 JSON으로 positive, negative 두개의 키가 있어야함",
            "프롬프트를 더 상세하게 설명해줘. 출력은 JSON으로 positive, negative 두개의 키가 있어야함",
            "프롬프트에서 주요 키워드를 추출해줘. 출력은 JSON으로 positive, negative 두개의 키가 있어야함"
        ]
        selected_prompt = st.selectbox("프롬프트 재해석 옵션", prompt_options)

        if st.button("분석"):
            if user_prompt and selected_prompt:
                result = call_gemini(user_prompt, selected_prompt)
                
                st.subheader("분석 결과")
                try:
                    json_result = extract_json_value(result)
                    st.json(json_result)

                    # 비동기적으로 이미지 생성
                    imagen2_response, imagen3_response = asyncio.run(generate_all_images(json_result['positive'], json_result['negative']))

                    # 결과 업데이트
                    with right_column:
                        st.subheader("생성된 이미지")
                        imagen2_col, imagen3_col = st.columns(2)
                        
                        with imagen2_col:
                            st.subheader("Imagen 2 결과")
                            imagen2_response.images[0].save('imagen2_image1.jpg')
                            imagen2_response.images[1].save('imagen2_image2.jpg')
                            st.image("imagen2_image1.jpg", use_column_width=True)
                            st.image("imagen2_image2.jpg", use_column_width=True)

                        with imagen3_col:
                            st.subheader("Imagen 3 결과")
                            imagen3_response.images[0].save('imagen3_image1.jpg')
                            imagen3_response.images[1].save('imagen3_image2.jpg')
                            st.image("imagen3_image1.jpg", use_column_width=True)
                            st.image("imagen3_image2.jpg", use_column_width=True)

                except json.JSONDecodeError:
                    st.text(result)

    with right_column:
        st.title("생성된 이미지")
        st.text("분석 버튼을 눌러 이미지를 생성하세요.")

if __name__ == "__main__":
    main()