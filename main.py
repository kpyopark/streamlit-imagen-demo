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
imagen3_model = ImageGenerationModel.from_pretrained(
    "imagen-3.0-generate-preview-0611")


async def generate_images(model, positive_prompt, negative_prompt, aspect_ratio):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(
            pool,
            lambda: model.generate_images(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                number_of_images=2,
                aspect_ratio=aspect_ratio
            )
        )
    return response


def extract_json_value(json_str):
    start_index = json_str.find('```json') + 7
    end_index = json_str.find('```', start_index)
    json_str = json_str[start_index:end_index].strip()
    return json.loads(json_str)


def call_gemini(prompt, instruction):
    prompt_template = f"""당신은 Imagen을 이용하여 광고 이미지를 구성하는 광고 담당자입니다. 
주어진 사용자 프롬프트를 분석하여 Imagen에서 그릴 수 있는 좋은 프롬프트로 변경해주세요.
결과는 json 형태로 나와야 하며, positive와 negative 두 개의 키가 있어야 합니다. 모든 결과는 영문으로 생성해주세요.

세부 인스트럭션을 따라서, output을 생성해 주세요. 

<사용자 프롬프트>
{prompt}
</사용자 프롬프트>

<세부 인스트럭션>
{instruction}
</세부 인스트럭션>
    """
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [prompt_template],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        }
    )
    return response.text


async def generate_all_images(positive_prompt, negative_prompt, aspect_ratio):
    tasks = [
        generate_images(imagen2_model, positive_prompt,
                        negative_prompt, aspect_ratio),
        generate_images(imagen3_model, positive_prompt,
                        negative_prompt, aspect_ratio)
    ]
    return await asyncio.gather(*tasks)


def main():
    st.set_page_config(layout="wide")

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.title("프롬프트 분석")

        user_prompt = st.text_input("이미지를 생성할 프롬프트를 입력하세요:")

        prompt_options = [
            "되도록 원문을 그대로 살려서 생성해줘.",
            "원문을 좀 더 묘사적으로 서술해줘.",
            "원문에서 주요 키워드를 중심으로 구성해줘.",
            "스타일, 톤, 느낌, 광각여부, 카메라크기, 해상도, 색감등을 좀 더 세밀하게 묘사해서 생성해줘.",
            "사용자 입력"  # 새로 추가된 옵션
        ]
        selected_prompt = st.selectbox("프롬프트 재해석 옵션", prompt_options)
        aspect_ratio_options = [
            "16:9",
            "4:3",
            "1:1"
        ]
        selected_aspect_ratio = st.selectbox("종횡비 선택", aspect_ratio_options)

        # 사용자 입력 옵션 선택 시 텍스트 에어리어 활성화
        if selected_prompt == "사용자 입력":
            user_input = st.text_area("임시 재해석 옵션 프롬프트를 입력하세요")
        else:
            user_input = None  # 사용자 입력 옵션이 아닐 때는 user_input 값을 None으로 설정

        if st.button("분석"):
            if user_prompt and selected_prompt:
                if selected_prompt == "사용자 입력":
                    result = call_gemini(user_prompt, user_input)
                else:
                    result = call_gemini(user_prompt, selected_prompt)

                st.subheader("분석 결과")
                try:
                    json_result = extract_json_value(result)
                    st.json(json_result)

                    # 비동기적으로 이미지 생성
                    imagen2_response, imagen3_response = asyncio.run(generate_all_images(
                        json_result['positive'], json_result['negative'], selected_aspect_ratio))

                    # 결과 업데이트
                    with right_column:
                        st.subheader("생성된 이미지")
                        imagen2_col, imagen3_col = st.columns(2)

                        with imagen2_col:
                            st.subheader("Imagen 2 결과")
                            if imagen2_response.images:
                                if len(imagen2_response.images) >= 1:
                                    imagen2_response.images[0].save(
                                        'imagen2_image1.jpg')
                                    st.image("imagen2_image1.jpg",
                                             use_column_width=True)
                                if len(imagen2_response.images) == 2:
                                    imagen2_response.images[1].save(
                                        'imagen2_image2.jpg')
                                    st.image("imagen2_image2.jpg",
                                             use_column_width=True)

                        with imagen3_col:
                            st.subheader("Imagen 3 결과")
                            if imagen3_response.images:
                                if len(imagen3_response.images) >= 1:
                                    imagen3_response.images[0].save(
                                        'imagen3_image1.jpg')
                                    st.image("imagen3_image1.jpg",
                                             use_column_width=True)
                                if len(imagen3_response.images) == 2:
                                    imagen3_response.images[1].save(
                                        'imagen3_image2.jpg')
                                    st.image("imagen3_image2.jpg",
                                             use_column_width=True)

                except json.JSONDecodeError:
                    st.text(result)

    with right_column:
        st.title("생성된 이미지")
        st.text("분석 버튼을 눌러 이미지를 생성하세요.")


if __name__ == "__main__":
    main()
