import streamlit as st
import json
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
from dotenv import load_dotenv
import os
from vertexai.preview.vision_models import ImageGenerationModel, Image
import asyncio
import concurrent.futures
from PIL import Image as PILImage

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")

vertexai.init(project=PROJECT_ID, location=LOCATION)

imagen2_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
imagen3_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

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
    print(response)
    return response

def extract_json_value(json_str):
    start_index = json_str.find('```json') + 7
    end_index = json_str.find('```', start_index)
    json_str = json_str[start_index:end_index].strip()
    return json.loads(json_str)

def resize_and_clip_image(image_path, output_path):
    with PILImage.open(image_path) as img:
        width, height = img.size
        
        target_ratio = 16 / 9
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            diff = width - new_width
            left = diff // 2
            right = width - (diff - left)
            img = img.crop((left, 0, right, height))
        else:
            new_height = int(width / target_ratio)
            diff = height - new_height
            top = diff // 2
            bottom = height - (diff - top)
            img = img.crop((0, top, width, bottom))
        
        resolutions = [(3840, 2160), (1920, 1080), (960, 540)]
        target_resolution = min(resolutions, key=lambda r: r[0] * r[1])
        
        img_resized = img.resize(target_resolution, PILImage.LANCZOS)
        img_resized.save(output_path)
    
    return output_path


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
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 32
        }
    )
    return response.text

async def generate_all_images(positive_prompt, negative_prompt, aspect_ratio):
    tasks = [
        generate_images(imagen2_model, positive_prompt, negative_prompt, aspect_ratio),
        generate_images(imagen3_model, positive_prompt, negative_prompt, aspect_ratio)
    ]
    return await asyncio.gather(*tasks)

def upscale_image(image_path, upscale_type, new_size=None, upscale_factor=None, model='imagen2', mime_type='image/png'):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    print("*** start upscaling ***")    
    image = Image(image_bytes=image_bytes)
    
    if model == 'imagen2':
        model = imagen2_model
    elif model == 'imagen3':
        model = imagen3_model
    
    if upscale_type == 'new_size':
        upscaled_image = model.upscale_image(
            image=image,
            new_size=int(new_size)
        )
    else:  # upscale_factor
        upscaled_image = model.upscale_image(
            image=image,
            upscale_factor=upscale_factor
        )
    
    print(upscaled_image)
    return upscaled_image

def get_image_resolution(image_path):
    with PILImage.open(image_path) as img:
        return img.size

def update_file_path(img_path, mime_type):
    # 기존 파일 이름과 디렉토리 분리
    directory, filename = os.path.split(img_path)
    filename_without_ext = os.path.splitext(filename)[0]
    if mime_type == 'image/jpeg' :
        new_extension = '.jpg'
    else:
        new_extension = '.png'
    new_filename = f'upscaled_{filename_without_ext}{new_extension}'
    upscaled_img_path = os.path.join(directory, new_filename)
    
    return upscaled_img_path

def main():
    #st.set_page_config(layout="wide")

    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.title("프롬프트 분석")

        user_prompt = st.text_input("이미지를 생성할 프롬프트를 입력하세요:")

        prompt_options = [
            "되도록 원문을 그대로 살려서 생성해줘.",
            "원문을 좀 더 묘사적으로 서술해줘.",
            "원문에서 주요 키워드를 중심으로 구성해줘.",
            "스타일, 톤, 느낌, 광각여부, 카메라크기, 해상도, 색감등을 좀 더 세밀하게 묘사해서 생성해줘.",
            "되도록 원문을 그대로 살리지만, 로고 캐릭터와 같이 Copyright에 걸린 정보는 세부적으로 내용만 서술하고, 이미지는 생성하지 않도록 해줘.",
            "사용자 입력"
        ]
        selected_prompt = st.selectbox("프롬프트 재해석 옵션", prompt_options)
        aspect_ratio_options = [
            "16:9",
            "4:3",
            "1:1"
        ]
        selected_aspect_ratio = st.selectbox("종횡비 선택", aspect_ratio_options)

        if selected_prompt == "사용자 입력":
            user_input = st.text_area("임시 재해석 옵션 프롬프트를 입력하세요")
        else:
            user_input = None

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

                    imagen2_response, imagen3_response = asyncio.run(generate_all_images(
                        json_result['positive'], json_result['negative'], selected_aspect_ratio))

                    st.session_state.generated_images = []
                    for i, img in enumerate(imagen2_response.images):
                        img_path = f'imagen2_image{i+1}.png'
                        img.save(img_path)
                        resolution = get_image_resolution(img_path)
                        st.session_state.generated_images.append(('Imagen 2', img_path, resolution))

                    for i, img in enumerate(imagen3_response.images):
                        img_path = f'imagen3_image{i+1}.png'
                        img.save(img_path)
                        resolution = get_image_resolution(img_path)
                        st.session_state.generated_images.append(('Imagen 3', img_path, resolution))

                except json.JSONDecodeError:
                    st.text(result)

        st.subheader("Upscale 설정")
        upscale_type = st.selectbox("Upscale 방식", ["new_size", "upscale_factor"])
        
        if upscale_type == "new_size":
            new_size = st.text_input("새 크기 입력 (예: 1024x1024)")
        else:
            upscale_factor = st.selectbox("Upscale 배수", ["x2", "x4"])

        mime_type = st.selectbox("Mime Type", ["image/png", "image/jpeg"])
        upscale_model = st.selectbox("Upscale 모델", ["imagen2", "imagen3"])

    with right_column:
        st.title("생성된 이미지")
        if st.session_state.generated_images:
            for i, (model_name, img_path, resolution) in enumerate(st.session_state.generated_images):
                st.subheader(f"{model_name} 결과 {i%2 + 1}")
                
                # Apply resize_and_clip_image function
                resized_img_path = f'resized_{img_path}'
                resized_img_path = resize_and_clip_image(img_path, resized_img_path)
                
                st.image(resized_img_path, use_column_width=True)
                new_resolution = get_image_resolution(resized_img_path)
                st.caption(f"원본 해상도: {resolution[0]}x{resolution[1]}")
                st.caption(f"조정된 해상도: {new_resolution[0]}x{new_resolution[1]}")
                
                upscale_key = f"upscale_{model_name}_{i}"
                if st.button(f"Upscale {model_name} Image {i%2 + 1}", key=upscale_key):
                    if upscale_type == "new_size":
                        upscaled_img = upscale_image(resized_img_path, upscale_type, new_size=new_size, model=upscale_model, mime_type=mime_type)
                    else:
                        upscaled_img = upscale_image(resized_img_path, upscale_type, upscale_factor=upscale_factor, model=upscale_model, mime_type=mime_type)
                    new_img_path = update_file_path(resized_img_path, mime_type)
                    upscaled_img_path = f'upscaled_{new_img_path}'
                    upscaled_img.save(upscaled_img_path)
                    upscaled_resolution = get_image_resolution(upscaled_img_path)
                    st.image(upscaled_img_path, use_column_width=True)
                    st.caption(f"업스케일된 해상도: {upscaled_resolution[0]}x{upscaled_resolution[1]}")
        else:
            st.text("분석 버튼을 눌러 이미지를 생성하세요.")

if __name__ == "__main__":
    main()