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

        display_options = [
            "원문 유지",
            "묘사적 서술",
            "키워드 중심",
            "세부 스펙 포함",
            "저작권 고려",
            "사용자 입력"
        ]

        # 실제 프롬프트에 사용될 상세 내용
        prompt_templates = {
            "원문 유지": """
        이미지 생성시 다음 사항을 준수해주세요:
        1. 원문의 문장 구조와 표현을 최대한 그대로 유지
        2. 불필요한 부가 설명이나 수식어구 최소화
        3. 원문의 맥락과 의도를 정확하게 반영
        4. 이미지 생성에 필수적인 세부 사항만 추가

        원문: {text}
        """,
            
            "묘사적 서술": """
        다음 지침에 따라 원문을 상세하게 재구성해주세요:
        1. 장면의 전반적인 분위기와 감성을 구체적으로 묘사
        2. 등장 요소들의 외형, 질감, 상태를 자세히 서술
        3. 공간감과 원근감을 표현하는 서술 추가
        4. 조명, 그림자, 시간대 등 환경적 요소 포함
        5. 감각적 표현과 은유적 표현 적절히 활용

        원문: {text}
        """,
            
            "키워드 중심": """
        원문에서 다음 요소들을 중심으로 프롬프트를 재구성해주세요:
        1. 핵심 주체/객체 추출
        2. 주요 행동과 상태 식별
        3. 중요 배경 요소 파악
        4. 분위기를 결정짓는 핵심 수식어 선별
        5. 각 키워드 간의 관계성 명시

        추출된 키워드들을 자연스럽게 연결하여 구성

        원문: {text}
        """,
            
            "세부 스펙 포함": """
        다음의 기술적 세부사항을 포함하여 프롬프트를 작성해주세요:

        [스타일 사양]
        - 이미지 스타일: (사진/일러스트/3D 등)
        - 아트 스타일: (사실적/만화적/초현실적 등)
        - 렌더링 스타일: (포토리얼리스틱/스케치/수채화 등)

        [카메라/구도 사양]
        - 촬영 각도: (정면/측면/부감/앙감)
        - 화각: (광각/표준/망원)
        - 거리: (클로즈업/중간/원경)

        [이미지 품질 사양]
        - 해상도: (8K/4K/FHD)
        - 디테일 수준: (초고해상도/일반/러프)
        - 노이즈/그레인: (없음/자연스러운/스타일리시)

        [색감/조명 사양]
        - 주조색: (따뜻한/차가운/모노톤)
        - 조명 스타일: (자연광/인공광/드라마틱)
        - 명암 대비: (강한/부드러운/평이한)

        원문: {text}
        """,
            
            "저작권 고려": """
        다음 지침에 따라 저작권을 고려한 프롬프트를 생성해주세요:

        [저작권 요소 처리 기준]
        1. 브랜드/로고
        - 구체적 명칭 → 일반적 형태 묘사
        - 예: "코카콜라 로고" → "붉은색 필기체 로고"

        2. 캐릭터
        - 고유명 → 일반적 특징 묘사
        - 예: "피카츄" → "노란색 몬스터 캐릭터", "스타워즈" → "우주 배경 미래 전투"

        3. 상표/디자인
        - 특정 제품명 → 제품 유형과 특징
        - 예: "아이폰" → "현대적 디자인의 스마트폰"

        4. 예술작품
        - 구체적 작품명 → 스타일과 주제 묘사
        - 예: "모나리자" → "르네상스 스타일의 여인 초상화"

        5. PG19
        - 잔인하거나, Gore한 느낌이 나는 경우, 그 이미지의 느낌만 가져오고 세부적인 사건 묘사는 제거해줘. 
        - 예 : "비내리는 피가 낭자한 사건 사진" → "비가 내리고 있는 어두운 느낌의 골목 사진" 

        원문: {text}
        """,
            
            "사용자 입력": "{text}"  # 사용자 정의 프롬프트는 그대로 사용
        }

        # Streamlit UI
        selected_display_option = st.selectbox("프롬프트 재해석 옵션", display_options)

        # 사용자 입력 옵션일 경우
        if selected_display_option == "사용자 입력":
            user_prompt = st.text_area("원하는 프롬프트 형식을 입력하세요")
            final_prompt_template = user_prompt
        else:
            final_prompt_template = prompt_templates[selected_display_option]

        aspect_ratio_options = [
            "16:9",
            "4:3",
            "1:1"
        ]
        selected_aspect_ratio = st.selectbox("종횡비 선택", aspect_ratio_options)

        if final_prompt_template == "사용자 입력":
            user_input = st.text_area("임시 재해석 옵션 프롬프트를 입력하세요")
        else:
            user_input = None

        if st.button("분석"):
            if user_prompt and final_prompt_template:
                if final_prompt_template == "사용자 입력":
                    result = call_gemini(user_prompt, user_input)
                else:
                    result = call_gemini(user_prompt, final_prompt_template)

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