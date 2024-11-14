from dotenv import load_dotenv
import os
import json
import base64
import requests
from vertexai.preview.generative_models import GenerativeModel, Part
from dataclasses import dataclass
from typing import List, Dict
from pprint import pprint

load_dotenv()

@dataclass
class ImageInfo:
    path: str
    subject_type: str = ""
    subject_description: str = ""

@dataclass
class GeminiResponse:
    image_infos: List[ImageInfo]
    positive_prompt: str
    negative_prompt: str

# Configuration variables
REGION = "us-central1"
CLOUD_PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_URI_PREFIX = f"https://{REGION}-preprod-aiplatform.googleapis.com"
ENDPOINT_URI = f"{ENDPOINT_URI_PREFIX}/v1/projects/{CLOUD_PROJECT_ID}/locations/{REGION}/publishers/google/models/imagen-3.0-capability-preview-0930:predict"

def remove_reference_image(obj):
    """
    재귀적으로 객체를 순회하면서 'referenceImage' 속성을 제거합니다.
    
    Args:
        obj: 처리할 Python 객체 (dict, list, 또는 기본 타입)
    
    Returns:
        referenceImage가 제거된 새로운 객체
    """
    # 입력받은 객체의 복사본을 만들어서 원본 데이터 보존
    if isinstance(obj, dict):
        new_obj = {}
        # 딕셔너리의 각 키-값 쌍을 순회
        for key, value in obj.items():
            # referenceImage 키는 제외
            if key != 'referenceImage':
                # 값에 대해 재귀적으로 처리
                new_obj[key] = remove_reference_image(value)
            else:
                new_obj[key] = "bytes"
        return new_obj
    elif isinstance(obj, list):
        # 리스트의 각 항목에 대해 재귀적으로 처리
        return [remove_reference_image(item) for item in obj]
    elif isinstance(obj, tuple):
        # 튜플의 각 항목에 대해 재귀적으로 처리
        return tuple(remove_reference_image(item) for item in obj)
    elif isinstance(obj, set):
        # 셋의 각 항목에 대해 재귀적으로 처리
        return {remove_reference_image(item) for item in obj}
    else:
        # 기본 타입은 그대로 반환
        return obj
    
def print_request_data(request_data):
    pprint(remove_reference_image(request_data))

def extract_json_value(json_str):
    start_index = json_str.find('```json') + 7
    end_index = json_str.find('```', start_index)
    json_str = json_str[start_index:end_index].strip()
    return json.loads(json_str)

def call_gemini_for_editing(image_infos: List[ImageInfo], user_prompt: str) -> GeminiResponse:
    images_parts = [
        Part.from_data(
            mime_type="image/png",
            data=encode_image(img_info.path)
        ) for img_info in image_infos
    ]
    
    prompt_template = f"""
당신은 Imagen을 이용하여 광고 이미지를 구성하는 광고 담당자입니다. 
주어진 여러 상품 사진들을 분석하고, 사용자의 요청에 맞는 배경 이미지를 생성하기 위한 영문 Imagen 프롬프트를 작성해줘. 

사용자 요청: {user_prompt}

<instructions>

<task1> 주어진 각 이미지에 대해 다음 정보를 분석하세요:
- 이미지의 주요 상품/객체 설명 (subject_description)
- 상품/객체 유형 (subject_type: SUBJECT_TYPE_PERSON, SUBJECT_TYPE_ANIMAL, SUBJECT_TYPE_PRODUCT, SUBJECT_TYPE_DEFAULT) </task1>

<task2> 모든 상품들의 공통된 특성과 차이점을 분석하세요. </task2>

<task3> 사용자의 요청사항을 분석하고 이에 맞는 배경 이미지 생성을 위한 positive_prompt를 작성하세요 (영문, 120 tokens 이내). </task3>

<task4> 이미지 품질을 저해할 수 있는 요소들에 대한 negative_prompt를 작성하세요 (영문, 60 tokens 이내). </task4>

</instructions>

<output>
{{
  "images": [
    {{
      "subject_description": "상품/객체에 대한 상세 설명",
      "subject_type": "SUBJECT_TYPE_XXX"
    }},
    ...
  ],
  "positive_prompt": "배경 이미지 생성을 위한 상세 프롬프트",
  "negative_prompt": "피해야 할 요소들에 대한 프롬프트"
}}
</output>
"""

    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [*images_parts, prompt_template],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.5,
            "top_p": 0.93,
            "top_k": 32
        }
    )
    
    result = extract_json_value(response.text)
    
    # Update image infos with Gemini's analysis
    for i, img_analysis in enumerate(result["images"]):
        image_infos[i].subject_type = img_analysis["subject_type"]
        image_infos[i].subject_description = img_analysis["subject_description"]
    
    return GeminiResponse(
        image_infos=image_infos,
        positive_prompt=result["positive_prompt"],
        negative_prompt=result["negative_prompt"]
    )

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_access_token():
    stream = os.popen('gcloud auth print-access-token')
    return stream.read().strip()

def make_prediction_request(endpoint_uri, access_token, request_data):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(endpoint_uri, headers=headers, json=request_data)
    response.raise_for_status()
    return response.json()

def convert_response_to_images(response):
    images = []
    if "predictions" in response:
        for prediction in response["predictions"]:
            if "bytesBase64Encoded" in prediction:
                image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
                images.append(image_bytes)
    return images

def product_editing(gemini_response: GeminiResponse):
    """
    Create a background image for multiple products using Imagen
    """
    access_token = get_access_token()
    
    # Prepare reference images data
    reference_images = []
    for idx, img_info in enumerate(gemini_response.image_infos):
        reference_images.append({
            "referenceType": "REFERENCE_TYPE_SUBJECT",
            "referenceId": 1,
            "referenceImage": {
                "bytesBase64Encoded": encode_image(img_info.path)
            },
            "subjectImageConfig": {
                "subjectDescription": img_info.subject_description,
                "subjectType": img_info.subject_type
            }
        })

    request_data = {
        "instances": [
            {
                "prompt": gemini_response.positive_prompt,
                "referenceImages": reference_images
            }
        ],
        "parameters": {
            "negativePrompt": gemini_response.negative_prompt,
            "seed": 12,
            "sampleCount": 4,
            "editConfig": {
                "baseSteps": 75
            },
            "promptLanguage": "en",
        }
    }

    print_request_data(request_data)

    response = make_prediction_request(ENDPOINT_URI, access_token, request_data)

    return convert_response_to_images(response)

def save_images(images: List[bytes], prefix: str = "generated"):
    """Save generated images to disk"""
    for idx, image_bytes in enumerate(images):
        with open(f"{prefix}_{idx}.png", "wb") as f:
            f.write(image_bytes)

# Example usage
if __name__ == "__main__":
    # Example with multiple images
    image_infos = [
        ImageInfo(path="product1.png"),
        ImageInfo(path="product2.png")
    ]
    
    # First, get analysis and prompts from Gemini
    user_prompt = "Create a modern minimalist background for these products"
    gemini_result = call_gemini_for_editing(image_infos, user_prompt)
    
    # Print the analysis results
    print("Gemini Analysis Results:")
    for img_info in gemini_result.image_infos:
        print(f"\nImage: {img_info.path}")
        print(f"Subject Type: {img_info.subject_type}")
        print(f"Description: {img_info.subject_description}")
    
    print(f"\nPositive Prompt: {gemini_result.positive_prompt}")
    print(f"Negative Prompt: {gemini_result.negative_prompt}")
    
    # Generate the background image
    generated_images = product_editing(gemini_result)
    
    # Save the results
    save_images(generated_images)