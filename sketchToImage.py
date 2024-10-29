from dotenv import load_dotenv
from pprint import pprint
import os
import json
import base64
import requests
from vertexai.preview.vision_models import ImageGenerationModel, Image
from vertexai.preview.generative_models import GenerativeModel, Part


load_dotenv()

# Configuration variables (replace with your actual values)
REGION = "us-central1"
CLOUD_PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_URI_PREFIX = f"https://{REGION}-preprod-aiplatform.googleapis.com" # Note the {} for formatting

SUBJECT_IMG_FILENAME = "/Users/markpark/Devel/streamlit-imagen-demo/imagen3_image2.png"  # Path to the subject image
SUBJECT_IMG_DESCRIPTION = "a model walking with yellow shirts"  # Description of the subject
#FACEMESH_IMG_FILENAME = "<PLEASE FILL>"  # Path to the face mesh image
#TEXT_PROMPT_TEMPLATE = "Create an image about [1] in the pose of control image [2] to match the description: A pencil style sketch of a full-body portrait of [1] with hatch-cross drawing, hatch drawing of portrait with 6B and graphite pencils, white background, pencil drawing, high quality, pencil stroke, looking at camera, natural human eyes"  # Template for prompt
TEXT_PROMPT_TEMPLATE = "Create an image about [1] in the pose of control image to match the description: [1] looking at camera, natural human eyes"  # Template for prompt
NEG_TEXT_PROMPT = "wrinkles, noise, Low quality, dirty, ugly, low res, multi face, nsfw, nude, rough texture, messy, messy background, weird hair, chinese clothes, chinese hair, traditional asia clothes, color background, photo realistic, photo, super realistic, signature, autograph, sign, text, characters, alphabet, letter"
ENDPOINT_URI = f"{ENDPOINT_URI_PREFIX.format(REGION)}/v1/projects/{CLOUD_PROJECT_ID}/locations/{REGION}/publishers/google/models/imagen-3.0-capability-preview-0930:predict"

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

def call_gemini_for_editing(image_path, prompt):
    image1 = Part.from_data(
        mime_type="image/png",
        data=encode_image(image_path))
    prompt_template = f"""
당신은 Imagen을 이용하여 광고 이미지를 구성하는 광고 담당자입니다. 
주어진 사진에서 사용자의 요청에 맞는 사진으로 수정하기 위한 영문 Imagen 프롬프트를 작성해줘. 

사용자 요청 : {prompt} 

<instructions>

<task1> Analyze the original photo and write a detailed description of it (within 60 tokens). </task1>

<task2> Identify and describe the most central object in the original photo (within 20 tokens). </task2>

<task3> Analyze the user request and identify the edit type:
STYLE_EDITING: Change the overall style of the image.
CONTROLLED_EDITING: Maintain the overall image composition but change color, tone, or convert it to a different art style (e.g., Sketch to Image).
SUBJECT_EDITING: Keep the main object but change its position or background.
RAW_EDITING: Change the main object itself.
Store this information in the edit_type variable. </task3>

<task4> Clearly explain the relationship between the user request and the extracted main object or background. Add this explanation to the edit_mode_selection_reason field. </task4>

<task5> Determine the edit mode based on the user request:
EDIT_MODE_INPAINT_INSERTION: Add to or change the main object or background.
EDIT_MODE_INPAINT_REMOVAL: Remove the main object or background.
EDIT_MODE_OUTPAINT: Extend the background.
NONE: Apply changes to the entire image.
Store this information in the edit_mode variable. </task5>

<task6> Determine the mask mode:
NONE: No mask needed for STYLE_EDITING.
MASK_MODE_FOREGROUND: Mask the main object for edits focused on it.
MASK_MODE_BACKGROUND: Mask the background for edits focused on it.
Store this information in the mask_mode variable. </task6>

<task7> Determine the subject type for SUBJECT_EDITING:
SUBJECT_TYPE_PERSON: If the main object is a person.
SUBJECT_TYPE_ANIMAL: If the main object is an animal.
SUBJECT_TYPE_PRODUCT: If the main object is a product.
SUBJECT_TYPE_DEFAULT: For all other cases.
Store this information in the subject_type variable. You must use one of the values [SUBJECT_TYPE_PERSON, SUBJECT_TYPE_ANIMAL, SUBJECT_TYPE_PRODUCT, SUBJECT_TYPE_DEFAULT] in the subject_type field. </task7>

<task8> Write prompts in English:
For STYLE_EDITING, focus on the overall style in the positive prompt (e.g., "Transform the image to have a Digital Stained Glass style").
For other cases, provide a detailed description of the final desired image in the positive prompt (within 120 tokens) and list important forbidden keywords in the negative prompt(within 60 Tokens). </task8>

<task9> All outputs should be in English. </task9>

</instructions>

<output>
{{
  "org_image_description" : ...,
  "main_object_description" : ...,
  "edit_type\" : ...,
  "edit_mode_selection_reason": ...,
  "edit_mode" : ...,
  "mask_mode\" : ...,
  "subject_type\" : ...,
  "positive_prompt\" : ...,
  "negative_prompt\" : ...,
}}
</output>
    """
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [image1, prompt_template],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.5,
            "top_p": 0.93,
            "top_k": 32
        }
    )
    return extract_json_value(response.text)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_access_token():
    # You might need to adjust this based on your authentication setup
    stream = os.popen('gcloud auth print-access-token')  # Get token from gcloud
    return stream.read().strip()

def make_prediction_request(endpoint_uri, access_token, request_data):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(endpoint_uri, headers=headers, json=request_data)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()

def convert_response_to_image(response):
    images = []
    if "predictions" in response:  # Check if the key exists
        for i, prediction in enumerate(response["predictions"]):
            if "bytesBase64Encoded" in prediction:
                image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
                images.append(Image(image_bytes))
            else:
                print(f"Warning: Prediction at index {i} does not contain 'bytesBase64Encoded' data.")
    else:
        print("Warning: Response does not contain 'predictions'.")
    return images

def save_images_from_response(response):
    if "predictions" in response:  # Check if the key exists
        for i, prediction in enumerate(response["predictions"]):
            if "bytesBase64Encoded" in prediction:
                image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
                with open(f"image_edited_{i}.png", "wb") as image_file:  # Save with index
                    image_file.write(image_bytes)
            else:
                print(f"Warning: Prediction at index {i} does not contain 'bytesBase64Encoded' data.")
    else:
        print("Warning: Response does not contain 'predictions'.")

def controlled_editing(prompt, negative_prompt, reference_image_paths, control_type):
    print('controlled_editing is progressing.')
    access_token = get_access_token()
    reference_images = [encode_image(path) for path in reference_image_paths]
    reference_images_obj = [
      {
        'referenceType': 'REFERENCE_TYPE_CONTROL',
        'referenceId': index,
        'referenceImage': {
          'bytesBase64Encoded': img_bytes
        },
        'controlImageConfig': {
          'controlType': control_type,
          'enableControlImageComputation': False
        }
      } for index, img_bytes in enumerate(reference_images)
    ]
    request_data = {
      "instances" : [
          {
              'prompt': f'Generate an image aligning with the scribble map to match the description: {prompt}',
              'referenceImages': reference_images_obj
          }
      ],
      'parameters': {
          'negativePrompt': negative_prompt,
          'seed': 1000,
          'sampleCount': 1,
          'promptLanguage': 'en'
      }
    }
    print_request_data(request_data)
    response = make_prediction_request(ENDPOINT_URI, access_token, request_data)
    images = convert_response_to_image(response)
    return images


def subject_editing(prompt, negative_prompt, subject_image_description, subject_image_paths):
    print('subject_editing is progressing.')
    access_token = get_access_token()
    subject_img_b64 = encode_image(subject_image_paths[0])
    request_data = {
        "instances": [
            {
                "prompt": prompt,
                "referenceImages": [
                    {
                        "referenceType": "REFERENCE_TYPE_SUBJECT",
                        "referenceId": 1,
                        "referenceImage": {"bytesBase64Encoded": subject_img_b64},
                        "subjectImageConfig": {
                            "subjectDescription": subject_image_description,
                            "subjectType": "SUBJECT_TYPE_ANIMAL"
                        }
                    }
                ]
            }
        ],
        "parameters": {
            "negativePrompt": negative_prompt,
            "seed": 1,
            "sampleCount": 2,
            "editConfig": {
                "baseSteps": 75
            },
            "promptLanguage": "en",
            "editMode": "EDIT_MODE_DEFAULT"
        }
    }
    print_request_data(request_data)
    response = make_prediction_request(ENDPOINT_URI, access_token, request_data)
    images = convert_response_to_image(response)
    return images

def raw_editing(prompt, negative_prompt, edit_mode, mask_mode, dilation, subject_image_paths):
    print('raw_editing is progressing.')
    access_token = get_access_token()
    subject_img_b64 = encode_image(subject_image_paths[0])
    parameters = {
            "negativePrompt": negative_prompt,
            "seed": 1,
            "sampleCount": 2,
            "editConfig": {
                "baseSteps": 75
            },
            "promptLanguage": "en",
        }

    if edit_mode != "NONE" :
        parameters["editMode"] = edit_mode

    instance_data = {
        "prompt": prompt,
        "referenceImages": [
            {
                "referenceType": "REFERENCE_TYPE_RAW",
                "referenceId": 1,
                "referenceImage": {"bytesBase64Encoded": subject_img_b64},
            }
        ]
    }

    if edit_mode != "NONE" and mask_mode != "NONE" :
        instance_data["referenceImages"].append(
            {
                'referenceType': 'REFERENCE_TYPE_MASK',
                'referenceId': 2,
                'maskImageConfig': {
                    'maskMode': mask_mode,
                    'dilation': dilation
                }
            }
        )

    request_data = {
        "instances": [
            instance_data
        ],
        "parameters": parameters
    }

    print_request_data(request_data)
    response = make_prediction_request(ENDPOINT_URI, access_token, request_data)
    images = convert_response_to_image(response)
    return images

def style_editing(prompt, negative_prompt, subject_image_paths):
    print('style_editing is progressing.')
    access_token = get_access_token()
    subject_img_b64 = encode_image(subject_image_paths[0])
    request_data = {
        "instances": [
            {
                "prompt": prompt,
                "referenceImages": [
                    {
                        "referenceType": "REFERENCE_TYPE_RAW",
                        "referenceId": 1,
                        "referenceImage": {"bytesBase64Encoded": subject_img_b64},
                    }
                ]
            }
        ],
        "parameters": {
            "negativePrompt": negative_prompt,
            "seed": 1,
            "sampleCount": 2,
            "editConfig": {
                "baseSteps": 25
            },
            "promptLanguage": "en",
        }
    }
    print_request_data(request_data)
    response = make_prediction_request(ENDPOINT_URI, access_token, request_data)
    images = convert_response_to_image(response)
    return images

if __name__ == "__main__":

    # Encode images
    #facemesh_img_b64 = encode_image(FACEMESH_IMG_FILENAME)
    formatted_text_prompt = TEXT_PROMPT_TEMPLATE.replace("[1]", SUBJECT_IMG_DESCRIPTION).replace("[2]", "face mesh image")  # Explicitly using "face mesh image"
    subject_editing(formatted_text_prompt, NEG_TEXT_PROMPT, SUBJECT_IMG_FILENAME)

    images = subject_editing("yellow shirt", "", SUBJECT_IMG_FILENAME)
    for index, image in enumerate(images):
        image.save(f"image_edited_{index}.png")

