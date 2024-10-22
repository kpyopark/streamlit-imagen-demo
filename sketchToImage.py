from dotenv import load_dotenv
import os
import json
import base64
import requests
from vertexai.preview.vision_models import ImageGenerationModel, Image

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
    response = make_prediction_request(ENDPOINT_URI, access_token, request_data)
    images = convert_response_to_image(response)
    return images

def subject_editing(prompt, negative_prompt, subject_image_path):
    access_token = get_access_token()
    subject_img_b64 = encode_image(subject_image_path)
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
                            "subjectDescription": SUBJECT_IMG_DESCRIPTION,
                            "subjectType": "SUBJECT_TYPE_PERSON"
                        }
                    }
                    # ,
                    # {
                    #     "referenceType": "REFERENCE_TYPE_CONTROL",
                    #     "referenceId": 2,
                    #     "referenceImage": {"bytesBase64Encoded": facemesh_img_b64},
                    #     "controlImageConfig": {
                    #         "controlType": "CONTROL_TYPE_FACE_MESH",
                    #         "enableControlImageComputation": True
                    #     }
                    # }
                ]
            }
        ],
        "parameters": {
            "negativePrompt": negative_prompt,
            "seed": 1,
            "sampleCount": 4,
            "promptLanguage": "en",
            "editMode": "EDIT_MODE_DEFAULT"
        }
    }
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

