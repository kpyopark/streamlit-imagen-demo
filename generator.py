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
    prompt_template = f"""You are an advertising specialist using Imagen to create advertising images. 
Analyze the given user prompt and transform it into a well-formed prompt suitable for Imagen. 
The output should be in JSON format and must contain two keys: "positive" and "negative". All results must be generated in English.

Based on the detailed instructions, generate the output.

<User Prompt>
{prompt}
</User Prompt>

<Detailed Instructions>
{instruction}
</Detailed Instructions>
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
    # st.set_page_config(layout="wide")

    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.title("Prompt Analysis")

        user_prompt = st.text_input("Enter the prompt for generating the image:")

        display_options = [
            "Keep Original",
            "Descriptive Narration",
            "Keyword Focus",
            "Detailed Specifications",
            "Copyright Considerations",
            "User Input"
        ]

        # Detailed content to be used in the actual prompt
        prompt_templates = {
            "Keep Original": """
        When generating the image, please adhere to the following guidelines:
        1. Maintain the original sentence structure and expression as much as possible.
        2. Minimize unnecessary additional explanations or modifiers.
        3. Accurately reflect the original context and intent.
        4. Add only essential details for image generation.

        Original Text: {text}
        """,
            
            "Descriptive Narration": """
        Please reconstruct the original text in detail, following these guidelines:
        1. Specifically describe the overall atmosphere and emotion of the scene.
        2. Describe the appearance, texture, and condition of the elements in detail.
        3. Add descriptions that express a sense of space and perspective.
        4. Include environmental elements such as lighting, shadows, and time of day.
        5. Appropriately utilize sensory and metaphorical expressions.

        Original Text: {text}
        """,
            
            "Keyword Focus": """
        Please reconstruct the prompt focusing on the following elements from the original text:
        1. Extract the core subjects/objects.
        2. Identify key actions and states.
        3. Identify important background elements.
        4. Select key modifiers that determine the atmosphere.
        5. Clearly state the relationships between each keyword.

        Connect the extracted keywords naturally to construct the prompt.

        Original Text: {text}
        """,
            
             "Detailed Specifications": """
        Please construct the prompt including the following technical specifications:

        [Style Specifications]
        - Image Style: (Photo/Illustration/3D, etc.)
        - Art Style: (Realistic/Cartoonish/Surreal, etc.)
        - Rendering Style: If it's too broad like "oil painting," please specify a particular artist/artwork style.

        [Camera/Composition Specifications]
        - Shooting Angle: (Front/Side/Overhead/Low Angle)
        - Focal Length: (Wide-angle/Standard/Telephoto)
        - Distance: (Close-up/Medium/Long Shot)

        [Image Quality Specifications]
        - Resolution: (8K/4K/FHD)
        - Level of Detail: (Ultra-High Resolution/Normal/Rough)
        - Noise/Grain: (None/Natural/Stylized)

        [Color/Lighting Specifications]
        - Dominant Color: (Warm/Cool/Monotone)
        - Lighting Style: (Natural/Artificial/Dramatic)
        - Contrast: (Strong/Soft/Flat)

        Original Text: {text}
        """,
            
            "Copyright Considerations": """
        Please generate a copyright-aware prompt according to the following guidelines:

        [Copyright Element Handling Guidelines]
        1. Brands/Logos
        - Specific names → general form descriptions
        - Example: "Coca-Cola logo" → "red cursive logo"

        2. Characters
        - Proper names → general characteristic descriptions
        - Example: "Pikachu" → "yellow monster character", "Star Wars" → "space-themed future battle"

        3. Trademarks/Designs
        - Specific product names → product type and features
        - Example: "iPhone" → "a smartphone with a modern design"

        4. Artworks
        - Specific artwork names → style and theme descriptions
        - Example: "Mona Lisa" → "a portrait of a woman in the Renaissance style"

        5. PG19 Content
        - If the content is violent or gore-like, extract only the feeling of the image and remove the specific event descriptions.
        - Example: "A photograph of a bloody scene in the rain" → "a photograph of a dark alley in the rain"

        6. Other
        - If the drawing style is too broad (e.g., oil painting), specify the particular artist or artwork style clearly.

        Original Text: {text}
        """,
            
            "User Input": "{text}"  # Use user-defined prompt as is
        }

        # Streamlit UI
        selected_display_option = st.selectbox("Prompt Reinterpretation Option", display_options)

        # If user input option is selected
        if selected_display_option == "User Input":
            user_prompt = st.text_area("Enter your desired prompt format:")
            final_prompt_template = user_prompt
        else:
            final_prompt_template = prompt_templates[selected_display_option]

        aspect_ratio_options = [
            "16:9",
            "4:3",
            "1:1"
        ]
        selected_aspect_ratio = st.selectbox("Select Aspect Ratio", aspect_ratio_options)

        if final_prompt_template == "User Input":
             user_input = st.text_area("Enter the prompt for temporary reinterpretation")
        else:
            user_input = None

        if st.button("Analyze"):
            if user_prompt and final_prompt_template:
                if final_prompt_template == "User Input":
                    result = call_gemini(user_prompt, user_input)
                else:
                    result = call_gemini(user_prompt, final_prompt_template)

                st.subheader("Analysis Result")
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
        
        st.subheader("Upscale Settings")
        upscale_type = st.selectbox("Upscale Method", ["new_size", "upscale_factor"])
        
        if upscale_type == "new_size":
            new_size = st.text_input("Enter new size (e.g., 1024x1024)")
        else:
             upscale_factor = st.selectbox("Upscale Factor", ["x2", "x4"])

        mime_type = st.selectbox("Mime Type", ["image/png", "image/jpeg"])
        upscale_model = st.selectbox("Upscale Model", ["imagen2", "imagen3"])


    with right_column:
        st.title("Generated Images")
        if st.session_state.generated_images:
            for i, (model_name, img_path, resolution) in enumerate(st.session_state.generated_images):
                st.subheader(f"{model_name} Result {i%2 + 1}")
                
                # Apply resize_and_clip_image function
                resized_img_path = f'resized_{img_path}'
                resized_img_path = resize_and_clip_image(img_path, resized_img_path)

                st.image(resized_img_path, use_column_width=True)
                new_resolution = get_image_resolution(resized_img_path)
                st.caption(f"Original Resolution: {resolution[0]}x{resolution[1]}")
                st.caption(f"Adjusted Resolution: {new_resolution[0]}x{new_resolution[1]}")

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
                    st.caption(f"Upscaled Resolution: {upscaled_resolution[0]}x{upscaled_resolution[1]}")

        else:
            st.text("Press the 'Analyze' button to generate images.")

if __name__ == "__main__":
    main()