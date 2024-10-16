import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import typing

def edit_image_app():
    st.title("Image Editing App")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Input")
        
        # Image upload
        edit_uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if edit_uploaded_file is not None:
            edit_image = Image(edit_uploaded_file.getvalue())
            st.image(edit_uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Text input for image path
        edit_image_path = st.text_input("Or enter the image file path")

        # Edit options
        edit_mode = st.selectbox(
            "Edit Mode",
            ["inpainting-insert", "inpainting-remove", "outpainting", "product-image"]
        )

        # Other parameters
        edit_prompt = st.text_input("Prompt")
        edit_negative_prompt = st.text_input("Negative Prompt")
        edit_number_of_images = st.number_input("Number of Images", min_value=1, max_value=8, value=1)
        edit_guidance_scale = st.slider("Guidance Scale", min_value=0.0, max_value=20.0, value=7.0)
        
        edit_mask_mode = st.selectbox("Mask Mode", [None, "background", "foreground", "semantic"])
        edit_segmentation_classes = st.text_input("Segmentation Classes (comma-separated)")
        edit_mask_dilation = st.slider("Mask Dilation", min_value=0.0, max_value=1.0, value=0.0)
        
        edit_product_position = st.selectbox("Product Position", [None, "fixed", "reposition"])
        edit_output_mime_type = st.selectbox("Output MIME Type", ["image/png", "image/jpeg"], index=0)
        
        # Conditionally show compression_quality based on output_mime_type
        edit_compression_quality = None
        if edit_output_mime_type == "image/jpeg":
            edit_compression_quality = st.slider("Compression Quality", min_value=0.0, max_value=1.0, value=0.8)
        
        edit_language = st.text_input("Language")
        edit_seed = st.number_input("Seed", value=None)
        edit_output_gcs_uri = st.text_input("Output GCS URI")
        
        edit_safety_filter_level = st.selectbox(
            "Safety Filter Level",
            ["block_most", "block_some", "block_few", "block_fewest"],
            index=1
        )
        
        edit_person_generation = st.selectbox(
            "Person Generation",
            ["dont_allow", "allow_adult", "allow_all"],
            index=1
        )

        if st.button("Edit Image"):
            if edit_uploaded_file is not None or edit_image_path:
                # Initialize the model
                model = ImageGenerationModel.from_pretrained("imagegeneration@006")
                
                params = {
                    "prompt": edit_prompt,
                    "base_image": edit_image if edit_uploaded_file else Image.load_from_file(edit_image_path),
                    "negative_prompt": edit_negative_prompt,
                    "number_of_images": edit_number_of_images,
                    "guidance_scale": edit_guidance_scale,
                    "edit_mode": edit_mode,
                    "segmentation_classes": edit_segmentation_classes.split(',') if edit_segmentation_classes else None,
                    "mask_dilation": edit_mask_dilation,
                    "product_position": edit_product_position,
                    "output_mime_type": edit_output_mime_type,
                    "compression_quality": edit_compression_quality,
                    "language": edit_language,
                    "seed": edit_seed,
                    "output_gcs_uri": edit_output_gcs_uri,
                    "safety_filter_level": edit_safety_filter_level,
                    "person_generation": edit_person_generation
                }
                
                # Add mask_mode only if it's not None
                if edit_mask_mode is not None:
                    params["mask_mode"] = edit_mask_mode
                
                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}
                
                # Generate the image
                edit_result = model.edit_image(**params)
                
                # Display the result
                with col2:
                    st.header("Output")
                    for i, edited_image in enumerate(edit_result.images):
                        img_path = f'edited_image{i+1}.png'
                        edited_image.save(img_path)
                        st.image(img_path, caption=f"Edited Image {i+1}", use_column_width=True)
            else:
                st.error("Please upload an image or provide an image path")

if __name__ == "__main__":
    edit_image_app()