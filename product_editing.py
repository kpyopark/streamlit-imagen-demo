import streamlit as st
from pathlib import Path
import tempfile
import os
from typing import List
import base64
from PIL import Image
import io

from imagen_editor import (
    ImageInfo, 
    call_gemini_for_editing,
    product_editing,
    save_images
)

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def display_image(image_data: bytes):
    """Display image from bytes data"""
    img = Image.open(io.BytesIO(image_data))
    st.image(img)

def main():
    st.title("Multi-Product Image Background Editor")
    st.write("Upload multiple product images and generate a custom background using Imagen AI")

    # Initialize session state variables
    if 'image_infos' not in st.session_state:
        st.session_state.image_infos = []
    if 'gemini_result' not in st.session_state:
        st.session_state.gemini_result = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Choose product images", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Create columns to display uploaded images
        cols = st.columns(len(uploaded_files))
        image_infos = []

        # Save and display uploaded images
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx]:
                st.image(uploaded_file, caption=f"Product {idx + 1}")
                temp_path = save_uploaded_file(uploaded_file)
                image_infos.append(ImageInfo(path=temp_path))

        # Always show editable fields for each image
        st.subheader("Product Information")
        for idx, _ in enumerate(uploaded_files):
            with st.expander(f"Product {idx + 1} Details", expanded=True):
                if st.session_state.analysis_done:
                    subject_type = st.text_input(
                        f"Subject Type for Product {idx + 1}",
                        value=st.session_state.gemini_result.image_infos[idx].subject_type
                    )
                    subject_desc = st.text_area(
                        f"Description for Product {idx + 1}",
                        value=st.session_state.gemini_result.image_infos[idx].subject_description
                    )
                    # Update the session state with edited values
                    st.session_state.gemini_result.image_infos[idx].subject_type = subject_type
                    st.session_state.gemini_result.image_infos[idx].subject_description = subject_desc
                else:
                    st.info("Click 'Analyze Images & Generate Background' to get AI analysis")

        # User prompt input
        user_prompt = st.text_area(
            "Describe how you want the background to look:",
            "Create a modern minimalist background for these products"
        )

        # Analysis button
        if st.button("Analyze Images & Generate Background"):
            with st.spinner("Analyzing images with Gemini..."):
                try:
                    # Get analysis and prompts from Gemini
                    st.session_state.gemini_result = call_gemini_for_editing(image_infos, user_prompt)
                    st.session_state.analysis_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)

        # Always show prompt editing section and generate button if analysis is done
        if st.session_state.analysis_done:
            st.subheader("Generated Prompts")
            edited_positive = st.text_area(
                "Edit positive prompt if needed:", 
                st.session_state.gemini_result.positive_prompt
            )
            edited_negative = st.text_area(
                "Edit negative prompt if needed:", 
                st.session_state.gemini_result.negative_prompt
            )

            # Update prompts in session state
            st.session_state.gemini_result.positive_prompt = edited_positive
            st.session_state.gemini_result.negative_prompt = edited_negative

        # Always show generate button if analysis is done
        if st.session_state.analysis_done:
            if st.button("Generate Final Image", key="generate_final"):
                with st.spinner("Generating background with Imagen..."):
                    try:
                        # Generate the background image
                        generated_images = product_editing(st.session_state.gemini_result)

                        # Display results
                        st.subheader("Generated Results")
                        for idx, img_data in enumerate(generated_images):
                            st.write(f"Result {idx + 1}")
                            display_image(img_data)

                            # Add download button for each image
                            btn = st.download_button(
                                label=f"Download Result {idx + 1}",
                                data=img_data,
                                file_name=f"generated_result_{idx}.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.exception(e)

        # Cleanup temporary files
        st.sidebar.write("Note: Temporary files will be cleaned up when you close the app.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Product Image Background Editor",
        page_icon="🎨",
        layout="wide"
    )
    main()