# Image Editing and Generation Tool

This repository contains a Streamlit application for various image editing and generation tasks using Google's Imagen models.

## Overview

The application is divided into four main tabs:

1.  **Image Generation:** This tab allows you to generate new images based on text prompts, with options for analyzing prompts using Gemini AI.
2.  **Product Editing:** This tab enables you to upload multiple product images and create a custom background for them using Imagen.
3.  **Mask Editing:** This tab provides various editing options, like inpainting, outpainting, and raw editing for an image.
4. **Sketch to Image:** This tab provides various image editing options, like controlled editing, subject editing, raw editing, and style editing.

## Features

### Image Generation
*   **Prompt Analysis:** Uses Gemini AI to analyze user-provided prompts and suggest optimized prompts for image generation.
*   **Image Generation:** Generates images using Imagen models based on the analyzed prompts.
*   **Image Upscaling:** Allows users to upscale generated images, controlling the upscale factor and resolution.

### Product Editing
*   **Multi-Product Image Upload:** Upload multiple product images for simultaneous background editing.
*   **Gemini Analysis:** Uses Gemini AI to analyze product images and user-provided background requirements.
*   **Custom Background Generation:** Generates custom backgrounds using Imagen AI based on the analysis.
*   **Downloadable Results:** Download the edited images for final use.

### Mask Editing
*   **Edit Model Selection:** Selection the Imagen model.
*   **Various Edit Modes:** Offers various edit modes, including inpainting, outpainting, and product-image editing.
*   **Customized Editing:** Provides text prompts, negative prompts, and mask configurations for personalized image editing.
*   **Output Options:** Adjust output image types and quality, including JPEG compression.

### Sketch to Image
*   **Various Editing Types:** Supports controlled, subject, raw, and style editing.
*   **Automated Parameter Extraction:** Automatically extracts parameters like prompt, negative prompt, and edit type.
*   **Fine-Grained Controls:** Offers various fine-grained controls for each editing type, such as edit mode, mask mode, and dilation.
*   **Image Upload and Preview:** Upload and preview your original image for editing.

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   pip
*   Google Cloud Platform (GCP) project with Vertex AI API enabled
*   gcloud CLI installed and authenticated

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/kpyopark/streamlit-imagen-demo.git
    cd streamlit-imagen-demo
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file in the root of the repository and add the following variables:
    ```
    PROJECT_ID=<YOUR_GCP_PROJECT_ID>
    LOCATION=<YOUR_GCP_REGION>
    OUTPUT_URI=<YOUR_GCS_BUCKET_URI>
    ```
    Replace `<YOUR_GCP_PROJECT_ID>`, `<YOUR_GCP_REGION>`, and `<YOUR_GCS_BUCKET_URI>` with your actual GCP project ID, region, and GCS bucket URI.

### Running the Application

1. Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```
2. Access the application through the provided URL in your browser.

## Usage

1.  Navigate to the desired tab using the tabs at the top of the application.
2.  Follow the instructions on each tab to upload, input text, and generate images.

## File Structure

*   `main.py`: Main entry point for the Streamlit application.
*   `generator.py`: Implements the image generation tab.
*   `edit.py`: Implements the mask editing tab.
*   `controlled_editing.py`: Implements the sketch to image tab.
*   `imagen_editor.py`: Includes functions for communicating with the Imagen API.
*   `sketchToImage.py`: Provides helper functions for gemini calls and sketch editing.
*   `product_editing.py`: Implements the product image editing tab.
*   `requirements.txt`: Lists all required Python packages.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, feel free to contact [toheavener01@gmail.com]