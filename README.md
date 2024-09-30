# Imaginify: Image Generation & Captioning

Imaginify is a desktop application that leverages Hugging Face models for image generation from text prompts and image captioning. It uses customtkinter for the user interface and provides a seamless experience for creating and captioning images. Users can generate images using Stable Diffusion and caption images using BLIP.

## Features
- Generate images from text prompts
- Caption images by uploading existing files
- Dark mode support
- Easy-to-use interface

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Shubham-Datta/Imaginify.git
    cd Imaginify
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Add your Hugging Face API token:**
   - Sign up or log in to Hugging Face and get your access token from [here](https://huggingface.co/docs/hub/security-tokens).
   - Open the `authtoken.py` file in the root directory and replace `"YOUR_HUGGINGFACE_USER_ACCESS_TOKEN"` with your actual token:
     ```python
     token = "your_hugging_face_token_here"
     ```

4. **Run the application:**
    ```sh
    python imaginify.py
    ```

## GPU Support

If you have a dedicated GPU and want to leverage it for faster image generation, you need to modify the code in `imaginify.py`:

1. Open `imaginify.py` in your preferred code editor.
2. Locate the line where the Stable Diffusion pipeline is moved to the device:
    ```python
    pipe = pipe.to("cpu")
    ```
3. Replace `"cpu"` with `"cuda"` to use the GPU:
    ```python
    pipe = pipe.to("cuda")
    ```

Your updated code in `imaginify.py` should look like this:
```python
# Initialize the stable diffusion model with Hugging Face token
def load_stable_diffusion_model():    
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token=token)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=token)
    pipe = pipe.to("cuda")  # Change "cpu" to "cuda" for GPU support
    return pipe
```

## Usage

- **Image Generation:**
  1. Enter a text prompt in the provided field.
  2. Click the "Generate" button to create an image.

- **Image Captioning:**
  1. Open an image file using the "Open Image" button.
  2. The generated caption will be displayed below the image.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues and pull requests to improve the application.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the models and API.
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter) for the UI framework.
