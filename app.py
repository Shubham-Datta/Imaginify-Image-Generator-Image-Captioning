import customtkinter as ctk
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from tkinter import filedialog, messagebox
from transformers import BlipProcessor, BlipForConditionalGeneration
from authtoken import token

# Initialize the stable diffusion model with Hugging Face token
def load_stable_diffusion_model():    
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token=token)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=token)
    pipe = pipe.to("cpu")
    return pipe

# Initialize the image captioning model
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Generate an image from text prompt
def generate_image(prompt, pipe):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image

# Generate caption for an image
def generate_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Save the image
def save_image():
    if generated_image is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            generated_image.save(save_path)
            messagebox.showinfo("Success", "Image saved successfully")
    else:
        messagebox.showwarning("Warning", "No image to save")

# Load models
pipe = load_stable_diffusion_model()
processor, model = load_captioning_model()

# Initialize the UI
app = ctk.CTk()
app.title("Imaginify: Image Generator & Image Captioning")
app.geometry("1000x700")

generated_image = None  # To hold the PIL Image object

# Function to update the generated image
def update_image():
    global generated_image
    prompt = prompt_entry.get()
    generated_image = generate_image(prompt, pipe)
    img = ImageTk.PhotoImage(generated_image)
    image_label.configure(image=img)
    image_label.image = img

# Function to switch theme
def switch_theme():
    if theme_switch.get() == 1:
        ctk.set_appearance_mode("dark")
    else:
        ctk.set_appearance_mode("light")

# Function to open an image and generate a caption
def open_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
    if image_path:
        caption = generate_caption(image_path, processor, model)
        caption_label.configure(text=caption)
        img = Image.open(image_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        caption_image_label.configure(image=img)
        caption_image_label.image = img

# Create frames
main_frame = ctk.CTkFrame(app)
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

left_frame = ctk.CTkFrame(main_frame, width=200)
left_frame.pack(side="left", fill="y")

tab_view = ctk.CTkTabview(main_frame, width=800)
tab_view.pack(side="right", fill="both", expand=True)

tab1 = tab_view.add("Image Generation")
tab2 = tab_view.add("Image Captioning")

# Left sidebar
save_button = ctk.CTkButton(left_frame, text="Save", command=save_image)
theme_switch = ctk.CTkSwitch(left_frame, text="Dark Mode", command=switch_theme)
open_button = ctk.CTkButton(left_frame, text="Open Image", command=open_image)

# UI Elements for Image Generation in tab1
translucent_box_gen = ctk.CTkFrame(tab1, corner_radius=15)
translucent_box_gen.pack(pady=20, padx=20, fill="both", expand=True)
translucent_box_gen.configure(fg_color=("gray75", "gray30"))

prompt_entry = ctk.CTkEntry(translucent_box_gen, width=400, placeholder_text="Enter your prompt here")
prompt_entry.pack(pady=20)

generate_button = ctk.CTkButton(translucent_box_gen, text="Generate", command=update_image)
generate_button.pack(pady=20)

image_label = ctk.CTkLabel(translucent_box_gen, text="")
image_label.pack(pady=20)

# UI Elements for Image Captioning in tab2
translucent_box_cap = ctk.CTkFrame(tab2, corner_radius=15)
translucent_box_cap.pack(pady=20, padx=20, fill="both", expand=True)
translucent_box_cap.configure(fg_color=("gray75", "gray30"))

caption_image_label = ctk.CTkLabel(translucent_box_cap, text="")
caption_image_label.pack(pady=20)

caption_label = ctk.CTkLabel(translucent_box_cap, text="Caption will appear here")
caption_label.pack(pady=20)

def update_sidebar():
    if tab_view.get() == "Image Generation":
        save_button.pack(pady=20)
        open_button.pack_forget()
    else:
        save_button.pack_forget()
        open_button.pack(pady=40)

tab_view.configure(command=update_sidebar)

# Initially update the sidebar based on the default tab
update_sidebar()

# Add theme switch at the bottom of the sidebar for both tabs
theme_switch.pack(side="bottom", pady=20)

# Run the application
app.mainloop()
