import os
import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from datetime import datetime
import devicetorch

import warnings
# Suppress specific timm deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

# Get the appropriate device
device = devicetorch.get(torch)

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0", trust_remote_code=True
)
birefnet = devicetorch.to(torch, birefnet)  # Move model to appropriate device

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def generate_filename():
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return f"no_bg_{timestamp}.png"

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    image = process(im)    
    unique_filename = generate_filename()
    image_path = os.path.join(output_folder, unique_filename)
    image.save(image_path)
    return (image, origin), image_path

def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0)
    input_images = devicetorch.to(torch, input_images) 
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    # Clean up GPU/MPS memory
    devicetorch.empty_cache(torch)
    return image
  
def process_file(f):
    unique_filename = generate_filename()
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    output_path = os.path.join(output_folder, unique_filename)
    transparent.save(output_path)
    return output_path

slider1 = ImageSlider(label="RMBG-2.0", type="pil")
slider2 = ImageSlider(label="RMBG-2.0", type="pil")
image = gr.Image(label="上传图片")
image2 = gr.Image(label="上传图片",type="filepath")
text = gr.Textbox(label="粘贴图像的URL")
png_file = gr.File(label="输出 PNG 文件")

tab1 = gr.Interface(
    fn, inputs=image, outputs=[slider1, gr.File(label="输出 PNG 文件")],
    allow_flagging="never"
)

tab2 = gr.Interface(
    fn, inputs=text, outputs=[slider2, gr.File(label="输出 PNG 文件")],
    allow_flagging="never"
)


header = """
<h1 style="font-size:24px;">RMBG-2.0 用于去除背景</h1>
<p style="font-size:16px;"><a href="https://space.bilibili.com/554863038" target="_blank">整合包@B站passion@AIDOBE</a></p>
"""
with gr.Blocks() as demo: 
    gr.Markdown(header) 
    gr.TabbedInterface( [tab1, tab2], ["输入图片", "输入URL"] )

demo.launch(share=False, inbrowser=True)