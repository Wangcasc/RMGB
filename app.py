# 华南理工大学
# 王熙来
# 开发时间：2025/3/27 16:52

import os
import gradio as gr
from gradio_imageslider import ImageSlider # Gradio组件模块，用于显示图片滑动效果
from loadimg import load_img # 用于加载图片的模块
from transformers import AutoModelForImageSegmentation # 用于加载预训练模型 https://www.jianshu.com/p/0b41e7d2de62
import torch
from torchvision import transforms
from datetime import datetime
import devicetorch

import warnings
# Suppress specific timm deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

# Get the appropriate device
device = devicetorch.get(torch)
print(f"Using device: {device}")

torch.set_float32_matmul_precision(["high", "highest"][0]) # 设置矩阵乘法的精度

current_dir = os.path.abspath(os.path.dirname(__file__)) # 获取当前脚本的绝对路径
model_dir = os.path.join(current_dir, "Models/RMBG-2.0") # 模型文件夹路径

# 加载预训练的背景移除模型
birefnet = AutoModelForImageSegmentation.from_pretrained(
    model_dir,
    trust_remote_code=True
)

birefnet = devicetorch.to(torch, birefnet)  # 将模型移动到合适的设备

# 定义图片预处理流程
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


output_folder = 'output_images'  # 定义输出文件夹
if not os.path.exists(output_folder):  # 如果文件夹不存在，则创建
    os.makedirs(output_folder)

def generate_filename():  # 生成唯一的文件名，用于保存处理后的图片
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")  # 获取当前时间戳
    return f"no_bg_{timestamp}.png"

def fn(image): # 主处理函数，用于处理图片并保存结果
    im = load_img(image, output_type="pil")  # 加载图片 格式为PIL 通道为RGB
    im = im.convert("RGB")  # 确保图片是RGB格式
    origin = im.copy()  # 复制原始图片
    image,mask = process(im)  # 处理图片
    unique_filename = generate_filename()  # 生成唯一的文件名
    image_path = os.path.join(output_folder, unique_filename) # 定义输出路径
    image_path_mask = os.path.join(output_folder, unique_filename.replace('.png', '_mask.png')) # 定义掩码输出路径
    image.save(image_path) # 保存处理后的图片
    mask.save(image_path_mask) # 保存掩码图片
    return (image, origin), image_path # 返回处理后的图片和原始图片的路径

def process(image): # 处理图片的函数
    image_size = image.size # 获取图片的大小
    input_images = transform_image(image).unsqueeze(0) # 对图片进行预处理
    input_images = devicetorch.to(torch, input_images)  # 将图片移动到合适的设备
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu() # 进行预测
    pred = preds[0].squeeze() # 获取预测结果
    pred_pil = transforms.ToPILImage()(pred) # 将预测结果转换为PIL格式
    mask = pred_pil.resize(image_size) # 将预测结果调整为原始图片的大小
    image.putalpha(mask) # 将预测结果作为透明度通道添加到原始图片中
    # Clean up GPU/MPS memory
    devicetorch.empty_cache(torch)
    return image,mask # 返回处理后的图片和掩码
  
def process_file(f):
    unique_filename = generate_filename()
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    output_path = os.path.join(output_folder, unique_filename)
    transparent.save(output_path)
    return output_path

# 创建Gradio的ImageSlider组件，用于显示处理前后的图片对比
slider1 = ImageSlider(label="RMBG-2.0", type="pil")
slider2 = ImageSlider(label="RMBG-2.0", type="pil")
# 创建Gradio的图片上传组件
image = gr.Image(label="上传图片")
# 创建Gradio的图片URL输入组件
image2 = gr.Image(label="上传图片",type="filepath")
# 创建Gradio的文本输入组件，用于输入图片的URL
text = gr.Textbox(label="粘贴图像的URL")
# 创建Gradio的文件输出组件，用于保存处理后的图片
png_file = gr.File(label="输出 PNG 文件")

# 定义第一个Tab（通过上传图片slider1进行处理） 绑定处理函数fn 以及输入输出组件
tab1 = gr.Interface(
    fn, inputs=image, outputs=[slider1, gr.File(label="输出 PNG 文件")],
    allow_flagging="never"
)

# 定义第二个Tab（通过输入URL进行处理） 绑定处理函数fn 以及输入输出组件
tab2 = gr.Interface(
    fn, inputs=text, outputs=[slider2, gr.File(label="输出 PNG 文件")],
    allow_flagging="never"
)

# 定义页面头部的HTML内容
header = """
<h1 style="font-size:24px;">RMBG-2.0 用于去除背景</h1>
<p style="font-size:16px;"><a href="https://github.com/Wangcasc" target="_blank">github@WXL.Wangcasc</a></p>
"""
with gr.Blocks() as demo: # 使用Gradio的Blocks布局创建应用
    gr.Markdown(header) # 添加页面头部HTML内容 markdown格式
    gr.TabbedInterface( [tab1, tab2], ["输入图片", "输入URL"] )  # 创建带标签页的界面 # 包含两个Tab # 标签页的标题

# 启动Gradio应用
demo.launch(share=False, inbrowser=True)
