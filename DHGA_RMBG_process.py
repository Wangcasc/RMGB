# @FileName :DHGA_RMBG_process.py
# @Time :2025/4/3 下午4:44
# @Author : SCUT Wang


import os
import gradio as gr
from gradio_imageslider import ImageSlider # Gradio组件模块，用于显示图片滑动效果
from loadimg import load_img # 用于加载图片的模块
from transformers import AutoModelForImageSegmentation # 用于加载预训练模型 https://www.jianshu.com/p/0b41e7d2de62
import torch
from torchvision import transforms
from datetime import datetime
import devicetorch

# Get the appropriate device
device = devicetorch.get(torch)
print(f"Using device: {device}")


torch.set_float32_matmul_precision(["high", "highest"][0])  # 设置矩阵乘法的精度

current_dir = os.path.abspath(os.path.dirname(__file__))  # 获取当前脚本的绝对路径
model_dir = os.path.join(current_dir, "Models/RMBG-2.0")  # 模型文件夹路径
birefnet = AutoModelForImageSegmentation.from_pretrained(  # 加载预训练的背景移除模型
    model_dir,
    trust_remote_code=True
)




