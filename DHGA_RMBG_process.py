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
from tqdm import tqdm
from PIL import Image

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

birefnet = devicetorch.to(torch, birefnet)  # 将模型移动到合适的设备

# 定义图片预处理流程
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def fn(image): # 主处理函数，用于处理图片并保存结果
    im = load_img(image, output_type="pil")  # 加载图片 格式为PIL 通道为RGB
    im = im.convert("RGB")  # 确保图片是RGB格式
    origin = im.copy()  # 复制原始图片
    image,mask = process(im)  # 处理图片

    return image, mask


def process(image):  # 处理图片的函数
    image_size = image.size  # 获取图片的大小
    input_images = transform_image(image).unsqueeze(0)  # 对图片进行预处理
    input_images = devicetorch.to(torch, input_images)  # 将图片移动到合适的设备
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()  # 进行预测
    pred = preds[0].squeeze() # 获取预测结果
    pred_pil = transforms.ToPILImage()(pred)  # 将预测结果转换为PIL格式
    mask = pred_pil.resize(image_size)  # 将预测结果调整为原始图片的大小

    # 将mask转换为灰度图像
    mask = mask.convert("L")
    # 使用PIL.Image.composite合成图片
    masked_image = Image.composite(image, Image.new("RGB", image_size, (0, 0, 0)), mask)

    # Clean up GPU/MPS memory
    devicetorch.empty_cache(torch)
    return masked_image,mask  # 返回处理后的图片和掩码


# 对color_hand中的所有子文件夹中的所有图片进行处理
dataset_root="./color_hand" # 数据集根目录
output_folder = './color_hand_masked_RMBG'  # 定义MASK后输出文件夹
output_folder_masked = './color_hand_mask_RMBG'  # 定义掩码输出文件夹
if not os.path.exists(output_folder):  # 如果文件夹不存在，则创建
    os.makedirs(output_folder)
if not os.path.exists(output_folder_masked):  # 如果文件夹不存在，则创建
    os.makedirs(output_folder_masked)

samples=os.listdir(dataset_root) # 获取文件夹中的所有子文件夹
for sample in tqdm(samples): # 遍历所有子文件夹
    sample_path=os.path.join(dataset_root,sample) # 获取子文件夹路径
    img_masked_output_path=os.path.join(output_folder,sample) # 获取输出路径
    mask_output_path=os.path.join(output_folder_masked,sample) # 获取掩码输出路径
    if not os.path.exists(img_masked_output_path):  # 如果文件夹不存在，则创建
        os.makedirs(img_masked_output_path)
    if not os.path.exists(mask_output_path):  # 如果文件夹不存在，则创建
        os.makedirs(mask_output_path)
    ###############处理图片################
    if os.path.isdir(sample_path):  # 如果是文件夹
        images=os.listdir(sample_path) # 获取子文件夹中的所有图片
        for image in images:
            image_path=os.path.join(sample_path,image) # 获取图片路径
            if os.path.isfile(image_path):
                img_masked, mask = fn(image_path) # 处理图片
                # 转为RGB格式
                img_masked = img_masked.convert("RGB")
                mask = mask.convert("L")
                img_masked.save(os.path.join(img_masked_output_path,image))
                mask.save(os.path.join(mask_output_path,image))
    else:
        print(f"{sample_path} is not a directory")
        continue







