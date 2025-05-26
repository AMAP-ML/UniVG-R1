import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import math
import requests
import copy
import torch
import sys
import warnings
import torch
import json
import re
import os

def extract_bbox(text):
    # r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]' for [[28, 586, 362, 793]]
    # r'<box>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</box>' for <box>826 704 993 984</box>
    # r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]' for [28, 586, 362, 793]
    # r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]' for [0.3, 0.2, 0.4, 0.28]
    matches = re.findall(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', text)
    if matches == []:
        matches = re.findall(r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]', text)
    if matches == []:
        matches = re.findall(r'\((\d+\.\d+),(\d+\.\d+)\),\((\d+\.\d+),(\d+\.\d+)\)', text)
    if matches == []:
        matches = re.findall(r'\((\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\)', text)
    if matches == []:
        matches = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
    if matches == []:
        matches = re.findall(r'<box>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</box>', text)
    try:
        if len(matches) == 1:
            boxes = [list(map(int, match)) for match in matches][0]
        else:
            matches = matches[:-1]
            boxes = [list(map(int, match)) for match in matches][0]
        return boxes
    except:
        return [0.0, 0.0, 0.0, 0.0]

def calculate(answer, result, total_count, obj):
    truth = obj['output'].strip().strip('.')
    label = truth in answer
    
    if label:
        result[obj['task']] = result.get(obj['task'], 0) + 1
    if obj['task'] not in total_count:
        total_count[obj['task']]=0
    total_count[obj['task']] += 1
    
    return result, total_count

def convert(text):
    return [obj/1000 for obj in text]

def calculate_iou(box1, box2):
    if box2 == []:
        return 0.0
    if isinstance(box2[0], list):
        box2 = box2[0]
    try:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
    except:
        print(f'[Notice]Error when calculating IOU:{box1},{box2}')
        return 0.0
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def resize(image_path, max_size=800):
    img = Image.open(image_path)
    width, height = img.size
    scaling_factor = min(max_size / width, max_size / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_img = img.resize((new_width, new_height))
    
    return resized_img

def normalize_bbox(image, info):
    # 获取图片的宽度和高度
    width, height = Image.open(image).size
    normalized_bboxes = []
    input_text = ''

    # 遍历 bbox 并进行归一化
    for bbox in info['bbox']:  # bbox 是 (x, y, w, h)
        x, y, w, h = bbox
        x1 = int((x / width) * 1000)
        y1 = int((y / height) * 1000)
        x2 = int(((x + w) / width) * 1000)
        y2 = int(((y + h) / height) * 1000)
        input_text += f'<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|> '

    return input_text

def split_image(image_path):
    """
    Split an image into four equal parts (quadrants) based on width and height.
    Returns a list of PIL Image objects.
    """
    img = Image.open(image_path)
    width, height = img.size
    half_width, half_height = width // 2, height // 2

    # Define the bounding boxes for the quadrants
    quadrants = [
        (0, 0, half_width, half_height),  # Top-left
        (half_width, 0, width, half_height),  # Top-right
        (0, half_height, half_width, height),  # Bottom-left
        (half_width, half_height, width, height)  # Bottom-right
    ]

    return [img.crop(box) for box in quadrants]

def get_image_index(text):
    patterns = {
        0: ['first', 'Image-1', 'Image1', 'Image 1', '1th'],
        1: ['second', 'Image-2', 'Image2', 'Image 2', '2th'],
        2: ['third', 'Image-3', 'Image3', 'Image 3', '3th'],
        3: ['fourth', 'Image-4', 'Image4', 'Image 4', '4th'],
        4: ['fifth', 'Image-5', 'Image5', 'Image 5', '5th'],
        5: ['sixth', 'Image-6', 'Image6', 'Image 6', '6th']
    }

    for index, keywords in patterns.items():
        if any(keyword in text for keyword in keywords):
            return index
    return -1


###########################################################################################################
############################################# InternVL2 series ############################################
###########################################################################################################
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def resize_image(image, target_size=500):
    width, height = image.size
    
    if max(width, height) > target_size:
        if width > height:
            new_width = target_size
            new_height = int(target_size * height / width)
        else:
            new_height = target_size
            new_width = int(target_size * width / height)
        
        resized_image = image.resize((new_width, new_height))
        return resized_image
    else:
        return image

def load_image_intenvl2(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    # image = resize_image(image)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    device_map['language_model.model.rotary_emb'] = 0

    return device_map
###########################################################################################################
############################################# InternVL2 series ############################################
###########################################################################################################