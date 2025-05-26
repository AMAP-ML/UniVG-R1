import torch
import warnings
import torch
import json
import datetime
import random
import copy
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from PIL import Image
import string
from decord import VideoReader, cpu 
import warnings
import os
from utils import *
### required by Migician ###
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, LlamaTokenizer

from transformers.image_utils import load_image

### required by LLaVA-OneVision ###
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
except:
    print('[WARNING] LLaVA hasn\'t been installed')

from modelscope import AutoTokenizer, AutoConfig, AutoModel

warnings.filterwarnings("ignore")

################# This implementation supports the evaluation of a series of mainstream multi-image models and diverse tasks #################
################# Note: Due to the conflicts of required transformers versions by different MLLMs, you may need to set the   #################
################# correct version for your target.                                                                           #################

PROMPT_TEMPLATE = {
    'format_univg': ' First output the thinking process in <think> </think> tags and then output the bounding box in <answer> </answer> tags.',
    'format_qwen': 'Format:<|box_start|>(x1,y1),(x2,y2)<|box_end|>. Don\'t generate addtional words.', # For more efficient evaluation
    'format_internvl2': 'Format:<box>[[x1,y1,x2,y2]]</box>. Don\'t generate addtional words.',
    'format_minicpm': 'Format:<box>x1 y1 x2 y2</box>. Don\'t generate addtional words.',
    'format_mantis': ' Format:<box>[x1, y1, x2, y2]</box>. Don\'t generate addtional words.',
    'format_cogvlm': ' Coordinates format:[[x0,y0,x1,y1]].'
}

def compute_iou(ground_truth, prediction, acc, task):
    iou = calculate_iou(ground_truth, prediction)
    if not acc.get(task):
        acc[task]={'IOU@0.7':0,'IOU@0.5':0,'IOU@0.3':0,'AVE_IOU':0,'TOTAL':0}
    if iou >= 0.7: acc[task]['IOU@0.7'] += 1
    if iou >= 0.5: acc[task]['IOU@0.5'] += 1
    if iou >= 0.3: acc[task]['IOU@0.3'] += 1
    acc[task]['AVE_IOU'] += iou
    acc[task]['TOTAL'] += 1
    return iou, acc

def replace_image(match):
    replace_image.counter += 1
    return f"Image-{replace_image.counter}"

def post_processing(acc, output_path, output):
    for task, obj in acc.items():
        print(f"✅ Results for [{task}]:")
        print(obj)
        for item, value in obj.items():
            if item != 'TOTAL':
                print(f"|——> {item}:{100*value/acc[task]['TOTAL']}%  ✨")
    output.append(acc)
    with open(output_path,'w') as file2:
        json.dump(output, file2, indent=4, ensure_ascii=False)

################# Input process functions for specific MLLMs #################
def qwen2_process(obj):
    messages = [{"role": "user","content": []}]
    for path in obj['images']:
        messages[0]["content"].append({"type": "image","image": path})
    
    obj['question'] += PROMPT_TEMPLATE['format_qwen']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def univgr1_process(obj):
    messages = [{"role": "user","content": []}]
    for path in obj['images']:
        messages[0]["content"].append({"type": "image","image": path})
    
    obj['question'] += PROMPT_TEMPLATE['format_univg']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def qwen2_respond(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

def internvl2_process(obj):
    prefix, num_patches_list, first = '', [], True
    for idx, image_path in enumerate(obj['images']):
        temp = load_image_intenvl2(image_path, max_num=12).to(torch.bfloat16).cuda()
        num_patches_list.append(temp.size(0))
        prefix += f'Image-{idx+1}: <image>\n'
        if first == True:
            pixel_values = temp
        else:
            pixel_values = torch.cat((pixel_values, temp), dim=0)
        first = False
    
    obj['question'] = prefix + obj['question'].replace('<image>','') + PROMPT_TEMPLATE['format_internvl2']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[\1,\2,\3,\4]', obj['question'])
    return pixel_values, question, num_patches_list

def minicpm_process(obj):
    content = []
    for image_path in obj['images']:
        temp = Image.open(image_path).convert('RGB')
        temp = resize_image(temp)
        content.append(temp)
    obj['question'] += PROMPT_TEMPLATE['format_minicpm']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[\1,\2,\3,\4]', obj['question'])
    content.append(question)
    return [{'role': 'user', 'content': content}]

def mantis_process(obj):
    images = []
    messages = [{"role": "user","content": []}]
    for image_path in obj['images']:
        image = load_image(image_path)
        images.append(image)
        messages[0]['content'].append({"type": "image"})
        
    obj['question'] += PROMPT_TEMPLATE['format_mantis']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[0.\1,0.\2,0.\3,0.\4]', obj['question'])
    messages[0]['content'].append({"type": "text","text":question})
    return messages, images

def llava_process(obj, config, tokenizer, image_processor, device):
    conv_template = "qwen_1_5"
    images = []
    prefix = ''
    for idx, image_path in enumerate(obj['images']):
        image = Image.open(image_path)
        prefix += f'image{idx+1}:<image>\n'
        images.append(image)
    image_tensor = process_images(images, image_processor, config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    obj['question'] = prefix + obj['question'].replace('<image>','') + PROMPT_TEMPLATE['format_mantis']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\(<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>\)', '', obj['question']) # [0.\1,0.\2,0.\3,0.\4]
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    print(prompt_question)
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size for image in images]
    
    return input_ids, image_tensor, image_sizes

def mplug_process(obj, config, tokenizer, processor, device):
    prefix, images = '', []
    for idx, image_path in enumerate(obj['images']):
        image = Image.open(image_path).convert('RGB')
        prefix += f'Image-{idx+1}:<|image|>\n'
        images.append(image)
    
    obj['question'] = prefix + obj['question'].replace('<image>','') + PROMPT_TEMPLATE['format_mantis']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\(<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>\)', '', obj['question']) # [0.\1,0.\2,0.\3,0.\4]
    
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""}
    ]
    inputs = processor(messages, images=images, videos=None).to(device)
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens': 256,
        'decode_text': True,
    })
    
    return inputs

def cogvlm_process(obj, model, tokenizer, device, task):
    prefix, images = '', []
    image = Image.open(obj['images'][0]).convert('RGB')
    
    obj['question'] = obj['question'].replace('<image>','').replace('This is a series of concated images separately with black lines.','These are a series of images separately with black lines.')
    obj['question'] = obj['question'].replace('<|box_start|>','').replace('<|box_end|>','').replace('<|object_ref_start|>','').replace('<|object_ref_end|>','')
    question = re.sub(r'\(<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>\)', r'[\[0.\1,0.\2,0.\3,0.\4]\]', obj['question']) # [0.\1,0.\2,0.\3,0.\4]
    question += PROMPT_TEMPLATE['format_cogvlm']
    
    if task == 'diff':
        question += 'Ground it in the right image.'
        
    print(question)
    inputs = model.build_conversation_input_ids(tokenizer, query=question, images=[image])
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": True}
    
    return inputs, gen_kwargs

################# Model responding functions for specific MLLMs #################
def qwen2_vl_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen2_vl_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = qwen2_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def univgr1_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"univgr1_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = univgr1_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def migician_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"Migician_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = qwen2_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        prediction = extract_bbox(response[0])
        ### We have implemented linear scaling on the y axis for Migician
        try:
            prediction[1] = int(round(min(prediction[1] /1000 *1.05, 0.999), 3) * 1000)
            prediction[3] = int(round(min(prediction[3]/ 1000*1.10, 0.999), 3) * 1000)
        except:
            pass
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def mplug_eval(model, tokenizer, config, processor, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"mplug-owl3_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        inputs = mplug_process(obj, config, tokenizer, processor, device)
        response = model.generate(**inputs)
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def mantis_eval(model, generation_kwargs, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"mantis_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages, images = mantis_process(obj)
        messages = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=messages, images=images, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, **generation_kwargs)
        response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def llava_eval(model, tokenizer, image_processor, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"llava-onevision_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}
    
    model = model.to(device)

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        input_ids, image_tensor, image_sizes = llava_process(obj, model.config, tokenizer, image_processor, device)
        cont = model.generate(input_ids, images=image_tensor, image_sizes=image_sizes, do_sample=False, temperature=0, max_new_tokens=100)
        response = tokenizer.batch_decode(cont, skip_special_tokens=True)
        
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)
    
def internvl2_eval(model, tokenizer, generation_config, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"internvl2_{test_data[0]['task']}_{current_time}.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)[:-1]
    output, acc = [], {}

    model.to(device)  # 确保模型只移动一次

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        pixel_values, question, num_patches_list = internvl2_process(obj)
        pixel_values = pixel_values.to(device)  # 确保输入在同一设备上
        response,_ = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        
        prediction = extract_bbox(response)  # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def minicpm_eval(model, tokenizer, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"minicpm_{test_data[0]['task']}_{current_time}.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)[:-1]
    output, acc = [], {}

    model.to(device)

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        msgs = minicpm_process(obj)
        response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        prediction = extract_bbox(response)
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def cogvlm_eval(model, tokenizer, test_data, device, output_path):
    
    ### Note: Cogvlm struggle with concated multi-image grounding ###
    
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = os.path.join(output_path, f"cogvlm_{current_time}.json")
    output, acc = [], {}
    
    with torch.no_grad():
        for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
            inputs, gen_kwargs = cogvlm_process(obj, model, tokenizer, device, test_data[0]['task'])
            response = model.generate(**inputs, **gen_kwargs)
            response = response[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(response[0])
            
            prediction = extract_bbox(response) # 正则匹配生成的bbox
            iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
            
            print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
            output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def model_selection(model_type, model_path, test_data, device, output_path):
    if model_type=='univg-r1':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=401408)
        univgr1_eval(model, test_data, processor, device, output_path)
    
    elif model_type=='migician':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=401408)
        migician_eval(model, test_data, processor, device, output_path)
        
    elif model_type=='qwen2_vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=401408)
        qwen2_vl_eval(model, test_data, processor, device, output_path)
    
    elif model_type=='internvl2_8b':
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        internvl2_eval(model, tokenizer, generation_config, test_data, device, output_path)
        
    elif model_type=='mantis':
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto")
        generation_kwargs = {"max_new_tokens": 1024, "num_beams": 5, "do_sample": True}
        mantis_eval(model, generation_kwargs, test_data, processor, device, output_path)

    elif model_type=='minicpm':
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        minicpm_eval(model, tokenizer, test_data, device, output_path)
    
    elif model_type=='llava_onevision':
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")
        model.eval()
        llava_eval(model, tokenizer, image_processor, test_data, device, output_path)
    
    elif model_type=='mplug_owl3':
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)
        mplug_eval(model, tokenizer, config, processor, test_data, device, output_path)
        
    elif model_type=='cogvlm':
        tokenizer = LlamaTokenizer.from_pretrained('vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True ).to(device).eval()
        cogvlm_eval(model, tokenizer, test_data, device, output_path)
        
    else:
        raise NotImplementedError
    
########################### Task-specific Calling ###########################
def MIG_all(model_type, model_path, test_data, device, output_path):
    model_selection(model_type, model_path, test_data, device, output_path)
    
def common_object(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'common_object']
    model_selection(model_type, model_path, test_data, device, output_path)

def diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def group_grounding(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'group_grounding']
    model_selection(model_type, model_path, test_data, device, output_path)

def multi_view(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'multi_view']
    model_selection(model_type, model_path, test_data, device, output_path)

def object_tracking(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'object_tracking']
    model_selection(model_type, model_path, test_data, device, output_path)

def view_diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'view_diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def refer_grounding(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'refer_grounding']
    model_selection(model_type, model_path, test_data, device, output_path)

def region(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'region']
    model_selection(model_type, model_path, test_data, device, output_path)

def correspondence(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'correspondence']
    model_selection(model_type, model_path, test_data, device, output_path)

def reasoning(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'reasoning']
    model_selection(model_type, model_path, test_data, device, output_path)

def worker(gpu_id, task_queue, model_type, test_data, model_path, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{str(gpu_id)}")
    
    while True:
        try:
            task = task_queue.get(timeout=5)
        except Empty:
            break
        task(model_type, model_path, test_data, device, output_path)
        task_queue.task_done()
        

if __name__ == "__main__":

    mp.set_start_method("spawn")
    
    with open('/path/to/your/revised_MIG_bench.json') as file:
        test_data = json.load(file)
        
    output_path = f'./eval_output' # log information output
    os.makedirs(output_path, exist_ok=True)
    
    ############################################## supported models ###############################################
    ### univg-r1, migician, qwen2_vl, internvl2_8b, llava_onevision, minicpm, mantis, cogvlm, mplug_owl3(transformers==4.37.2)
    ### we support multi-processing parallel for migician, qwen2_vl, minicpm on MIG-Bench
    num_gpus, model_type = 8, 'univg-r1'
    model_path = "/path/to/your/univg-r1-model"
    print(f'😊 In case you forget~ The model you are evaluating is: {model_type} ❤️\n📂 From path: {model_path}')
    ###############################################################################################################
    
    ############################################### supported tasks ###############################################
    ### MIG-Bench ###
    ### MIG_all, common_object, view_diff, correspondence, diff, group_grounding
    ### multi_view, region, refer_grounding, object_tracking, reasoning
    tasks = [common_object, view_diff, correspondence, diff, group_grounding, multi_view, region, refer_grounding, object_tracking, reasoning]
    ###############################################################################################################
    
    processes = []
    
    task_queue = mp.JoinableQueue()
    for task in tasks:
        task_queue.put(task)

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, task_queue, model_type, test_data, model_path, output_path))
        processes.append(p)
        p.start()

    task_queue.join()
    for p in processes:
        p.join()
        
    print(f'😊 In case you forget~ The model you are evaluating is: {model_type} ❤️\n📂 From path: {model_path}')