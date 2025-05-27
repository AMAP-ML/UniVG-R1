from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = 'GD-ML/UniVG-R1'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)
processor = AutoProcessor.from_pretrained(model_path, max_pixels=401408)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "figs/case076_1.jpg",
            },
            {
                "type": "image",
                "image": "figs/case076_2.jpg",
            },
            {
                "type": "text",
                "text": "You are now presented with two objects. For the area marked by the red bounding box in the first image, identify and locate the corresponding area in the second image that serves a similar function or shares a similar meaning.  First output the thinking process in <think> </think> tags and then output the bounding box in <answer> </answer> tags."
            }
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
inputs = inputs.to("cuda:0")

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
