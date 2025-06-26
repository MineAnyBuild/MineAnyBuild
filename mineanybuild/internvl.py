import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import requests
from io import BytesIO
import math
import argparse


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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
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
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map



def test_architectures(task, input_root='architectures.json', task_root='./task', output_root='_test_assets'):
    '''
    Query MLLM-based agents to obtain the output results of three types of tasks:
    1. Executable Spatial Plan Generation
    2. Creativity
    3. Spatial Understanding

    This example is for task data and architectures data provided by MineAnyBuild.

    Parameters:
    ----------
    task : str
        The task type, can be one of the following:
        ["Spatial_Understanding", "Creativity", "Executable_Spatial_Plan_Generation"]

    input_root : str
        The root directory of the input data, which contains the architecture data (several JSON files, each of which contains an architecture.), or the architectures.json file we provided that contains the architecture data.

    task_root : str
        The root directory of the task data, where the task-specific data is stored.

    output_root : str
        The root directory of the output data, where the results will be saved.
    '''

    output_root_1 = os.path.join(output_root, task, 'llm_response')
    task_hash = {"Spatial_Understanding": "Task_Spatial_Understanding.json", "Creativity": "Task_Creativity.json", "Executable_Spatial_Plan_Generation": "Task_Spatial_Planning.json"}

    task_json_file = os.path.join(task_root, task_hash[task])
    with open(task_json_file, 'r') as f1:
        task_data = json.load(f1)

    with open(input_root, 'r') as f2:
        architectures_raw_data = json.load(f2)
    architectures_data = {item['id']: item for item in architectures_raw_data}

    # you can evaluate partly by splitting the data.
    # task_data = random.sample(task_data, 100)

    for arch in task_data:
        output_dir = os.path.join(output_root_1, arch["id"])
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=True)
        instruction = arch["instruction"]
        block_materials = architectures_data[arch["id"]]["block_materials"]
        block_types = {**{"air": -1}, **{block_materials[idx]:(idx+1) for idx in range(len(block_materials))}}
        arch_image = architectures_data[arch["id"]]["image"]

        output_path = os.path.join(output_dir, args.model_path.split('/')[-1]+'.json')
        if os.path.exists(output_path):
            continue

        if task == "Spatial_Understanding":
            prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation.\n\nBuild the architecture based on the instruction and reference image. The instruction provides the relative coordinates of every block in reference image, and please identify the structure and summarize it as an overall blueprint matrix. The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\nHere is an example.\nBlock materials: {"oak_planks": 1}\nInstruction: build a 3*3*4 (width, length, height) wooden house layer by layer from bottom to top. Layer 1: oak_planks: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]. Layer 2: oak_planks: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]. Layer 3: oak_planks: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]. Layer 4: oak_planks: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)].\nOutput: '''json
            [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
            '''
            IMPORTANT: You must only use blocks from the given block materials dictionary. Make full use of them.\n
            IMPORTANT: You MUST output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX). You MUST NOT answer your reasons.\n
            IMPORTANT: Your results should only follow the format of the example and not be influenced by the content of the examples.\n
            IMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position.\n"""+f"""
            Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nReference image: <image>\nOutput: """
        elif task == "Creativity":
            prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation.\n\nBuild the architecture based on the instruction and let your imagination run wild and use your creativity to build your best architecture. Before providing your final blueprint matrix, plan your solution considering the following spatial planning perspectives:\n
            1. How to best interpret and implement the build specification.\n
            - List key elements from the build specification\n
            - Brainstorm block combinations for different parts of the structure\n
            - Outline a rough structure layout\n
            2. Creative ways to use the available blocks to achieve the desired aesthetic.\n
            3. How to ensure the mapping correctness in your blueprint matrix.\n
            4. Ways to maximize creativity and the dynamic range of possible builds.\n
            5. Consider potential challenges and solutions\n
            The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\n
            Here is an output example.\n
            Block materials = ["oak_planks", "cobblestone", "red_wool"]\n
            Instruction: build a 3*3*4 (width, length, height) wooden house. \n
            Output:\n
            Planning Reasons: Let's build a simple wooden house. I use cobblestone as the material for the floor and oak_planks for the wall and roof. \n
            Selected_block_materials = {"oak_planks": 1, "cobblestone": 2}\n
            Blueprint:
            '''json
            [[[2,2,2],[2,2,2],[2,2,2]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
            '''

            IMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\n
            IMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""+f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nOutput:"""
        elif task == "Executable_Spatial_Plan_Generation":
            if arch["id"].split('_')[-1] == '0':
                prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation.\n\nBuild the architecture based on the instruction and reference image. The instruction divides the architecture in the reference image by structure and demands. Please analyze and plan how to build the corresponding sub-structures according to the divided structure and demands, and give the ONLY ONE OVERALL blueprint matrix of the total architecture. The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\n
                Here is an example.\n
                Block materials: {"oak_planks": 1}\n
                Instruction: build a 3*3*4 (width, length, height) wooden house. We want it to be a simple boxy house. The roof and floor should be solid and there is some space that player can go inside the house.\n
                output: \n
                Planning: The floor and roof of this wooden house can be made of 3*3 oak_planks as a square. Make the house hollow with air in the layer 2 and 3 and leave the space for entrance towards west (negative x). The two-layer walls are also made of 7*2=14 oak_planks. I can build it layer by layer so that I can truly understand the spatial structure of this house. Then, here is the overall blueprint matrix of the whole architeture. I'm sure that the 3-dim list has correct template and fill in -1 as "air" to simulate no block here.\n
                Blueprint:
                '''json
                [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
                '''

                IMPORTANT: You must only use blocks from the given block materials dictionary. Make full use of them.\n
                IMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\n
                IMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""+f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nReference image: <image>\nOutput: """
            elif arch["id"].split('_')[-1] == '1':
                prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation.\n\nBuild the architecture based on the instruction and reference image. You should understand the structure of the reference image and output your planning(reasoning) and overall blueprint matrix. The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\nHere is an example.\nBlock materials: {"oak_planks": 1}\nInstruction: build a 3*3*4 (width, length, height) wooden house.\nOutput: 
                Planning: We want it to be built with solid walls on all sides for the bottom layer. Make the hollow interior with only the outer walls for middle layers and a solid roof on the top layer.\n
                Blueprint:
                '''json
                [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
                '''

                IMPORTANT: You must only use blocks from the given block materials dictionary. Make full use of them.\n
                IMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\n
                IMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""+f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nReference image: <image>\nOutput: """

        pixel_values = load_image(arch_image, max_num=12).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        print(response)
        with open(output_path, 'w') as f2:
            json.dump(response, f2, indent=4)



def test_spatial_reasoning(images_root='./data/images/reasoning', task_data='./task/Task_Spatial_Reasoning.json', output_root='_spatial_reasoning'):
    '''
    Query MLLM-based agents to obtain the output results of Spatial Reasoning tasks.
    '''

    output_path = os.path.join(output_root, 'results_' + args.model_path.split('/')[-1] + '.json')
    with open(task_data, 'r') as f1:
        vqa_data = json.load(f1)

    out = {}
    for task in tqdm(vqa_data):
        image = images_root + task["options_image"]
        print(image)
        instruction = task["instruction"]
        if task["id"].split("SR_")[1].split("_")[0] == '3':
            prompt = f"You are an expert Minecraft builder and player in a flat Minecraft Java 1.20.4 server.\nYou need to answer this question with a visual image.\nQuestion: {instruction}\n<image>\nYou must output ONLY one option (from True,False) without any reason based on the question. IMPORTANT: You can only answer one word (from True,False). Your answer:"
        else:
            prompt = f"You are an expert Minecraft builder and player in a flat Minecraft Java 1.20.4 server.\nYou need to answer this question with a visual image.\nQuestion: {instruction}\n<image>\nYou must output ONLY one option (from A,B,C,D) without any reason based on the question. IMPORTANT: You can only answer one letter(from A,B,C,D). Your answer:"
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        print(response)
        out[task["id"]] = response

    with open(output_path, 'w') as f2:
        json.dump(out, f2, indent=4)


def test_spatial_commonsense(output_root='./_spatial_commonsense', input_root='./data/images/commonsense', task_data='./task/Task_Spatial_Commonsense.json'):
    '''
    Query MLLM-based agents to obtain the output results of Spatial Commonsense tasks.
    '''

    with open(task_data,'r') as f1:
        data = json.load(f1)

    out = {}
    for item in tqdm(data):
        task_id = item["id"]
        instruction = item["question"]
        images = item["images"]
        imgs_desp = item["image_desp"]

        visual_image = os.path.join(input_root, images[0])
        image_content = f"The next image is {imgs_desp[0]}. <image>"
        prompt = f"""You are an expert in Minecraft and interior design, familiar with real-life common sense.\nYou will receive a question and an image. Answer the question based on the image, focusing on spatial commonsense. Your response must not exceed 70 words. Do not include any additional content or thoughts. Now, take a breath and continue.\nInstruction: {instruction}\n"""+image_content+"""\nYour answer:"""

        pixel_values = load_image(visual_image).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        out[task_id] = response

    if not os.path.exists(output_root): 
        os.makedirs(output_root, exist_ok=True)

    output_path = os.path.join(output_root, 'results_'+args.model_path.split('/')[-1]+'.json')
    with open(output_path, 'w') as f2:
        json.dump(out, f2, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference scripts for MineAnyBuild tasks")
    parser.add_argument('--task', type=str, choices=['Spatial_Understanding', 'Creativity', 'Executable_Spatial_Plan_Generation', 'Spatial_Commonsense', 'Spatial_Reasoning'], required=True, help="Tasks of MineAnyBuild")
    parser.add_argument('--model_path', type=str, default='/models/InternVL2_5-4B', help="Path to the model directory")
    args = parser.parse_args()

    model_path = args.model_path
    device_map = split_model(args.model_path.split('/')[-1])

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
        device_map=device_map).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, device_map='auto')
    generation_config = dict(max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    task_map = {
        'Spatial_Understanding': lambda: test_architectures("Spatial_Understanding"),
        'Creativity': lambda: test_architectures("Creativity"), 
        'Executable_Spatial_Plan_Generation': lambda: test_architectures("Executable_Spatial_Plan_Generation"),
        'Spatial_Commonsense': test_spatial_commonsense,
        'Spatial_Reasoning': test_spatial_reasoning
    }
    
    task_map[args.task]()
