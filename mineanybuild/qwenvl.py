import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import requests
from io import BytesIO
import math
import argparse


def test_demo():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)



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

        messages = [{"role": "user",
                "content": [
                    {"type": "image", "image": arch_image},
                    {"type": "text", "text": prompt},
                ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_text[0])
        with open(output_path, 'w') as f2:
            json.dump(output_text[0], f2, indent=4)



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
        messages = [{"role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_text[0])
        out[task["id"]] = output_text[0]

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

        messages = [{"role": "user",
                "content": [
                    {"type": "image", "image": visual_image},
                    {"type": "text", "text": prompt},
                ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_text[0])
        out[task_id] = output_text[0]

    if not os.path.exists(output_root): 
        os.makedirs(output_root, exist_ok=True)

    output_path = os.path.join(output_root, 'results_'+args.model_path.split('/')[-1]+'.json')
    with open(output_path, 'w') as f2:
        json.dump(out, f2, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference scripts for MineAnyBuild tasks")
    parser.add_argument('--task', type=str, choices=['Spatial_Understanding', 'Creativity', 'Executable_Spatial_Plan_Generation', 'Spatial_Commonsense', 'Spatial_Reasoning'], required=True, help="Tasks of MineAnyBuild")
    parser.add_argument('--model_path', type=str, default='/models/Qwen2.5-VL-7B-Instruct', help="Path to the model directory")
    args = parser.parse_args()

    model_path = args.model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16, device_map="auto")

    min_pixels = 32*32
    max_pixels = 768*768
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    task_map = {
        'Spatial_Understanding': lambda: test_architectures("Spatial_Understanding"),
        'Creativity': lambda: test_architectures("Creativity"), 
        'Executable_Spatial_Plan_Generation': lambda: test_architectures("Executable_Spatial_Plan_Generation"),
        'Spatial_Commonsense': test_spatial_commonsense,
        'Spatial_Reasoning': test_spatial_reasoning
    }
    
    task_map[args.task]()
