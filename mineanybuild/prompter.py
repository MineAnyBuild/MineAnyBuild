import os
import numpy as np
import sys
import requests
import base64
import json
import time

# Fill in your api_url and api_key here
api_url = ''
api_key = ''


# proprietary MLLM 
class MLLMAgent:
    def __init__(self, model_name='gpt-4o-mini'):  
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.model = model_name
        print(self.model)

    def __call__(self, data):    
        '''
        Query the MLLM based on the payload of the prompt.
        '''
        response = self.post_request(data=data)
        if response.status_code == 200:
            result = response.json()
            if 'content' not in result['choices'][0]['message']:
                print(result['choices'][0]['message'])
                return result['choices'][0]['message']
            else:
                print(result['choices'][0]['message']['content'])
                return result['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}"

    def post_request(self, data):
        while True:
            try:
                response = requests.post(self.api_url, data=json.dumps(data), headers=self.headers)
                return response
            except requests.exceptions.Timeout:
                print("last request timeout, retrying...")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(str(e))
                print("failed to connect, retrying...")
                time.sleep(10)

    def encode_image(self, image_path):
        if "http" in image_path:
            response = requests.get(image_path)
            if response.status_code != 200:
                raise ValueError(f"Failed to download images, HTTP status code:{response.status_code}")
            return base64.b64encode(response.content).decode('utf-8')
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

    def llm_query(self, prompt):
        '''
        Format of the payload for the LLM query.
        '''
        payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

        return payload

    def vlm_query(self, prompt, images):
        '''
        Format of the payload for the MLLM query. (w/ images)
        '''
        image_messages = []
        for image_path in images:
            temp_message = {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64, {self.encode_image(image_path)}"}
            }
            image_messages.append(temp_message)
        content = [{"type": "text", "text": prompt}] + image_messages
        payload = {
                "model": self.model,
                "messages":[{"role":"user", "content": content}]
        }

        return payload



    def Task_Spatial_Understanding(self, instruction, block_types, arch_image):
        """
        Payload for Spatial Understanding task.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation."""

        user_prompt_1 = """Build the architecture based on the instruction and reference image. The instruction provides the relative coordinates of every block in reference image, and please identify the structure and summarize it as an overall blueprint matrix. The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\nHere is an example.\nBlock materials: {"oak_planks": 1}\nInstruction: build a 3*3*4 (width, length, height) wooden house layer by layer from bottom to top. Layer 1: oak_planks: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]. Layer 2: oak_planks: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]. Layer 3: oak_planks: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]. Layer 4: oak_planks: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)].\nOutput: '''json [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\nIMPORTANT: You must only use blocks from the given block materials dictionary.\nIMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\nIMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""

        user_prompt_2 = f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nReference image: """
        user_prompt_3 = """Output: """

        img_base64 = self.encode_image(arch_image)
        
        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_2},
                # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": user_prompt_3},
                ]
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content}
            ]
        }
        return payload


    def Task_Spatial_Plan_concrete(self, instruction, block_types, arch_image):
        """
        Payload for Executable Spatial Plan Generation task, with specific explanations as instruction input.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation."""

        user_prompt_1 = """Build the architecture based on the instruction and reference image. The instruction divides the architecture in the reference image by structure and demands. Please analyze and plan how to build the corresponding sub-structures according to the divided structure and demands, and give the ONLY ONE OVERALL blueprint matrix of the total architecture. The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\nHere is an example.\nBlock materials: {"oak_planks": 1}\nInstruction: build a 3*3*4 (width, length, height) wooden house. We want it to be a simple boxy house. The roof and floor should be solid and there is some space that player can go inside the house.\noutput: \nPlanning: The floor and roof of this wooden house can be made of 3*3 oak_planks as a square. Make the house hollow with air in the layer 2 and 3 and leave the space for entrance towards west (negative x). The two-layer walls are also made of 7*2=14 oak_planks. I can build it layer by layer so that I can truly understand the spatial structure of this house. Then, here is the overall blueprint matrix of the whole architeture. I'm sure that the 3-dim list has correct template and fill in -1 as "air" to simulate no block here.\nBlueprint:'''json [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\nIMPORTANT: You must only use blocks from the given block materials dictionary.\nIMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\nIMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""

        user_prompt_2 = f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nReference image: """
        user_prompt_3 = """Output: """

        img_base64 = self.encode_image(arch_image)
        
        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_2},
                # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": user_prompt_3},
                ]
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content}
            ]
        }
        return payload


    def Task_Spatial_Plan_simple(self, instruction, block_types, arch_image):
        """
        Payload for Executable Spatial Plan Generation task, with simple instruction.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation."""

        user_prompt_1 = """Build the architecture based on the instruction and reference image. You should understand the structure of the reference image and output your planning(reasoning) and overall blueprint matrix. The blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\nHere is an example.\nBlock materials: {"oak_planks": 1}\nInstruction: build a 3*3*4 (width, length, height) wooden house.\nOutput: Planning: We want it to be built with solid walls on all sides for the bottom layer. Make the hollow interior with only the outer walls for middle layers and a solid roof on the top layer.\nBlueprint: '''json [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\nIMPORTANT: You must only use blocks from the given block materials dictionary.\nIMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\nIMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""

        user_prompt_2 = f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\nReference image: """
        user_prompt_3 = """Output: """

        img_base64 = self.encode_image(arch_image)
        
        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_2},
                # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": user_prompt_3},
                ]
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content}
            ]
        }
        return payload


    def Task_Creativity(self, instruction, block_types):
        """
        Payload for Creativity task.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and Python coding. Your goal is to produce a Minecraft architecture, considering aspects such as architecture structure, block variety, symmetry and asymmetry, overall aesthetics, and most importantly, adherence to the platonic ideal of the requested creation."""

        user_prompt_1 = """Build the architecture based on the instruction and let your imagination run wild and use your creativity to build your best architecture. Before providing your final blueprint matrix, plan your solution considering the following spatial planning perspectives:\n1. How to best interpret and implement the build specification.\n- List key elements from the build specification\n- Brainstorm block combinations for different parts of the structure\n- Outline a rough structure layout\n2. Creative ways to use the available blocks to achieve the desired aesthetic.\n3. How to ensure the mapping correctness in your blueprint matrix.\n4. Ways to maximize creativity and the dynamic range of possible builds.\n5. Consider potential challenges and solutions\nThe blueprint is a three-dimension list, and the dimensions of the blueprint matrix are in the order of height(positive y), length(positive z) and width(positive x). Fill in "-1" into the blueprint to denote as "air". The elements of the list matrix are integers filled according to the given mapping table.\nHere is an output example.\nBlock materials = ["oak_planks", "cobblestone", "red_wool"]\nInstruction: build a 3*3*4 (width, length, height) wooden house. \nOutput:\nPlanning Reasons: Let's build a simple wooden house. I use cobblestone as the material for the floor and oak_planks for the wall and roof. \nSelected_block_materials = {"oak_planks": 1, "cobblestone": 2}\nBlueprint: '''json [[[2,2,2],[2,2,2],[2,2,2]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\nIMPORTANT: You must output ONLY ONE BLUEPRINT following the example format (A 3-DIM MATRIX).\nIMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix if no block is placed in the corresponding position."""

        user_prompt_2 = f"""Now, take a breath and continue.\nBlock materials: {block_types}\nInstruction: {instruction}\n"""
        user_prompt_3 = """Output: """


        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_2},
                {"type": "text", "text": user_prompt_3},
                ]
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content}
            ]
        }
        return payload


    def Task_Spatial_Commonsense(self, instruction, images, imgs_desp, img_root):
        """
        Payload for Spatial Commonsense task.
        """
        system_prompt = """You are an expert in Minecraft and interior design, familiar with real-life common sense. """

        user_prompt_1 = f"""You will receive a question and an image. Answer the question based on the image, focusing on spatial commonsense. Your response must not exceed 70 words. Do not include any additional content or thoughts. Now, take a breath and continue.\nInstruction: {instruction}\n"""

        user_prompt_3 = """\nYour answer: """

        img_content = []
        for i in range(len(images)):
            img_prompt = f"""The next image is {imgs_desp[i]}"""
            img_content.append({"type": "text", "text": img_prompt})
            img_base64 = self.encode_image(os.path.join(img_root, images[i]))
            # img_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})
            img_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})


        content_system = [{"type": "text", "text": system_prompt}]
        content = [{"type": "text", "text": user_prompt_1}, *img_content, {"type": "text", "text": user_prompt_3}]
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content}
            ]
        }
        return payload




    def Task_Spatial_Reasoning(self, instruction, input_image):
        """
        Payload for Spatial Reasoning task.
        """
        system_prompt = """You are an expert Minecraft builder and player in a flat Minecraft Java 1.20.4 server."""

        user_prompt_1 = f"""You need to answer this question with a visual image.\nQuestion: {instruction}\n"""

        user_prompt_2 = f"""You must output ONLY one option (from A,B,C,D,True,False) without any reason. IMPORTANT: You can only answer one letter or one word (from A,B,C,D,True,False). Your answer:"""

        img_base64 = self.encode_image(input_image)

        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": user_prompt_2},
                ]
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content}
            ]
        }
        return payload


#################
# Test script examples for different tasks


def test_architectures_1(task, input_root='architectures.json', task_root='./task', output_root='_test_assets', proprietary_models=['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    '''
    Query MLLM-based agents to obtain the output results of three types of tasks:
    1. Executable Spatial Plan Generation
    2. Creativity
    3. Spatial Understanding

    This example is for task data and architectures data provided by MineAnyBuild.
    For your curated data, please refer to the test_architectures_2 function.

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

    proprietary_models : list
        Default is a list of 7 proprietary models. Can be modified to test other models.
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
    
    OurLLM = {pm:MLLMAgent(model_name=pm) for pm in proprietary_models}    

    for arch in task_data:
        output_dir = os.path.join(output_root_1, arch["id"])
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=True)
        instruction = arch["instruction"]
        block_materials = architectures_data[arch["id"]]["block_materials"]
        block_types = {**{"air": -1}, **{block_materials[idx]:(idx+1) for idx in range(len(block_materials))}}
        arch_image = architectures_data[arch["id"]]["image"]
        for pm in proprietary_models:
            output_path = os.path.join(output_dir, pm+'.json')
            if os.path.exists(output_path):
                continue
            print(pm)
            if task == "Spatial_Understanding":
                payload = OurLLM[pm].Task_Spatial_Understanding(instruction, block_types, arch_image)
            elif task == "Creativity":
                payload = OurLLM[pm].Task_Creativity(instruction, block_types)
            elif task == "Executable_Spatial_Plan_Generation":
                if arch["id"].split('_')[-1] == '0':
                    payload = OurLLM[pm].Task_Spatial_Plan_concrete(instruction, block_types, arch_image)
                elif arch["id"].split('_')[-1] == '1':
                    payload = OurLLM[pm].Task_Spatial_Plan_simple(instruction, block_types, arch_image)
            output = OurLLM[pm](payload)
            with open(output_path, 'w') as f2:
                json.dump(output, f2, indent=4)
                


def test_architectures_2(task, input_root='/architectures', task_root='./task', output_root='_test_assets', proprietary_models=['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    '''
    Query MLLM-based agents to obtain the output results of three types of tasks:
    1. Executable Spatial Plan Generation
    2. Creativity
    3. Spatial Understanding

    This example is for your curated data.
    For task & architectures data provided by MineAnyBuild, please refer to the test_architectures_1 function.

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

    proprietary_models : list
        Default is a list of 7 proprietary models. Can be modified to test other models.
    '''

    output_root_1 = os.path.join(output_root, task, 'llm_response')


    task_hash = {"Spatial_Understanding": "Task_Spatial_Understanding.json", "Creativity": "Task_Creativity.json", "Executable_Spatial_Plan_Generation": "Task_Spatial_Planning.json"}

    task_json_file = os.path.join(task_root, task_hash[task])
    with open(task_json_file, 'r') as f1:
        task_data = json.load(f1)

    # you can evaluate partly by splitting the data.
    # task_data = random.sample(task_data, 100)
    
    OurLLM = {pm:MLLMAgent(model_name=pm) for pm in proprietary_models}    

    for arch in os.listdir(task_data):
        input_path = os.path.join(input_root, arch) # edit it based on your data format.
        output_dir = os.path.join(output_root_1, arch["id"])
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=True)

        with open(input_path, 'r') as f1: # edit it based on your data format.
            architecture_data = json.load(f1)

        instruction = arch["instruction"]
        block_materials = architecture_data["block_materials"] # edit it based on your data format.
        block_types = {**{"air": -1}, **{block_materials[idx]:(idx+1) for idx in range(len(block_materials))}}
        arch_image = os.path.join(input_root, architecture_data["image"])

        for pm in proprietary_models:
            output_path = os.path.join(output_dir, pm+'.json')
            if os.path.exists(output_path):
                continue
            print(pm)
            if task == "Spatial_Understanding":
                payload = OurLLM[pm].Task_Spatial_Understanding(instruction, block_types, arch_image)
            elif task == "Creativity":
                payload = OurLLM[pm].Task_Creativity(instruction, block_types)
            elif task == "Executable_Spatial_Plan_Generation":
                if arch["id"].split('_')[-1] == '0':
                    payload = OurLLM[pm].Task_Spatial_Plan_concrete(instruction, block_types, arch_image)
                elif arch["id"].split('_')[-1] == '1':
                    payload = OurLLM[pm].Task_Spatial_Plan_simple(instruction, block_types, arch_image)
            output = OurLLM[pm](payload)
            with open(output_path, 'w') as f2:
                json.dump(output, f2, indent=4)



def test_spatial_commonsense(output_root='./_spatial_commonsense', input_root='./data/images/commonsense', task_data='./task/Task_Spatial_Commonsense.json', proprietary_models = ['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    '''
    Query MLLM-based agents to obtain the output results of Spatial Commonsense tasks.

    Parameters:
    ----------
    output_root : str
        The root directory of the output data, where the results will be saved.

    input_root : str
        The root directory of the input data, which contains images of each tasks of Spatial Commonsense task.

    task_data : str
        Path to the JSON file containing task-specific data.

    proprietary_models : list
        Default is a list of 7 proprietary models. Can be modified to test other models.
    '''


    OurLLM = {pm:MLLMAgent(model_name=pm) for pm in proprietary_models}    

    with open(task_data,'r') as f1:
        data = json.load(f1)

    out = {}
    for item in data:
        temp = {}
        task_id = item["id"]
        instruction = item["instruction"]
        images = item["image"]
        imgs_desp = item["image_desp"]
        for pm in proprietary_models:
            payload = OurLLM[pm].Task_Spatial_Commonsense(instruction, images, imgs_desp, input_root)
            output = OurLLM[pm](payload)
            temp[pm] = output
        out[task_id] = temp

    if not os.path.exists(output_root): 
        os.makedirs(output_root, exist_ok=True)

    output_path = os.path.join(output_root, 'results.json')
    with open(output_path, 'w') as f2:
        json.dump(out, f2, indent=4)



def test_spatial_reasoning(images_root='./data/images/reasoning', task_data='./task/Task_Spatial_Reasoning.json', output_root='_spatial_reasoning/results.json', proprietary_models=['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    from tqdm import tqdm
    OurLLM = {pm:MLLMAgent(model_name=pm) for pm in proprietary_models}
    with open(task_data, 'r') as f1:
        vqa_data = json.load(f1)

    # you can evaluate partly by splitting the data.s
    # vqa_data = random.sample(vqa_data, 100)

    out = {}
    for task in tqdm(vqa_data):
        image = os.path.join(images_root, task["options_image"])
        instruction = task["question"]
        temp_out = {}
        for pm in proprietary_models:
            payload = OurLLM[pm].Task_Spatial_Reasoning(instruction, image)
            output = OurLLM[pm](payload)
            temp_out[pm] = output
        out[task["id"]] = temp_out

    with open(output_root, 'w') as f2:
        json.dump(out, f2, indent=4)




if __name__ == '__main__':
    ##########################
    # Uncomment the statements below to test tasks
    ##########################
    
    # test_architectures_1("Spatial_Understanding")
    # test_architectures_1("Creativity")
    # test_architectures_1("Executable_Spatial_Plan_Generation")
    # test_architectures_2("Spatial_Understanding")
    # test_architectures_2("Creativity")
    # test_architectures_2("Executable_Spatial_Plan_Generation")
    # test_spatial_commonsense()
    test_spatial_reasoning()