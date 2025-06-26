import os
import numpy as np
import sys
import requests
import base64
import json
import time
from tqdm import tqdm
import argparse
from ast import literal_eval


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


    def Critic_Scoring_Spatial_Commonsense(self, response, answer):
        """
        Evaluation prompt for Spatial Commonsense task.
        """
        system_prompt = """You are an expert in the field of multi-modal large language models and answer proofreading. You can well compare the differences between the output results of large models and the standard answers and score them."""

        user_prompt_1 = f"""You will get the output result of a multi-modal large language model and a standard answer. You need to compare the two and score the output result of the MLLM. What you need to note is: \n1)Evaluate the matching degree between the output result of the large model and the standard answer. It is not necessary for the contents of the two to be completely the same, but the tendency of the answers must be the same to be considered a correct match.\n2)You need to score the matching degree, with a full score of 10. If it is a correct match, please score at least 8 points or more. If it is a wrong match, please score at least 3 points or less. For example, if the output of the large model is yes, and the standard answer is no, then it is obviously a wrong match.\n3)You need to carefully check the key information in the standard answer, such as spatial position and direction, action tendency, and spatial common sense reasoning. If the output of the large model meets all of them, please add points; if there are any that are not met, please deduct points as appropriate within the range.\n"""
        user_prompt_2 = """Please output the score and scoring reason (within 70 words) following this JSON format:\n{"score": 5, "reason": ""}\n"""

        user_prompt_3 = f"""Standard answer: {answer}\nMLLM response: {response}\nYour result in JSON format:"""


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


    def Critic_Scoring_Creativity(self, instruction, input_image):
        """
        Evaluation prompt for Creativity task.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and an expert architecture critic."""

        user_prompt_1 = f"""Give a grade from 1 to 10 to the following Minecraft architectures from different views. You should give the grade based on how well they are presented and correspond together to the building instructions in the following aspects:\n
        - Creativity: from *boring, dull*(1) to *mediocre, normal*(5) and *blue sky thinking, inspiring*(10).\n
        - Completeness: from *nothing, abandoned*(1) to *partial, incomplete*(5) and *masterfully completed, perfectly realized*(10).\n
        - Complexity: from *simplistic, basic*(1) to *straightforward, moderate *(5) and *challenging, hardcore*(10).\n
        - Architecture Structure: from *boxy, rudimentary*(1) to *intuitive, modest*(5) and *sophisticated, intricate*(10).\n
        - Overall Aesthetic, Atmosphere and Fidelity: from *stark, bare*(1) to *appealing, unusual*(5) and *epic, masterpiece*(10)."""

        example_json = {
            "Creativity": {
                "grade": 6,
                "comment": "The building uses the same material but different forms of block types to enrich the architectural design. The design of this building is based on reality but beyond reality."
            },
            "Completeness": {
                "grade": 5,
                "comment": "The architecture well follows the building instructions and achieves a good finish. The architecture is not broken and is built completely."
            },
            "Complexity": {
                "grade": 6,
                "comment": "This architecture has several advanced building techniques like using several stairs upside down and half slabs to present some structures with half a block."
            },
            "Architecture Structure": {
                "grade": 6, 
                "comment": "The design of the whole building is based on the vertical line as the axis and symmetrical in the center, which has the sense of extending upward."
            },
            "Overall Aesthetic, Atmosphere and Fidelity": {
                "grade": 5,
                "comment": "The selection and placement of blocks have a certain aesthetic sense, reveal the feeling of ancient, in line with the requirements of the given instructions."
            }}

        user_prompt_2 = f"""Building instructions: {instruction}\nYou must ONLY return the results in the following JSON format, but do not refer to the grades in this example and just follow the FORMAT: {example_json}\nYou must not output any other reasoning or analysis.\nOutput:"""

        user_prompt_image_input_intro = """Give the grades based on the following image showing the Minecraft architecture in JSON format."""

        img_base64 = self.encode_image(input_image)


        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_image_input_intro},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
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
    

    def Critic_Scoring_Spatial_Plan(self, instruction, input_image, ref_image):
        """
        Evaluation prompt for Executable Spatial Plan Generation task.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and an expert architecture critic."""

        user_prompt_1 = f"""Give a grade from 1 to 10 to the following Minecraft architectures from different views. The scores of reference human-annotated architectures are all 8 by default, as a reference for comparison. You should give the grade based on how well they are presented and correspond together to the building instructions in the following aspects:\n
        - Completeness(Instruction Following): from *nothing, abandoned*(1) to *partial, incomplete*(5) and *masterfully completed, perfectly realized*(10).\n
        - Complexity: from *simplistic, basic*(1) to *straightforward, moderate *(5) and *challenging, hardcore*(10).\n
        - Overall Aesthetic, Atmosphere and Fidelity: from *stark, bare*(1) to *appealing, unusual*(5) and *epic, masterpiece*(10)."""

        example_json = {
            "Completeness(Instruction Following)": {
                "grade": 8,
                "comment": "The architecture well follows the building instructions and achieves a good finish, which is similar to the reference architecture."
            },
            "Complexity": {
                "grade": 8,
                "comment": "This architecture has several advanced building techniques like using several stairs upside down and half slabs to present some structures with half a block."
            },
            "Overall Aesthetic, Atmosphere and Fidelity": {
                "grade": 8,
                "comment": "The selection and placement of blocks have a certain aesthetic sense, reveal the feeling of ancient, in line with the requirements of the given instructions."
            }}

        user_prompt_2 = f"""Building instructions: {instruction}\nYou must ONLY return the results in the following JSON format, but do not refer to the grades in this example and just follow the FORMAT: {example_json}\nYou must not output any other reasoning or analysis.\nOutput:"""

        user_prompt_image_ref_intro = """The following image is the ground-truth human-annotated reference image."""

        user_prompt_image_input_intro = """Give the grades based on the following image showing the Minecraft architecture in JSON format."""

        ref_img_base64 = self.encode_image(ref_image)
        img_base64 = self.encode_image(input_image)


        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_image_ref_intro},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_img_base64}"}},
                {"type": "text", "text": user_prompt_image_input_intro},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
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



    def Critic_Scoring_Spatial_Understanding(self, instruction, input_image, ref_image):
        """
        Evaluation prompt for Spatial Understanding task.
        """
        system_prompt = """You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 server and an expert architecture critic."""

        user_prompt_1 = f"""Give a grade from 1 to 10 to the following Minecraft architectures from different views. The scores of reference human-annotated architectures are all 10 by default, as a reference for comparison. You should give the grade based on how well they are presented and correspond together to the building instructions in the following aspects:\n
        - Instruction Following(Completeness): from *nothing, abandoned*(1) to *partial, incomplete*(5) and *masterfully completed, perfectly realized*(10)."""

        example_json = {
            "Instruction Following(Completeness)": {
                "grade": 8,
                "comment": "The architecture well follows the building instructions and achieves a good finish, which is similar to the reference architecture."
            }}

        user_prompt_2 = f"""Building instructions: {instruction}\nYou must ONLY return the results in the following JSON format, but do not refer to the grades in this example and just follow the FORMAT: {example_json}\nYou must not output any other reasoning or analysis.\nOutput:"""

        user_prompt_image_ref_intro = """The following image is the ground-truth human-annotated reference image."""

        user_prompt_image_input_intro = """Give the grades based on the following image showing the Minecraft architecture in JSON format."""

        ref_img_base64 = self.encode_image(ref_image)
        img_base64 = self.encode_image(input_image)


        content_system = [{"type": "text", "text": system_prompt}]
        content = [
                {"type": "text", "text": user_prompt_1},
                {"type": "text", "text": user_prompt_image_ref_intro},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_img_base64}"}},
                {"type": "text", "text": user_prompt_image_input_intro},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
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

def critic_spatial_tasks(task, clipped_images_root='/selected-frames', input_root='architectures.json', output_root='_output/proprietary/test_assets', proprietary_models = ['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    '''
    Evaluate MLLM-based agents of the following tasks:
    1. Executable Spatial Plan Generation
    2. Creativity
    3. Spatial Understanding

    '''

    gpt_41 = MLLMAgent(model_name='gpt-4.1')
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    with open(input_root, 'r') as f1:
        architectures_raw_data = json.load(f1)    
    architectures_data = {item['id']: item for item in architectures_raw_data}

    gt_image_root = os.path.join(os.path.dirname(input_root), "task", "images")

    Results_of_Tasks = {pm:{} for pm in proprietary_models}
    output_path = os.path.join(output_root, '/Results_Task_' + task.replace(" ", "_") + '.json')

    images = os.listdir(clipped_images_root)
    for image in tqdm(images):
        arch_image = os.path.join(clipped_images_root, image)
        task = image.split("$")[0]
        arch_id = image.split("$")[1].split('.json')[0]
        model_name = image.split("$")[2].split('.jpg')[0]

        if task in ['Spatial_Understanding', 'Executable_Spatial_Plan_Generation']:
            ref_image = gt_image_root + architectures_data[arch_id]["image"]
        instruction = architectures_data[arch_id]["instructions"]

        
        if task == 'Creativity':
            payload = gpt_41.Critic_Scoring_Creativity(instruction, arch_image)
            output = gpt_41(payload)
            Results_of_Tasks[model_name][arch_id] = output
        elif task == 'Spatial Understanding':
            payload = gpt_41.Critic_Scoring_Spatial_Understanding(instruction, arch_image, ref_image)
            output = gpt_41(payload)
            Results_of_Tasks[model_name][arch_id] = output
        elif task == 'Executable Spatial Plan Generation':
            payload = gpt_41.Critic_Scoring_Spatial_Plan(instruction, arch_image, ref_image)
            output = gpt_41(payload)
            Results_of_Tasks[model_name][arch_id] = output

    with open(output_path, 'w') as f1:
        json.dump(Results_of_Tasks, f1, indent=4) 



def critic_spatial_commonsense(results_root='_spatial_commonsense/results.json', task_root='./task/Task_Spatial_Commonsense.json', output_path='_spatial_commonsense/critic_results.json'):
    '''
    Evaluate MLLM-based agents of Spatial Commonsense tasks.

    Parameters:
    ----------
    results_root : str
        Path to the JSON file containing MLLM response results for spatial commonsense tasks.

    task_root : str
        Path to the JSON file containing ground truth data and task definitions.

    output_path : str
        Path where the evaluation scores and critic results will be saved.
    '''

    from ast import literal_eval
    with open(results_root, 'r') as f1:
        results_data = json.load(f1)
    with open(task_root, 'r') as f2:
        gt_data = json.load(f2)

    gt_answers = {item["id"]:item["answer"] for item in gt_data}
    gpt_41 = MLLMAgent(model_name='gpt-4.1')

    out = {}
    for k, v in results_data.items():
        temp = {}
        gt_answer = gt_answers[k]
        for k1, v1 in v.items():
            payload = gpt_41.Critic_spatial_commonsense(v1, gt_answer)
            output = literal_eval(gpt_41(payload))
            temp[k1] = output
        out[k] = temp

    with open(output_path, 'w') as f3:
        json.dump(out, f3, indent=4)


def critic_spatial_commonsense_opensource(results_root='_spatial_commonsense', task_data='./task/Task_Spatial_Commonsense.json', output_path='_spatial_commonsense/critic_results.json', models=["InternVL2_5-8B", "llava-onevision-qwen2-7b-ov", "llava-onevision-qwen2-0.5b-ov", "Qwen2.5-VL-7B-Instruct"]):
    '''
    Evaluate open-source MLLM-based agents of Spatial Commonsense tasks.
    Open-source MLLM: internvl, qwen, llavaonevision

    Parameters:
    ----------
    results_root : str
        Path to the JSON file containing MLLM response results for spatial commonsense tasks.

    task_data : str
        Path to the JSON file containing ground truth data and task definitions.

    output_path : str
        Path where the evaluation scores and critic results will be saved.
    '''

    from ast import literal_eval
    results = {}
    for pm in models:
        with open(f'{results_root}/spat_commonsense_results_{pm}.json', 'r') as f1: # edit with your specific results
            results_data = json.load(f1)
        results[pm] = results_data
    with open(task_data, 'r') as f2:
        gt_data = json.load(f2)

    gt_answers = {item["id"]:item["answer"] for item in gt_data}
    gpt_41 = MLLMAgent(model_name='gpt-4.1')
    
    out = {}
    for k, v in results.items():
        temp = {}
        for k1, v1 in v.items():
            gt_answer = gt_answers[k1]
            payload = gpt_41.Critic_spatial_commonsense(v1, gt_answer)
            output = literal_eval(gpt_41(payload))
            temp[k1] = output
        out[k] = temp

    with open(output_path, 'w') as f3:
        json.dump(out, f3, indent=4)
    

def calculate_scores_spatial_reasoning(input_results_filepath, eval_data_filepath, proprietary_models = ['claude-3-5-sonnet', 'claude-3-7-sonnet-latest']):
    with open(input_results_filepath, 'r') as f1:
        results = json.load(f1)

    with open(eval_data_filepath, 'r') as f2:
        data = json.load(f2)

    answers = {item["id"]:item["answer"] for item in data}

    total = {k:0 for k in proprietary_models}
    for k, v in results.items():
        ans = answers[k]
        print(k)
        for k1, v1 in v.items():
            if ans == v1.strip():
                total[k1] += 1

    scores = {k:v/len(data) for k, v in total.items()}
    print(scores)



def calculate_scores_spatial_commonsense(input_results_filepath, models = ["InternVL2_5-1B", "InternVL2_5-2B", "InternVL2_5-4B", "Qwen2.5-VL-3B-Instruct"]):
    with open(input_results_filepath, 'r') as f1:
        data = json.load(f1)

    out = {k:0 for k in models}
    sr = {k:0 for k in models}

    for k, v in data.items():   # k: model_name
        for k1, v1 in v.items():
            out[k] += v1['score']
            if v1['score'] >= 7: # we specify the score>7 as success.
                sr[k] += 1

    scores = {k:v/50 for k, v in out.items()}
    print(scores)
    srs = {k:v/50 for k, v in sr.items()}
    # print(srs)



def calculate_scores_creativity(input_dir, models=['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    with open(input_dir, 'r') as f1:
        data = json.load(f1)
    out = {k:0 for k in models}
    test_num = {k:0 for k in models}
    for k1, v1 in data.items():
        test_num[k1] = len(v1)
        for k2, v2 in v1.items():
            v2 = literal_eval(v2)
            score = v2["Creativity"]["grade"]*0.8 + v2["Completeness"]["grade"]*0.05 + v2["Complexity"]["grade"]*0.05 + v2["Architecture Structure"]["grade"]*0.05 + v2["Overall Aesthetic, Atmosphere and Fidelity"]["grade"]*0.05
            out[k1] += score

    out = {k:v/test_num[k] for k, v in out.items()}
    print(out)


def calculate_scores_spatial_plan(input_dir, models=['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    with open(input_dir, 'r') as f1:
        data = json.load(f1)
    out = {k:0 for k in models}
    test_num = {k:0 for k in models}
    for k1, v1 in data.items():
        test_num[k1] = len(v1)
        for k2, v2 in v1.items():
            v2 = literal_eval(v2)
            score = v2["Completeness(Instruction Following)"]["grade"]*0.3 + v2["Complexity"]["grade"]*0.3 + v2["Overall Aesthetic, Atmosphere and Fidelity"]["grade"]*0.4
            out[k1] += score

    out = {k:v/test_num[k] for k, v in out.items()}

    print(out)


def calculate_scores_spatial_understanding(input_dir, models=['claude-3-5-sonnet', 'claude-3-7-sonnet-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini']):
    with open(input_dir, 'r') as f1:
        data = json.load(f1)
    out = {k:0 for k in models}
    test_num = {k:0 for k in models}
    for k1, v1 in data.items():
        test_num[k1] = len(v1)
        for k2, v2 in v1.items():
            v2 = literal_eval(v2)
            score = v2["Instruction Following(Completeness)"]["grade"]
            out[k1] += score
    out = {k:v/test_num[k] for k, v in out.items()}

    print(out)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation scripts for MineAnyBuild tasks")
    parser.add_argument('--task', type=str, choices=['Spatial_Understanding', 'Creativity', 'Executable_Spatial_Plan_Generation', 'Spatial_Commonsense', 'Spatial_Reasoning'], required=True, help="Tasks of MineAnyBuild")
    args = parser.parse_args()

    task_map = {
        'Spatial_Understanding': lambda: critic_spatial_tasks("Spatial_Understanding"),
        'Creativity': lambda: critic_spatial_tasks("Creativity"), 
        'Executable_Spatial_Plan_Generation': lambda: critic_spatial_tasks("Executable_Spatial_Plan_Generation"),
        'Spatial_Commonsense': critic_spatial_commonsense
    }
    
    task_map[args.task]()




