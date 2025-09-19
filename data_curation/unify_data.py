import json
import os
import argparse

#############################################################
###   Data Standardization for MineAnyBuild tasks
#############################################################

def task_CR():
    '''
    Data Standardization for Creativity Task
    '''

    output = []
    output_path = "./task/Task_Creativity.json"

    ### Grabcraft
    grabcraft_data_root = './data/data_grabcraft/data_processed'
    archs = os.listdir(grabcraft_data_root)
    
    for arch in archs:
        with open(os.path.join(grabcraft_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        for i in range(3):
            temp_out = {}
            temp_out["id"] = "TSK_CR_"+temp_arch_data["id"].split('_')[1]+f"_{i}"
            temp_out["AR_id"] = temp_arch_data["id"]
            temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
            temp_out["instruction"] = temp_arch_data["instructions"]["simple"][i]
            temp_out["block_materials"] = temp_arch_data["object_materials"]
            temp_out["image"] = None
            temp_out["image_desp"] = None
            temp_out["options_image"] = None
            temp_out["options"] = None
            temp_out["metadata"] = None
            output.append(temp_out)

    ### Wiki
    wiki_data_root = './data/data_official_wiki/data_processed'
    wiki_archs = os.listdir(wiki_data_root)

    for arch in wiki_archs:
        with open(os.path.join(wiki_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        old_block_materials = temp_arch_data["block_materials"]
        new_reversed_block_materials = {}
        new_object_materials = []
        for k, v in old_block_materials.items():
            if k == "air":
                continue
            pure_k = k.split('[')[0]
            if pure_k not in new_object_materials:
                new_object_materials.append(pure_k)
            new_reversed_block_materials[v] = new_object_materials.index(pure_k)+1
        
        for i in range(3):
            temp_out = {}
            temp_out["id"] = "TSK_CR_"+temp_arch_data["id"].split('_')[1]+f"_{i}"
            temp_out["AR_id"] = temp_arch_data["id"]
            temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
            temp_out["instruction"] = temp_arch_data["instructions"]["simple"][i]
            temp_out["block_materials"] = new_object_materials
            temp_out["image"] = None
            temp_out["image_desp"] = None
            temp_out["options_image"] = None
            temp_out["options"] = None
            temp_out["metadata"] = None
            output.append(temp_out)

    ### Reakon Assets
    asset_data_root = './data/data_assets_decorations/data_processed'
    assets_archs = os.listdir(asset_data_root)

    for arch in assets_archs:
        
        with open(os.path.join(asset_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        for i in range(3):
            temp_out = {}
            temp_out["id"] = "TSK_CR_"+temp_arch_data["id"].split('_')[1]+f"_{i}"
            temp_out["AR_id"] = temp_arch_data["id"]
            temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
            temp_out["instruction"] = temp_arch_data["instructions"]["simple"][i]
            temp_out["block_materials"] = temp_arch_data["object_materials"]
            temp_out["image"] = None
            temp_out["image_desp"] = None
            temp_out["options_image"] = None
            temp_out["options"] = None
            temp_out["metadata"] = None
            output.append(temp_out)


    sorted_output = sorted(output,key=lambda x: int(x["id"].split("_")[2]))
    with open(output_path, 'w', encoding="utf-8") as f2:
        json.dump(sorted_output, f2, indent=4)


def task_SP():
    '''
    Data Standardization for Executable Spatial Plan Generation Task
    '''

    output = []
    output_path = "./task/Task_Spatial_Planning.json"

    ### Grabcraft
    grabcraft_data_root = './data/data_grabcraft/data_processed'
    archs = os.listdir(grabcraft_data_root)
    
    for arch in archs:
        with open(os.path.join(grabcraft_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        temp_out = {}
        temp_out["id"] = "TSK_SP_"+temp_arch_data["id"].split('_')[1]+f"_0"
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        if "spatial_planning" in temp_arch_data["instructions"]:
            temp_out["instruction"] = temp_arch_data["instructions"]["spatial_planning"]
        elif "spatial_plan" in temp_arch_data["instructions"]:
            temp_out["instruction"] = temp_arch_data["instructions"]["spatial_plan"]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["image"] = "/data/images/grabcraft/"+arch.split('.json')[0].replace("'", "")+".png"
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)

        temp_out = {}
        temp_out["id"] = "TSK_SP_"+temp_arch_data["id"].split('_')[1]+f"_1"
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["simple"][0]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["image"] = "/data/images/grabcraft/"+arch.split('.json')[0]+".png"
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)
        
    ### Wiki
    wiki_data_root = './data/data_official_wiki/data_processed'
    wiki_archs = os.listdir(wiki_data_root)

    for arch in wiki_archs:
        with open(os.path.join(wiki_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        old_block_materials = temp_arch_data["block_materials"]
        new_reversed_block_materials = {}
        new_object_materials = []
        for k, v in old_block_materials.items():
            if k == "air":
                continue
            pure_k = k.split('[')[0]
            if pure_k not in new_object_materials:
                new_object_materials.append(pure_k)
            new_reversed_block_materials[v] = new_object_materials.index(pure_k)+1
        temp_out = {}
        temp_out["id"] = "TSK_SP_"+temp_arch_data["id"].split('_')[1]+f"_0"
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["spatial_plan"]
        temp_out["block_materials"] = new_object_materials
        temp_out["image"] = "/data/images/wiki/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)

        temp_out = {}
        temp_out["id"] = "TSK_SP_"+temp_arch_data["id"].split('_')[1]+f"_1"
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["simple"][0]
        temp_out["block_materials"] = new_object_materials
        temp_out["image"] = "/data/images/wiki/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)


    ### Reakon Assets
    asset_data_root = './data/data_assets_decorations/data_processed'
    assets_archs = os.listdir(asset_data_root)

    for arch in assets_archs:
        
        with open(os.path.join(asset_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)

        temp_out = {}
        temp_out["id"] = "TSK_SP_"+temp_arch_data["id"].split('_')[1]+f"_0"
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["spatial_plan"]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["image"] = "/data/images/assets/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)

        temp_out = {}
        temp_out["id"] = "TSK_SP_"+temp_arch_data["id"].split('_')[1]+f"_1"
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["simple"][0]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["image"] = "/data/images/assets/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)


    sorted_output = sorted(output,key=lambda x: int(x["id"].split("_")[2]))
    with open(output_path, 'w', encoding="utf-8") as f2:
        json.dump(sorted_output, f2, indent=4)


def task_SU():
    '''
    Data Standardization for Spatial Understanding Task
    '''

    output = []
    output_path = "./task/Task_Spatial_Understanding.json"

    ### Grabcraft
    grabcraft_data_root = './data/data_grabcraft/data_processed'
    archs = os.listdir(grabcraft_data_root)
    
    for arch in archs:
        with open(os.path.join(grabcraft_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        temp_out = {}
        temp_out["id"] = "TSK_SU_"+temp_arch_data["id"].split('_')[1]
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        if "instr_follow" in temp_arch_data["instructions"]:
            temp_out["instruction"] = temp_arch_data["instructions"]["instr_follow"]
        elif "concrete_easy" in temp_arch_data["instructions"]:
            temp_out["instruction"] = temp_arch_data["instructions"]["concrete_easy"]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["image"] = "/data/images/grabcraft/"+arch.split('.json')[0].replace("'", "")+".png"
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)

    ### Wiki
    wiki_data_root = './data/data_official_wiki/data_processed'
    wiki_archs = os.listdir(wiki_data_root)

    for arch in wiki_archs:
        with open(os.path.join(wiki_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        old_block_materials = temp_arch_data["block_materials"]
        new_reversed_block_materials = {}
        new_object_materials = []
        for k, v in old_block_materials.items():
            if k == "air":
                continue
            pure_k = k.split('[')[0]
            if pure_k not in new_object_materials:
                new_object_materials.append(pure_k)
            new_reversed_block_materials[v] = new_object_materials.index(pure_k)+1
        temp_out = {}
        temp_out["id"] = "TSK_SU_"+temp_arch_data["id"].split('_')[1]
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["instr_follow"]
        temp_out["block_materials"] = new_object_materials
        temp_out["image"] = "/data/images/wiki/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)



    ### Reakon Assets
    asset_data_root = './data/data_assets_decorations/data_processed'
    assets_archs = os.listdir(asset_data_root)

    for arch in assets_archs:
        
        with open(os.path.join(asset_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)

        temp_out = {}
        temp_out["id"] = "TSK_SU_"+temp_arch_data["id"].split('_')[1]
        temp_out["AR_id"] = temp_arch_data["id"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["instruction"] = temp_arch_data["instructions"]["instr_follow"]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["image"] = "/data/images/assets/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_desp"] = None
        temp_out["options_image"] = None
        temp_out["options"] = None
        temp_out["metadata"] = None
        output.append(temp_out)



    sorted_output = sorted(output,key=lambda x: int(x["id"].split("_")[2]))
    with open(output_path, 'w', encoding="utf-8") as f2:
        json.dump(sorted_output, f2, indent=4)



def task_SC():
    '''
    Data Standardization for Spatial Commonsense Task
    '''

    output_path = "./task/Task_Spatial_Commonsense.json"
    with open('./data/data_spatial_commonsense/task_spat_commonsense.json', 'r') as f1:
        ori_data = json.load(f1)
    output = []
    for item in ori_data:
        temp_dict = {}
        temp_dict["id"] = item["id"]
        temp_dict["env"] = item["env"]
        temp_dict["difficulty_factor"] = None
        temp_dict["instruction"] = item["question"]
        temp_dict["answer"] = item["ref_ans"]
        temp_dict["image"] = [f"/data/images/commonsense/{img}" for img in item["images"]]
        temp_dict["image_desp"] = item["image_desp"]
        temp_dict["options_image"] = None
        temp_dict["options"] = None
        temp_dict["metadata"] = None
        output.append(temp_dict)

    with open(output_path, 'w', encoding="utf-8") as f2:
        json.dump(output, f2, indent=4)



def task_SR():
    '''
    Data Standardization for Spatial Reasoning Task
    '''

    output_path = "./task/Task_Spatial_Reasoning.json"
    with open('./data/data_stimuli_vqa/SpaRea_VQAs.json', 'r') as f1:
        ori_data = json.load(f1)
    
    output = []
    for item in ori_data:
        temp_dict = {}
        temp_dict["id"] = item["id"]
        temp_dict["metadata"] = item["metadata"]
        temp_dict["difficulty_factor"] = None
        temp_dict["instruction"] = item["instruction"]
        temp_dict["answer"] = item["answer"]
        temp_dict["options_image"] = "/data/images/reasoning/"+item["options_image"]
        MR_num = item["metadata"].split("_")[0]
        temp_dict["options"] = [f"/data/images/stimuli/{MR_num}/{img}" for img in item["options"]]
        temp_dict["image_desp"] = None
        temp_dict["env"] = None
        temp_dict["image"] = None
        output.append(temp_dict)

    with open(output_path, 'w', encoding="utf-8") as f2:
        json.dump(output, f2, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scripts for MineAnyBuild tasks")
    parser.add_argument('--task', type=str, choices=['Spatial_Understanding', 'Creativity', 'Executable_Spatial_Plan_Generation', 'Spatial_Commonsense', 'Spatial_Reasoning'], required=True, help="Tasks of MineAnyBuild")
    args = parser.parse_args()

    task_map = {
        'Spatial_Understanding': lambda: task_SU(),
        'Creativity': lambda: task_CR(), 
        'Executable_Spatial_Plan_Generation': lambda: task_SP(),
        'Spatial_Commonsense': task_SC(),
        'Spatial_Reasoning': task_SR()
    }
    
    task_map[args.task]()
