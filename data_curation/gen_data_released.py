import json
import os
from PIL import Image, ImageDraw, ImageFont
import random
import shutil
import copy

#############################################################
###        Data curation codes for released data          ### 
#############################################################



#############################################################
### Data Curation for Architectures and Planning tasks
#############################################################



def address_wiki_data(arch_name, original_input):
    '''
    Data curation function for released Minecraft Wiki data.
    '''

    # the input path is obtained by the previous step.
    input_root = './benchmark_data/original_wiki_rawdata'
    output_root = './benchmark_data/original_wiki_addressed'

    if not os.path.exists(output_root):
        os.makedirs(output_root)
    with open(os.path.join(input_root, f"{arch_name}.json"), 'r') as f1:
        data = json.load(f1)
    if not data:
        return
    out = {}

    start_pos = original_input[arch_name]["start_pos"]
    end_pos = original_input[arch_name]["end_pos"]

    out["data_source"] = "Minecraft Official Wiki"
    out["metadata"] = original_input[arch_name]["metadata"]
    out["type"] = original_input[arch_name]["metadata"].split('/')[2]
    out["biome"] = original_input[arch_name]["metadata"].split('/')[1]
    out["name"] = arch_name
    block_amount = 0
    object_materials = []
    ob_cnt = 1
    object_materials_hash = {"air": -1}

    width = end_pos[0]-start_pos[0]+1
    height = end_pos[1]-start_pos[1]+1
    depth = end_pos[2]-start_pos[2]+1
    out["3d_info"] = {"width": width, "height": height, "depth": depth}
    # print(out["3d_info"])
    out_blueprint = [[[] for d in range(depth)] for h in range(height)]

    for k, v in data.items():
        w, h, d = int(k.split('_')[0])-start_pos[0], int(k.split('_')[1])-start_pos[1], int(k.split('_')[2])-start_pos[2]
        # print(w, h, d)
        if v["displayName"] == "Jigsaw Block":
            final_state = v["entity"]["value"]["final_state"]["value"]
            if final_state == "minecraft:structure_void":
                out_blueprint[h][d].append(-1)
            else:
                if final_state.split('minecraft:')[1] not in object_materials:
                    object_materials.append(final_state.split('minecraft:')[1])
                    object_materials_hash[final_state.split('minecraft:')[1]] = ob_cnt
                    ob_cnt += 1
                out_blueprint[h][d].append(object_materials_hash[final_state.split('minecraft:')[1]])
                block_amount += 1
        else:
            if v["name"] != "air":
                block_amount += 1
                obj_name = v["name"]
                if v["_properties"]:
                    temp_str = ""
                    for k1, v1 in v["_properties"].items():
                        temp_str += f"{k1}={v1},".replace("False", "false").replace("True", "true")
                    obj_name = f"{obj_name}[{temp_str[:-1]}]"
                if obj_name not in object_materials:
                    object_materials.append(obj_name)
                    object_materials_hash[obj_name] = ob_cnt
                    ob_cnt += 1
            
                out_blueprint[h][d].append(object_materials_hash[obj_name])
            else:
                out_blueprint[h][d].append(-1)
    out["object_materials"] = object_materials
    out["block_materials"] = object_materials_hash
    out["block_amount"] = block_amount
    out["blueprint"] = out_blueprint
    

    with open(os.path.join(output_root, f'{arch_name}.json'), 'w') as f2:
        json.dump(out, f2, indent=4)


def address_assets_data(arch_name, original_input):
    '''
    Data curation function for released assets data.
    '''

    input_root = './benchmark_data/original_assets_rawdata'
    output_root = './benchmark_data/original_assets_addressed'
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    json_path = os.path.join(input_root, f"{arch_name}.json")
    with open(json_path, 'r') as f1:
        data = json.load(f1)
    if not data:
        return
    out = {}

    start_pos = original_input[arch_name]["start_pos"]
    end_pos = original_input[arch_name]["end_pos"]

    out["data_source"] = "Reakon"
    out["description"] = original_input[arch_name]["description"]
    out["name"] = arch_name
    block_amount = 0
    object_materials = []
    ob_cnt = 1
    object_materials_hash = {"air": -1}

    width = end_pos[0]-start_pos[0]+1
    height = end_pos[1]-start_pos[1]+1
    depth = end_pos[2]-start_pos[2]+1
    out["3d_info"] = {"width": width, "height": height, "depth": depth}
    # print(out["3d_info"])
    out_blueprint = [[[] for d in range(depth)] for h in range(height)]

    for k, v in data.items():
        w, h, d = int(k.split('_')[0])-start_pos[0], int(k.split('_')[1])-start_pos[1], int(k.split('_')[2])-start_pos[2]
        # print(w, h, d)
        if v["displayName"] == "Jigsaw Block":
            final_state = v["entity"]["value"]["final_state"]["value"]
            if final_state == "minecraft:structure_void":
                out_blueprint[h][d].append(-1)
            else:
                if final_state.split('minecraft:')[1] not in object_materials:
                    object_materials.append(final_state.split('minecraft:')[1])
                    object_materials_hash[final_state.split('minecraft:')[1]] = ob_cnt
                    ob_cnt += 1
                out_blueprint[h][d].append(object_materials_hash[final_state.split('minecraft:')[1]])
                block_amount += 1
        else:
            if v["name"] != "air":
                block_amount += 1
                obj_name = v["name"]
                if v["_properties"]:
                    temp_str = ""
                    for k1, v1 in v["_properties"].items():
                        temp_str += f"{k1}={v1},".replace("False", "false").replace("True", "true")
                    obj_name = f"{obj_name}[{temp_str[:-1]}]"
                if obj_name not in object_materials:
                    object_materials.append(obj_name)
                    object_materials_hash[obj_name] = ob_cnt
                    ob_cnt += 1
            
                out_blueprint[h][d].append(object_materials_hash[obj_name])
            else:
                out_blueprint[h][d].append(-1)
    out["object_materials"] = object_materials
    out["block_materials"] = object_materials_hash
    out["block_amount"] = block_amount
    out["blueprint"] = out_blueprint
    

    with open(os.path.join(output_root, f'{arch_name}.json'), 'w') as f2:
        json.dump(out, f2, indent=4)



#############################################################
# Data Curation for Spatial Reasoning task (VQA)
#############################################################


def address_mental_rotation_data(arch_name, original_input):
    '''
    codes for preprocessing the mental rotation data
    '''

    input_root = './benchmark_data/original_mental_rotation_rawdata'
    output_root = './benchmark_data/original_mental_rotation_addressed'
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    json_path = os.path.join(input_root, f"{arch_name}.json")
    with open(json_path, 'r') as f1:
        data = json.load(f1)
    if not data:
        return
    out = {}

    start_pos = original_input[arch_name]["start_pos"]
    end_pos = original_input[arch_name]["end_pos"]

    out["data_source"] = "Mental Rotation"
    out["name"] = arch_name
    block_amount = 0
    object_materials = []
    ob_cnt = 1
    object_materials_hash = {"air": -1}

    width = end_pos[0]-start_pos[0]+1
    height = end_pos[1]-start_pos[1]+1
    depth = end_pos[2]-start_pos[2]+1
    out["3d_info"] = {"width": width, "height": height, "depth": depth}
    out_blueprint = [[[] for d in range(depth)] for h in range(height)]

    for k, v in data.items():
        w, h, d = int(k.split('_')[0])-start_pos[0], int(k.split('_')[1])-start_pos[1], int(k.split('_')[2])-start_pos[2]
        if v["displayName"] == "Jigsaw Block":
            final_state = v["entity"]["value"]["final_state"]["value"]
            if final_state == "minecraft:structure_void":
                out_blueprint[h][d].append(-1)
            else:
                if final_state.split('minecraft:')[1] not in object_materials:
                    object_materials.append(final_state.split('minecraft:')[1])
                    object_materials_hash[final_state.split('minecraft:')[1]] = ob_cnt
                    ob_cnt += 1
                out_blueprint[h][d].append(object_materials_hash[final_state.split('minecraft:')[1]])
                block_amount += 1
        else:
            if v["name"] != "air":
                block_amount += 1
                if v["name"] not in object_materials:
                    object_materials.append(v["name"])
                    object_materials_hash[v["name"]] = ob_cnt
                    ob_cnt += 1
            out_blueprint[h][d].append(object_materials_hash[v["name"]])
    out["object_materials"] = object_materials
    out["block_materials"] = object_materials_hash
    out["block_amount"] = block_amount
    out["blueprint"] = out_blueprint
    

    with open(os.path.join(output_root, f'{arch_name}.json'), 'w') as f2:
        json.dump(out, f2, indent=4)



def concat_image(output_root, output_name, base_img_path, option_img_paths, gap=5, font_size=40):
    '''
    codes for concatenating images for Spatial Reasoning tasks, with multiple choices from A,B,C,D.
    '''

    base_img = Image.open(base_img_path)
    option_imgs = [Image.open(path) for path in option_img_paths]
    
    opt_width, opt_height = option_imgs[0].size
    base_img = base_img.resize((opt_width, opt_height)) 
    
    middle_gap = gap * 2  
    caption_height = 30  
    
    total_width = opt_width + middle_gap + (opt_width * 2 + gap)
    total_height = max(
        (opt_height * 2 + gap * 1),  
        (opt_height + caption_height)  
    )
 
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    base_y = (total_height - opt_height - caption_height) // 2
    canvas.paste(base_img, (0, base_y))
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    # text_width, text_height = draw.textsize("Original Image", font=font)
    text_bbox = draw.textbbox((0, 0), "Original Image", font=font)
    text_width = text_bbox[2] - text_bbox[0]  # right - left
    text_height = text_bbox[3] - text_bbox[1] # bottom - top
    text_x = (opt_width - text_width) // 2  
    text_y = base_y + opt_height + 5  
    draw.text((text_x, text_y), "Original Image", fill=(0,0,0), font=font)
    
    option_start_x = opt_width + middle_gap
    positions = [
        (option_start_x, 0),
        (option_start_x + opt_width + gap, 0),
        (option_start_x, opt_height + gap),
        (option_start_x + opt_width + gap, opt_height + gap)
    ]
    
    for idx, (img, pos) in enumerate(zip(option_imgs, positions)):
        img = img.resize((opt_width, opt_height))
        canvas.paste(img, pos)

        label = chr(65 + idx)  # A,B,C,D
        draw.text((pos[0]+10, pos[1]+10), label, 
                 fill=(0,0,0), font=font,
                 stroke_width=1, stroke_fill=(255,255,255))
    
    output_path = f"{output_root}/{output_name}.jpg"
    canvas.save(output_path, quality=95)
    print(f"{output_path}")
    return f"{output_name}.jpg"


def concat_single_image(output_root, output_name, base_img_path, input_img_path, gap=10):
    '''
    codes for concatenating images for Spatial Reasoning tasks, for True/False VQAs.
    '''

    base_img = Image.open(base_img_path)
    input_img = Image.open(input_img_path)
    
    base_width, base_height = base_img.size
    input_img = input_img.resize((base_width, base_height))
    
    total_width = base_width + gap + base_width
    total_height = base_height
    
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    canvas.paste(base_img, (0, 0))
    canvas.paste(input_img, (base_width + gap, 0))
    
    output_path = f"{output_root}/{output_name}.jpg"
    canvas.save(output_path)
    print(f"{output_path}")
    return f"{output_name}.jpg"


def generate_mental_rotation_task_1():
    '''
    codes for generating VQA data for Spatial Reasoning task (question type 1)
    question: Which option is the same as the original image, aside from its orientation?
    '''

    input_root = "./stimuli"
    output_root = "./stimuli_vqa"
    task_cnt = 1
    qa_out = []

    same_group = [["N_1", "RX_0", "RY_2", "RZ_3"],["N_1", "RX_0", "RY_3", "RZ_2"],["N_1", "RX_2", "RY_0", "RZ_3"],["N_1", "RX_2", "RY_3", "RZ_0"],["N_1", "RX_3", "RY_0", "RZ_2"],["N_1", "RX_3", "RY_2", "RZ_0"],["N_2", "RX_0", "RY_1", "RZ_3"],["N_2", "RX_0", "RY_3", "RZ_1"],["N_2", "RX_1", "RY_0", "RZ_3"],["N_2", "RX_1", "RY_3", "RZ_0"],["N_2", "RX_3", "RY_0", "RZ_1"],["N_2", "RX_3", "RY_1", "RZ_0"],["N_3", "RX_0", "RY_1", "RZ_2"],["N_3", "RX_0", "RY_2", "RZ_1"],["N_3", "RX_1", "RY_0", "RZ_2"],["N_3", "RX_1", "RY_2", "RZ_0"],["N_3", "RX_2", "RY_0", "RZ_1"],["N_3", "RX_2", "RY_1", "RZ_0"]]

    for num in range(48):
        random.seed(num+1000)
        mr_dir = f"{input_root}/MR{num+1}"
        for sg in range(len(same_group)):
            out = {}
            out["id"] = f"TSK_SR_1_{task_cnt:04d}"
            out["instruction"] = "Which option is the same as the original image, aside from its orientation?"
            shuffled = random.sample(same_group[sg], len(same_group[sg])) 
            out["metadata"] = f'MR{num+1}_'+''.join([item.replace("_", "") for item in shuffled])
            out["original"] = "N_0.png"
            out["answer"] = chr(65+shuffled.index(same_group[sg][0]))
            pre_concat_images = [f"{mr_dir}/{item}.png" for item in shuffled]
            out["options_image"] = concat_image(output_root, out["id"], f"{mr_dir}/N_0.png", pre_concat_images)
            out["options"] = [f"{item}.png" for item in shuffled]
            qa_out.append(out)
            task_cnt += 1

    with open(f"{output_root}/SpaRea_VQAs_1.json", 'w') as f1:
        json.dump(qa_out, f1, indent=4)


def generate_mental_rotation_task_2():
    '''
    codes for generating VQA data for Spatial Reasoning task (question type 2)
    question: Which option is different from the original image, aside from its orientation?
    '''
    
    input_root = "./stimuli"
    output_root = "./stimuli_vqa"
    task_cnt = 1
    qa_out = []

    same_group = [["N_1", "N_2", "N_3", "RX_0"],["N_1", "N_2", "N_3", "RX_1"],["N_1", "N_2", "N_3", "RX_2"],["N_1", "N_2", "N_3", "RX_3"],["N_1", "N_2", "N_3", "RY_0"],["N_1", "N_2", "N_3", "RY_1"],["N_1", "N_2", "N_3", "RY_2"],["N_1", "N_2", "N_3", "RY_3"],["N_1", "N_2", "N_3", "RZ_0"],["N_1", "N_2", "N_3", "RZ_1"],["N_1", "N_2", "N_3", "RZ_2"],["N_1", "N_2", "N_3", "RZ_3"]]

    for num in range(48):
        random.seed(num+2000)
        mr_dir = f"{input_root}/MR{num+1}"
        for sg in range(len(same_group)):
            out = {}
            out["id"] = f"TSK_SR_2_{task_cnt:04d}"
            out["instruction"] = "Which option is different from the original image, aside from its orientation?"
            shuffled = random.sample(same_group[sg], len(same_group[sg])) 
            out["metadata"] = f'MR{num+1}_'+''.join([item.replace("_", "") for item in shuffled])
            out["original"] = "N_0.png"
            out["answer"] = chr(65+shuffled.index(same_group[sg][3]))
            pre_concat_images = [f"{mr_dir}/{item}.png" for item in shuffled]
            out["options_image"] = concat_image(output_root, out["id"], f"{mr_dir}/N_0.png", pre_concat_images)
            out["options"] = [f"{item}.png" for item in shuffled]
            qa_out.append(out)
            task_cnt += 1

    with open(f"{output_root}/SpaRea_VQAs_2.json", 'w') as f1:
        json.dump(qa_out, f1, indent=4)


def generate_mental_rotation_task_3():
    '''
    codes for generating VQA data for Spatial Reasoning task (question type 3)
    question: Can the right image be obtained by rotating the original image? True or False?
    '''

    input_root = "./stimuli"
    output_root = "./stimuli_vqa"
    task_cnt = 1
    qa_out = []

    same_group = ["N_1", "N_2", "N_3"]
    diff_group = ["RX_1", "RY_2", "RZ_3"]
    for num in range(48):
        random.seed(num+3000)
        mr_dir = f"{input_root}/MR{num+1}"
        for sg in range(len(same_group)):
            out = {}
            out["id"] = f"TSK_SR_3_{task_cnt:04d}"
            out["instruction"] = "Can the right image be obtained by rotating the original image? True or False?"
            shuffled = same_group[sg]
            out["metadata"] = f'MR{num+1}_'+''.join([item.replace("_", "") for item in shuffled])
            out["original"] = "N_0.png"
            out["answer"] = "Ture"
            pre_concat_image = f"{mr_dir}/{shuffled}.png"
            out["options_image"] = concat_single_image(output_root, out["id"], f"{mr_dir}/N_0.png", pre_concat_image)
            out["options"] = f"{shuffled}.png"
            qa_out.append(out)
            task_cnt += 1

        for sg in range(len(diff_group)):
            out = {}
            out["id"] = f"TSK_SR_3_{task_cnt:04d}"
            out["instruction"] = "Can the right image be obtained by rotating the original image? True or False?"
            shuffled = diff_group[sg]
            out["metadata"] = f'MR{num+1}_'+''.join([item.replace("_", "") for item in shuffled])
            out["original"] = "N_0.png"
            out["answer"] = "False"
            pre_concat_image = f"{mr_dir}/{shuffled}.png"
            out["options_image"] = concat_single_image(output_root, out["id"], f"{mr_dir}/N_0.png", pre_concat_image)
            out["options"] = f"{shuffled}.png"
            qa_out.append(out)
            task_cnt += 1

    with open(f"{output_root}/SpaRea_VQAs_3.json", 'w') as f1:
        json.dump(qa_out, f1, indent=4)



#############################################################
### Data Standardization for buildings/assets
#############################################################

def data_archs():
    '''
    Data standardization for released architectures and assets data.
    '''

    output = []
    output_path = "./data/architectures.json"
    image_output_path = './data/images'
    if not os.path.exists(os.path.join(image_output_path, 'grabcraft')):
        os.makedirs(os.path.join(image_output_path, 'grabcraft'))
    if not os.path.exists(os.path.join(image_output_path, 'wiki')):
        os.makedirs(os.path.join(image_output_path, 'wiki'))
    if not os.path.exists(os.path.join(image_output_path, 'assets')):
        os.makedirs(os.path.join(image_output_path, 'assets'))

    ### Grabcraft
    raw_data_root = "./data/raw_html_data"
    grabcraft_data_root = './data/data_grabcraft/data_processed'
    archs = os.listdir(grabcraft_data_root)
    
    for arch in archs:
        temp_out = {}
        with open(os.path.join(grabcraft_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        temp_out["id"] = temp_arch_data["id"]
        temp_out["name"] = arch.split('.json')[0].replace("'", "")
        temp_out["description"] = temp_arch_data["name"]
        temp_out["data_resource"] = "Grabcraft"
        temp_out["3d_info"] = temp_arch_data["3d_info"]
        temp_out["type"] = temp_arch_data["type"]
        temp_out["biome"] = None
        img_name = temp_arch_data["image_urls"][0].split('/')[-1]
        img_path = os.path.join(raw_data_root, temp_arch_data["type"], arch.split('.json')[0], img_name)
        output_img_path = os.path.join(image_output_path, "grabcraft", temp_out["name"]+'.png')
        shutil.copy(img_path, output_img_path)
        temp_out["image"] = "./images/grabcraft/"+temp_out["name"]+".png"
        temp_out["image_urls"] = temp_arch_data["image_urls"]
        temp_out["metadata"] = temp_arch_data["data_source"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["blueprint"] = temp_arch_data["blueprint"]
        output.append(temp_out)

    ### Wiki
    wiki_data_root = './data/data_official_wiki/data_processed'
    wiki_images_root = './data/data_official_wiki'
    wiki_archs = os.listdir(wiki_data_root)

    for arch in wiki_archs:
        temp_out = {}
        with open(os.path.join(wiki_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        temp_out["id"] = temp_arch_data["id"]
        temp_out["name"] = arch.split('.json')[0]
        temp_out["description"] = temp_arch_data["name"]
        temp_out["data_resource"] = "Minecraft Official Wiki"
        temp_out["3d_info"] = temp_arch_data["3d_info"]
        temp_out["type"] = temp_arch_data["type"]
        temp_out["biome"] = temp_arch_data["biome"]
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        img_path = os.path.join(wiki_images_root, temp_arch_data["image"])
        output_img_path = os.path.join(image_output_path, "wiki", temp_arch_data["image"].split('/')[1])
        shutil.copy(img_path, output_img_path)
        temp_out["image"] = "./images/wiki/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_urls"] = None
        temp_out["metadata"] = temp_arch_data["metadata"]

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
        new_blueprint = copy.deepcopy(temp_arch_data["blueprint"])
        for i in range(len(temp_arch_data["blueprint"])):
            for j in range(len(temp_arch_data["blueprint"][i])):
                for k in range(len(temp_arch_data["blueprint"][i][j])):
                    if temp_arch_data["blueprint"][i][j][k] == -1:
                        continue
                    new_blueprint[i][j][k] = new_reversed_block_materials[temp_arch_data["blueprint"][i][j][k]]

        temp_out["block_materials"] = new_object_materials
        temp_out["blueprint"] = new_blueprint
        output.append(temp_out)

    ### Reakon Assets
    asset_data_root = './data/data_assets_decorations/data_processed'
    asset_images_root = './data/data_assets_decorations'
    assets_archs = os.listdir(asset_data_root)

    for arch in assets_archs:
        temp_out = {}
        with open(os.path.join(asset_data_root, arch), 'r') as f1:
            temp_arch_data = json.load(f1)
        temp_out["id"] = temp_arch_data["id"]
        temp_out["name"] = arch.split('.json')[0]
        temp_out["description"] = temp_arch_data["description"]
        temp_out["data_resource"] = "Reakon"
        temp_out["3d_info"] = temp_arch_data["3d_info"]
        temp_out["type"] = None
        temp_out["biome"] = None
        temp_out["difficulty_factor"] = temp_arch_data["difficulty_factor"]
        img_path = os.path.join(asset_images_root, temp_arch_data["image"])
        output_img_path = os.path.join(image_output_path, "assets", temp_arch_data["image"].split('/')[1])
        shutil.copy(img_path, output_img_path)
        temp_out["image"] = "./images/assets/"+temp_arch_data["image"].split('/')[1]
        temp_out["image_urls"] = None
        temp_out["metadata"] = None
        temp_out["block_materials"] = temp_arch_data["object_materials"]
        temp_out["blueprint"] = temp_arch_data["blueprint"]
        output.append(temp_out)


    sorted_output = sorted(output,key=lambda x: int(x["id"].split("_")[1]))
    with open(output_path, 'w', encoding="utf-8") as f2:
        json.dump(sorted_output, f2, indent=4)



if __name__ == "__main__":
    ### Functions for data curation for released data of planning tasks
    # address_wiki_data()
    # address_assets_data()
    # data_archs()

    ### Functions for data curation for released data of spatial reasoning tasks
    address_mental_rotation_data()
    # generate_mental_rotation_task_1()
    # generate_mental_rotation_task_2()
    # generate_mental_rotation_task_3()


