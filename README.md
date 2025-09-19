<div align="center">
<h2 align="center">
   <b>MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents</b>
</h2>
<b><font size=3>NeurIPS 2025 Datasets and Benchmarks Track</font></b> 
<div>
<a href="http://sadil13.github.io/" target="_blank">Ziming&#160;Wei</a><sup>1*</sup>,
<a href="https://expectorlin.github.io/" target="_blank">Bingqian&#160;Lin</a><sup>2*</sup>,
<a href="https://openreview.net/profile?id=~Zijian_Jiao1" target="_blank">Zijian&#160;Jiao</a><sup>1*</sup>,
<a href="https://scholar.google.com/citations?user=jV19-sIAAAAJ" target="_blank">Yunshuang&#160;Nie</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=tHRExZ8AAAAJ" target="_blank">Liang&#160;Ma</a><sup>3</sup>,
<br>
<a href="https://openreview.net/profile?id=~Yuecheng_Liu1" target="_blank">Yuecheng&#160;Liu</a><sup>4</sup>,
<a href="https://scholar.google.com/citations?user=ny9KAREAAAAJ" target="_blank">Yuzheng&#160;Zhuang</a><sup>4</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=voxznZAAAAAJ">Xiaodan&#160;Liang</a><sup>1&#9993</sup>
</div>
<sup>1</sup>Shenzhen Campus of Sun Yat-Sen University,&#160;
<sup>2</sup>Shanghai Jiao Tong University,&#160;
<br>
<sup>3</sup>Mohamed bin Zayed University of Artificial Intelligence,&#160;
<sup>4</sup>Huawei Noah’s Ark Lab
<br />
<sup>*&#160;</sup>Equal contribution&#160;&#160;</span>
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://mineanybuild.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Project-MineAnyBuild-red" alt="Project"></a>
    <a href="https://arxiv.org/abs/2505.20148" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://huggingface.co/datasets/SaDil/MineAnyBuild">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-orange" alt="Dataset"></a>
    <a href="https://github.com/MineAnyBuild/MineAnyBuild">
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version"></a>
</div>

</div>

______________________________________________________________________

<font size=2>
Spatial Planning is a crucial part in the field of spatial intelligence, which requires the understanding and planning about object arrangements in space perspective. AI agents with the spatial planning ability can better adapt to various real-world applications, including robotic manipulation, automatic assembly, urban planning <i>etc</i>.  Recent works have attempted to construct benchmarks for evaluating the spatial intelligence of Multimodal Large Language Models (MLLMs). Nevertheless, these benchmarks primarily focus on spatial reasoning based on typical Visual Question-Answering (VQA) forms, which suffers from the gap between abstract spatial understanding and concrete task execution. In this work, we take a step further to build a comprehensive benchmark called <b>MineAnyBuild</b>, aiming to evaluate the spatial planning ability of open-world AI agents in the <i>Minecraft</i> game. Specifically, MineAnyBuild requires an agent to generate <i>executable architecture building plans</i> based on the given multi-modal human instructions. It involves 4,000 curated spatial planning tasks and also provides a paradigm for infinitely expandable data collection by utilizing rich player-generated content. MineAnyBuild evaluates spatial planning through four core supporting dimensions: spatial understanding, spatial reasoning, creativity, and spatial commonsense. Based on MineAnyBuild, we perform a comprehensive evaluation for existing MLLM-based agents, revealing the severe limitations but enormous potential in their spatial planning abilities. We believe our MineAnyBuild will open new avenues for the evaluation of spatial intelligence and help promote further development for open-world AI agents capable of spatial planning.</font>

![motivation](assets/overview.png)


## 📰 Updates
- [09/2025] 🎉🎉🎉 Our paper has been accepted as Poster by **NeurIPS 2025 Datasets and Benchmarks Track**.
- [05/2025] [Arxiv paper](https://arxiv.org/abs/2505.20148) released.



## 🚨 TODOs (all done and these items will be deleted in few weeks)
- [x] Remaining codes for evaluation of MLLM-based agents (5 tasks).
- [x] Remaining codes for data curation (w/ example JSON files).
- [x] Remaining codes for inference of MLLM-based agents (5 tasks).
- [x] Icons&urls for HuggingFace datasets, project webpage.
- [x] Upload files to Hugging Face (e.g., pure map for evaluation).
- [x] Curation codes for released data.


______________________________________________________________________


# Contents
- [Contents](#contents)
- [🔧Installation](#installation)
  - [Install Minecraft](#install-minecraft)
  - [Install packages](#install-packages)
  - [Create a Conda env and packages](#create-a-conda-env-and-packages)
- [📥Downloads](#downloads)
- [🚀Running codes](#running-codes)
  - [📊Evaluation](#evaluation)
    - [Executable Spatial Plan Generation, Creativity and Spatial Understanding tasks](#executable-spatial-plan-generation-creativity-and-spatial-understanding-tasks)
    - [Spatial Reasoning task](#spatial-reasoning-task)
    - [Spatial Commonsense task](#spatial-commonsense-task)
  - [🗃️Data curation](#️data-curation)
    - [Executable Spatial Plan Generation, Creativity and Spatial Understanding tasks](#executable-spatial-plan-generation-creativity-and-spatial-understanding-tasks-1)
    - [Spatial Reasoning task](#spatial-reasoning-task-1)
    - [Data standardization](#data-standardization)
  - [🤖Inference of MLLM-based agents](#inference-of-mllm-based-agents)
    - [Proprietary MLLMs](#proprietary-mllms)
    - [Open-source MLLMs](#open-source-mllms)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)




# 🔧Installation

## Install Minecraft

Ensure you have a valid Microsoft account and Minecraft JAVA Edition.

1. Visit the [Minecraft Official Website](https://www.minecraft.net/en-us) and navigate to the "Get Minecraft" section.
2. Select your platform (Windows, Linux or macOS) and choose **JAVA Edition**.
3. Download the launcher and install it, and then log in using your Microsoft account.
4. Start the launcher, select the release version (1.20.4) compatible with Mineflayer and click **Play** to load the game.
5. Load the test environment map we provide (recommanded) or create a new world with settings.
6. Open the **Pause Menu**, click the **Open to LAN**, and following the settings:
     - Take note of the **Port Number** displayed (in the lower versions of Minecraft) or enter a **Port Number** that is consistent with the code (in the latest versions of Minecraft).
     - Set the game mode to **Creative**.
     - [IMPORTANT] Enable **Cheats** to allow command usage required for MLLM-based agents.

For more detailed instructions, visit the [Minecraft Help Center](https://help.minecraft.net/hc/en-us) for help.


## Install packages

1. Download and install the appropriate version of [Node.js](https://nodejs.org/en).
2. Run the following command to install [Mineflayer](https://github.com/PrismarineJS/mineflayer) on your terminal:
```
npm install mineflayer
```

## Create a Conda env and packages

```
conda create -n mineanybuild python=3.10
conda activate mineanybuild
pip install -r requirements.txt
```

# 📥Downloads

Download Links |  Description |
:--- | :--- |
[Hugging Face](https://huggingface.co/datasets/SaDil/MineAnyBuild)|Datasets for MineAnyBuild tasks, including task instructions, architectures and stimuli data. |
[Hugging Face](https://huggingface.co/datasets/SaDil/MineAnyBuild)|Pure maps in Maps_Evaluation.zip for evaluation. |
[Hugging Face](https://huggingface.co/datasets/SaDil/MineAnyBuild-Raw_data)|Raw HTML data fetched from Grabcraft, namely Grabcraft_raw_data.rar.|
<!-- [Hugging Face](111)|111| -->



# 🚀Running codes

<!-- (**Sorry for the delay due to some final exams. We'll make up this part as soon as possible.**) -->

## 📊Evaluation
You should run the [Inference of MLLM-based agents](#inference-of-mllm-based-agents) first to generate the responses of MLLM-based agents.

For different tasks in our MineAnyBuild, please conduct evaluation according to the following steps respectively.

### Executable Spatial Plan Generation, Creativity and Spatial Understanding tasks


1. Start the Minecraft game and record the video with Replay Mod (recommended currently). 
    <span style="font-family: 'Georgia', serif; font-size: 14px;">
    1\) Start the recording in Pause Menu of Minecraft game.
    2\) Run `/mineanybuild/mineflayer.ipynb` and specify the starting frame.
    3\) Stop the recording when the notebook cells finish executing.
    4\) Save and render the video in Replay Mod Menu.</span>
    <span style="font-family: 'Times New Roman', serif; font-size: 14px;"><i> (For the concrete instructions, please refer to the [Replay Mod documentation](https://www.replaymod.com/) and Section B.3.2 in the Supplementary Material. We provide a document for usage instructions of Replay Mod in [docs/replay_mod.md](docs/replay_mod.md).)</i></span>
   

2. Video Segmentation
    <span style="font-family: 'Georgia', serif; font-size: 14px;">Split the video into multiple frames and match each frame to its corresponding building structure. We provide an example of using this in `frame_clipper()` and `select_frames()` functions of `/mineanybuild/utils.py`. You can use it to match automatically or conduct manual matching selection (recommended for accuracy) if less data is to be tested.</span>


3. Run the functions/scripts in `/mineanybuild/evaluator.py` following the below instructions.
    ```
    python /mineanybuild/evaluator.py --task [Spatial_Understanding|Creativity|Executable_Spatial_Plan_Generation]
    ```


4. Run the function `process_json_critic_scores()` in `/mineanybuild/utils.py` to process the JSON files generated by the previous step.

5. Run the functions `calculate_scores_creativity()`, `calculate_scores_spatial_plan()`, `calculate_scores_spatial_understanding()` to calculate the final scores of the three tasks respectively.
    


### Spatial Reasoning task

1. Run the function `calculate_scores_spatial_reasoning()` in `/mineanybuild/evaluator.py` to calculate the *Accuracy* of the Spatial Reasoning task.



### Spatial Commonsense task
1. Run the functions `critic_spatial_commonsense()` (or `critic_spatial_commonsense_opensource()`) in `/mineanybuild/evaluator.py` following the below instructions.
    ```
    python /mineanybuild/evaluator.py --task Spatial_Commonsense
    ```

2. Run the function `calculate_scores_spatial_commonsense()` in `/mineanybuild/evaluator.py` to calculate the scores of the Spatial Commonsense task.



## 🗃️Data curation

### Executable Spatial Plan Generation, Creativity and Spatial Understanding tasks

For the released data on HuggingFace, you can refer the codes in the dirpath `/data_curation`, e.g., [data_curation/gen_data_released.py](data_curation/gen_data_released.py).

For data you want to curate by yourself, please follow the below steps.

1. Prepare your own architectures/buildings/assets in your Minecraft maps. You can follow the template in the example file in `/examples/village_input.json` to annotate your data. 

    ```js
    "plains_meeting_point_3": {       // architecture name
        "start_pos": [1, 5, -796],    // the X,Y,Z position of the left-bottom block of the architecture
        "end_pos": [11, 13, -786]     // the X,Y,Z position of the right-up block of the architecture
    },
    ...
    ```

    By manually annotating the above data, you can directly obtain curated data through the following automated pipeline.

2. Run `/data_curation/customize_data.ipynb` to obtain the curated data. 

   *Optional: If the JSON format is not completely correct, run the function `process_json()` in `/data_curation/utils.py` to process the data and format it in JSON.*


3. Run the main function `curate_architecture_datasets()` for reference in `/data_curation/gen_data_customized.py` to curate the datasets and generate instructions for different tasks.


4. [Optional] Calculate the difficulty factor of new data.
    ```
    python /data_curation/calculate_difficulty.py
    ```


### Spatial Reasoning task

For the released data on HuggingFace, you can refer codes in [data_curation/gen_data_released.py](data_curation/gen_data_released.py).

For data you want to curate by yourself, please follow the below steps.


1. Prepare your own stimuli structures built in your Minecraft map. You can follow the template in the example file in `/examples/mental_rotation_input.json` to annotate your data. 

2. Run `/data_curation/customize_data.ipynb` to obtain the curated data. 

   *Optional: If the JSON format is not completely correct, run the function `process_json()` in `/data_curation/utils.py` to process the data and format it in JSON.*


3. Run the main function `curate_spatial_reasoning_datasets()` for reference in `/data_curation/gen_data_customized.py` to curate the datasets and generate VQAs data for this task.



### Data standardization
Run the functions/scripts in `/data_curation/unify_data.py` following the below instructions.

```
python unify_data.py --task [Spatial_Understanding|Creativity|Executable_Spatial_Plan_Generation|Spatial_Reasoning|Spatial_Commonsense]
```


## 🤖Inference of MLLM-based agents
Specify your API keys for each models.

Run the following codes to perform inference for MLLM-based agents.

### Proprietary MLLMs

1. Run `/mineanybuild/prompter.py` specifying the task of MineAnyBuild. Please specify the input and output file/directories for each task function.

```
python /mineanybuild/prompter.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
```


2. Run the `json_parser_blueprint()` function in `/mineanybuild/utils.py` to transform the response of MLLM-based agents into *blueprint* 3D matrix in JSON format.



### Open-source MLLMs

1. Run the following codes based on the type of open-source MLLMs you want to run.

```
python /mineanybuild/internvl.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
python /mineanybuild/qwenvl.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
python /mineanybuild/llavaov.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
```

2. Run the `json_parser_blueprint()` function in `/mineanybuild/utils.py` to transform the response of MLLM-based agents into *blueprint* 3D matrix in JSON format.



<details>
    <summary><strong>✅Planned Future Updates</strong></summary>
    <ul>
        <li>More detailed annotations</li>
        <li>Docs of instructions for Replay Mod</li>
        <li>Optimization of the use of visualization tools (timer in Replay Mod code, viewer)</li>
        <li>RL environment codes for Mineflayer</li>
        <li>Fix known bugs</li>
        <li>Debugging and provide adaptation to normal python codes (w/o Jupyter notebook)</li>
        <li>MineRL/MineDojo codes</li>
    </ul>
</details>




# Citation
If you find this work useful, please consider citing:
```bibtex
@article{wei2025mineanybuild,
  title={MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents},
  author={Wei, Ziming and Lin, Bingqian and Jiao, Zijian and Nie, Yunshuang and Ma, Liang and Liu, Yuecheng and Zhuang, Yuzheng and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2505.20148},
  year={2025}
}
```

# Acknowledgement
Some of the codes are built upon [APT](https://github.com/spearsheep/APT-Architectural-Planning-LLM-Agent). Part of our datasets are built upon [GrabCraft](https://www.grabcraft.com), [Minecraft Wiki](https://minecraft.fandom.com/wiki/Minecraft_Wiki) and [Raekon](https://www.patreon.com/Raekon). Thanks them for their great works and resources!