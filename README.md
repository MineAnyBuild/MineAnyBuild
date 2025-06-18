<div align="center">
<h2 align="center">
   <b>MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents
   <!-- <br /> <font size=3>Under Review</font></b>  -->
</h2>
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
- [05/2025] [Arxiv paper](https://arxiv.org/abs/2505.20148) released.



## 🚨 TODOs (Urgency, To-be-completed in June 2025)
**Sorry for the delay due to some final exams. We'll make up this part as soon as possible.**
- [ ] Remaining codes for evaluation of MLLM-based agents (5 tasks)
- [ ] Remaining codes for data curation (w/ example JSON files)
- [x] Remaining codes for inference of MLLM-based agents (5 tasks)
- [x] Icons&urls for HuggingFace datasets, project webpage.
- [ ] Upload files to Google Drive (e.g., pure map for evaluation).


## ✅ TODOs (Future, in several months)

- [ ] Docs of instructions for Replay Mod (June~July 2025)
- [ ] Optimization of the use of visualization tools (timer in Replay Mod code, viewer)
- [ ] RL environment codes for Mineflayer (Jun.~Aug. 2025)
- [ ] Fix known bugs
- [ ] Debugging and provide adaptation to normal python codes (w/o Jupyter notebook)
- [ ] MineRL/MineDojo codes (in several months)



______________________________________________________________________


# Contents
- [Contents](#contents)
- [Installation](#installation)
  - [Install Minecraft](#install-minecraft)
  - [Install packages](#install-packages)
  - [Create a Conda env and packages](#create-a-conda-env-and-packages)
- [Running codes](#running-codes)
  - [Evaluation](#evaluation)
    - [Executable Spatial Plan Generation, Creativity and Spatial Understanding tasks](#executable-spatial-plan-generation-creativity-and-spatial-understanding-tasks)
    - [Spatial Reasoning task](#spatial-reasoning-task)
    - [Spatial Commonsense task](#spatial-commonsense-task)
  - [Data curation](#data-curation)
  - [Inference of MLLM-based agents](#inference-of-mllm-based-agents)
    - [Proprietary MLLMs](#proprietary-mllms)
    - [Open-source MLLMs](#open-source-mllms)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)




# Installation

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
conda create -n mineflayer python=3.10
conda activate mineflayer
pip install -r requirements.txt
```



# Running codes

(**Sorry for the delay due to some final exams. We'll make up this part as soon as possible.**)

## Evaluation
For different tasks in our MineAnyBuild, please conduct evaluation according to the following steps respectively.

### Executable Spatial Plan Generation, Creativity and Spatial Understanding tasks


1. Start the Minecraft game and record the video with Replay Mod (recommended currently). 
    <span style="font-family: 'Georgia', serif; font-size: 14px;">1\) Start the recoding in Pause Menu of Minecraft game.
    2\) Run `/mineanybuild/mineflayer.ipynb` and specify the starting frame.
    3\) Stop the recording when the notebook cells finish executing.
    4\) Save and render the video in Replay Mod Menu.</span>
    <span style="font-family: 'Times New Roman', serif; font-size: 14px;"><i> (For the concrete instructions, please refer to the [Replay Mod documentation](https://www.replaymod.com/) and Section B.3.2 in the Supplementary Material. We will provide a Doc for usage instructions of Replay Mod in several weeks.)</i></span>
   

2. Video Segmentation
    <span style="font-family: 'Georgia', serif; font-size: 14px;">Split the video into multiple frames and match each frame to its corresponding building structure. We provide an example of using this in `/mineanybuild/utils.py`. You can use it to match automatically or conduct manual matching selection if less data is to be tested.</span>



3. Run the functions/scripts in `/mineanybuild/evaluator.py` following the below instructions.
    ```
    python /mineanybuild/evaluator.py --task [Spatial_Understanding|Creativity|Executable_Spatial_Plan_Generation]
    ```


### Spatial Reasoning task


1. Cauculate the SR(Success Rate) of the Spatial Reasoning task by running the following code.
```

```


### Spatial Commonsense task
1. 
    ```
    python /mineanybuild/evaluator.py --task Spatial_Commonsense
    ```

2. Cauculate the SR(Success Rate) of the Spatial Reasoning task by running the following code.





## Data curation


1. run `/data_curation/customize_data.ipynb` to


2. instruction


3. difficulty factor




## Inference of MLLM-based agents
Run the following code to perform inference for MLLM-based agents.

### Proprietary MLLMs

Run `prompter.py` specifying the task of MineAnyBuild. Please specify the input and output file/directories for each task function.

```
python /mineanybuild/prompter.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
```

### Open-source MLLMs

Run the following code based on the type of open-source MLLMs you want to run.

```
python /mineanybuild/internvl.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
python /mineanybuild/qwenvl.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
python /mineanybuild/llavaov.py --task [Spatial_Understanding|Spatial_Reasoning|Creativity|Executable_Spatial_Plan_Generation|Spatial_Commonsense]
```




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
Some of the codes are built upon [APT](https://github.com/spearsheep/APT-Architectural-Planning-LLM-Agent). Thanks them for their great works!