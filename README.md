# Example-based Motion Synthesis via Generative Motion Matching, ACM Transactions on Graphics (Proceedings of SIGGRAPH 2023)

#####  <p align="center"> [Weiyu Li*](https://wyysf-98.github.io/), [Xuelin Chen*†](https://xuelin-chen.github.io/), [Peizhuo Li](https://peizhuoli.github.io/), [Olga Sorkine-Hornung](https://igl.ethz.ch/people/sorkine/), [Baoquan Chen](https://cfcs.pku.edu.cn/baoquan/)</p>
 
#### <p align="center">[Project Page](https://wyysf-98.github.io/GenMM) | [ArXiv](https://arxiv.org/abs/2306.00378) | [Paper](https://wyysf-98.github.io/GenMM/paper/Paper_high_res.pdf) | [Video](https://youtu.be/lehnxcade4I)</p>

<p align="center">
  <img src="https://wyysf-98.github.io/GenMM/assets/images/teaser.png"/>
</p>

<p align="center"> All Code and demo will be released in this week(still ongoing...) 🏗️ 🚧 🔨</p>

- [x] Release main code
- [x] Release blender addon
- [x] Detailed README and installation guide
- [ ] Release skeleton-aware component, WIP as we need to split the joints into groups manually.
- [ ] Release codes for evaluation

## Prerequisite

<details> <summary>Setup environment</summary>

:smiley: We also provide a Dockerfile for easy installation, see [Setup using Docker](./docker/README.md).

 - Python 3.8
 - PyTorch 1.12.1
 - [unfoldNd](https://github.com/f-dangel/unfoldNd)

Clone this repository.

```sh
git clone git@github.com:wyysf-98/GenMM.git
```

Install the required packages.

```sh
conda create -n GenMM python=3.8
conda activate GenMM
conda install -c pytorch pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 && \
pip install -r docker/requirements.txt
pip install torch-scatter==2.1.1
```

</details>

## Quick inference demo
For local quick inference demo using .bvh file, you can use

```sh
python run_random_generation.py -i './data/Malcolm/Gangnam-Style.bvh'
```
More configuration can be found in the `run_random_generation.py`.
We use an Apple M1 and NVIDIA Tesla V100 with 32 GB RAM to generate each motion, which takes about ~0.2s and ~0.05s as mentioned in our paper.

## Blender add-on
You can install and use the blender add-on with easy installation as our method is efficient and you do not need to install CUDA Toolkit.
We test our code using blender 3.22.0, and will support 2.8.0 in the future.

Step 1: Find yout blender python path. Common paths are as follows
```sh
(Windows) 'C:\Program Files\Blender Foundation\Blender 3.2\3.2\python\bin'
(Linux) '/path/to/blender/blender-path/3.2/python/bin'
(Windows) '/Applications/Blender.app/Contents/Resources/3.2/python/bin'
```

Step 2: Install required packages. Open your shell(Linux) or powershell(Windows), 
```sh
cd {your python path} && pip3 install -r docker/requirements.txt && pip3 install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
```
, where ${CUDA} should be replaced by either cpu, cu117, or cu118 depending on your PyTorch installation.
On my MacOS with M1 cpu,

```sh
cd /Applications/Blender.app/Contents/Resources/3.2/python/bin && pip3 install -r docker/requirements_blender.txt && pip3 install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```

Step 3: Install add-on in blender. [Blender Add-ons Official Tutorial](https://docs.blender.org/manual/en/latest/editors/preferences/addons.html). `edit -> Preferences -> Add-ons -> Install -> Select the downloaded .zip file`

Step 4: Have fun! Click the armature and you will find a `GenMM` tag.

(GPU support) If you have GPU and CUDA Toolskits installed, we automatically dectect the running device.

Feel free to submit an issue if you run into any issues during the installation :)

## Acknowledgement

We thank [@stefanonuvoli](https://github.com/stefanonuvoli/skinmixer) for the help for the discussion of implementation about `Motion Reassembly` part (we eventually manually merged the meshes of different characters). And [@Radamés Ajna](https://github.com/radames) for the help of a better huggingface demo. 


## Citation

If you find our work useful for your research, please consider citing using the following BibTeX entry.

```BibTeX
@article{10.1145/weiyu23GenMM,
    author     = {Li, Weiyu and Chen, Xuelin and Li, Peizhuo and Sorkine-Hornung, Olga and Chen, Baoquan},
    title      = {Example-Based Motion Synthesis via Generative Motion Matching},
    journal    = {ACM Transactions on Graphics (TOG)},
    volume     = {42},
    number     = {4},
    year       = {2023},
    articleno  = {94},
    doi = {10.1145/3592395},
    publisher  = {Association for Computing Machinery},
}
```
