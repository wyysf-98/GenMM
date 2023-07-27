# Example-based Motion Synthesis via Generative Motion Matching, ACM Transactions on Graphics (Proceedings of SIGGRAPH 2023) (still ongoing...)

#####  <p align="center"> [Weiyu Li*](https://wyysf-98.github.io/), [Xuelin Chen*†](https://xuelin-chen.github.io/), [Peizhuo Li](https://peizhuoli.github.io/), [Olga Sorkine-Hornung](https://igl.ethz.ch/people/sorkine/), [Baoquan Chen](https://cfcs.pku.edu.cn/baoquan/)</p>
 
#### <p align="center">[Project Page](https://wyysf-98.github.io/GenMM) | [ArXiv](https://arxiv.org/abs/2306.00378) | [Paper](https://wyysf-98.github.io/GenMM/paper/Paper_high_res.pdf) | [Video](https://youtu.be/lehnxcade4I)</p>

<p align="center">
  <img src="https://wyysf-98.github.io/GenMM/assets/images/teaser.png"/>
</p>

<p align="center"> All Code and demo will be released in this week(still ongoing...) 🏗️ 🚧 🔨</p>


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
```


</details>

## Quick inference demo
For local quick inference demo using .bvh file, you can use

```sh
python run_random_generation.py -i './data/Malcolm/Gangnam-Style.bvh'
```
More configuration can be found in the `run_random_generation.py`

## Optimization
We provide a colab for a demo
<p>
  <a href="https://colab.research.google.com/github/wyysf-98/Sin3DGen/blob/main/colab_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

We use an Apple M1 and NVIDIA Tesla V100 with 32 GB RAM to generate each motion, which takes about ~0.5s and ~0.02s as mentioned in our paper.


## Acknowledgement

We thank [@stefanonuvoli](https://github.com/stefanonuvoli/skinmixer) for the help for the discussion of implementation about `Motion Reassembly` part (we eventually manually merged the meshes of different characters). And [@Radamés Ajna](https://github.com/radames) for the help of a better huggingface demo. 


## Citation

If you find our work useful for your research, please consider citing using the following BibTeX entry.

```BibTeX
@article{weiyu23GenMM,
    author    = {Weiyu Li and Xuelin Chen and Peizhuo Li and Olga Sorkine-Hornung and Baoquan Chen},
    title     = {Example-based Motion Synthesis via Generative Motion Matching},
    journal   = {ACM Transactions on Graphics (TOG)},
    year      = {2023},
    publisher = {ACM}
}
@article{10.1145/weiyu23GenMM,
    author     = {Li, Weiyu and Chen, Xuelin and Li, Peizhuo and Sorkine-Hornung, Olga and Chen, Baoquan},
    title      = {Example-Based Motion Synthesis via Generative Motion Matching},
    year       = {2023},
    issue_date = {August 2023},
    publisher  = {Association for Computing Machinery},
    address    = {New York, NY, USA},
    volume     = {42},
    number     = {4},
    issn       = {0730-0301},
    url        = {https://doi.org/10.1145/3592395},
    doi        = {10.1145/3592395},
    journal    = {ACM Trans. Graph.},
    month      = {jul},
    articleno  = {94},
    numpages   = {12},
}
```