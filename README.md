<h2 align="center">Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography</h2>

<div align="center">

<!-- <a href="https://arxiv.org/abs/2506.22753">
  <img src="https://img.shields.io/badge/ArXiv-2506.22753-red">
</a>&nbsp&nbsp&nbsp&nbsp<a href="https://dmdiff.github.io/">
  <img src="https://img.shields.io/badge/Project-Page-green.svg" />
</a>&nbsp&nbsp&nbsp&nbsp<a href="https://pan.sjtu.edu.cn/web/share/ebafd0ff28a601db09b58744a5b914d1">
  <img src="https://img.shields.io/badge/Data-Download-blue.svg" />
</a> -->

[![arXiv](https://img.shields.io/badge/ArXiv-2506.22753-red)](https://arxiv.org/abs/2506.22753)
&nbsp;&nbsp;&nbsp;&nbsp;
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://dmdiff.github.io/)
&nbsp;&nbsp;&nbsp;&nbsp;
[![Data Download](https://img.shields.io/badge/Data-Download-blue.svg)](https://pan.sjtu.edu.cn/web/share/ebafd0ff28a601db09b58744a5b914d1)


[Jianing Zhang]()<sup>2,3</sup>, [Jiayi Zhu]()<sup>1</sup>, [Feiyu Ji]()<sup>1</sup>, [Xiaokang Yang]()<sup>1</sup>, [Xiaoyun Yuan]()<sup>1\*</sup>

<sup>1</sup>MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University<br> <sup>2</sup>Fudan University<br> <sup>3</sup>Tsinghua University<br>* Corresponding author
</div>

We have released the code. Our implementation is based on the <a href="https://github.com/ArcticHare105/S3Diff">S3Diff project</a>, and we gratefully acknowledge the authors of S3Diff for their excellent work.


:star: If DMDiff is helpful for you, please help star this repo. Thanks! :hugs:

:star: Since our code has been re-tuned, retrained, and evaluated on the updated version of the dataset, the reported metrics may differ slightly from those presented in the paper. Thanks! :hugs:

## :book: Table Of Contents

- [Update](#update)
- [TODO](#todo)
- [Abstract](#abstract)
- [Framework Overview](#framework_overview)
- [Visual Comparison](#visual_comparison)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)

<!-- - [Installation](#installation)
- [Inference](#inference) -->

## <a name="update"></a>:new: Update

- **2026.2.27**: The checkpoint is released :fire:
- **2026.1.18**: The code is released :fire:
<!-- - [**History Updates** >]() -->

## <a name="todo"></a>:hourglass: TODO

- [ ] Release Checkpoints :link:
- [x] Release Code :computer:

## <a name="abstract"></a>:fireworks: Abstract

> Metalenses offer significant potential for ultra-compact computational imaging but face challenges from complex optical degradation and computational restoration difficulties. Existing methods typically rely on precise optical calibration or massive paired datasets, which are non-trivial for real-world imaging systems. Furthermore, a lack of control over the inference process often results in undesirable hallucinated artifacts. We introduce Degradation-Modeled Multipath Diffusion for tunable metalens photography, leveraging powerful natural image priors from pretrained models instead of large datasets. Our framework uses positive, neutral, and negative-prompt paths to balance high-frequency detail generation, structural fidelity, and suppression of metalens-specific degradation, alongside pseudo data augmentation. A tunable decoder enables controlled trade-offs between fidelity and perceptual quality. Additionally, a spatially varying degradation-aware attention (SVDA) module adaptively models complex optical and sensor-induced degradation. Finally, we design and build a millimeter-scale MetaCamera for real-world validation. Extensive results show that our approach outperforms state-of-the-art methods, achieving high-fidelity and sharp image reconstruction. More materials: https://dmdiff.github.io/.

## <a name="framework_overview"></a>:eyes: Framework Overview

<img src=assets/pic/multipath_diffusion_model.png>
<img src=assets/pic/training_and_inference.png>

:star: To tackle the challenges of metalens-based imaging, we propose a degradation-modeled multipath diffusion framework that leverages pretrained large-scale generative diffusion models for tunable metalens photography. Our approach addresses three key challenges: complex metalens degradations, limited paired training data, and hallucinations in generative models. With the powerful natural image priors from the base generative diffusion model, our method reconstructs vivid and realistic images using a small training dataset. To further enhance restoration, we propose a Spatially Varying Degradation-Aware (SVDA) attention module, which quantifies optical aberrations and sensor-induced noise to guide the restoration process. Additionally, we introduce a Degradation-modeled Multipath Diffusion (DMDiff) framework, incorporating positive, neutral, and negative-prompt paths to balance detail enhancement and structural fidelity while mitigating metalens-specific distortions. Finally, we design an instantly tunable decoder, enabling dynamic control over reconstruction quality to suppress hallucinations.


<!-- </details> -->

## <a name="setup"></a> ⚙️ Setup
```bash
conda create -n dmdiff python=3.10
conda activate dmdiff
pip install -r requirements.txt
```
Or use the conda env file that contains all the required dependencies.

```bash
conda env create -f environment.yaml
```

:star: Since we employ peft in our code, we highly recommend following the provided environmental requirements, especially regarding diffusers.

## <a name="training"></a> :wrench: Training

#### Step1: Download the pretrained models
We enable automatic model download in our code, if you need to conduct offline training, download the pretrained model [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)

#### Step2: Prepare training data
We have released the DMDiff dataset. Please download it <a href="https://pan.sjtu.edu.cn/web/share/ebafd0ff28a601db09b58744a5b914d1">here</a>. 

#### Step3: Training for S3Diff

Please modify the paths to training datasets in `configs/sr.yaml`
Replace lines 27 to 28 of configs/sr.yaml with your actual path.
```
gt_path: 'path_to_your_gt_folder/'
lr_path: 'path_to_your_lr_folder'
```
We first train a model to generate pseudo, change line 208 of src/train.py as you need.
```
args.neg_prob = 1 for persuo training
args.neg_prob = 0.1 for normal training
```

Then run:

```bash
sh run_training.sh
```


## <a name="inference"></a> 💫 Inference

#### Step1: Download [datasets](https://pan.sjtu.edu.cn/web/share/ebafd0ff28a601db09b58744a5b914d1) for inference

#### Step2: Download the [pretrained models](https://pan.sjtu.edu.cn/web/share/ebafd0ff28a601db09b58744a5b914d1)

Download the pretrained [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) model

Download the pretrained DMDiff model 

#### Step3: Inference for DMDiff

Please add the paths to evaluate datasets in `configs/sr.yaml`
Replace lines 57 to 58 of configs/sr.yaml with your actual path.
```
gt_path: 'path_to_your_gt_folder/'
lr_path: 'path_to_your_lr_folder'
```

Then run:

```bash
sh run_inference.sh
```



## :smiley: Citation

Please cite us if our work is useful for your research.

```
@InProceedings{Zhang_2025_ICCV,
    author    = {Zhang, Jianing and Zhu, Jiayi and Ji, Feiyu and Yang, Xiaokang and Yuan, Xiaoyun},
    title     = {Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {25914-25924}
}
```

## :notebook: License

This project is released under the [Apache 2.0 license](LICENSE).


## :envelope: Contact

If you have any questions, please feel free to contact yuanxiaoyun@sjtu.edu.cn.

