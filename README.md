<div align="center">
  
  <div>
  <h1>Advancing Prompt-Based Methods for Replay-Independent General Continual Learning</h1>
  </div>

  <div>
      Zhiqi Kang, Liyuan Wang, Xingxing Zhang, Karteek Alahari
  </div>

  <div>
      <h4>
          ICLR 2025
      </h4>
  </div>
  <br/>

</div>

Official PyTorch implementation of our ICLR 2025 paper for general continual learning "[Advancing Prompt-Based Methods for Replay-Independent General Continual Learning](https://openreview.net/forum?id=V6uxd8MEqw)". (ArXiv/Camera-Ready version is coming soon!)

Our proposed MISA（**M**ask and **I**nitial-**S**ession **A**daptation) consists of the forgetting-aware initial session adaptation and the non-parametric logit mask to facilitate general continual learning, as presented in the following figure:

<p align="center">
    <img width="600" src="https://github.com/kangzhiq/MISA/blob/main/Imgs/overview.png" alt="MISA">
</p>


## How to run MISA?

### Build the conda environment

Please make sure that necessary packages in the environment.yml file are available.

### Two stage of MISA


#### Initial session adaptation

To warm up the prompt parameters by our initial session adaptation and our proposed forgetting-aware minimization, please run:

    . scripts/misa_fam.sh

The warmed-up prompts will be stored in the pretrained_prompt/ folder.

We provide the pretrained prompt parameters in the pretrained_prompt/ folder to faciliate the reproduction of our results.

#### Training on downstram dataset and test

To test different methods with different datasets, simply run the corresponding script with the specific dataset entry in the file:

    . scripts/misa.sh

## Acknolegment
This implementation is developed based on the source code of [MVP](https://github.com/KHU-AGI/Si-Blurry/tree/main).

## CITATION
If you find our codes or paper useful, please consider giving us a star or citing our work.

```bibtex
@inproceedings{kang2025advancing,
title={Advancing Prompt-Based Methods for Replay-Independent General Continual Learning},
author={Zhiqi KANG and Liyuan Wang and Xingxing Zhang and Karteek Alahari},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}

