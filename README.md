# NavOL: Navigation Policy with Online Imitation Learning

> **Code is coming soon.** This repository will host the official implementation of *NavOL*, an online imitation learning framework for visual navigation built on IsaacLab and a pre-trained navigation diffusion policy.

## Project Page
🌐 **[https://logosroboticsgroup.github.io/NavOL/](https://logosroboticsgroup.github.io/NavOL/)**

## Paper
**NavOL: Navigation Policy with Online Imitation Learning**  
Xiaofei Wei\*, Chun Gu\*, Li Zhang  
*School of Data Science, Fudan University · Shanghai Innovation Institute*  
**ICML 2026 (Accepted)**

## TL;DR
NavOL fine-tunes a pre-trained navigation diffusion policy (NavDP) via massively parallel rollouts in IsaacLab, supervised online by a privileged global path planner. The rollout&ndash;update loop trains on the policy's own visited state distribution, removing the need for reward design and mitigating distribution shift in offline imitation learning. With **8 RTX 4090 GPUs**, the system collects more than **2,000 high-quality trajectories per hour** across **50 indoor 3D scenes**.

## What's coming
- Training code (rollout&ndash;update loop on top of IsaacLab + NavDP)
- Indoor navigation benchmark on 3D-Front
- Pre-trained checkpoints
- Real-world deployment scripts

## Citation
```bibtex
@inproceedings{wei2026navol,
  title     = {{NavOL}: Navigation Policy with Online Imitation Learning},
  author    = {Wei, Xiaofei and Gu, Chun and Zhang, Li},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
