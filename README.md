# Diffusion-LM
This repository contains my work and experimentation in implementing a Diffusion-based (Small) Language Model (Diffusion-LM) from scratch. 

![Diffusion-process](images/diffusion.png)
## Tasks Achieved:
- Implemented GPT2 following Karpathy's Tutorial
- Implemented a very basic version of [Diffusion-LM](https://arxiv.org/abs/2205.14217)
- Added 2D embedding space visuals using t-SNE along with POS based color-coding. 
- Improved Diffusion Training by fixing training data.
- Training on E2E dataset and ROCStories.

### Some Weird t-SNE Plots of Embedding Space
<table>
  <tr>
    <td><img src="images/spiral_2.png" alt="Spiral 2" width="250"/></td>
    <td><img src="images/spiral_3.png" alt="Spiral 3" width="250"/></td>
    <td><img src="images/spiral_4.png" alt="Spiral 4" width="250"/></td>
  </tr>
  <tr>
    <td><img src="images/spiral_embed.jpeg" alt="Spiral Embedding" width="250"/></td>
    <td><img src="images/e2e_1k_50k.png" alt="E2E Training" width="250"/></td>
  </tr>
</table>

### Plot of E2E training after 20k epochs with T=1000
<img src="images/emd_spc_30k.png" alt="E2E Training" width="750"/>

<!-- 
<div align="center">
  <img src="images/output1.jpeg" alt="E2E training inference"/>
  <p><i>E2E dataset inference after training for 50k epochs with 1k time steps</i></p>
</div> -->

## TO-DOs:
- Explore Anchoring and Discrete Diffusion methods
- Benchmark with GPT2 on the E2E and ROCStories Dataset.
- Benchmark using HF Diffusion SLMs on the same datasets.
- Try to beat/match SOTA!
