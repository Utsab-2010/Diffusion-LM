# Diffusion-LM
This repository contains my work and experimentation in implementing a Diffusion-based Language Model (Diffusion-LM) from scratch. 

![Diffusion-process](images/diffusion.png)
## Tasks Achieved:
- Implemented GPT2 following Karpathy's Tutorial
- Trained on the ROCStories Dataset.
- Implemented a very basic version of [Diffusion-LM](https://arxiv.org/abs/2205.14217)
- Added 2D and 3D embedding space visuals suing t-SME and PCA
- Improved Diffusion Training via fix training data.
- Trained on E2E dataset

### Some Fun Stuff
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

<div align="center">
  <img src="images/output1.jpeg" alt="E2E training inference"/>
  <p><i>E2E dataset inference after training for 50k epochs with 1k time steps</i></p>
</div>

## TO-DOs:
- Explore Anchoring and Discrete Diffusion methods
- Benchmark with GPT2 on the ROCStories Dataset.
- Document the work and organise the workspace/scripts
