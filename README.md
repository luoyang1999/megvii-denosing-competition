# Restormer-Skff-SSA Denosing Network

### Table of contents
  * [Network Architecture](#network)
  * [Experimental Results](#result)
  * [Dependencies](#dependencies)
  * [Model](#model)
  * [Test](#test)
  
<a id="network"></a>
## Network Architecture
**Encoder-Decoder Residual Network (EDRN)**  
![Overview of Network](/figs/network.png)

<a id="result"></a>
## MEGVII2022 Real Denosing Challenge Results  
**Quantitative Results**  

 Method | PSNR (dB) | Score
 :---------------:|:----------:|:---------:
 Baseline | 26.89 | 0.78 | --
 EDRN (ours) | 28.79 | 0.84 | 47.08

<a id="dependencies"></a>
# About Our Source Code & Trained Model
## Dependencies
  * Python==3.7
  * MegEngine==1.8.2
  
Our code is tested on Ubuntu 18.04 environment with an NVIDIA RTX 3090 GPU.

<a id="model"></a>
## Check model parameter
`$ python model.py`

<a id="test"></a>
## Test
Quick start (Demo) to reproduce our results.
Fix test_dir=='YOUR TEST DIR'

`$ python prdict.py`
