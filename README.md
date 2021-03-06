# The 4th Solution of Megvii2022 Denosing Competation

### Table of contents
  * [Network Architecture](#network)
  * [Experimental Results](#result)
  * [Dependencies](#dependencies)
  * [Model](#model)
  * [Test](#test)
  
<a id="network"></a>
## Network Architecture
**Restormer-Skff-SSA Denosing Network**  
![Network Architecture](https://github.com/luoyang1999/megvii-denosing-competition/blob/main/network.png)

<a id="result"></a>
## Real Denosing Challenge Results  
**Quantitative Results**  

 Method | PSNR (dB) | Score
 :---------------:|:----------:|:---------:
 (ours) | 43.87 | 9.00

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
