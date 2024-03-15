# TernaryMamba: 
A ternary quantized language model based on the Mamba architecture, enabling model compression and efficient inference by representing weights as {-1, 0, 1}.


### Ternary Quantization
The model's weights are quantized to three values: {-1, 0, 1}. This enables model compression and efficient inference by representing the weights in a compact form.

### Mamba Architecture
The language model is constructed using Mamba blocks, which consist of:
- LayerNorm
- SwiGLU (Swish activation + Gated Linear Unit)
- Mamba State Space Model (SSM)

### OpenWebText Dataset
- The model is trained on 30~50% of the OpenWebText dataset.
- The data is preprocessed and stored in binary format, which is loaded during training.

### Model Training
- The model is trained using the Adam optimizer.
- During the training process:
  - Train/val losses are periodically evaluated.
  - The loss graph is updated.

<br><br>



## Dependency
Install the required dependencies using the following command:


```pip install mamba-ssm causal-conv1d```

##  Run local 

#### Local UBUNTU TESTED, RTX 4090, train on 30% of the downloaded OpenWebText dataset 


```python MAIN/local_r2_TernaryMamba_32_30_layer10_embed512.py```


##  Run colab

#### If you have sufficient compute units on Colab Pro or higher, select A100 and train on 50% of the downloaded OpenWebText dataset


```colab_TernaryMamba_data50.ipynb```


<br>


## Disclaimer

This is an experimental work that attempts to apply the ideas from the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" on top of the foundation from "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". Due to limited GPU resources, extensive testing has not been conducted, so there could be many limitations.








