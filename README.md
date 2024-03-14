# TernaryMamba: 
A ternary quantized language model based on the Mamba architecture, enabling model compression and efficient inference by representing weights as {-1, 0, 1}.

<br><br>
## dependency

 pip install mamba-ssm causal-conv1d



## run
<br>


🟧🟧🟧🟧🟧  로컬 UBUNTU TESTED,   rtx 4090, TernaryMamba   =  OpenWebText 다운로드후  3/10 만 훈련, 대략 5GB 의 train 데이터셋 

🟧 ```python MAIN/local_r2_TernaryMamba_32_30_layer10_embed512``` 

💙💙 결과 








🟧🟧🟧🟧🟧 OpenWebText 다운로드후  4/10 만 훈련, 대략 6.8GB 의 train 데이터셋 (리소스제한등) 해보기 


🟧 ```TernaryMamba_local-v10-5.py```  : 로컬 BitNet-b1.58-mamba   rtx4090용 버전 vram 24g  : batch 64-> 32    : 테스팅중 , killed , colab에서 해야? 배치싸이즈 32로 하고?  실패


🟧 ```TernaryMamba_colab-v10-4.ipynb```  :  기본 BitNet-b1.58-mamba  colab용  :::배치싸이즈 32      ,      테스팅중

















