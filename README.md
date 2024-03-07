# TernaryMamba: 
A ternary quantized language model based on the Mamba architecture, enabling model compression and efficient inference by representing weights as {-1, 0, 1}.

<br><br>
## dependency

 pip install mamba-ssm causal-conv1d



## run
<br>

🟧 ```b158_mamba_v2.py```  : 기본 BitNet-b1.58-mamba   : 현재까지 추천 

🟧 ```b158_mamba_v2a.py```  : 기본 BitNet-b1.58-mamba   : 현재까지 추천 + safetensor로 저장하기 추가 

🟧 ```b158_mamba_v3.py```  : 기본 BitNet-b1.58-mamba   : 현재까지 추천  + Shakespeare 직적다운후에 진행 
<br>


🟧🟧🟧🟧🟧 OpenWebText 다운로드후  1/10 만 훈련, 대략 1.7GB 의 train 데이터셋 (리소스제한등) 

🟧 ```mamba_OpenWeb_colab-v10.ipynb```  : 기본 mamba  colab용    : 

🟧 ```b158_mamba_OpenWeb_colab-v10.ipynb```  : 기본 BitNet-b1.58-mamba  colab용    : 현재까지 추천, 꼭  a100, 25g vram 사용 , loss 그래프 나옴 

💙💙 결과 




🟧🟧🟧🟧🟧 OpenWebText 다운로드후  5/10 만 훈련, 대략 8.5GB 의 train 데이터셋 (리소스제한등) 해보기 


🟧 ```TernaryMamba_local-v10-5.py```  : 로컬 BitNet-b1.58-mamba   rtx4090용 버전 vram 24g  : batch 64-> 48    : 테스팅중











