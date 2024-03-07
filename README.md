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

🟧 ```b158_mamba_OpenWeb_colab-v9.ipynb```  : 기본 BitNet-b1.58-mamba  colab용    : 현재까지 추천,  a100,v100 등 추천  25g vram 사용

🟧 로컬용 






```b158_mamba_v6.py``` : 기본 BitNet-b1.58-mamba + Straight-Through Estimator (STE) 기법추가 : 돌려보니 비추하는 기법

다만 STE는 편향(bias)이 있는 그래디언트 추정치를 사용하므로, 이로 인한 수렴 속도 저하나 수렴 안정성 저하 등의 잠재적 문제점이 있음을 고려해야 합니다. 따라서 Mamba 모델에 STE를 적용할 경우 세심한 하이퍼파라미터 튜닝과 모니터링이 필요할 것입니다.



## test : 
해도 에러나네 이거. 


🟧 ```python inference.py --model_path model.safetensors --text "Once upon a time" --max_length 200```   

