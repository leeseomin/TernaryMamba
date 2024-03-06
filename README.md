# BitNet-b1.58-mamba

<br><br>
## dependency

 pip install mamba-ssm causal-conv1d



## run
<br>

```b158_mamba_v2.py```  : 기본 BitNet-b1.58-mamba   : 현재까지 추천 

```b158_mamba_v6.py``` : 기본 BitNet-b1.58-mamba + Straight-Through Estimator (STE) 기법추가 : 돌려보니 비추하는 기법

다만 STE는 편향(bias)이 있는 그래디언트 추정치를 사용하므로, 이로 인한 수렴 속도 저하나 수렴 안정성 저하 등의 잠재적 문제점이 있음을 고려해야 합니다. 따라서 Mamba 모델에 STE를 적용할 경우 세심한 하이퍼파라미터 튜닝과 모니터링이 필요할 것입니다.


