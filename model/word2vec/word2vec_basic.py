import sys
sys.path.append('../..')
import numpy as np
from model.common.layers import MatMul

# sample data
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]]) # [배치, 단어들]
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]]) # [배치, 단어들]

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 계층 생성 (cbow의 경우 주위 2개의 인풋)
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
h0 = in_layer0.forward(c0)
h1 = in_layer0.forward(c1)
h = (h0 + h1) / 2
s = out_layer.forward(h)
print(s)
