import numpy as np


# 분기 노드 : 길이가 D인 배열 N번 복사(forward) 후 더함(backward)
D, N = 8, 7
x = np.random.randn(1, D)  # 입력
y = np.repeat(x, N, axis=0)  # 순전파

dy = np.random.randn(N, D)  # 무작위로 기울기 세팅
dx = np.sum(dy, axis=0, keepdims=True)  # 역전파


# Sum 노드 : 길이가 D인 배열 N번 더한(forward) 후 복사(backward)
D, N = 8, 7
x = np.random.randn(N, D)
y = np.sum(x, axis=0, keepdims=True)

dy = np.random.randn(1, D)
dx = np.repeat(dy, N, axis=0)


# Matmul 노드 :
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]  # 인수로 들어오는 행렬의 값들을 0으로 바꿈
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        """

        :param dout: dL/dy (그전 단계까지 미분한 값)
        :return:
        """
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# Sigmoid 계층
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / ()