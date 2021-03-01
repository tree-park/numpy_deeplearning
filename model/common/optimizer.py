from model.common.np import *


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        네트워크에 저장된 신경망 가중치와 계산된 역전파를 이용해
        확률적 경사 하강법 (Stochastic Gradient Descent)으로 가중치(파라미터)를 조정함.
          - 무작위로 선택된 데이터(미니배치)에 대한 기울기 이용
        :param params: 신경망의 가중치
        :param grads: 역전파된 기울기
        :return:
        """
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:  # 초기값..?
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / \
               (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
