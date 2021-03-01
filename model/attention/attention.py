import numpy as np
from model.common.layers import Softmax


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.catch = None

    def forward(self, hs, a):
        """
        input으로 hidden state와 attention 가중치를 받음
        :param hs: 배치 * 시퀀스길이 * 히든스테이트
        :param a: 배치 * 시퀀스길이 (각 단어의 가중치)
        :return:
        """
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.catch = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.catch
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)  # c sum의 역전파
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)  # ar repeat의 역전파
        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.catch = None

    def forward(self, hs, h):
        N, T, H = hs.shape
        hr = h.reshape(N, 1, H).repeat(T, axis=1)  # 해당하는 단어 계산 편의를 위해 반복
        t = hs * hr  # Key들
        s = np.sum(t, axis=2)  # h에 대한 hs의 가중치
        a = self.softmax.forward(s)

        self.catch = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.catch
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention:
    def __init__(self):
        self.parms, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

