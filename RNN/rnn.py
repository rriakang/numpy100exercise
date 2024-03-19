#rnn 클래스의 초기화와 순전파메서드

import numpy as np

class RNN :
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b] #파라미터 초기화
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache = None

    def foward(self,x,h_prev):
        Wx, Wh, b = self.params 
        t = np.matmul(h_prev,Wh) + np.matmul(x,Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next