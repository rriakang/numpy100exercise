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

    def backward(self, dh_next) :
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1-h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T,dt)
        dh_prev = np.matmul(dt,Wh.T)
        dWx = np.matmul(x.T,dt)
        dx = np.matmul(dt.Wx.T)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev
