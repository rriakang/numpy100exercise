#node 구현

#repeat node

import numpy as np

D,N = 8,7
 
x = np.random.randn(1,D) #입력
y = np.repeat(x,N,axis=0) #순전파 -> 원소 복제 수행 x를 N번 수행가능 이떄 axis를 지정하여 어느 축 방향으로 복제할지 조정가능


dy = np.random.randn(N,D) #무작위 기울기
dx = np.sum(dy,axis=0,keepdims=True) #역전파


#sum node

import numpy as np

D,N = 8,7
 
  
x = np.random.randn(N,D) #입력
y = np.repeat(x,axis=0,keepdims=True) #순전파 -> 원소 복제 수행 x를 N번 수행가능 이떄 axis를 지정하여 어느 축 방향으로 복제할지 조정가능

dy = np.random.randn(1,D) #무작위 기울기
dx = np.sum(dy,N,axis=0) #역전파



#MatMul node


class Matul:
    def __init__(self,W):
        self.params = [W] #학습하는 매개변수 보관
        self.grads = [np.zeros_like(W)] #위 매개변수에 대응하는 기울기 grads
        self.x = None
    
    def forward(self,x):
        W, =  self.params
        out = np.matmul(x,W)
        self.x = x
        return out
    
    def backward(self,dout):
        W, = self.params
        dx = np.matmul(dout,W.T)
        dW = np.matmul(self.x.T,dout)
        self.grads[0] = dW # 얕은 복사 이루어짐
        self.grads[0][...] = dW #생략기호  -> 깊은 복사 이루어짐
        return dx
    

#sigmoid 계층
    

class Sigmoid :
    def __init__(self) :
        self.params, self.grads =[],[]
        self.out = None
    
    def forward(self,x):
        out = 1 / ( 1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(sefl,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    

#Affine 계층 
    
class Affine :
    def __init__(self,W,b) :
        self.params = [W,b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self,x) :
        W,b = self.params
        out = np.matmul(x,W) + b
        self.x = x
        return out
    
    def backward(self,dout):
        W, b = self.params
        dx = np.matmul(dout,W.T)
        dW = np.matmul(self.x.T,dout)
        db = np.sum(dout,axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db


        return dx
    

#SGD -> 확률적경사하강법
    

class SGD :
    def __init__(self,lr=0.01):
        self.lr = lr
    
    def update(self, parms,grads):
        for i in range(len(parms)):
            parms[i] = -self.lr * grads[i]