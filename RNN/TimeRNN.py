import numpy as np
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

class TimeRNN :
    def __init__(self,Wx,Wh,b,stateful=False):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers = None #다수의 rnn 계층을 리스트로 저장하는 용도

        #변수 h는 forward() 메서드를 불렀을 때의 마지막 RNN 계층의 은닉 상태를 저장,
        self.h, self.dh = None, None
        self.stateful = stateful

    #은닉 상태의 기울기를 저장 
    def set_state(self,h):
        self.h = h
    #은닉 상태를 초기화하는 메서드
    def reset_state(self):
        self.h = None

    # 긴 시계열 데이터를 처리할 때는 RNN의 은닉 상태를 유지해야함. 이처럼 은닉 상태를 유지하는 기능을 흔히 'stateful'이라는 단어로 표현
    # 많은 딥러닝 프레임워크에서 RNN 계층을 살펴보면 인수로 stateful이라는것을 받으며, 이를 통해 이전 시각의 은닉상태를 유지할지 지정할수잇음
        

    # TimeRNN 계층의 forward() 메서드가 불리면, 인스턴스 변수 h에는 마지막 RNN 계층의 은닉 상태가 저장
    # 그래서 다음번 forward() 메서드 호출 시 stateful이 True면 먼저 저장된 h값이 그대로 이용되고, stateful이 False면 h가 다시 영행렬로 초기화
    # -> truncated BPTT 
    # xs T개 분량의 시계열 데이터를 하나로 모은것
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # 배치 크기, 시퀀스 길이, 입력 데이터의 차원
        D, H = Wx.shape  # 입력 데이터의 차원, 은닉 상태의 차원

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)

            self.h = layer.forward(xs[:, t, :])
            # 문장에서 출력값을 담을 그릇을 준비
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs
    
    def backward(self, dhs) :
        Wx,Wh,b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0,0,0]
        for t in reversed(range(T)): #역전파 시작
            layer = self.layers[t]
            # RNN 계층의 순전파에서는 출력이 2개로 분기
            # 순전파 시 분기했을 경우, 역전파에서는 각 기울기가 합산되어 전해짐
            dx, dh = layer.backward(dhs[:, t,:] + dh) # 합산된 기울기
            dxs[:, t, :] = dx
            # 각 시간 단계의 역전파가 완료될 때마다, 해당 시간 단계의 가중치 기울기를 grads에 합산합니다. 
            #이는 모든 시간 단계에서 발생하는 가중치 변경을 고려하기 위함

            for i, grad in enumerate(layer.grads) :
                grads[i] += grad
        # 최종적으로 계산된 가중치 기울기를 self.grads에 저장
        # 이 기울기들은 모델의 가중치를 업데이트하는 데 사용
        for i, grad in enumerate(grads):
            # [..] -> 위치에서 나머지 모든 차원을 포함
            # self.grads[i] 배열 전체를 grad로 교체
            self.grads[i][...] = grad
        self.dh = dh
           

        return dxs
