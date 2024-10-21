##학습용코드

import sys
sys.path.append('...')
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

#하이퍼파라미터 설정

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_size = 1.0



#데이터 읽기, 모델과 옵티마이저 생성
x,t = spiral.load()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_size)

data_size = len(x)
max_iters = data_size //batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    #데이터 뒤섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 기울기를 구해 매개변수 갱신
        loss = model.forward(batch_x,batch_t)
        model.backward()
        optimizer.update(model.param,model.grads)
        total_loss += loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss /loss_count
            print(' | epoch %d | repeat %d / %d | loss %.2f | ' % (epoch + 1, iters+1,max_iters,avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0,0

