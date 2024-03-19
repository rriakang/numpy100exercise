
np.zeros : 0으로 가득찬 Array를 배출한다. 즉, 여기에는 튜플, int, 혹은 list의 값이 들어와야 한다 그렇게 되면 해당하는 shape으로 형태를 만들어준다음 Array를 return 한다. 만약 여기에 np.zeros_like 처럼 변수를 넣어주면 오류가 나온다.

np.zeros_like : 어떤 변수만큼의 사이즈인 0 으로 가득 찬 Array를 배출한다. 즉, 여기에는 변수가 들어와야한다. 여기는 변수 말고도 그냥 [2,3,3] 이렇게 parameter 로 넣어줘도 되는데 이때는 단, 2,3,3 shape을 가진 array가 나오는 것이 아니라 [0, 0, 0] 인 numpy array 가 나온다.