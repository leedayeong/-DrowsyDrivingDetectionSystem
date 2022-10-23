
# coding: utf-8

# In[15]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
#%matplotlib inline
#plt.style.use('ggplot')

knn = cv2.ml.KNearest_create()

def start(sample_size=25) :
    train_data = generate_data(sample_size)
    #print("train_data :",train_data)
    labels = classify_label(train_data)
    power, nomal, short = binding_label(train_data, labels)
    print("Return true if training is successful :", knn.train(train_data, cv2.ml.ROW_SAMPLE, labels))
    return power, nomal, short

def run(new_data, power, nomal, short):
    a = np.array([new_data])
    b = a.astype(np.float32)
    #plot_data(power, nomal, short)    
    ret, results, neighbor, dist = knn.findNearest(b, 5) # Second parameter means 'k'
    #print("Neighbor's label : ", neighbor)
    print("predicted label : ", results)
    #print("distance to neighbor : ", dist)
    #print("what is this : ", ret)
    #plt.plot(b[0,0], b[0,1], 'm*', markersize=14);
    return int(results[0][0])
    
#'num_samples' 생성할 데이터 포인터 수
def generate_data(num_samples, num_features = 2) :
    """randomly generates a number of data points"""    
    data_size = (num_samples, num_features)
    data = np.random.randint(0,30, size = data_size)
    return data.astype(np.float32)

#운전 사고를 예방할 수 있는 시간을 기준으로졸음 운전 위험도를 결정
def classify_label(train_data):
    labels = []
    for data in train_data :
        if data[1] < data[0]-5:
            labels.append(2)
        elif data[1] >= (data[0] + 5) :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(labels)

def binding_label(train_data, labels) :
    power = train_data[labels==0]
    nomal = train_data[labels==1]
    short = train_data[labels==2]
    return power, nomal, short

def plot_data(po, no, sh) :
    plt.figure(figsize = (10,6))
    plt.scatter(po[:,0], po[:,1], c = 'r', marker = 's', s = 50)
    plt.scatter(no[:,0], no[:,1], c = 'g', marker = '^', s = 50)
    plt.scatter(sh[:,0], sh[:,1], c = 'b', marker = 'o', s = 50)
    plt.xlabel('x is second for alarm term')
    plt.ylabel('y is 10s for time to close eyes')


