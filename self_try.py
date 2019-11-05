# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#损失函数————————》求骗到函数——————————》梯度下降函数
def data_processing(x):
    x=x.reshape(-1,1)
    add_b=np.ones((len(x),1))
    c=np.hstack((add_b,x))
    x1=x**2
    d=np.hstack((c,x1))
    np.random.seed(22)
    theta=np.random.randint(0,10,d.shape[1])
    return d,theta

def cost_fun(x_b,theta,y):  ##返回一个具体的数字
    res=(y-x_b.dot(theta))**2
    return np.sum(res)/len(y)

def get_gradient(x_b,theta,y):##返回和theta一样数量的向量。
    g=np.arange(len(theta))
    # g[0]=2*(np.sum((y-x_b.dot(theta))))/len(y)
    for i in range(0,len(theta)):
        g[i]=2*(y-x_b.dot(theta)).dot(x_b[:,i])/len(y)  ##滴定要注意，后面没得求和公式，因为后面有与xi的内积，而前面没有！！！
    return -g                                              ##一定要注意减数与被减数的关系

def gradient_descent(x_b,theta,y,learning_rate,max_inter_num=10000000,epsilon=0.000001):
    count=0
    while(count<max_inter_num):
        # gradient=get_gradient(x_b,theta,y)
        last_theta=theta
        theta=theta-get_gradient(x_b,theta,y)*learning_rate
        count+=1
        if ( abs(cost_fun(x_b,theta,y)-cost_fun(x_b,last_theta,y))<epsilon):
            break
    return theta,count
###########################################################################################################
np.random.seed(26)
x=np.linspace(0,50,100)+0.2 #从0-50的区间中均匀取出100个数
y=2*x+1+np.random.randint(-5,5,100) #从-5，5的区间中随机取出100个数
x=x.reshape((-1,1))
add_b=np.ones((len(x),1))
x_b=np.hstack((add_b,x)) #将左边的矩阵拼接到右边的矩阵上，并且是水平拼接。
theta=np.random.randint(-50,50,x_b.shape[1])##theta中的参数个数与x_b中的个数相关。
final_theta,total_count=gradient_descent(x_b,theta,y,0.001)
# p
# ##一定要主义学习率的选择，选择的结果可能会导致结果不能收敛。范围太大不一定是程序的问题。
print(final_theta)
print(total_count)
y1=x*final_theta[1]+final_theta[0]
plt.scatter(x,y,color='blue')
plt.plot(x,y1,'r-',linewidth=5)
plt.show()
#########################################################################################################


