import numpy as np
a=np.array([1,2,3,4,5])
b=np.array([5,4,3,2,1])

def cos_similary(a,b):
    a=np.array(a)
    b=np.array(b)
    powa=np.power(a,2)
    powb=np.power(b,2)
    dot_ab=a.dot(b)
    result=dot_ab/(np.sqrt(np.sum(powa))*np.sqrt(np.sum(powb)))
    return result

c=cos_similary(a,b)
print(c)

def batchnorm_forward(x,gamma,beta,eps):
    n,d=x.shape
    avg=1/n*np.sum(x,axis=0)
    x_=x-avg
    var=1/n*np.sum(np.power(x,2),axis=0)
    norm_var=np.sqrt(var+eps)
    ivar=1/norm_var
    x_hat=x_*ivar
    gammax=gamma*x_hat
    out=gammax+beta
    return out