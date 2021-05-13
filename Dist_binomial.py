import numpy as np
from numpy.random import binomial
from scipy.stats import binom
from math import factorial
import matplotlib.pyplot as plt
#Lanzar 3 monedas y que salgan dos caras
def my_binomial(k,n,p):
    return factorial(n)/(factorial(k)*factorial(n-k))*pow(p,k)*pow(1-p,n-k)

def plot_hist(num_trials):
    values=[0,1,2,3]
    #Crea un arreglo donde dice cuantas caras salieron en un rango de num_trials
    arr=[binomial(n,p) for _ in range(num_trials)]
    #Cuenta cuantas veces salieron cada cara 
    est=np.unique(arr, return_counts=True)
    #Divide el numero de hechos específicos por el numero de intentos
    est=est[1]/len(arr)
    #Muestra teoricamente la probabilidad que estamos haciendo
    teo= [binom(3,0.5).pmf(k) for k in values]
    plt.bar(values, est, color = 'red')
    plt.bar(values, teo, alpha = 0.5, color='blue')
    plt.title('{} experimentos'.format(num_trials))
    plt.show()

print(my_binomial(2,3,0.5))

#Lanzar 3 monedas y que salgan dos caras
dist = binom(3,0.5)
print(dist.pmf(2))

#Lanzar 3 monedas y que dos o menos sean caras
dist = binom(3,0.5)
print(dist.cdf(2))

p=0.5
n=3
print(binomial(n,p))

plot_hist(1000)

 