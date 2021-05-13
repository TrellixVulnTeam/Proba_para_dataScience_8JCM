import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import normal
from numpy import hstack
from sklearn.neighbors import KernelDensity

#Función gaussiana de manera manual
def gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2)*np.pi)*np.exp(-0.5*pow((x-mu),2))

x=np.arange(-4,4,0.1)
y=gaussian(x,1,1)

#Función gaussiana distribuida por medio de scipy
dist=norm(0,1)
y1=[dist.pdf(value) for value in x]
plt.plot(x,y1)
plt.title("Función gaussiana distribuida")
plt.show()

#Función gaussiana acumulada por medio de scipy
y2=[dist.cdf(value) for value in x]
plt.plot(x,y2)
plt.title("Función gaussiana acumulada")
plt.show()

#Extraer una tabla de excel y ponerla en df desde el item 4 en adelante

df=pd.read_excel('s057.xls')
arr=df['Normally Distributed Housefly Wing Lengths'].values[4:]
#unique sirve para contar cuantas veces se repite un elemento
values,dist = np.unique(arr, return_counts=True)
plt.bar(values,dist)
plt.title("Distribución gaussiana de longitud de alas de las moscas")
plt.show()
#estimación de distribución
mu=arr.mean()
sigma=arr.std()
x2=np.arange(30,60,0.1)
dist2=norm(mu,sigma)
y3=[dist2.pdf(values) for values in x2]
plt.plot(x2,y3)
plt.bar(values,dist/len(arr))
plt.title("Estimación de la distribución de longitud de alas")
plt.show()


#Muestras creadas (estimación paramétrica)
sample=normal(loc=50, scale=5, size = 1000) #Generador loc=promedio, scale=sigma
mu=sample.mean()
sigma=sample.std()
dist=norm(mu,sigma)
values = [value for value in range(30,70)]
probabilidades = [dist.pdf(value) for value in values]#pdf muestra la probabilidad de que salga cierto valor
plt.hist(sample,bins=30,density=True)#density pasa los numeros del eje y a probabilidad
plt.plot(values,probabilidades)
plt.title("Estimación de distribución gaussiana aleatoria")
plt.show()

#Construir una función bimodal (no paramétrica=combinación de varias distribuciones)
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = hstack((sample1, sample2)) #suma dos arreglos en una gráfica

model = KernelDensity(bandwidth=2, kernel='gaussian') #bandwith=parametro de suavizado, kernel= indicar la función
sample=sample.reshape(len(sample),1) # Arregla la estructura de datos para poder trabajar con ella
model.fit(sample)#Ajusta nuestro modelo con los datos

values = np.asarray([value for value in range(1,60)])#Asarray es para convertir la lista en arreglo y poder trabajar con ella en la librería de sklearn
values = values.reshape((len(values),1))
probabilities=model.score_samples(values)#Probabilidad logarítmica (Es mejor para el calculo del pc)
probabilities=np.exp(probabilities)#Obtenemos la probabilidad normal haciendo la exponecial de la probabilidad logarítmica

plt.hist(sample,bins=50,density=True)
plt.plot(values, probabilities)
plt.title("Estimación de distribución bimodal")
plt.show()