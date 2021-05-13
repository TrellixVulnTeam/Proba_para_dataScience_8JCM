from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def likelihood(y,yp):
    return yp*y+(1-yp)*(1-y)

fig=plt.figure()
#Se crea plano en 3d
ax=fig.add_subplot(projection='3d')

Y=np.arange(0,1,0.01)
YP=np.arange(0,1,0.01)
#Crea los ejes Y y YP para ser usados en 3D
Y,YP=np.meshgrid(Y,YP)
#Pone a prueba la ecuación de verosimilitud
Z=likelihood(Y,YP)
#Se gráfica la máxima verosimilitud con un problema tipo bernoulli para justificar la ecuación de likelihood()
surf=ax.plot_surface(Y,YP,Z,cmap=cm.coolwarm)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.show()

#Ejemplo iris
atrib_names = ['sepal length','sepal width','petal length','petal width']
X,y=load_iris(return_X_y=True)

clf= LogisticRegression(random_state=10, solver='liblinear').fit(X[:100], y[:100])#fit ajusta el modelo a nuestros valores (va hasta 100 para que solo hayan dos tipos de flores)
print(clf.coef_)# clf.coef da los coeficientes (pesos) para operar con los atributos y decir cual es el más importante