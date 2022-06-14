## Diplomado de Python Aplicado ala Ingenieria

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

import pandas as pd
from sklearn import linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Preprocesamiento de los datos
scale=StandardScaler()
from sklearn.metrics import r2_score
#from scipy import stats


#Leemos el archivo csv con pandas y creamos el dataframe
cars_𝑑𝑓 = 𝑝d.𝑟𝑒𝑎𝑑_𝑐𝑠𝑣("cars2.csv")

# Definimos las variables independientes
𝑋 = cars_𝑑𝑓[['Weight']]
#  Definimos la variable dependiente
𝑦 = cars_𝑑𝑓['CO2']


# Se realiza el escalado de los datos teniendo en cuenta la media y la desviacion
# estandar de los datos 
scaled_car = scale.fit_transform(X)


scale_x_car = ( cars_df [ "Weight" ] - cars_df [ "Weight" ]. mean ()) / cars_df [ "Weight" ].std()


#------- train _test-------------

# se hace la division del 25% para train y 25% para test
train_X = scale_x_car [:25]
train_y = y[:25]

# test 12%
test_X = scaled_car[25:]
test_y = y[25:]


scaled_car = scale.transform([[2300]])

# # mostramos el diagrama de dispersion 
# # para nuestro cojunto de entrenamiento
plt.scatter(train_X,train_y)
# #plt.scatter(x, y)
plt.show()

# # mostramos el diagrama de dispersion para nuestro cojunto de test
plt.scatter(test_X ,test_y)
plt.show()


#Modelo de Regresión polinomial

mymodel = np.poly1d(np.polyfit(train_X, train_y, 4))

# # Definimos el espaciamiento para la linea
myline = np.linspace(0, 2, 100)

# Nuevos valores de Y

poli_new_y = mymodel(myline)


# # mostramos el diagrama de dispersion del Modelo Polinomial
plt.scatter(train_X,train_y)
# #plt.scatter(x, y)
plt.show()

plt.plot(myline, poli_new_y)
plt.show()

print("")
# #imprimir la forma del polinomio
print("El Modelo polinomial es:")
print(mymodel)
print("")

# # R-cuadrado​----#
print("El R^2  Train es:")
print(r2_score(train_y, mymodel(train_X)))
print("")


# # R-cuadrado​----#
print("El R^2  Test es:")
print(r2_score(test_y, mymodel(test_X)))
print("")

# #---------Predecir valores futuros---
print("La Predicion es:")
speed_pred = mymodel(0.2)
print(speed_pred)
print("")