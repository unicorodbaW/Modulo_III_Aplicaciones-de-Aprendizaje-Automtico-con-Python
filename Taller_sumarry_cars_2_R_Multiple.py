
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

#------- train _test-------------

# se hace la division del 25% para train y 25% para test
train_X = scaled_car [:25]
train_y = y[:25]

# test 12%
test_X = scaled_car[25:]
test_y = y[25:]


#Modelo de Regresión Múltiple

# Se crea el modelo de regresion para las variables independientes 
# y la variable dependiente

reg_model=lm.LinearRegression()

# Se ajusta el modelo a los datos escalados
reg_model.fit(train_X,train_y)

# Estandarizamos los valores para hacer la prediccion
# Carro de 2300 kg 
scaled_car = scale.transform([[2300]])

# Se realiza la prediccion 
predictCo2 = reg_model.predict([test_X[0]])

# Imprimimos el resultado de la prediccion
print("")
print("La Predicion es:")
print(predictCo2)
print("")

# R^2 score Relacion 
r2_train = r2_score(train_y,reg_model.predict(train_X))
print("La Relacion Train:")
print(r2_train)
print("")

r2_test = r2_score(test_y,reg_model.predict(test_X))
print("La Relacion Test:")
print(r2_test)
print("")

# # mostramos el diagrama de dispersion 
# # para nuestro cojunto de entrenamiento
plt.scatter(train_X,train_y)
# #plt.scatter(x, y)
plt.show()

# # mostramos el diagrama de dispersion para nuestro cojunto de test
plt.scatter(test_X ,test_y)
plt.show()

