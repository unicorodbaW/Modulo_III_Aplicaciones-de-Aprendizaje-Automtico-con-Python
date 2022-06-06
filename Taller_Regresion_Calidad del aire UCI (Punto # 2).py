# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:14:10 2022

@author: Wendy Mendozita
"""

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 


import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats
import pandas as pd
import numpy 

# ------- leemos archivos excel con pandas
df = pd.read_excel('AirQualityUCI.xlsx')
print(df)


#------------------Regresion Lineal -----------------------#

# Declaramos las varaibles x y Y
x  = df['C6H6(GT)']
y  = df['NO2(GT)']

# Variable estadistico retornados del metodo linregress
slope, intercept, r,p, std_err = stats.linregress(x,y)

# Crear una funcion para crear la linea de regresion
def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
plt.xlabel('C6H6(GT)')
plt.ylabel('NO2(GT)')
plt.title('C6H6(GT) & NO2(GT)')


# # #------------R de Relación----

slope, intercept, r,p, std_err = stats.linregress(x,y)
print("")
print("El Valor de R Lineal es:")
print(r)
print("")

# # #------- Predecir valores futuros-----

predicion_val_futuro = myfunc(15)
print("La Predicion_ Lineal es:")
print(predicion_val_futuro)
print("")


# #------------------Regresion Polinomial -----------------------#

# Modelo polinomial​
mymodel = numpy.poly1d(numpy.polyfit(x, y, 5))

# Definimos el espaciamiento para la linea
myline = numpy.linspace(1, 22, 100)

# Nuevos valores de Y
poli_new_y = mymodel(myline)

# # Dibujamos la línea de regresión polinomial

plt.plot(myline, poli_new_y)
plt.show()
plt.xlabel('C6H6(GT)')
plt.ylabel('NO2(GT)')
plt.title('C6H6(GT) & NO2(GT)')


# # imprimir la forma del polinomio
print("Ecuacion polinomial : ")
print(mymodel)
print("")

# #--------------- R-cuadrado​----#
print("El R cuadrado es:")
print(r2_score(y, mymodel(x)))
print("")

# # ------------   Predecir valores futuros -------------- 

predicion_polinomial = mymodel(10)
print("La predicion Polinomial es:")
print(predicion_polinomial)
print("")


# -----Regresion Multiple---------


# Hacer una lista de las variables independiente
k = df[['NO2(GT)']]

# lista de variable dependiente
z = df ['C6H6(GT)']

k1 = numpy.array(k)
z1 = numpy.array(z)

# Regresion 

reg_mod = linear_model.LinearRegression()
reg_mod.fit(k1, z1)

# Predicion 

predic_multiple = reg_mod.predict([[50]])

print("La Regresion Multiple es:")
print(predic_multiple )
print("")

# ----- Imprimir _coeficiente-----

print("El coeficiente es:")
print(reg_mod.coef_)
print("")




