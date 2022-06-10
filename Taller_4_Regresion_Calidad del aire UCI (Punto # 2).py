
## Diplomado de Python Aplicado ala Ingenieria

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 


import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats
import pandas as pd
import numpy as np

# ------- leemos archivos excel con pandas
df = pd.read_excel('AirQualityUCI.xlsx')
print(df)


#------------------Regresion Lineal -----------------------#

# Declaramos las varaibles x y Y
x  = df['C6H6(GT)'][:8001]
y  = df['NO2(GT)'][:8001]


# Variable estadistico retornados del metodo linregress
slope, intercept, r,p, std_err = stats.linregress(x,y)

# Crear una funcion para crear la linea de regresion
def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))


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

predicion_val_futuro = myfunc(40)
print("La Predicion_ Lineal es:")
print(predicion_val_futuro)
print("")


# #------------------Regresion Polinomial -----------------------#

# Modelo polinomial​
mymodel = np.poly1d(np.polyfit(x, y, 3))

# Definimos el espaciamiento para la linea
myline = np.linspace(100, 1000, 8000)

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

predicion_polinomial = mymodel(16)
print("La predicion Polinomial es:")
print(predicion_polinomial)
print("")


# -----Regresion Multiple---------

x,y = np.array(x).reshape(-1,1), np.array(y)

# Regresion 

reg_mod = linear_model.LinearRegression()
reg_mod.fit(x,y)

# Predicion 

predic_multiple = reg_mod.predict([[102]])

print("La Regresion Multiple es:")
print(predic_multiple )
print("")

# ----- Imprimir _coeficiente-----

print("El coeficiente es:")
print(reg_mod.coef_)
print("")




