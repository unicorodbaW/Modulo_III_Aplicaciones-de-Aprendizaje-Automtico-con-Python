
## Diplomado de Python Aplicado ala Ingenieria

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

# importamos librerias 
import pandas as pd
from sklearn import linear_model as lm
import numpy 
from sklearn.preprocessing import StandardScaler # Preprocesamiento de los datos
scale=StandardScaler()
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# semilla para los datos aleatorios
#numpy.random.seed(2)

#Leemos el archivo csv con pandas y creamos el dataframe
stud_𝑑𝑓 = 𝑝d.𝑟𝑒𝑎𝑑_excel("students_data.xlsx")

# Definimos las variables independientes
x = stud_𝑑𝑓[['absences', 'freetime']]
#  Definimos la variable dependiente
𝑦 = stud_𝑑𝑓['G3']

# Estandarizacion de los datos 
scaled_stude = scale.fit_transform(x)

# Train / Test
train_x = scaled_stude[:300]
train_y = y[:300]

test_x = scaled_stude[300:]
test_y = y[300:]


# Se crea el modelo de Regresion Multiple

reg_model=lm.LinearRegression()

# Se ajusta el modelo a los datos escalados
reg_model.fit(train_x,train_y)

scaled = scale.transform([[4,2]])


# Se realiza la prediccion 
predictG3 = reg_model.predict([scaled[0]])

# Imprimimos el resultado de la prediccion
print("la Predicion es:")
print(predictG3)
print("")

# R_ Relacion del modelo train
r2_train = r2_score(train_y, reg_model.predict(train_x))
# Mostramos r de relacion
print("R_ Relacion Train es:")
print(r2_train)
print("")


# R_ Relacion del modelo test
r2_tes = r2_score(test_y, reg_model.predict(test_x))
# Mostramos r de relacion
print("R_ Relacion Test es:")
print(r2_tes)
print("")

