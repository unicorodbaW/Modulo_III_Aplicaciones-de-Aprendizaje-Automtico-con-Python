## Diplomado de Python Aplicado ala Ingenieria

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

# importamos librerias 
import pandas as pd
from sklearn import linear_model as lm
import numpy  as np
from sklearn.preprocessing import StandardScaler # Preprocesamiento de los datos
scale=StandardScaler()
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# semilla para los datos aleatorios
#numpy.random.seed(2)

#Leemos el archivo csv con pandas y creamos el dataframe
neflix_𝑑𝑓 = 𝑝d.𝑟𝑒𝑎𝑑_csv("netflix_titles.csv")

d={'Movie':1, 'TV Show':2}
# Mapeamos nuestra variable nacionalidad y reemplazamos los valores del directorio
# en el dataframe
neflix_𝑑𝑓['type']= neflix_𝑑𝑓['type'].map(d)
# Creamos el directorio para la variable Go

neflix_𝑑𝑓["duracion"] = pd.to_numeric(neflix_𝑑𝑓['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')

condiciones = [
    (neflix_𝑑𝑓["duration"].str.contains("Season").astype(np.bool_)),
    (neflix_𝑑𝑓["duration"].str.contains("min").astype(np.bool_))
    ]
selecciones = [1, 2]
neflix_𝑑𝑓["duration_type_pelicula"] =  np.select(condiciones, selecciones, default='Not Specified')

# Definimos las variables independientes
x = neflix_𝑑𝑓[['type', 'duracion']][:2000] 
#  Definimos la variable dependiente
𝑦 = neflix_𝑑𝑓['duration_type_pelicula'][:2000] 

# Estandarizacion de los datos 
scaled_neflix = scale.fit_transform(x)

# Train / Test
train_x = scaled_neflix[:1500]
train_y = y[:1500]

test_x = scaled_neflix[1500:]
test_y = y[1500:]


# Se crea el modelo de Regresion Multiple

reg_model=lm.LinearRegression()

# Se ajusta el modelo a los datos escalados
reg_model.fit(train_x,train_y)

scaledn = scale.transform([[2,2]])


# Se realiza la prediccion 
predictdurationtype = reg_model.predict([scaledn[0]])

# Imprimimos el resultado de la prediccion
print("la Predicion es:")
print(predictdurationtype)
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


