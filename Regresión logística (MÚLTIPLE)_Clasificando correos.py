## Diplomado de Python Aplicado ala Ingenieria
#------------------------#-----------------------
## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

#-----------------------------------------

# importamos librerias de Tratamiento de Datos
import pandas as pd
import numpy as np

# importamos librerias Grafias 
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Importamos Libreria de Preprocesado y modelado
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

#  Configuración matplotlib
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

#Configuración warnings
import warnings
warnings.filterwarnings('ignore')


#llamamos database por un URL de master/data/spam.csv'
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/' \
       + 'Estadistica-machine-learning-python/master/data/spam.csv'

# leemos la database spam.csv
datos = pd.read_csv(url)

# Mostramos las 3 primeras líneas de dataframe spam.csv
datos.head(3)

# Se codifica la variable respuesta como 1 si es spam 
# y 0 si no lo es, y se identifica cuantas observaciones hay de cada clase.

#clasificamos el type de Correo asi 1 si es spam y 0 si no lo es

datos['type'] = np.where(datos['type'] == 'spam', 1, 0)

# Imprimimos el Número de observaciones por clase realizando un conteo
print("Número de observaciones por clase")
print(datos['type'].value_counts())
print("")

#Imprimimos el porcentaje de observaciond e clases realizando un conteo
print("Porcentaje de observaciones por clase")
print(100 * datos['type'].value_counts(normalize=True))

# El 66.6% de los correos no son spam y el 39.4% sí lo son.
# Un modelo de clasificación que sea útil debe de ser capaz de predecir 
# correctamente un porcentaje de observaciones por encima del porcentaje
# de la clase mayoritaria. En este caso, el umbral de referencia que 
# se tiene que superar es del 66.6%.

# Creamos el Modelo de regresion logistica Multiple con el objetivo de predecir 
# si un correo es spam en función de todas las variables disponibles

# División de los datos en train y test

#Definimos las variables independientes
X = datos.drop(columns = 'type')

# Definimos la variable dependiente
y = datos['type']

# Declaramos los Train_Test(x-y)
#Bloques "bloque de entrenamiento " y "bloque de pruebas" 

# bloque de entrenamiento el 75% de los registros, y al bloque de pruebas el 25% restante.

X_train, X_test, y_train, y_test = train_test_split( 
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8, # el porcentaje del total de registros a incluir.
                                        
                                        random_state = 1234, # generador de números aleatorios, lo que reproducir la función
                                        
                                        shuffle      = True # specifica si los registros deberán ser desordenados previamente 
                                                            # o no (True- False)
                                    )


# Creamos el modelo utilizando matrices como en scikitlearn
# la matriz que vamos a predecir  se le tiene que añadir una columna de 1s para el intercept del modelo

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)

# Se ajusta el modelo a los datos escalados
modelo = modelo.fit()


# imprimimos un resumen usando el método model.summary. con el objetos TensorFlow.keras.models.Model.
print(modelo.summary())

#------------------Predicion----------------

# vamos a obtener predicciones para nuevos datos. con el modelos de statsmodels que nos  permiten calcular los intervalos de
# confianza asociados a cada predicción.

# Calculamos la predicion de los intervalo de confianza 
predicciones = modelo.predict(exog = X_train)

# Clasificación predicha
clasificacion = np.where(predicciones<0.5, 0, 1)
#Imprimimos la clasificacion predecida

print(clasificacion)


# ----------------------Accuracy de test---------------

# calcula el porcentaje de aciertos que tiene el modelo al predecir las observaciones de test (accuracy).

# Accuracy de test del modelo 
X_test = sm.add_constant(X_test, prepend=True)

# predecimos el x_test del modelo
predicciones = modelo.predict(exog = X_test)

# llamamos la Clasificación predicha
clasificacion = np.where(predicciones<0.5, 0, 1)
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
# Imprimimos el l accuracy de test
print("")
print(f"El accuracy de test es: {100*accuracy}%")


# Calculamos la Matriz de confusión de las predicciones de test

#con  pandas.crosstab()a creamos la matriz, agregando una fila adicional a mi marco de datos (Y_actual=0, Y_predicted= 1).
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    clasificacion,
    rownames=['Real'],# Renombrar  con Real las filas de la matriz
    colnames=['Predicción'] # Renombrar conPredicion las Columnas de la matriz
)

#Imprimimos elvalor de la  confusion_matrix

print(confusion_matrix)

# CONCLUSION #

# El modelo logístico creado para predecir la probabilidad de que un correo sea spam es en conjunto significativo
# (Likelihood ratio p-value = 0). El porcentaje de clasificación correcta en el conjunto del test es del 92.8%, 
# un valor muy por encima del umbral de 66.6% esperado por azar.

