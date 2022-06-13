# Diplomado Python Aplicado a la Ingenieria (UPB)

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

# importamos librerias
import statsmodels.api as sm  
import  numpy  as  np
# import pandas  as  pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# llamamos la dataset
carseats = sm.datasets.get_rdataset("Carseats", "ISLR")
datos = carseats.data
print(carseats.__doc__)

datos['ventas_altas'] = np.where(datos.Sales > 8, 0, 1)
datos = datos.drop(columns = 'Sales')

## Convertimos las columnas NO numericas "ShelveLoc ,Urban,US" e "Ir" 
# en valores numericos

d={ 'Bad' : 0 , 'Medium' : 1 , 'Good' : 2 }
## Mapeamos nuestra variable nacionalidad y reemplazamos los valores del directorio
# en el dataframe
datos ["ShelveLoc"] = datos["ShelveLoc" ].map(d)

d1 = { 'Yes' : 1 , 'No' : 0 }
# Mapeamos nuestra variable nacionalidad y reemplazamos los valores del directorio
# en el dataframe
datos ["Urban"] = datos["Urban"].map(d1)

d2 = { 'Yes' : 1 , 'No' : 0 }
# Mapeamos nuestra variable nacionalidad y reemplazamos los valores del directorio
# en el dataframe
datos ["US"] = datos["US"].map(d2)

print("")
print(datos)

#Se definen las caracteristicas (Columnas desde donde intentamos predecir)
features =["CompPrice", "Income" , "Advertising" , "Population" , "Price" , "ShelveLoc" , "Age" , "Education" , "Urban" , "US" ]

# Creamos nuestra variable X con los valores del DataFrame utilizando
# las features
X  =  datos[features]
y  =  datos["ventas_altas"]

# se hace la division del 320 % para train y 320 % para test
train_x  =  X [: 320 ]
train_y  =  y [: 320 ]

test_x  =  X [ 320 :]
test_y  =  y [ 320 :]

dtree  = DecisionTreeClassifier ()

# se ajusta los datos al modelo 
dtree = dtree.fit(train_x ,train_y)

# Predicion con los datos test
predicion=(dtree.predict([[ 138 , 73 , 11 , 276 , 120 , 0 , 42 , 17 , 1 , 1 ]]))
print("la predicion Aproximada es:")
print(predicion)
print("[0] means Ventas Altas")
print("[1] means NO")
# Exportar datos para poderlo graficar en el diagrama de flujo
datos = tree.export_graphviz(dtree, out_file=None, feature_names = features)

# se Crea la Grafica
graph = pydotplus.graph_from_dot_(datos)

# Guardamos la grafica 
graph.write_png('Arboldedecisión_silla.png')

# Abrir la grafica y la mostramos por pantalla
img=pltimg.imread('Arboldedecisión_silla.png')
imgplot = plt.imshow(img)
plt.show()






