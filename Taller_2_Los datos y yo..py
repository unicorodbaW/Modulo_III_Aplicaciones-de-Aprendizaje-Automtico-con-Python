# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:06:19 2022

@author: Wendy Mendozita
"""

## Nombre:  Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

import pandas as pd
import random
import numpy as np

#Lea la base de datos netflix_titles usando la librería “pandas”.
my_data= pd.read_csv('netflix_titles.csv')
print(my_data)

# Imprima por consola las primeras 5 filas del arreglo. 

print(my_data.head(5))

# Imprima por consola las Ultima 5 filas del arreglo. 

print(my_data.tail(5))

# Imprima cada uno de los tipos de dato asociado a las etiquetas.

print(my_data.dtypes)

# Guarde un archivo .xlsx, en el cual el nombre del archivo sea “Netflix_list” y el nombre de la hoja sea “títulos #.
 
my_data.to_excel("netflix_titles.xlsx",sheet_name="titulos",index= False)

my_data_2= pd.read_excel("netflix_titles.xlsx", sheet_name="titulos")

# Cree una nueva data frame en el cual segmente únicamente: el tipo, la duración, la descripción y el país.

nueva_data = my_data[['type','duration','description','country']]

# Haga un filtro para las películas que tienen una duración superior a 100 min.

my_data["duracion"] = pd.to_numeric(my_data['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')

# Haga un filtro para los “TV Shows” que tienen más de 3 temporadas.

tv_show = my_data[my_data['type'].str.contains('TV Show', na=False)]
tv_show_3_seasons= tv_show[tv_show['duracion']>3]

# Haga un filtro en el cual solo tenga en cuenta 2 categorías/etiquetas (libre elección)

categoria  =  my_data . loc [my_data ['listed_in' ]. isin ([ 'Romantic TV Shows, TV Dramas' , 'Documentaries'])]

# Modifique los valores del ID de las 5 primeras y las 5 últimas “shows” y de cualquier otra etiqueta de su elección (solo un valor).

my_data.iloc[:5, 0] = 's2'
my_data.loc[:2968,'listed_in' ] = 'Anime Series, International TV Shows'

# Añada una nueva columna “Visto”, en la cual debe colocar 1/0 (aleatorio) si vio o no el show (simulación).

my_data["Visto"] = np.random.randint(0, 2, my_data.shape[0])