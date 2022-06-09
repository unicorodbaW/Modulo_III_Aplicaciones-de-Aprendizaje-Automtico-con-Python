# Diplomado Python Aplicado a la Ingenieria UPB

## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 


# Importamos las Librerias que vamos a utilizar
from sklearn import linear_model
import pandas as pd
import numpy as np

# -----Regresion Multiple----

# leemos archivos csv con pandas
df_veh =  pd.read_csv("cars.csv")


condition_list = [
    (df_veh["Car"] == "Toyota"),
    (df_veh["Car"] == "Mitsubishi"),
    (df_veh["Car"] == "Skoda"),
    (df_veh["Car"] == "Fiat"),
    (df_veh["Car"] == "Mini"),
    (df_veh["Car"] == "VW"),
    (df_veh["Car"] == "Mercedes"),
    (df_veh["Car"] == "Ford"),
    (df_veh["Car"] == "Audi"),
    (df_veh["Car"] == "Hyundai"),
    (df_veh["Car"] == "Suzuki"),
    (df_veh["Car"] == "Honda"),
    (df_veh["Car"] == "Hundai"),
    (df_veh["Car"] == "Opel"),
    (df_veh["Car"] == "BMW"),
    (df_veh["Car"] == "Mazda"),
    (df_veh["Car"] == "Volvo")
]

sele_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# selecion de Marca especifica 
df_veh['clasif_car'] = np.select(condition_list, sele_list)

# Crear un Nuevo Dataframe
nuevo_df = pd.DataFrame()

nuevo_df["marca"] = df_veh["Car"].drop_duplicates()

nuevo_df["clasif_car"] = sele_list

# Hacer una lista de las variables independiente
x = df_veh[['Volume','Weight','CO2']]

# valores de la variable dependiente
y = df_veh['clasif_car']

x = np.array(x)
y = np.array(y)

#----- Regresion -------
 
reg_mod = linear_model.LinearRegression()

reg_mod.fit(x, y)

predicted_Car = reg_mod.predict([[2000, 1725, 114]])

# selecion aleatoria
car_m=int(np.round(predicted_Car,decimals = 0))

name = nuevo_df[nuevo_df["clasif_car"].isin([car_m])]

# Mostrar dataframe
print(df_veh)

# Imprimir datos de la marca
print("\n"+"Es probable que la Marca sea:",name["marca"].values[0])


# ----- Imprimir _coeficiente-----
print("El valor de los coeficiente son: = "+str(reg_mod.coef_))