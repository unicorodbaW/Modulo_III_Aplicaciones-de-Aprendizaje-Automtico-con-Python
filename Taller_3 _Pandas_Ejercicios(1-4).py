# -*- coding: utf-8 -*-
"""
Created on Mon May 30 18:35:07 2022
cap=360000*(12%/(100+1))x10^2
@author: Wendy Mendozita
"""

import pandas as pd
df = pd.read_excel('Datos_D_F.xlsx')
print(df)

def imc(x):
    imc=x['Peso kg']/(x['Altura']**2)
    return round(imc,2)

def capital(x):
    cap=x['Inversion'] * (x['Interes']/(100+1 ))**x['Años']
    return round(cap, 6)

def descuento(x):
    des=0
    if x['Hora']<=6:
        des=10
    elif x['Hora']<=12:
        des=20
    elif x['Hora']<=18:
        des=30
    elif x['Hora']<=24:
        des=40
    return des 

def precio(x):
    res=(x['Descuento']/100)*15000
    return res

def telefono(x):
    tel=""
    if x['sexo']=="M":
        tel=f'{x["Telefono"]}-11'
    elif x['sexo']=="F":
        tel=f'{x["Telefono"]}-10'
    return tel 
    
print("\n\nCalculos Imc")
df["Imc"] = df.apply(imc, axis=1)
print(df) 

print("\n\nCalculos Capital")
df["Capital_Final"] = df.apply(capital, axis=1)
print(df)

print("\n\nCalculos Descuento")
df["Descuento"] = df.apply(descuento, axis=1)
print(df)


print("\n\nCalculos Precio")
df["Precio"] = df.apply(precio, axis=1)
print(df)

print("\n\nCalculos Precio")
df["Telefono Nuevo"] = df.apply(telefono, axis=1)
print(df)