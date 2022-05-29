# -*- coding: utf-8 -*-
"""
Created on Sat May 28 14:44:25 2022

@author: Wendy Mendozita
"""

a=int(input("Ingrese el valor de a: "))
b=int(input("Ingrese el valor de b: "))
c=int(input("Ingrese el valor de c: "))
d=int(input("Ingrese el valor de d: "))
e=int(input("Ingrese el valor de e: "))
f=int(input("Ingrese el valor de f: "))

print("")

ecu1=(a+(b/c))/(d+(e/f))
ecu2=(a-(b/(c-d)))
print("")
print("Ecuacion 1")
print(ecu1)
print("")
print("Ecuacion 2")
print(ecu2)

print("")
aux= ecu1
ecu1= ecu2
ecu2=aux
#print(ecu1)
#print(ecu2)

print(f"valor de ecu1 es: {ecu1}")
print(f"valor de ecu2 es: {ecu2}")