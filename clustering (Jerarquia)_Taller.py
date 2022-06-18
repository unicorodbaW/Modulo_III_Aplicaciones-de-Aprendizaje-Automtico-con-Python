# Diplomado Python Aplicado a la Ingenieria (UPB)
#--------------#---------------------------
## Nombre: Wendy Paola Mendoza Barrera
## ID: 502216
## Email: wendy.mendozab@upb.edu.co 

#--------------------------------------
# Tarea de clustering
# Enviar Por Git

# Importamos la librerias tratamiento de los Datos
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Importamos Librerias Graficas
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Importamos libreria de procesos y modelados 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

# Esta función extrae la información de un modelo AgglomerativeClustering
# y representa su dendograma con la función dendogram de scipy.cluster.hierarchy

# Crear matriz de ligamiento y luego trazar el dendrograma
def plot_dendrogram(model, **kwargs):
 
# crear los conteos de muestras debajo de cada nodo

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot
    # Se utiliza para trazar el dendrograma
    dendrogram(linkage_matrix, **kwargs)


# Simulación de datos
X, y = make_blobs(
        n_samples    = 200, # el número total de puntos divididos equitativamente entre grupos
        n_features   = 2,  # El número de características para cada muestra.
        centers      = 4, # El número de centros a generar, o las ubicaciones de los centros fijos
        cluster_std  = 0.60, # La desviación estándar de los conglomerados.
        shuffle      = True, # Mezcla las muestras.
        random_state = 0 # números aleatorios para la creación de conjuntos de datos
       )

# Visualizamos los datos de simulacion 
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
    
# Mostrar etiquetas de titulos
ax.set_title('Datos simulados')
ax.legend();

# Escalamos los datos
X_scaled = scale(X)

# Calculamos  y comparamos el vinculo entre los resultados con los
# linkages complete, ward y average, utilizando la distancia euclídea como métrica de similitud

# linkages complete
modelo_hclust_complete = AgglomerativeClustering(
                            affinity = 'euclidean', # métrica utilizada como distancia
                            linkage  = 'complete',  # Tipo de linkage utilizado. Puede ser “ward”, “complete”, “average” o “single”
                            distance_threshold = 0, # Distancia (altura del dendograma) 
                            n_clusters         = None # número de clusters que se van a generar
                        )
modelo_hclust_complete.fit(X=X_scaled)

# linkages average

modelo_hclust_average = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'average',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_average.fit(X=X_scaled)

# linkages ward

modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            distance_threshold = 0,
                            n_clusters         = None
                     )
modelo_hclust_ward.fit(X=X_scaled)

# establecer el valor de distance_threshold para garantizar que completamos el árbol completo.

AgglomerativeClustering(distance_threshold=0, n_clusters=None)

# Visualizamos los tres tipos de linkage identifican claramente 4 clusters en un 
# Dendrogramas
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
# Mostrar el dendrogram linkages  usando el metodo average
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
axs[0].set_title("Distancia euclídea, Linkage average")

# Mostrar el dendrogram linkages usando el metodo complete
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")

# Mostrar el dendrogram linkages usando el metodo  ward
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();


# identificar el número de clusters, es inspeccionar visualmente el dendograma
# y decidir a qué altura se corta para generar los clusters

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# Definimos la altura de corte de Dendrogramas
altura_corte = 6

# Visualizamos el Dendrogramas 
plot_dendrogram(modelo_hclust_ward, color_threshold=altura_corte, ax=ax)
ax.set_title("Distancia euclídea, Linkage ward")
ax.axhline(y=altura_corte, c = 'black', linestyle='--', label='altura corte')
ax.legend();

# Usamos el Método silhouette para identificar el número óptimo de clusters
range_n_clusters = range(2, 15)
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = AgglomerativeClustering(
                    affinity   = 'euclidean',
                    linkage    = 'ward',
                    n_clusters = n_clusters
              )
    
#Ajustar el clustering jerárquico a partir de las características o la
# matriz de distancia.

    cluster_labels = modelo.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    valores_medios_silhouette.append(silhouette_avg)
    
# Visualizamos la evolución de media de los índices silhouette    
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.plot(range_n_clusters, valores_medios_silhouette, marker='o')
ax.set_title("Evolución de media de los índices silhouette")
ax.set_xlabel('Número clusters')
ax.set_ylabel('Media índices silhouette');

# Identificamos  el número óptimo de clusters, se reentrena el modelo indicando este valor

# creamos el Modelo
# Calculamos el vinculo entre los datos (ward) teniendo en cuenta 
# la distanncia euclidiana
modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            n_clusters = 4
                      )

modelo_hclust_ward.fit(X=X_scaled)

AgglomerativeClustering(n_clusters=4)
