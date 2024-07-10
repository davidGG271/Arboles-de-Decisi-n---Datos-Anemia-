import pandas as pd
import numpy as np
from graphviz import Digraph

class Nodo:
    def __init__(self, etiqueta=None, atributo=None):
        self.etiqueta = etiqueta
        self.atributo = atributo
        self.hijos = {}

    def agregar_hijo(self, valor, hijo):
        self.hijos[valor] = hijo

def entropia(clases):
    valores_unicos, cuentas = np.unique(clases, return_counts=True)
    probabilidades = cuentas / len(clases)
    return -np.sum(probabilidades * np.log2(probabilidades))

def ganancia_informacion(conjuntos, distribucion_clases, atributo):
    entropia_antes = entropia(distribucion_clases)
    valores_unicos = conjuntos[atributo].unique()
    entropia_ponderada_despues = 0

    for valor in valores_unicos:
        subconjunto_clases = distribucion_clases[conjuntos[atributo] == valor]
        entropia_ponderada_despues += len(subconjunto_clases) / len(distribucion_clases) * entropia(subconjunto_clases)

    return entropia_antes - entropia_ponderada_despues

class ArbolDecisionID3:
    def __init__(self):
        self.raiz = None

    def ajustar(self, conjuntos, distribucion_clases, atributos):
        self.raiz = self._ajustar(conjuntos, distribucion_clases, atributos)

    def _ajustar(self, conjuntos, distribucion_clases, atributos):
        if len(set(distribucion_clases)) == 1:
            return Nodo(etiqueta=distribucion_clases.iloc[0])

        mejor_atributo = None
        mejor_ganancia = -1

        for atributo in atributos:
            ganancia = ganancia_informacion(conjuntos, distribucion_clases, atributo)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo

        nodo = Nodo(atributo=mejor_atributo)

        for valor in conjuntos[mejor_atributo].unique():
            subconjunto_conjuntos = conjuntos[conjuntos[mejor_atributo] == valor]
            subconjunto_clases = distribucion_clases[conjuntos[mejor_atributo] == valor]

            hijo_nodo = self._ajustar(subconjunto_conjuntos, subconjunto_clases, [atr for atr in atributos if atr != mejor_atributo])
            nodo.agregar_hijo(valor, hijo_nodo)

        return nodo

    def imprimir_arbol(self, nodo=None, indent=""):
        if nodo is None:
            nodo = self.raiz

        if nodo.atributo is not None:
            print(indent + f"Atributo: {nodo.atributo}")
            indent += "  "
            for valor, hijo_nodo in nodo.hijos.items():
                print(indent + f"Valor: {valor}")
                self.imprimir_arbol(hijo_nodo, indent + "  ")
        else:
            print(indent + f"Etiqueta: {nodo.etiqueta}")

    def exportar_arbol(self, nombre_archivo='arbol'):
        dot = Digraph(comment='Árbol de Decisión')
        self._exportar_arbol(dot, self.raiz)
        dot.render(nombre_archivo, format='png', cleanup=True)

    def _exportar_arbol(self, dot, nodo, nombre_padre=None, etiqueta_arista=None):
        if nodo.atributo is not None:
            if nombre_padre is not None:
                dot.node(nodo.atributo, label=nodo.atributo)
                dot.edge(nombre_padre, nodo.atributo, label=etiqueta_arista, fontsize='8')

            for valor, hijo_nodo in nodo.hijos.items():
                # Agregar una etiqueta única para cada rama
                etiqueta_rama = f'{nodo.atributo}_{valor}'
                self._exportar_arbol(dot, hijo_nodo, nodo.atributo, etiqueta_rama)
        else:
            # Agregar una etiqueta única para cada hoja
            etiqueta_hoja = f'{nodo.etiqueta}_{nombre_padre}'
            dot.node(etiqueta_hoja, label=nodo.etiqueta, shape='box')
            dot.edge(nombre_padre, etiqueta_hoja, label=etiqueta_arista,fontsize='8')

# Cargar datos desde un archivo CSV
ruta_archivo = 'datos.csv'
df = pd.read_csv(ruta_archivo)

# Crear una instancia del árbol de decisión
arbol_decision = ArbolDecisionID3()

#df.drop('Anemia', axis=1) : devuelve el df eliminando la columna de Anemia   ||    axis = 0 fila || axis = 1 columna
#df['Anemia'] : devuelve la columna de Anemia
#df.columns[:-1] : devuelve un objeto Index de pandas con las etiquetas de las columnas del df menos la ultima
# Ajustar el árbol de decisión al conjunto de datos
arbol_decision.ajustar(df.drop('Anemia', axis=1), df['Anemia'], df.columns[:-1])

# Imprimir el árbol de decisión
arbol_decision.imprimir_arbol()

# Exportar el árbol de decisión como una imagen
arbol_decision.exportar_arbol()