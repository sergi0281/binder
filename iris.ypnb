from sklearn.datasets import load_iris
import pandas as pd

# Carregar el dataset Iris
iris = load_iris()

# Convertir-lo a un DataFrame de Pandas
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Afegir la columna de la classe (tipus de flor)
df['species'] = iris.target

# Veure les primeres files
df.head()
