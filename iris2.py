!pip install scikit-learn

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.read_csv("iris.csv")
# Convertir-lo a un DataFrame de Pandas
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Afegir la columna de la classe (tipus de flor)
df['species'] = iris.target

df.head()

df.info()

df.describe()

sns.scatterplot(data=df, x="petal length (cm)", y="sepal length (cm)")

colors_personalitzats = {
    0: "red",
    1: "green",
    2: "blue"
}
sns.scatterplot(data=df, x="petal length (cm)", y="sepal length (cm)",hue="species", palette=colors_personalitzats)

sns.pairplot(df, hue="species")

sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean", hue="species", palette="coolwarm")
