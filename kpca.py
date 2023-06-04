import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Cargamos los datos del archivo CSV usando Pandas
    dt_heart = pd.read_csv('data/heart.csv')
    
    # Verificamos si los datos se han importado correctamente imprimiendo la forma y los primeros 5 registros
    print("Shape of data:", dt_heart.shape)
    print("Head of data:\n", dt_heart.head(5))
    
    # Guardamos nuestro dataset sin la columna de target
    features = dt_heart.drop(['target'], axis=1)
    
    # Este será nuestro dataset, pero solo con la columna de target
    target = dt_heart['target']
    
    # Normalizamos los datos para que tengan media cero y desviación estándar igual a uno
    features = StandardScaler().fit_transform(features)
    
    # Partimos el conjunto de datos en un conjunto de entrenamiento y otro de prueba, utilizando una proporción de 70/30
    # Además, usamos el random state para asegurarnos de que obtendremos los mismos resultados cada vez que se ejecute el código
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Aplicamos la función de kernel de tipo polinomial para reducir la dimensionalidad de nuestros datos a solo 4 componentes principales
    kpca = KernelPCA(n_components=4, kernel='poly')
    
    # Ajustamos nuestro modelo KPCA a los datos de entrenamiento
    kpca.fit(X_train)
    
    # Transformamos nuestros datos de entrenamiento y prueba utilizando nuestro modelo KPCA ajustado
    train_transformed = kpca.transform(X_train)
    test_transformed = kpca.transform(X_test)
    
    # Creamos un modelo de regresión logística para clasificar nuestros datos reducidos en 4 dimensiones
    logistic = LogisticRegression(solver='lbfgs')
    
    # Entrenamos nuestro modelo de regresión logística en nuestros datos reducidos en 4 dimensiones
    logistic.fit(train_transformed, y_train)
    
    # Imprimimos el puntaje de precisión del modelo de regresión logística en los datos de prueba
    print("SCORE KPCA: ", logistic.score(test_transformed, y_test))

if __name__ == "__main__":
    main()
