from sklearn.neighbors import KNeighborsClassifier

def classificar_knn(X_train_scaled, y_train, X_test_scaled):

    # Inicializando o classificador KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=5) # vizinhos a serem verificados

    # Treinando o modelo KNN
    knn_classifier.fit(X_train_scaled, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = knn_classifier.predict(X_test_scaled)

    return knn_classifier, y_pred


