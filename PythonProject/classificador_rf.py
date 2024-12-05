from sklearn.ensemble import RandomForestClassifier

def classificar_randomforest(X_train_scaled, y_train, X_test_scaled):

    # Inicializa o classificador Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=150) # n_estimators = número de arvores criadas

    # Treina o modelo Random Forest
    rf_classifier.fit(X_train_scaled, y_train)

    # Faz previsões no conjunto de teste
    y_pred = rf_classifier.predict(X_test_scaled)

    return rf_classifier, y_pred