import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from dadosCliente import caracteristicasDetalhadas, caracteristicas , caracteristicasValores
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def plotar_grafico_dispersao(feature_index,X_train_scaled,y_train):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_scaled[:, feature_index], y_train, c=y_train, cmap='viridis', edgecolor='k')
    plt.xlabel(caracteristicasDetalhadas[feature_index])
    plt.ylabel('Risco')
    plt.title(f'Heart Dataset - {caracteristicas[feature_index]} vs Risco')
    plt.colorbar(label='Risco Cardíaco')
    plt.show()

def plotar_grafico_barras(feature_index,dataset):
    coluna = dataset.columns[feature_index]
    # Agrupa os dados por gênero e calcula a contagem de cada classe de risco
    grouped_data = dataset.groupby([coluna, 'target']).size().unstack()

    # Cria o gráfico de barras empilhadas
    grouped_data.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.xlabel(caracteristicasDetalhadas[feature_index])
    plt.ylabel('Contagem')
    plt.title(f'Heart Dataset - Distribuição do Risco por {caracteristicas[feature_index]}')
    plt.legend(['Sem Risco', 'Com Risco'])
    rotulos = caracteristicasValores[feature_index].split(',')
    plt.xticks(range(len(rotulos)), rotulos)
    plt.show()

def plotar_graficos(X_train_scaled,dataset, y_train, y_test, y_pred):
    # Plotando gráficos de dispersão para as características dos dados
    indices_barras = [1, 2, 5, 6, 8, 10, 11, 12]
    for feature_index in range(X_train_scaled.shape[1]):
        if feature_index in indices_barras:
            plotar_grafico_barras(feature_index, dataset)
        else:
            plotar_grafico_dispersao(feature_index, X_train_scaled, y_train)

    # --------------

    # Avaliando o modelo
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # --------------

    # Exibindo os resultados
    print("Acurácia:", accuracy) # Exibe a acuracia
    print("Relatório de Classificação:")
    print(class_report)

    # Calculando a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Obtendo os valores da matriz de confusão
    # tn, fp, fn, tp = cm.ravel()

    # Plotando a matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Valor Previsto')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão')
    plt.show()