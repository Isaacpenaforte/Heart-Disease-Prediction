import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classificador_knn import classificar_knn
from classificador_rf import classificar_randomforest
from relatorio_paciente import coletar_dados_paciente, fazer_predicao
from graficos import plotar_graficos
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Tratamento de dados

# Recebe da tabela com resultados e riscos
dataset = pd.read_csv('./src/assets/heart_dataset.csv')

resultados_exames = dataset.iloc[:, :-1]  # Define uma variavel para todas as colunas de menos a ultima
risco = dataset['target']  # Define uma variavel para a ultima coluna

# Dividindo os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(resultados_exames, risco, test_size=0.3, random_state=40)

# Pré-processamento: Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classificação KNN
knn_classifier, y_pred_knn = classificar_knn(X_train_scaled, y_train, X_test_scaled)

# Classificação Arvore de Decisão
rf_classifier, y_pred_rf = classificar_randomforest(X_train_scaled, y_train, X_test_scaled)

print('Heart Care - Oque deseja fazer?')
print('1- Realizar predição')
print('2- Plotar gráficos')
escolha = int(input())

if escolha == 1:
    algoritmo=int(input('Qual algoritmo deseja usar?\n1- K-Nearest Neighbors\n2- Random Forest\n'))
    # Coleta dados do paciente
    dados_paciente = coletar_dados_paciente()
    if algoritmo == 1:
        # Faz a predição knn
        fazer_predicao(dados_paciente, knn_classifier, scaler)
    elif algoritmo == 2:
        # Faz a predição randomforest
        fazer_predicao(dados_paciente, rf_classifier, scaler)
    else:
        print('Entrada invalida, tente novamente\n')

elif escolha == 2:
    algoritmo=int(input('Qual algoritmo deseja usar?\n1- K-Nearest Neighbors\n2- Random Forest\n'))
    if escolha == 1:
        # Plota grafico knn
        plotar_graficos(X_train_scaled, X_train, dataset, y_train, y_test, y_pred_knn)
    elif escolha == 2:
        # Plota grafico randomforest
        plotar_graficos(X_train_scaled, X_train, dataset, y_train, y_test, y_pred_rf)
    else:
        print('Entrada invalida, tente novamente\n')

else:
    print('Ate mais!')