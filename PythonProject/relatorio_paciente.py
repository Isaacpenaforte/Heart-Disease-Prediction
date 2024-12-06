import numpy as np

from dadosCliente import caracteristicasDetalhadas, caracteristicasValores

def coletar_dados_paciente():
    dados_paciente = {}

    for i in range(len(caracteristicasDetalhadas)):
        while True:
            valor = input(f"\nDigite o valor/resultado para {caracteristicasDetalhadas[i]}:\n Entradas possiveis: {caracteristicasValores[i]} ")
            try:
                valor = float(valor)
                dados_paciente[i] = valor
                break
            except ValueError:
                print("Valor inválido. Por favor, digite um número.")

    print(dados_paciente)
    return dados_paciente

def fazer_predicao(dados_paciente, modelo, scaler):

    # Realiza uma predição para um novo paciente.
    # Recebe a lista de dados, modelo de predição e o padronizador de dados

    # Converte a lista dados_paciente
    dados = list(dados_paciente.values())
    dados = np.array(dados).reshape(1, -1)

    # Padroniza os dados
    dados_padronizados = scaler.transform(dados)

    # Faz a predição
    predicao = modelo.predict(dados_padronizados)

    # Revela a predição ao usuário
    if predicao == 0:
        print("\nParabéns! De acordo com nossa análise, o paciente não apresenta risco significativo de doença cardíaca no momento.")
    else:
        print("\nAtenção! Nossa análise indica que o paciente apresenta risco elevado de doença cardíaca.")
        print("\nÉ fundamental que o paciente adote hábitos de vida mais saudáveis para controlar essas condições.")
        print("\nProcure um médico para avaliação e acompanhamento em uma unidade de saúde mais proxíma de sua casa.")
        print("\nTente substituir alimentos processados e fast food por por frutas, legumes, grãos integrais e carnes magras. Reduza o consumo de sal e alimentos industrializados.")
        print("\nA prática regular de exercícios, como caminhar, nadar ou pedalar, pode ajudar a controlar o colesterol e a pressão arterial.")

