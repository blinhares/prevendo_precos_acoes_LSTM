import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pathlib import Path
from threading import Thread
from rich.console import Console
from rich.theme import Theme
from rich.table import Table

custom_theme = Theme({"ok": "green", "nok": "bold red"})
console = Console(theme=custom_theme)
table = Table(title="Resumo das Ações")
table.add_column("Titulo")
table.add_column("Prev. de Var. %")


PATH_IMG = Path(__file__).parent / "src" / "imgs"


RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"

print(RED + "ERROR!" + RESET + "Something went wrong...")

DIAS_A_PRE_TEST = 25
QTD_DIAS_REGRESSOR = 70
UITIS_MEM_MODEL = 128
N_DIAS_APOS_PREV_MODELO = 3  # NUMERO DE DIAS APOS DATA ATUAL
PERIODO_DADOS_PAPEL_ANALIZADO = "5y"

titulos = primeira_coluna = [
    "ITUB4",
    "BRBI11",
    "RRRP3",
    "LREN3",    
]


def get_stock_close_data(titulo: str, periodo: int):
    # Criando o data frame com os dados de saida.
    stock = pd.DataFrame(yf.Ticker(titulo).history(period=periodo))
    stock = pd.DataFrame(stock["Close"], columns=["Close"])
    return stock


def estimar_valor_seguinte(
    stock_close: pd.DataFrame,
    dias_regressor: int,
    scala: MinMaxScaler,
    model: Sequential,
):

    # definindo a data
    data_dia_seguinte = stock_close.index[len(stock_close.index) - 1] + pd.Timedelta(
        1, unit="d"
    )
    # print(f'Realizaremos a previsao do dia: {data_dia_seguinte}...')

    # Pegar dos dados inciais os ultimios valores de acordo com a variavel de n regressores.
    # print(f'Coletando as ultimas observacoes dos {dias_regressor} ultimos dias...')
    previsao_dia_seg = stock_close[len(stock_close) - dias_regressor : len(stock_close)]
    # escalar os dados... de retorno a funcao retornar um array
    previsao_dia_seg = scala.transform(
        previsao_dia_seg)
    # transforma o array em uma linha com varios valores.
    previsao_dia_seg = previsao_dia_seg.reshape(1, -1)
    # transorma os dados no formato que deve ser inserido no model
    previsao_dia_seg = np.reshape(
        previsao_dia_seg,
        (previsao_dia_seg.shape[0], 
         previsao_dia_seg.shape[1], 1)
    )
    # prevendo o valor
    valor_previsto = model.predict(previsao_dia_seg)
    # retornando a escalo normal
    valor_previsto = scala.inverse_transform(valor_previsto)
    # print(f'O vlaor previsto para o dia {data_dia_seguinte} é : {valor_previsto[0][0]}')

    # criando um DF com os dados gerados
    df_previsto = pd.DataFrame(
        [[None, valor_previsto[0][0]]],
        columns=["Close", "Predictions"],
        index=[data_dia_seguinte],
    )
    return df_previsto


def prev_lstm(titulo: str):

    stock_close = get_stock_close_data(titulo, PERIODO_DADOS_PAPEL_ANALIZADO)

    # escalando valores
    scala = MinMaxScaler(feature_range=(0, 1))
    stock_close_scalado = scala.fit_transform(stock_close)

    # aqui vamos separa as observacoes em duas partes vamos tentar prever os ultimos 60 dias

    stock_close_treino = stock_close[0 : (len(stock_close.index) - DIAS_A_PRE_TEST)]

    stock_close_teste = stock_close[(len(stock_close.index) - DIAS_A_PRE_TEST) :]

    # aqui se criam duas listas vazias que vao receber os valores

    x_train, y_train = [], []

    for i in range(QTD_DIAS_REGRESSOR, len(stock_close_treino)):

        x_train.append(stock_close_scalado[i - QTD_DIAS_REGRESSOR : i, 0])

        y_train.append(stock_close_scalado[i, 0])

    # x train é um array com o comjunto numero equivalente a regressor de observacoes por linha
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Criando o Modelo
    model = Sequential()

    model.add(
        LSTM(
            units=UITIS_MEM_MODEL,
            return_sequences=True,
            input_shape=(QTD_DIAS_REGRESSOR, 1),
        )
    )

    model.add(LSTM(units=1))

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x=x_train, y=y_train, epochs=4, batch_size=1, verbose=2)

    inputs = stock_close[
        len(stock_close) - len(stock_close_teste) - QTD_DIAS_REGRESSOR :
    ]
    inputs = scala.transform(inputs)

    x_test = []

    for i in range(QTD_DIAS_REGRESSOR, inputs.shape[0]):  # de 90 a 150
        x_test.append(inputs[i - QTD_DIAS_REGRESSOR : i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    closing_price = model.predict(x=x_test)

    closing_price = scala.inverse_transform(closing_price)

    stock_close_teste["Predictions"] = closing_price

    for _ in range(N_DIAS_APOS_PREV_MODELO):
        df_previsto = estimar_valor_seguinte(
            stock_close,
            QTD_DIAS_REGRESSOR,
            scala,
            model
        )
        # concatenando os dados ao df de previsoes
        stock_close_teste = pd.concat(
            [stock_close_teste, df_previsto])

        ######## ATUALIZANDO A TABELA DE STOCK ORIGINAL ########
        df_previsto["Close"] = df_previsto["Predictions"]
        stock_close = pd.concat(
            [stock_close, df_previsto["Close"]])

    stock_close.drop(
        stock_close.index[N_DIAS_APOS_PREV_MODELO * -1 :], axis=0, inplace=True
    )

    valor_d_plus_1 = stock_close_teste["Predictions"].values[
        N_DIAS_APOS_PREV_MODELO * -1
    ]

    valor_d = stock_close_teste["Predictions"].values[
        N_DIAS_APOS_PREV_MODELO * -1 - 1
    ]  # type: ignore

    prev_var_percentual_amanha = ((valor_d_plus_1 - valor_d) / valor_d) * 100

    # Visualizando a Previsão
    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(20, 10))

    plt.ylabel("Preço da Ação")
    plt.xlabel("Data")
    plt.suptitle(
        titulo
        + " DR: "
        + str(QTD_DIAS_REGRESSOR)
        + " Unit: "
        + str(UITIS_MEM_MODEL)
        + f" Var: {prev_var_percentual_amanha :.2f} %"
    )
    # plt.plot(stock_close_treino['Close'], label = "Treino")
    plt.plot(stock_close.tail(DIAS_A_PRE_TEST * 2)["Close"], label="Treino")
    plt.plot(stock_close_teste["Close"], label="Observado")
    plt.plot(stock_close_teste["Predictions"], label="Previsão")
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )

    plt.savefig(PATH_IMG / str(titulo + ".png"))

    if prev_var_percentual_amanha > 0.5:

        # valor previsto de hoje - valor de ontem / valor de ontem
        print(
            f"{GREEN}{titulo} - Previsao de variacao de: {prev_var_percentual_amanha} %{RESET}"
        )
        table.add_row(titulo, f"{prev_var_percentual_amanha :.2f}", style="ok")

    else:
        table.add_row(titulo, f"{prev_var_percentual_amanha :.2f}", style="nok")
        print(
            f"{RED}{titulo} - Previsao de variacao de: {prev_var_percentual_amanha} %{RESET}"
        )
    print(stock_close_teste.tail(10))


for titulo in titulos:
    # lista.append(Thread(target=prev_lstm,args=(titulo + '.SA',)))
    # Thread(target=prev_lstm,args=(titulo + '.SA',)).start()
    prev_lstm(titulo + ".SA")
# prev_lstm('BRBI11.SA')
console.print(table)
