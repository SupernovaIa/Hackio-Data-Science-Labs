# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Visualizaciones
# -----------------------------------------------------------------------
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Analisis Exploratorio Series Temporales
# -----------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Modelo Series Temporales
# -----------------------------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from itertools import product


# Otros
# -----------------------------------------------------------------------
from tqdm import tqdm

class TimeSeriesAnalysis:
    def __init__(self, dataframe, temporal_column, value_column):
        """
        Inicializa el objeto TimeSeriesAnalysis.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            El DataFrame que contiene los datos de la serie temporal.
        temporal_column : str
            Nombre de la columna con las fechas o tiempo.
        value_column : str
            Nombre de la columna con los valores de la serie temporal.
        """
        self.data = dataframe.copy()
        self.temporal_column = temporal_column
        self.value_column = value_column

        # Asegurar que la columna temporal es de tipo datetime
        self.data[self.temporal_column] = pd.to_datetime(self.data[self.temporal_column])
        self.data.set_index(self.temporal_column, inplace=True)
    
    def exploracion_datos(self):
        """
        Realiza una exploración básica de los datos.
        """
        print(f"El número de filas es {self.data.shape[0]} y el número de columnas es {self.data.shape[1]}")
        print("\n----------\n")
        
        if self.data.duplicated().sum() > 0:
            print(f"En este conjunto de datos tenemos {self.data.duplicated().sum()} valores duplicados")
        else:
            print("No hay duplicados")
        
        print("\n----------\n")
        if self.data.isnull().sum().sum() > 0:
            print("Las columnas con valores nulos y sus porcentajes son:")
            nulos = self.data.isnull().sum()
            display((nulos[nulos > 0] / self.data.shape[0]) * 100)
        else:
            print("No hay valores nulos")
        
        print("\n----------\n")
        print("Estadísticas de las variables numéricas:")
        display(self.data.describe().T)
    
    def comprobar_serie_continua(self):
        """
        Comprueba si la serie temporal es continua.
        """
        fecha_completa = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq="MS")
        mes_anio_actual = self.data.index.to_period("M")
        mes_anio_completo = fecha_completa.to_period("M")
        meses_faltantes = mes_anio_completo.difference(mes_anio_actual)

        if len(meses_faltantes) == 0:
            print("La serie temporal es continua, no faltan meses.")
        else:
            print("La serie temporal NO es continua.")
            print("Meses-Años faltantes:", meses_faltantes)
    
    def graficar_serie(self):
        """
        Grafica la serie temporal original.
        """
        fig = px.line(
            self.data,
            x=self.data.index,
            y=self.value_column,
            title="Serie Temporal Original",
            labels={self.temporal_column: "Fecha", self.value_column: "Valores"}
        )
        fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Valores")
        fig.show()
    
    def graficar_media_movil(self, window=30):
        """
        Grafica la media móvil de la serie temporal.
        
        Parameters:
        -----------
        window : int
            Tamaño de la ventana para calcular la media móvil.
        """
        self.data["rolling_window"] = self.data[self.value_column].rolling(window=window).mean()
        fig = px.line(
            self.data,
            x=self.data.index,
            y=[self.value_column, "rolling_window"],
            title="Evolución con Media Móvil",
            labels={self.temporal_column: "Fecha", self.value_column: "Valores"}
        )
        fig.data[0].update(name="Valores Originales")
        fig.data[1].update(name=f"Media Móvil ({window} días)", line=dict(color="red"))
        fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Valores")
        fig.show()
    
    def detectar_estacionalidad(self, figsize = (12, 10)):
        """
        Detecta visualmente si la serie temporal tiene un componente estacional.
        """
        decomposition = seasonal_decompose(self.data[self.value_column], model='additive', period=12)
        
        # Crear figura y subplots
        fig, axes = plt.subplots(4, 1, figsize= figsize, sharex=True)
        
        # Serie original
        axes[0].plot(self.data[self.value_column], color="blue", linewidth=2)
        axes[0].set_title("Serie Original", fontsize=14)
        axes[0].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Tendencia
        axes[1].plot(decomposition.trend, color="orange", linewidth=2)
        axes[1].set_title("Tendencia", fontsize=14)
        axes[1].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Estacionalidad
        axes[2].plot(decomposition.seasonal, color="green", linewidth=2)
        axes[2].set_title("Estacionalidad", fontsize=14)
        axes[2].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Ruido
        axes[3].plot(decomposition.resid, color="red", linewidth=2)
        axes[3].set_title("Ruido (Residuo)", fontsize=14)
        axes[3].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Ajustar diseño
        plt.suptitle("Descomposición de la Serie Temporal", fontsize=16, y=0.95)
        plt.xlabel("Fecha", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def graficar_acf_pacf(self, lags=40):
        """
        Grafica las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
        
        Parameters:
        -----------
        lags : int
            Número de rezagos a graficar.
        """
        plt.figure(figsize=(12, 10))
        plot_acf(self.data[self.value_column].dropna(), lags=lags)
        plt.title("Función de Autocorrelación (ACF)")
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 10))
        plot_pacf(self.data[self.value_column].dropna(), lags=lags, method="ywm")
        plt.title("Función de Autocorrelación Parcial (PACF)")
        plt.grid()
        plt.show()
    
    def prueba_estacionariedad(self):
        """
        Aplica la prueba de Dickey-Fuller aumentada para verificar estacionariedad.
        """
        result = adfuller(self.data[self.value_column].dropna())
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        print("Valores Críticos:")
        for key, value in result[4].items():
            print(f"{key}: {value}")
        if result[1] < 0.05:
            print("Rechazamos la hipótesis nula. La serie es estacionaria.")
        else:
            print("No podemos rechazar la hipótesis nula. La serie NO es estacionaria.")


class SARIMAModel:
    def __init__(self):
        self.best_model = None
        self.best_params = None

    def generar_parametros(self, p_range, q_range, seasonal_order_ranges):
        """
        Genera combinaciones de parámetros SARIMA de forma automática.

        Args:
            p_range (range): Rango de valores para el parámetro p.
            q_range (range): Rango de valores para el parámetro q.
            seasonal_order_ranges (tuple of ranges): Rango de valores para los parámetros estacionales (P, D, Q, S).

        Returns:
            list of tuples: Lista con combinaciones en formato (p, q, (P, D, Q, S)).
        """
        P_range, D_range, Q_range, S_range = seasonal_order_ranges

        parametros = [
            (p, q, (P, D, Q, S))
            for p, q, (P, D, Q, S) in product(
                p_range, q_range, product(P_range, D_range, Q_range, S_range)
            )
        ]

        return parametros

    def evaluar_modelos(self, y_train, y_test, parametros, diferenciacion, df_length, variable):
        """
        Evalúa combinaciones de parámetros SARIMA, devuelve un DataFrame con los resultados,
        y genera una visualización de las predicciones comparadas con los valores reales.

        Args:
            y_train (pd.Series): Serie temporal de entrenamiento.
            y_test (pd.Series): Serie temporal de prueba.
            parametros (list of tuples): Lista de combinaciones de parámetros en formato [(p, q, (P, D, Q, S)), ...].
            diferenciacion (int): Valor para el parámetro `d` de diferenciación.
            df_length (int): Longitud total del dataset para calcular los índices de predicción.

        Returns:
            pd.DataFrame: DataFrame con las combinaciones de parámetros y los errores RMSE.
        """
        results = []

        for p, q, seasonal_order in tqdm(parametros):
            try:
                # Crear y entrenar el modelo SARIMAX
                modelo_sarima = SARIMAX(
                    y_train,
                    order=(p, diferenciacion, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)


                start_index = len(y_train)
                end_index = df_length - 1
                pred_test = modelo_sarima.predict(start=start_index, end=end_index)
                pred_test = pd.Series(pred_test, index=y_test.index)  # Convertir a Serie de pandas

                # Calcular RMSE para el conjunto de prueba
                error = np.sqrt(mean_squared_error(y_test, pred_test))
                results.append({"p": p, "q": q, "seasonal_order": seasonal_order, "RMSE": error})

                # Guardar el mejor modelo
                if self.best_model is None or error < self.best_model["RMSE"]:
                    self.best_model = {
                        "modelo": modelo_sarima,
                        "RMSE": error,
                        "pred_test": pred_test,
                    }
                    self.best_params = {"p": p, "q": q, "seasonal_order": seasonal_order}

            except Exception as e:
                # Manejar errores durante el ajuste
                results.append({"p": p, "q": q, "seasonal_order": seasonal_order, "RMSE": None})

        # Convertir los resultados a un DataFrame
        results_df = pd.DataFrame(results)

        # Visualizar las predicciones del mejor modelo
        self._visualizar_predicciones_test(y_test, variable)
        return results_df


    def _visualizar_predicciones_test(self, y_test, variable):
        """
        Visualiza las predicciones del mejor modelo SARIMA comparando
        los valores reales y predicciones del conjunto de prueba, incluyendo
        el intervalo de confianza.

        Args:
            y_test (pd.Series): Serie temporal de prueba.
            variable (str): Nombre de la variable objetivo.
        """
        if self.best_model is None:
            raise ValueError("No se ha ajustado ningún modelo aún. Llama a 'evaluar_modelos' primero.")

        # Obtener las predicciones y el intervalo de confianza
        modelo = self.best_model["modelo"]
        forecast = modelo.get_forecast(steps=len(y_test))
        pred_test = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Crear la figura
        plt.figure(figsize=(14, 7))

        # Graficar valores reales
        sns.lineplot(x=y_test.index, y=y_test[variable], label="Valores Reales", color="blue", linewidth=2)

        # Graficar predicciones
        sns.lineplot(x=y_test.index, y=pred_test, label="Predicciones SARIMA", color="red", linestyle="--", linewidth=2)

        # Graficar intervalo de confianza
        plt.fill_between(
            y_test.index,
            conf_int.iloc[:, 0],  # Límite inferior
            conf_int.iloc[:, 1],  # Límite superior
            color="pink",
            alpha=0.3,
            label="Intervalo de Confianza",
        )

        # Personalización
        plt.title("Comparación de Predicciones vs Valores Reales (Conjunto de Prueba)", fontsize=16)
        plt.xlabel("Fecha", fontsize=14)
        plt.ylabel("Valores", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

