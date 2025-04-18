# Importación de librerías 
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb


class AnalisisModelosRegresion:
    def __init__(self, dataframe, variable_dependiente):
        """
        Inicializa el análisis, dividiendo los datos y configurando los modelos.
        """
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]

        # División del conjunto de datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42, shuffle=True
        )

        # Configuración de modelos
        self.modelos = {
            "regresion": LinearRegression(n_jobs=-1),
            "tree": DecisionTreeRegressor(),
            "random_forest": RandomForestRegressor(),
            "gradient_boosting": GradientBoostingRegressor(),
            "xgboost": xgb.XGBRegressor(),
        }

        # Almacenamiento de predicciones y mejores modelos
        self.predicciones = {modelo: {"train": None, "test": None} for modelo in self.modelos}
        self.mejor_modelo = {modelo: None for modelo in self.modelos}
        self.resultados = None

    def ajustar_modelo(self, modelo, param_grid=None):
        """
        Ajusta el modelo especificado usando GridSearchCV si se proporcionan hiperparámetros.
        """
        if modelo not in self.modelos:
            raise ValueError(f"Modelo '{modelo}' no reconocido.")

        estimator = self.modelos[modelo]

        if param_grid:
            grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring="neg_mean_squared_error")
            grid_search.fit(self.X_train, self.y_train)
            self.mejor_modelo[modelo] = grid_search.best_estimator_
        else:
            estimator.fit(self.X_train, self.y_train)
            self.mejor_modelo[modelo] = estimator

        # Predicciones
        self.predicciones[modelo]["train"] = self.mejor_modelo[modelo].predict(self.X_train)
        self.predicciones[modelo]["test"] = self.mejor_modelo[modelo].predict(self.X_test)

    def obtener_resultados(self):
        """
        Genera un DataFrame con las predicciones y los residuos de los modelos ajustados.
        """
        def crear_resultados(real, predicho, conjunto, modelo):
            return pd.DataFrame({
                "Real": real,
                "Predicho": predicho,
                "Conjunto": conjunto,
                "Modelo": modelo,
                "Residuos": real - predicho,
            })

        resultados = []
        for modelo, pred in self.predicciones.items():
            if pred["train"] is not None and pred["test"] is not None:
                resultados.extend([
                    crear_resultados(self.y_train, pred["train"], "Train", modelo),
                    crear_resultados(self.y_test, pred["test"], "Test", modelo),
                ])

        if not resultados:
            raise ValueError("Debe ajustar al menos un modelo antes de obtener resultados.")
        
        self.resultados = pd.concat(resultados, axis=0)
        return self.resultados

    def calcular_metricas(self, modelo):
        """
        Calcula métricas de evaluación (R2, MAE, MSE, RMSE) para un modelo ajustado.
        """
        if modelo not in self.predicciones:
            raise ValueError(f"Modelo '{modelo}' no reconocido.")

        pred = self.predicciones[modelo]
        if pred["train"] is None or pred["test"] is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo}' antes de calcular métricas.")

        metricas = {
            "train": {
                "R2": r2_score(self.y_train, pred["train"]),
                "MAE": mean_absolute_error(self.y_train, pred["train"]),
                "MSE": mean_squared_error(self.y_train, pred["train"]),
                "RMSE": np.sqrt(mean_squared_error(self.y_train, pred["train"])),
            },
            "test": {
                "R2": r2_score(self.y_test, pred["test"]),
                "MAE": mean_absolute_error(self.y_test, pred["test"]),
                "MSE": mean_squared_error(self.y_test, pred["test"]),
                "RMSE": np.sqrt(mean_squared_error(self.y_test, pred["test"])),
            },
        }
        return pd.DataFrame(metricas)

    def plot_residuos(self, modelo):
        """
        Plotea los residuos del modelo ajustado.
        """
        if self.resultados is None:
            raise ValueError("Debe obtener los resultados antes de plotear.")

        data = self.resultados[self.resultados["Modelo"] == modelo]
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data[data["Conjunto"] == "Train"], x="Residuos", kde=True, ax=ax[0], color="blue")
        sns.histplot(data[data["Conjunto"] == "Test"], x="Residuos", kde=True, ax=ax[1], color="green")
        ax[0].set_title(f"Residuos - Train ({modelo})")
        ax[1].set_title(f"Residuos - Test ({modelo})")
        plt.tight_layout()
        plt.show()

    def importancia_predictores(self, modelo):
        """
        Muestra y plotea la importancia de los predictores de modelos basados en árboles.
        """
        if modelo not in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            raise ValueError("Solo se admite para modelos basados en árboles.")

        if self.mejor_modelo[modelo] is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo}' antes de obtener la importancia de predictores.")

        importancias = self.mejor_modelo[modelo].feature_importances_
        importancia_df = pd.DataFrame({"Predictor": self.X_train.columns, "Importancia": importancias})
        importancia_df = importancia_df.sort_values(by="Importancia", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importancia", y="Predictor", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Predictores - {modelo}")
        plt.show()
        return importancia_df