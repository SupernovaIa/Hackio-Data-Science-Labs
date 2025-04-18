from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


# Cargar el modelo
with open('transformers/mejor_modelo.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar los transformers
with open('transformers/transformer_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('transformers/transformer_target.pkl', 'rb') as f:
    target = pickle.load(f)

with open('transformers/transformer_one.pkl', 'rb') as f:
    one = pickle.load(f)


variables_one = ['Gender', 'ProductCategory']


@app.route("/")
def home():
    return jsonify({"mensaje": "API de predicción en funcionamiento.",
                    "endpoints": {"/predict": "Usa este endopint para realizar predicciones."}})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_pred = pd.DataFrame(data, index=[0])

        df_pred['DiscountsAvailed'] = df_pred['DiscountsAvailed'].astype('category')
        col_num = df_pred.select_dtypes(include=np.number).columns
        df_pred[col_num] = scaler.transform(df_pred[col_num])

        df_one = pd.DataFrame(one.transform(df_pred[variables_one]).to_array(), columns = one.get_feature_names_out())
        df_pred = pd.concat([df_pred, df_one], axis = 1)
        df_pred.drop(columns=variables_one, axis=1, inplace=True)

        df_pred = target.transform(df_pred)

        prediccion = model.predict(df_pred).to_list()[0]
        prob = model.predict_proba(df_pred).to_list()[0][1]

        print("HOLA")
        return jsonify({"prediccion": prediccion,
                        "probabilidad": prob})

    except:
        print("ADIOS")
        return jsonify({"respuesta": "Ha habido un problema en el envío de los datos."})

    


if __name__== "__main__":
    app.run(debug=True)