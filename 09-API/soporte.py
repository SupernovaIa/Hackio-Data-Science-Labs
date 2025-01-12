import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim

import os
import dotenv

pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames

import warnings
warnings.filterwarnings('ignore')

import requests

import time
import random

api_key = os.getenv('token')


def get_locations(towns):
    """
    Obtiene las coordenadas geográficas y la dirección de una lista de localidades.

    Parámetros:
    - towns (list): Lista de nombres de las localidades para las cuales se quiere obtener la información geográfica.

    Retorna:
    - (list): Lista de diccionarios, cada uno con las claves 'Nombre', 'Direccion', 'Latitud' y 'Longitud', que contienen el nombre de la localidad, su dirección completa, y sus coordenadas geográficas.
    """
    geolocator = Nominatim(user_agent="my_app")
    locations = []

    for town in tqdm(towns):

        location = geolocator.geocode(town)
        dc = {}
        dc['Nombre'] = town
        dc['Direccion'] = location.address
        dc['Latitud'] = location.latitude
        dc['Longitud'] = location.longitude

        locations.append(dc)

    return locations


def check_locations(towns, locations):
    """
    Verifica si las coordenadas de latitud y longitud en una lista de ubicaciones son válidas.

    Parámetros:
    - towns (list): Lista de nombres de las localidades correspondientes a cada ubicación.
    - locations (list): Lista de diccionarios que contienen las coordenadas de las localidades, con claves 'Latitud' y 'Longitud'.

    Retorna:
    - None: No retorna ningún valor, pero imprime un mensaje de error con el nombre de la localidad si alguna coordenada es inválida, o un mensaje de éxito si todas son válidas.
    """
    ok = True

    for i in range(len(locations)):

        lat = locations[i]['Latitud']
        lon = locations[i]['Longitud']
        if lat > 90 or lat < -90:
            ok = False
            print(f'Error en {towns[i]}')

        elif lon > 180 or lon < -180:
            ok = False
            print(f'Error en {towns[i]}')

    if ok:
        print('Todas las coordenadas concuerdan')


def buscar_lugares_cercanos(coordenadas, categoria=None, query=None, distancia=1000):
    """
    Busca lugares cercanos a unas coordenadas específicas utilizando la API de Foursquare.

    Parámetros:
    - coordenadas (tuple): Tupla con latitud y longitud de la ubicación de referencia.
    - categoria (str, opcional): Categoría de los lugares a buscar (ej. 'food', 'cafe'). Por defecto es None.
    - query (str, opcional): Consulta de búsqueda para especificar un tipo de lugar o nombre. Por defecto es None.
    - distancia (int, opcional): Radio de búsqueda en metros. El valor por defecto es 1000 metros.

    Retorna:
    - (dict): Respuesta de la API en formato JSON con los resultados de la búsqueda.
    """
    
    url = "https://api.foursquare.com/v3/places/search"
    
    headers = {
        "accept": "application/json",
        "Authorization": api_key
    }
    
    # Parámetros: latitud, longitud y radio (en metros)
    params = {
        "ll": f"{coordenadas[0]},{coordenadas[1]}",
        "radius": distancia
    }
    
    # Agregar la categoría si está definida
    if categoria:
        params["categories"] = categoria

    elif query:
        params["query"] = query
    
    # Realizar la solicitud a Foursquare
    response = requests.get(url, headers=headers, params=params)

    # Si el código de estado es 200
    if response.status_code == 200:
        # Devolvemos la respuesta en json
        return response.json()
    
    else:
        print(f'Error {response.status_code}')


def sacar_valor(dc):
    """
    Extrae el valor de 'formatted_address' de un diccionario. Si ocurre un error, devuelve NaN.

    Parameters:
    - dc (dict): Diccionario del cual se extraerá el valor.

    Returns:
    - (str or NaN): El valor de 'formatted_address' si existe, o NaN en caso de error.
    """
    try:
        return dc.get('formatted_address')
    except:
        return np.nan
    

def limpieza_df(df):
    """
    Limpia un DataFrame eliminando duplicados, extrayendo coordenadas y dirección, y eliminando columnas innecesarias.

    Parameters:
    - df (DataFrame): El DataFrame que contiene los datos a limpiar.

    Returns:
    - (DataFrame): El DataFrame limpio con las columnas renombradas y ajustadas.
    """
    # Eliminamos duplicados. Como están ordenados por distancia nos quedamos con el primero
    df.drop_duplicates(['name'], keep='first', inplace=True)

    # Coordenadas
    df['Coordenadas'] = df['geocodes'].apply(lambda x: x['main'])
    df['Latitud'] = df['Coordenadas'].apply(lambda x: x.get('latitude'))
    df['Longitud'] = df['Coordenadas'].apply(lambda x: x.get('longitude'))

    # Descomprimir
    df['Direccion'] = df['location'].apply(sacar_valor)

    # Quitamos las columnas innecesarias
    df.drop(columns=['fsq_id', 'categories', 'chains', 'closed_bucket', 'link', 'related_places', 'timezone'], inplace=True)
    df.drop(columns='geocodes', inplace=True)
    df.drop(columns=['location', 'Coordenadas'], inplace=True)

    # Renombrar columnas
    df.rename(columns={'distance': 'Distancia', 'name': 'Nombre'}, inplace=True)

    return df


def obtener_df_lugares(coordenadas, query=None, categoria=None, distancia=1000):
    """
    Obtiene una lista de lugares cercanos basados en coordenadas, consulta y/o categoría especificada.

    Parámetros:
    - coordenadas (list): Lista de tuplas o listas con coordenadas (latitud, longitud) de los lugares a buscar.
    - query (str, opcional): Término de búsqueda opcional para filtrar los lugares por nombre o descripción.
    - categoria (str, opcional): Categoría opcional para filtrar los lugares (e.g., 'restaurante', 'hotel').
    - distancia (int, opcional): Radio de búsqueda en metros alrededor de cada coordenada. Por defecto es 1000.

    Retorna:
    - (DataFrame): Un DataFrame con la información de los lugares encontrados, procesada y limpia.
    """

    resultados = []

    for i in tqdm(range(len(coordenadas))):

        # Tiempo de espera aleatorio entre llamadas
        time.sleep(random.uniform(0.5, 1.5))  # más eficiente que randint para tiempos aleatorios en este rango

        # Hacemos la llamada a la API
        res = buscar_lugares_cercanos(coordenadas[i], categoria=categoria, query=query, distancia=distancia)

        # Si hay resultados, los procesamos
        for lugar in res.get('results', []):
            if lugar not in resultados:  # Añadimos si no está en la lista
                resultados.append(lugar)

    df = pd.DataFrame(resultados)
    return limpieza_df(df)