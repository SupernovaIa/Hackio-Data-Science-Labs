from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

from time import sleep
import random
# Selenium para establecer la configuración del driver
# -----------------------------------------------------------------------
from selenium import webdriver

# Para generar una barra de proceso en los bucles for
# -----------------------------------------------------------------------
from tqdm import tqdm

# Para trabajar con ficheros
# -----------------------------------------------------------------------
import os

import re

import zipfile
import shutil

import requests
import pandas as pd

import json


def ine_datos_economicos(anio):

    # Configuración de preferencias
    chrome_options = webdriver.ChromeOptions()

    # Establacemos las preferencias
    prefs = {
        "download.default_directory": "/Users/javi/Documents/Hackio/Laboratorios/Laboratorio-ETL-Extraccion/datos/DatosEconomicos", # DatosDescargados la crea si no existe
        "download.prompt_for_download": False, # Desactiva el diálogo que Chrome normalmente muestra para pedir confirmación del usuario antes de descargar un archivo
        "directory_upgrade": True # Chrome actualiza el directorio de descarga predeterminado a la nueva ubicación especificada por download.default_directory si esta ha cambiado
    }

    url_economicos = "https://www.ine.es/dynt3/inebase/es/index.htm?padre=10426&capsel=10429"

    # Añadir las opciones 
    chrome_options.add_experimental_option("prefs", prefs)

    # Abrir navegador
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url_economicos)
    driver.maximize_window()

    # Click en la url
    sleep(random.uniform(1,2))
    driver.find_element(By.XPATH, "//a[text()='P.I.B. a precios de mercado y valor añadido bruto a precios básicos por ramas de actividad: Precios corrientes por provincias y periodo. ']").click()

    # XPATH basado en el texto "Seleccionar todos" para identificar los botones
    sleep(random.uniform(1,2))
    buttons = driver.find_elements(By.XPATH, "//button[.//span[text()='Seleccionar todos']]")

    # Recorrer tods los botones
    for button in buttons:
        button.click()

    driver.find_element("id", "periodo").click()
    driver.find_element("id", "periodo").send_keys(str(anio))
    sleep(random.uniform(1,2))
    driver.find_element("id", "periodo").send_keys(Keys.ENTER)

    # Acceso al menú de descarga
    sleep(random.uniform(1,2))
    driver.find_element('xpath', '//*[@id="btnDescargaForm"]').click()

    # Cambiamos el iframe y descargamos
    sleep(random.uniform(1,2))

    # Iframes
    iframes = driver.find_elements(By.TAG_NAME, "iframe")

    # Cambiar al primer iframe
    driver.switch_to.frame(iframes[0])

    # Descargar
    sleep(random.uniform(1,2))
    label = driver.find_element(By.XPATH, "//label[contains(text(), 'CSV: separado por ;')]")
    label.click()

    sleep(random.uniform(1,2))
    driver.close()


def ine_datos_demograficos(anio):

    # Configuración de preferencias
    chrome_options = webdriver.ChromeOptions()

    # Establacemos las preferencias
    prefs = {
        "download.default_directory": "/Users/javi/Documents/Hackio/Laboratorios/Laboratorio-ETL-Extraccion/datos/DatosDemograficos",  # DatosDescargados la crea si no existe
        "download.prompt_for_download": False,   # Desactiva el diálogo que Chrome normalmente muestra para pedir confirmación del usuario antes de descargar un archivo
        "directory_upgrade": True    # Chrome actualiza el directorio de descarga predeterminado a la nueva ubicación especificada por download.default_directory si esta ha cambiado
    }

    url_demograficos = "https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177012&menu=resultados&idp=1254734710990"

    # Añadir las opciones 
    chrome_options.add_experimental_option("prefs", prefs)

    # Abrir navegador
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url_demograficos)
    driver.maximize_window()

    # Hacemos click en 'por provincia'
    # Selector usando la estructura de la etiqueta 'a' dentro de 'li' con el texto específico
    driver.find_element(By.XPATH, "//a[text()='Por provincia']").click()

    # Hacemos scroll para ver el siguiente lugar donde hacer click
    sleep(random.uniform(1,2))
    driver.execute_script('window.scrollBy(0, 200)')

    # Acceder a la pagina de descarga
    sleep(random.uniform(1,2))
    driver.find_element(By.XPATH, "//a[text()='Población por provincias, edad (3 grupos de edad), españoles/Extranjeros, Sexo y año']").click()


    # Hacemos scroll
    sleep(random.uniform(1,2))
    driver.execute_script('window.scrollBy(0, 200)')

    # XPATH basado en el texto "Seleccionar todos"
    buttons = driver.find_elements(By.XPATH, "//button[.//span[text()='Seleccionar todos']]")

    # Recorrer tods los botones
    for button in buttons:
        button.click()  # Realiza la acción en cada botón (ej: click)

    # Seleccionamos el año
    driver.find_element("id", "periodo").click()
    driver.find_element("id", "periodo").send_keys(str(anio))
    sleep(random.uniform(1,2))
    driver.find_element("id", "periodo").send_keys(Keys.ENTER)

    # Le damos a descargar
    sleep(random.uniform(1,2))
    driver.find_element('xpath', '//*[@id="btnDescargaForm"]').click()

    # En la nueva pantalla hay un iframe
    sleep(random.uniform(1,2))

    # Iframes
    iframes = driver.find_elements(By.TAG_NAME, "iframe")

    # Cambiar al primer iframe (si es el que contiene el botón)
    driver.switch_to.frame(iframes[0])

    # Click en descargar como CSV separado por ;
    sleep(random.uniform(1,2))
    label = driver.find_element(By.XPATH, "//label[contains(text(), 'CSV: separado por ;')]")
    label.click()

    # Cerrar navegador
    sleep(random.uniform(1,2))
    driver.close()

def cambiar_nombre(anio, tipo):

    ruta = f'datos/Datos{tipo}'

    file_name = os.listdir(f'datos/Datos{tipo}/')[-1]

    ruta_archivo = os.path.join(ruta, file_name)
    ruta_archivo_nuevo = os.path.join(ruta, str(anio) + '.csv')

    os.rename(ruta_archivo, ruta_archivo_nuevo)


# Limpieza

def recorrer_categorias(archivo: str):
    """
    Procesa un archivo JSON de categorías y devuelve un DataFrame con los valores y sus categorías.

    Parameters:
    - archivo (str): Ruta al archivo JSON que contiene las categorías.

    Returns:
    - (pd.DataFrame): DataFrame con los valores de cada categoría y el nombre de la categoría.
    """

    df_categories = pd.DataFrame()
    
    # Intentar abrir y cargar el archivo JSON
    try:
        with open(archivo, 'r') as a:
            categories = json.load(a)['included']
    except FileNotFoundError:
        print(f"Archivo no encontrado: {archivo}")
        return df_categories
    except json.JSONDecodeError:
        print(f"Error al decodificar el archivo JSON: {archivo}")
        return df_categories

    # Recorremos todas las categorías
    for category in categories:

        data = category['attributes']['values']
        df = pd.DataFrame(data)

        if len(categories) > 1:
            # Solo añadimos el nombre de la categoría si hay más de una
            df['category'] = category['attributes']['title']

        df_categories = pd.concat([df_categories, df])

    return df_categories


def recorrer_archivos(anio: str):
    """
    Recorre todos los archivos en un directorio de año específico, procesa sus categorías y devuelve un DataFrame consolidado.

    Parameters:
    - anio (str): Año correspondiente al nombre del directorio que contiene los archivos a procesar.

    Returns:
    - (pd.DataFrame): DataFrame consolidado con los datos de todas las categorías y regiones del directorio del año especificado.
    """

    df_anio = pd.DataFrame()

    # Intentar listar archivos en el directorio del año
    try:
        archivos = os.listdir(anio)
    except FileNotFoundError:
        print(f"El directorio para el año {anio} no existe: {anio}")
        return df_anio

    # Recorremos todos los archivos
    for archivo in archivos:
        ruta_archivo = os.path.join(anio, archivo)

        df_categories = recorrer_categorias(ruta_archivo)
        df_categories['region'] = archivo.split('.')[0]
        
        df_anio = pd.concat([df_anio, df_categories])

    return df_anio


def recorrer_anios(ruta: str):
    """
    Recorre los directorios de cada año en la ruta especificada, procesa los archivos de cada año y devuelve un DataFrame consolidado con los datos.

    Parameters:
    - ruta (str): Ruta al directorio principal que contiene subdirectorios organizados por año.

    Returns:
    - (pd.DataFrame): DataFrame consolidado con los datos de todas las categorías, regiones y años encontrados en la ruta especificada.
    """

    df_total = pd.DataFrame()

    # Intentamos acceder al directorio que contiene los años
    try:
        anios = os.listdir(ruta)

    except FileNotFoundError:
        print(f"No existe la ruta: {ruta}")
        return df_total

    # Recorremos los años
    for anio in anios:
        ruta_anio = os.path.join(ruta, anio)
        df_anio = recorrer_archivos(ruta_anio)
        df_total = pd.concat([df_total, df_anio])

    return df_total


def limpiar_df(df: pd.DataFrame):
    """
    Limpia y transforma un DataFrame, agregando identificadores de región y campos de fecha.

    Parameters:
    - df (pd.DataFrame): DataFrame con columnas `region` y `datetime` que necesitan limpieza y transformación.

    Returns:
    - (pd.DataFrame): DataFrame transformado con un identificador de región (`region_id`), y columnas de fecha (`year`, `month`), sin las columnas originales `region` y `datetime`.
    """

    id_comunidades = {'Ceuta': 8744,
                    'Melilla': 8745,
                    'Andalucía': 4,
                    'Aragón': 5,
                    'Cantabria': 6,
                    'Castilla - La Mancha': 7,
                    'Castilla y León': 8,
                    'Cataluña': 9,
                    'País Vasco': 10,
                    'Principado de Asturias': 11,
                    'Comunidad de Madrid': 13,
                    'Comunidad Foral de Navarra': 14,
                    'Comunitat Valenciana': 15,
                    'Extremadura': 16,
                    'Galicia': 17,
                    'Illes Balears': 8743,
                    'Canarias': 8742,
                    'Región de Murcia': 21,
                    'La Rioja': 20}

    # Cambiar el id de la región
    df['region_id'] = df['region'].apply(lambda x: id_comunidades[x])
    # Transformar la parte de la fecha
    df['datetime'] = df['datetime'].str.split('T', expand=True)[0]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    # Eliminar las columnas irrelevantes
    df.drop(columns=['region', 'datetime'], inplace=True)

    return df


def df_ine(ruta: str):
    """
    Carga y concatena archivos CSV de un directorio en un único DataFrame.

    Parámetros:
    - ruta (str): Ruta al directorio que contiene archivos CSV organizados por año.

    Retorna:
    - (pd.DataFrame): Un DataFrame concatenado con los datos de todos los archivos CSV en el directorio.
    """

    # Acceder a las carpetas correspondientes
    try:
        anios = os.listdir(ruta)

    except FileNotFoundError:
        print(f"No existe la ruta: {ruta}")

    lista = [] 

    # Recorremos los años
    for anio in anios:
        ruta_anio = os.path.join(ruta, anio)
        df = pd.read_csv(ruta_anio, sep=';', encoding='latin-1')

        lista.append(df)

    # Juntar los dataframes
    return pd.concat(lista)


def limpieza_ine_econ(df_econ):
    """
    Limpia y transforma un DataFrame de datos económicos, generando identificadores de provincias y regiones.

    Parámetros:
    - df_econ (pd.DataFrame): DataFrame que contiene datos económicos, incluyendo las columnas 'Provincias', 'periodo', y 'Total'.

    Retorna:
    - (tuple): Una tupla de dos elementos:
    - df_econ (pd.DataFrame): DataFrame con columnas limpias y datos transformados, que incluye identificadores para provincias y regiones.
    - df_provincias (pd.DataFrame): DataFrame único de provincias con las columnas 'provincia' y 'provincia_id'.
    """

    dc_provincias = {
        'Albacete': 7,                  # Castilla - La Mancha
        'Alicante/Alacant': 15,         # Comunitat Valenciana
        'Almería': 4,                   # Andalucía
        'Araba/Álava': 10,              # País Vasco
        'Asturias': 11,                 # Principado de Asturias
        'Ávila': 8,                     # Castilla y León
        'Badajoz': 16,                  # Extremadura
        'Balears, Illes': 8743,         # Illes Balears
        'Barcelona': 9,                 # Cataluña
        'Bizkaia': 10,                  # País Vasco
        'Burgos': 8,                    # Castilla y León
        'Cáceres': 16,                  # Extremadura
        'Cádiz': 4,                     # Andalucía
        'Cantabria': 6,                 # Cantabria
        'Castellón/Castelló': 15,       # Comunitat Valenciana
        'Ciudad Real': 7,               # Castilla - La Mancha
        'Córdoba': 4,                   # Andalucía
        'Coruña, A': 17,                # Galicia
        'Cuenca': 7,                    # Castilla - La Mancha
        'Gipuzkoa': 10,                 # País Vasco
        'Girona': 9,                    # Cataluña
        'Granada': 4,                   # Andalucía
        'Guadalajara': 7,               # Castilla - La Mancha
        'Huelva': 4,                    # Andalucía
        'Huesca': 5,                    # Aragón
        'Jaén': 4,                      # Andalucía
        'León': 8,                      # Castilla y León
        'Lleida': 9,                    # Cataluña
        'Lugo': 17,                     # Galicia
        'Madrid': 13,                   # Comunidad de Madrid
        'Málaga': 4,                    # Andalucía
        'Murcia': 21,                   # Región de Murcia
        'Navarra': 14,                  # Comunidad Foral de Navarra
        'Ourense': 17,                  # Galicia
        'Palencia': 8,                  # Castilla y León
        'Palmas, Las': 8742,            # Canarias
        'Pontevedra': 17,               # Galicia
        'Rioja, La': 20,                # La Rioja
        'Salamanca': 8,                 # Castilla y León
        'Santa Cruz de Tenerife': 8742, # Canarias
        'Segovia': 8,                   # Castilla y León
        'Sevilla': 4,                   # Andalucía
        'Soria': 8,                     # Castilla y León
        'Tarragona': 9,                 # Cataluña
        'Teruel': 5,                    # Aragón
        'Toledo': 7,                    # Castilla - La Mancha
        'Valencia/València': 15,        # Comunitat Valenciana
        'Valladolid': 8,                # Castilla y León
        'Zamora': 8,                    # Castilla y León
        'Zaragoza': 5,                  # Aragón
        'Ceuta': 8744,                  # Ceuta
        'Melilla': 8745                 # Melilla
    }

    # Provincias
    df_econ['provincia'] = df_econ['Provincias'].str.split(r'\s(\D+)', regex=True, expand=True)[1]
    df_econ['provincia_id'] = df_econ['Provincias'].str.split(r'\s(\D+)', regex=True, expand=True)[0]
    # Limpiar periodo (2021 tiene un (P))
    df_econ['periodo'] = df_econ['periodo'].apply(lambda x: int(x.replace('(P)', '')) if type(x) == str else x)
    # Convertir total a float
    df_econ['Total'] = df_econ['Total'].str.replace('.', '').astype(float)
    # Region_id
    df_econ['region_id'] = df_econ['provincia'].apply(lambda x: dc_provincias[x])
    # Reset_index
    df_econ.reset_index(drop=True, inplace=True)

    # Aprovechamos a obtener un dataframe para las provincias
    df_provincias = df_econ[['provincia', 'provincia_id']].drop_duplicates().reset_index(drop=True)
    df_provincias['region_id'] = df_provincias['provincia'].apply(lambda x: dc_provincias[x])

    # Eliminar columnas sobrantes
    df_econ.drop(columns=['Provincias', 'provincia'], inplace=True)

    # Renombrar columnas
    df_econ.rename(columns={'periodo': 'year', 'Total': 'total', 'Ramas de actividad': 'categories'}, inplace=True)

    return df_econ, df_provincias


def limpieza_ine_demo(df_demo):
    """
    Limpia y transforma un DataFrame de datos demográficos, aplicando filtros y renombrando columnas clave.

    Parámetros:
    - df_demo (pd.DataFrame): DataFrame que contiene datos demográficos, incluyendo las columnas 'Provincias', 'Edad (3 grupos de edad)', 'Españoles/Extranjeros', y 'Sexo'.

    Retorna:
    - (pd.DataFrame): DataFrame transformado que contiene datos demográficos filtrados, con columnas renombradas, identificadores de provincia, y sin las entradas de totales.
    """

    # Filtramos para quedarnos solo con las entradas que nos interesan
    cond1 = df_demo['Provincias'] != 'TOTAL ESPAÑA'
    cond2 = df_demo['Edad (3 grupos de edad)'] != 'TOTAL EDADES'
    cond3 = df_demo['Españoles/Extranjeros'] != 'TOTAL'
    cond4 = df_demo['Españoles/Extranjeros'] != r'% Extranjeros'
    cond5 = df_demo['Sexo'] != 'Ambos sexos'

    df_demo = df_demo[cond1 & cond2 & cond3 & cond4 & cond5]

    # Total
    df_demo['Total'] = df_demo['Total'].str.replace('.','').astype(float)
    # Provincias
    df_demo['provincia'] = df_demo['Provincias'].str.split(r'\s(\D+)', regex=True, expand=True)[1]
    df_demo['provincia_id'] = df_demo['Provincias'].str.split(r'\s(\D+)', regex=True, expand=True)[0]
    # Quitar columnas
    df_demo.drop(columns='Provincias', inplace=True)
    df_demo.reset_index(drop=True, inplace=True)

    # Renombrar columnas
    df_demo.rename(columns={'Edad (3 grupos de edad)': 'grupo_edad', 'Españoles/Extranjeros': 'origen', 'Sexo': 'sexo', 'Año': 'year', 'Total': 'total'}, inplace=True)

    return df_demo