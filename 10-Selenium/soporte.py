import pandas as pd
from time import sleep
from selenium import webdriver  # Selenium es una herramienta para automatizar la interacción con navegadores web.
from webdriver_manager.chrome import ChromeDriverManager  # ChromeDriverManager gestiona la instalación del controlador de Chrome.
from selenium.webdriver.common.keys import Keys  # Keys es útil para simular eventos de teclado en Selenium.
from selenium.webdriver.support.ui import Select  # Select se utiliza para interactuar con elementos <select> en páginas web.
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def crear_df(tabla):
    """
    Convierte una cadena de texto de tabla cruda en un DataFrame de pandas formateado con nombres de columnas y tipos de datos adecuados.

    Parámetros:
    - tabla (str): Una cadena de texto multilínea que contiene datos meteorológicos con valores separados por espacios y unidades.

    Retorna:
    - (pd.DataFrame): Un DataFrame con columnas para diversas métricas meteorológicas como temperatura, humedad, velocidad, presión y precipitación, indexado por fecha. Las unidades se eliminan y los valores se convierten a float para análisis.
    """
    df = pd.DataFrame(tabla.split("\n"))
    df.drop([0,1], inplace=True)
    df[0] = df[0].str.replace(' °F', '')
    df[0] = df[0].str.replace(' mph', '')
    df[0] = df[0].str.replace(' in', '')
    df[0] = df[0].str.replace(' %', '')
    df = df[0].str.split(' ', expand=True)
    df.columns = ["Date", "High Temp (ºF)", "Avg Temp (ºF)", "Low T (ºF)", "High Dew Pt (ºF)", "Avg Dew Pt (ºF)", "Low Dew Pt (ºF)", "High Hum (%)", "Avg Hum (%)", "Low Hum (%)", "High Spe (mph)", "Avg Spe (mph)", "Low Spe (mph)", "High Press (in)", "Low Press (in)", "Sum Prec (In)"]
    df.set_index('Date', inplace=True)
    df = df.applymap(lambda x: float(x))
    return df


def obtener_codigo(driver, municipio):
    """
    Obtiene el código del municipio desde la página web de Wunderground utilizando un navegador controlado por Selenium.

    Parámetros:
    - driver (WebDriver): Instancia del navegador de Selenium.
    - municipio (str): Nombre del municipio para construir la URL y buscar la información correspondiente.

    Retorna:
    - (str): El código del municipio extraído de la página web de Wunderground.
    """

    # Definimos la web
    url = f"https://www.wunderground.com/weather/es/{municipio}"
    driver.get(url)
    driver.maximize_window()

    # Dar 10 segundos hasta encontrar la ventana de cookies
    iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located(('xpath','//*[@id="sp_message_iframe_1165301"]')))
    # Cambiar al 'iframe' de cookies
    driver.switch_to.frame(iframe)
    # Esperar
    sleep(3)
    # Rechazar cookies
    driver.find_element("css selector","#notice > div.message-component.message-row.cta-buttons-container > div.message-component.message-column.cta-button-column.reject-column > button").click()

    # Vovler al contenido principal
    driver.switch_to.default_content() 

    # Esperar
    sleep(3)
    # Hacer click en la estación
    driver.find_element("css selector", "#inner-content > div.region-content-top > lib-city-header > div:nth-child(1) > div > div > a.station-name").click()
    # Esperar
    sleep(3)
    # Tomar el código
    codigo_municipio = driver.find_element("css selector", "#inner-content > div.region-content-top > app-dashboard-header > div.dashboard__header.small-12.ng-star-inserted > div > div.heading > h1").text.split(' - ')[1]
    
    return codigo_municipio


def obtener_dfs(driver, codigo):
    """
    Obtiene una lista de DataFrames mensuales desde la página web de Wunderground utilizando el código de un municipio y Selenium.

    Parámetros:
    - driver (WebDriver): Instancia del navegador de Selenium.
    - codigo (str): Código del municipio para construir la URL y acceder a los datos históricos.

    Retorna:
    - (list[pd.DataFrame]): Una lista de DataFrames que contienen los datos meteorológicos mensuales para cada mes del año 2024.
    """
    dfs = []
    tablas = []
    
    for i in tqdm(range(1,10)):
        # Redireccionar
        val = f'https://www.wunderground.com/dashboard/pws/{codigo}/table/2024-{i}-1/2024-{i}-1/monthly'
        wait = WebDriverWait(driver, 10)
        driver.get(val)
        wait.until(EC.url_to_be(val))
        # Obtener tabla
        tabla = driver.find_element("css selector", "#main-page-content > div > div > div > lib-history > div.history-tabs > lib-history-table > div > div").text
        tablas.append(tabla)
        df = crear_df(tabla)
        dfs.append(df)

    return dfs


def obtener_dfs_municipio(municipio):
    """
    Obtiene un DataFrame consolidado con los datos meteorológicos mensuales de un municipio específico utilizando Selenium.

    Parámetros:
    - municipio (str): Nombre del municipio para buscar y obtener sus datos históricos desde la página web de Wunderground.

    Retorna:
    - (pd.DataFrame): Un DataFrame que contiene los datos meteorológicos mensuales concatenados para el municipio especificado.
    """

    # Iniciar proceso
    driver = webdriver.Chrome()
    # Obtener código
    codigo_municipio = obtener_codigo(driver, municipio)
    # Obtener DataFrames
    dfs = obtener_dfs(driver, codigo_municipio)
    # Cerrar ventana
    driver.close()

    return pd.concat(dfs)