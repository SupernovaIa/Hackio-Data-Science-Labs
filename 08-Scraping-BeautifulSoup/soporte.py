import requests
from bs4 import BeautifulSoup

def get_info(clave, obj):
    """
    Recupera información específica de un objeto web según la clave proporcionada.

    Parámetros:
    - clave (str): La clave que representa el tipo de información a recuperar ('Nombre', 'Categoria', 'Seccion', 'Descripcion', 'Dimensiones' o 'Imagen').
    - obj (objeto BeautifulSoup): El objeto HTML parseado del cual extraer la información.

    Retorna:
    - (str): El texto o la URL de la imagen extraída según la clave proporcionada.
    """

    if clave == 'Nombre':
        return obj.findAll('a', {'class': 'title'})[0].getText()

    elif clave == 'Categoria':
        return obj.findAll('a', {'class': 'tag'})[0].getText()
    
    elif clave == 'Seccion':
        secciones = []

        for sec in obj.findAll('div', {'class': 'cat-sec'}):
            secciones.append(sec.getText())
        
        return None if len(secciones) == 0 else " ".join(secciones)
    
    elif clave == 'Descripcion':
        return obj.findAll('div', {'class': 'article-container style-1'})[0].getText().strip('\n')
    
    elif clave == 'Dimensiones':
        return obj.findAll('div', {'class': 'price'})[0].getText().strip('\n')
    
    elif clave == 'Imagen':
        return f'https://atrezzovazquez.es/{obj.findAll('img')[0].get('src')}'
    
    else:
        raise KeyError('Clave no válida')
    

def llenar_objetos(items, lista_dc = []):
    """
    Llena una lista de diccionarios con información extraída de los elementos proporcionados.

    Parámetros:
    - items (list): Lista de elementos HTML parseados (BeautifulSoup objects) de los que se extraerá la información.
    - lista_dc (list, opcional): Lista de diccionarios a la cual se añadirán los nuevos diccionarios llenos. Por defecto, es una lista vacía.

    Retorna:
    - (list): Lista de diccionarios, cada uno con la información extraída ('Nombre', 'Categoria', 'Seccion', 'Descripcion', 'Dimensiones', 'Imagen') de los elementos.
    """
    # Lista de categorías
    claves = ['Nombre', 'Categoria', 'Seccion', 'Descripcion', 'Dimensiones', 'Imagen']

    # Recorremos cada item de los items
    for item in items: 

        # Definimos un diccionario vacío
        dc = {}

        # Llenamos el diccionario con la información que haya disponible
        for clave in claves:
            try:
                dc[clave] = get_info(clave, item)
            except:
                dc[clave] = None

        # Añadimos el diccionario lleno a la lista
        lista_dc.append(dc)

    # Devolvemos la lista
    return lista_dc


def rascar_pagina(pagina, lista_dc=[]):
    """
    Rasca una página web para extraer información de los elementos disponibles y llena una lista de diccionarios con esa información.

    Parámetros:
    - pagina (int): Número de página que se desea rascar.
    - lista_dc (list, opcional): Lista de diccionarios donde se añadirá la información extraída. Por defecto, es una lista vacía.

    Retorna:
    - (list or str): Lista de diccionarios con la información extraída de los objetos de la página. En caso de error, devuelve un mensaje con el código de estado HTTP.
    """

    # URL
    url_atr = f'https://atrezzovazquez.es/shop.php?search_type=-1&search_terms=&limit=48&page={pagina}'

    # Petición
    response_atr = requests.get(url_atr)

    # Si no está bien salimos
    if response_atr.status_code != 200:
        return f'Status code: {response_atr.status_code}'

    # Sopa
    sopa_atr = BeautifulSoup(response_atr.content, 'html.parser')

    # Sacamos los objetos de la página
    objetos = sopa_atr.findAll("div", {'class': "col-md-3 col-sm-4 shop-grid-item"})

    return llenar_objetos(objetos, lista_dc=lista_dc)