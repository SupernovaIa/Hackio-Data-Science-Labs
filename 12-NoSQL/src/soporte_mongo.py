# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd

# Para trabajar con Mongo
# ------------------------------------------------------------------------------
import pymongo


def query_near(punto_referencia, radio, coleccion, col_geometria_coleccion):
    """
    Realiza una consulta a una colección de MongoDB para encontrar documentos
    que estén cerca de un punto de referencia dado dentro de un radio específico.

    Args:
        punto_referencia (dict): Coordenadas del punto de referencia en formato de diccionario
            con las claves 'type' (tipo de geometría, por ejemplo, "Point") y 'coordinates'
            (coordenadas en formato [longitud, latitud]).
        radio (int): Radio de búsqueda en metros.
        coleccion: Colección de MongoDB sobre la que realizar la consulta.

    Returns:
        pandas.DataFrame: DataFrame que contiene los documentos encontrados cercanos al punto
        de referencia dentro del radio especificado.
    """
    # Realiza una consulta a la colección buscando documentos cercanos al punto de referencia
    # dentro del radio especificado utilizando el operador $near de MongoDB.
    find_near_df = pd.DataFrame(coleccion.find(
        {'geometry': {
            '$near': {f'${col_geometria_coleccion}': punto_referencia, '$maxDistance': radio}
        }}
    ))

    return find_near_df


def query_geonear(punto_referencia, nombre_nuevo_campo, radio, coleccion):
    """
    Realiza una consulta geoespacial utilizando el método geoNear de MongoDB.

    Args:
        punto_referencia (dict): Coordenadas del punto de referencia en formato GeoJSON.
        nombre_nuevo_campo (str): Nombre del nuevo campo que contendrá las distancias calculadas.
        radio (float): Radio máximo de búsqueda en metros.
        coleccion (pymongo.collection.Collection): Colección en la que se realizará la consulta.

    Returns:
        pandas.DataFrame: DataFrame con los documentos encontrados y las distancias calculadas.
    """
    
    # Define la consulta utilizando el método aggregate de la colección
    query = [{
        "$geoNear": {
            'near': punto_referencia,  # Punto de referencia para la búsqueda
            'distanceField': nombre_nuevo_campo,  # Campo que contendrá las distancias calculadas
            'maxDistance': radio,  # Distancia máxima de búsqueda en metros
            'spherical': True  # Especifica que se deben calcular las distancias en una esfera
        }
    }]
    
    # Ejecuta la consulta utilizando el método aggregate y convierte el resultado en un DataFrame de Pandas
    return pd.DataFrame(coleccion.aggregate(query))


def conexion_mongo(nombre_base_datos, nombre_coleccion):
    """
    Establece una conexión con MongoDB y selecciona una base de datos y una colección.

    Esta función intenta conectarse a MongoDB utilizando pymongo.MongoClient().
    Verifica si la base de datos y la colección especificadas existen en el servidor MongoDB.
    Si la conexión es exitosa, devuelve el cliente de MongoDB y la conexión a la colección especificada.

    Args:
        nombre_base_datos (str): El nombre de la base de datos a la que se desea conectar.
        nombre_coleccion (str): El nombre de la colección dentro de la base de datos a la que se desea conectar.

    Returns:
        pymongo.MongoClient: El cliente de MongoDB.
        pymongo.collection.Collection: La conexión a la colección especificada en MongoDB.

    Raises:
        ValueError: Si la base de datos o la colección especificada no existen en el servidor MongoDB.
    """
    try:
        # Intentar conectarse a MongoDB
        cliente = pymongo.MongoClient()
        
        # Verificar si la base de datos existe
        if nombre_base_datos not in cliente.list_database_names():
            raise ValueError(f"La base de datos '{nombre_base_datos}' no existe.")
        
        # Conectarse a la base de datos
        bbdd = cliente[nombre_base_datos]

        # Verificar si la colección existe
        if nombre_coleccion not in bbdd.list_collection_names():
            raise ValueError(f"La colección '{nombre_coleccion}' no existe en la base de datos '{nombre_base_datos}'.")
        
        # Conectarse a la colección
        con = bbdd[nombre_coleccion]

        return cliente, con

    except Exception as e:
        print(f"Error al conectar con MongoDB: {e}")



def crear_bbbd_colecciones(conexion, nombre_base_datos, nombre_colecciones):
    """
    Crea una nueva base de datos y sus colecciones en MongoDB.

    Esta función utiliza la conexión proporcionada para crear una nueva base de datos y sus colecciones.
    Si alguna de las colecciones ya existe, se imprime un mensaje informativo y se omite su creación.

    Args:
        conexion: pymongo.MongoClient o pymongo.collection.Collection. La conexión a la instancia de MongoDB o la base de datos existente en la que se crearán las colecciones.
        nombre_base_datos (str): El nombre de la base de datos que se creará o en la que se agregarán las colecciones.
        nombre_colecciones (list): Una lista de nombres de colecciones que se crearán en la base de datos.

    Returns:
        pymongo.collection.Collection: La instancia de la base de datos donde se han creado las colecciones.

    Raises:
        TypeError: Si la conexión proporcionada no es válida (no es una instancia de pymongo.MongoClient o pymongo.collection.Collection).
    """
    try:
        # Verificar el tipo de conexión proporcionada
        if not isinstance(conexion, (pymongo.MongoClient, pymongo.collection.Collection)):
            raise TypeError("La conexión proporcionada no es válida. Debe ser una instancia de pymongo.MongoClient o pymongo.collection.Collection.")

        # Crear una nueva base de datos
        sitios = conexion[nombre_base_datos]

        # Crear las colecciones
        for categoria in nombre_colecciones:
            try: 
                sitios.create_collection(categoria)
            except pymongo.errors.CollectionInvalid:
                print(f"La colección {categoria} ya existe.")
        
        return sitios

    except Exception as e:
        print(f"Error al crear la base de datos y colecciones: {e}")


def insertar_datos_en_colecciones(df, base_datos, nombre_categorias):
    """
    Inserta datos en las colecciones creadas en MongoDB.

    Esta función inserta los datos en las colecciones correspondientes según la categoría proporcionada.

    Args:
        resultados (DataFrame): El DataFrame que contiene los datos a insertar en las colecciones.
        bbdd_api (pymongo.collection.Collection): La conexión a la base de datos donde se encuentran las colecciones.
        categorias (list): Una lista de categorías que se utilizarán para filtrar y insertar los datos en las colecciones.

    Returns:
        None
    """
    for categoria in nombre_categorias: 
        # Filtramos el DataFrame
        df_insertar = df[df["category"] == categoria]

        # Convertimos el DataFrame filtrado en una lista de diccionarios
        lista_diccionarios = df_insertar.T.to_dict().values()

        # Cambiamos el nombre de la clave 'fsq_id' por '_id' para que Mongo no genere automáticamente IDs
        lista_diccionarios = [{"_id": d.pop("fsq_id"), **d} for d in lista_diccionarios]

        for documento in lista_diccionarios:
            try: 
                # Insertamos los datos en las colecciones con el método 'insert_many()'
                insercion = base_datos[categoria].insert_one(documento)
            
            except pymongo.errors.DuplicateKeyError:
                print(f"Este 'ID' ({documento['_id']}) ya existe en la base de datos.")
            except Exception as e:
                print(f"Error al insertar datos en la colección {categoria}: {e}")
