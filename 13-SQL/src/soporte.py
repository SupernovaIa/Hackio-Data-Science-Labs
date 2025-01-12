from tqdm import tqdm
from geopy.geocoders import Nominatim

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

        # Añadimos España a la query para obtener solamente resultados de España
        geocode = lambda query: geolocator.geocode("%s, Spain" % query)
        location = geocode(town)
        dc = {}
        dc['Nombre'] = town
        dc[0] = location.latitude
        dc[1] = location.longitude
        dc[2] = location.address

        locations.append(dc)

    return locations