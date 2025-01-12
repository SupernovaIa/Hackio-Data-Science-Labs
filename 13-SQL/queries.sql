-- Ejercicio 1. Queries Generales

-- 1.1. Calcula el promedio más bajo y más alto de temperatura.
SELECT max(temperatura_c), min(temperatura_c)
FROM tiempo t ; 
-- La temperatura máxima 32ºC y la mínima 9ºC


-- 1.2. Obtén los municipios en los cuales coincidan las medias de la sensación térmica y de la temperatura.
SELECT m.nombre, round(avg(t.temperatura_c), 1) AS mediatemp, round(avg(t.sensacion_termica_c), 1) AS mediasenstermica
FROM tiempo t
NATURAL JOIN municipios m 
GROUP BY m.municipio_id 
HAVING avg(t.temperatura_c) = avg(t.sensacion_termica_c) ;


-- 1.3. Obtén el local más cercano de cada municipio
SELECT m.nombre AS municipio, u.name AS local_name, u.distance
FROM ubicaciones u
JOIN (
    SELECT municipio_id, MIN(distance) AS min_distance
    FROM ubicaciones
    GROUP BY municipio_id
) AS min_distances
ON u.municipio_id = min_distances.municipio_id
AND u.distance = min_distances.min_distance
JOIN municipios m
ON u.municipio_id = m.municipio_id ;


-- 1.4. Localiza los municipios que posean algún localizador a una distancia mayor de 2000 y que posean al menos 25 locales.
SELECT *
FROM (
	SELECT u.municipio_id, count(u.fsq_id)
	FROM ubicaciones u 
	NATURAL JOIN municipios m 
	GROUP BY u.municipio_id 
	HAVING count(u.fsq_id) >= 25) 
	AS numeromun
INNER JOIN municipios m 
	ON numeromun.municipio_id = m.municipio_id 
INNER JOIN ubicaciones u 
	ON u.municipio_id = m.municipio_id 
WHERE u.distance > 2000 ;


-- 1.5. Teniendo en cuenta que el viento se considera leve con una velocidad media de entre 6 y 20 km/h, moderado con una media de entre 21 y 40 km/h, fuerte con media de entre 41 y 70 km/h y muy fuerte entre 71 y 120 km/h. 
-- Calcula cuántas rachas de cada tipo tenemos en cada uno de los días. 
-- Este ejercicio debes solucionarlo con la sentencia CASE de SQL.
SELECT
    fecha,
    COUNT(CASE WHEN (vel_viento BETWEEN 6 AND 20) THEN 1 END) AS rachas_leves,
    COUNT(CASE WHEN (vel_viento BETWEEN 21 AND 40) THEN 1 END) AS rachas_moderaadas,
    COUNT(CASE WHEN (vel_viento BETWEEN 41 AND 70) THEN 1 END) AS rachas_fueres,
    COUNT(CASE WHEN (vel_viento BETWEEN 71 AND 120) THEN 1 END) AS rachas_muy_fueres
FROM tiempo t 
GROUP BY fecha ;


-- Ejercicio 2. Vistas

-- 2.1. Crea una vista que muestre la información de los locales que tengan incluido el código postal en su dirección.
CREATE VIEW VistaLocalesCodigoPostal AS 
SELECT * 
FROM ubicaciones u 
WHERE u.address LIKE '%28___%' ;


-- 2.2. Crea una vista con los locales que tienen más de una categoría asociada.
CREATE VIEW VistaLocalesCategorias AS 
SELECT name, address , count(DISTINCT category_id) AS NumCat
FROM ubicaciones u 
GROUP BY name, address 
HAVING count(DISTINCT category_id) > 1 ;
-- No me sale ninguno porque he eliminado duplicados con el mismo fsq_id
-- Aun así, todas las entradas tienen un valor único de categoría salvo un parque/monumento


-- 2.3. Crea una vista que muestre el municipio con la temperatura más alta de cada día
CREATE VIEW VistaMunicipiosCalidos AS 
SELECT m.nombre , t.fecha, max(maxtemp)
FROM (
	SELECT fecha, max(temperatura_c) AS maxtemp
	FROM tiempo t 
	GROUP BY fecha ) AS TempMax
INNER JOIN tiempo t ON t.fecha = TempMax.fecha
INNER JOIN municipios m ON m.municipio_id = t.municipio_id 
WHERE t.temperatura_c = TempMax.maxtemp
GROUP BY m.nombre, t.fecha
ORDER BY t.fecha 
-- Vemos que algunos están empatados en el máximo lo cual tiene sentido al ser municipios próximos geográficamente


-- 2.4. Crea una vista con los municipios en los que haya una probabilidad de precipitación mayor del 100% durante mínimo 7 horas.

-- ???


-- 2.5. Obtén una lista con los parques de los municipios que tengan algún castillo.
CREATE VIEW VistaMunicipiosCastillosParques AS 
SELECT u."name" AS Nombre, u.address AS Direccion , m.nombre AS Municipio, u.category_id 
FROM (
	SELECT municipio_id 
	FROM ubicaciones u 
	WHERE category_id = 3) AS TablaCastillos
INNER JOIN ubicaciones u ON TablaCastillos.municipio_id = u.municipio_id
INNER JOIN municipios m ON u.municipio_id = m.municipio_id 
WHERE category_id = 0;
-- Castillos es categoría 3 y parque es 0


-- Ejercicio 3. Tablas Temporales

-- 3.1. Crea una tabla temporal que muestre cuántos días han pasado desde que se obtuvo la información de la tabla AEMET.
create temporary table DiasTranscurridos as
SELECT (current_date - fecha) AS DiasTranscurridos, *
FROM tiempo t ;


-- 3.2. Crea una tabla temporal que muestre los locales que tienen más de una categoría asociada e indica el conteo de las mismas
create temporary table CategoriasDobles as
SELECT name, address , count(DISTINCT category_id) AS NumCat
FROM ubicaciones u 
GROUP BY name, address 
HAVING count(DISTINCT category_id) > 1 ;
-- Al igual que en 2.2 no me sale ninguno porque he eliminado duplicados con el mismo fsq_id
-- Aun así, todas las entradas tienen un valor único de categoría salvo un parque/monumento


-- 3.3. Crea una tabla temporal que muestre los tipos de cielo 
-- para los cuales la probabilidad de precipitación mínima de los promedios de cada día es 5.
create temporary table TiposCieloPrecip as
SELECT cielo_id, MIN(promediodiario) AS probminima
FROM (
    SELECT cielo_id, 
           DATE(fecha) AS dia,
           AVG(prob_precip::integer) AS promediodiario
    FROM tiempo
    WHERE prob_precip != 'Riesgo'
    GROUP BY cielo_id, DATE(fecha)) AS promedios
GROUP BY cielo_id
HAVING MIN(promediodiario) >= 5
ORDER BY cielo_id ;


-- 3.4. Crea una tabla temporal que muestre el tipo de cielo más y menos repetido por municipio.
create temporary table TiposCieloMunicipios as
SELECT max_repetidos.municipio_id, 
       max_repetidos.cielo_id AS cielo_mas_repetido, 
       max_repetidos.max_count AS max_repeticiones,
       min_repetidos.cielo_id AS cielo_menos_repetido, 
       min_repetidos.min_count AS min_repeticiones
FROM (
    SELECT municipio_id, cielo_id, count(cielo_id) AS max_count
    FROM tiempo t
    GROUP BY municipio_id, cielo_id
    HAVING count(cielo_id) = (
        SELECT MAX(conteo)
        FROM (
            SELECT count(cielo_id) AS conteo
            FROM tiempo t2
            WHERE t2.municipio_id = tiempo.municipio_id
            GROUP BY t2.cielo_id
        ) AS max_subquery
    )
) AS max_repetidos
JOIN (
    SELECT municipio_id, cielo_id, count(cielo_id) AS min_count
    FROM tiempo
    GROUP BY municipio_id, cielo_id
    HAVING count(cielo_id) = (
        SELECT MIN(conteo)
        FROM (
            SELECT count(cielo_id) AS conteo
            FROM tiempo t3
            WHERE t3.municipio_id = tiempo.municipio_id
            GROUP BY t3.cielo_id
        ) AS min_subquery
    )
) AS min_repetidos
ON max_repetidos.municipio_id = min_repetidos.municipio_id
ORDER BY max_repetidos.municipio_id ;
-- Dos subtablas unidas con los cielos más y menos repetidos por municipio


-- Ejercicio 4. SUBQUERIES

-- 4.1. Necesitamos comprobar si hay algún municipio en el cual no tenga ningún local registrado.
SELECT m.nombre AS Municipio, tablaids.mun_id AS ID_Municipio
FROM (
	SELECT fsq_id, m.municipio_id AS mun_id
	FROM ubicaciones u 
	RIGHT JOIN municipios m
		ON m.municipio_id = u.municipio_id 
	ORDER BY m.municipio_id) AS tablaids
INNER JOIN municipios m ON tablaids.mun_id = m.municipio_id
WHERE fsq_id IS NULL ;

-- Vemos que hay 89 municipios sin locales


-- 4.2. Averigua si hay alguna fecha en la que el cielo se encuentre "Muy nuboso con tormenta".
SELECT DISTINCT fecha
FROM tiempo
WHERE cielo_id IN (
	SELECT cielo_id 
	FROM cielo c 
	WHERE estado = 'Muy nuboso con tormenta') ;
-- Vemos que solamente el 29 de agosto

	
-- 4.3. Encuentra los días en los que los avisos sean diferentes a "Sin riesgo".
SELECT fecha 
FROM tiempo t 
WHERE avisos != 'Sin riesgo'
GROUP BY fecha ;
-- No necesito subquery
	

-- 4.4. Selecciona el municipio con mayor número de locales.
SELECT nombre AS Municipio, numero
FROM (
	SELECT count(fsq_id) AS numero, municipio_id 
	FROM ubicaciones u 
	GROUP BY municipio_id ) AS tablacount
INNER JOIN municipios m 
	ON m.municipio_id = tablacount.municipio_id
ORDER BY numero DESC 
LIMIT 1 ;
-- Para sorpresa de nadie es Madrid


-- 4.5. Obtén los municipios cuya media de sensación térmica sea mayor que la media total.
SELECT  m.nombre , round(tablatemps.st, 1) AS SensacionTermica, round(tablatemps.tem, 1) AS TemperaturaMedia
FROM (
	SELECT t.municipio_id AS id , avg(sensacion_termica_c) AS st, avg(temperatura_c) AS tem
	FROM tiempo t 
	GROUP BY municipio_id ) AS tablatemps
INNER JOIN municipios m 
	ON m.municipio_id = tablatemps.id
WHERE tablatemps.st > tablatemps.tem ;


-- 4.6. Selecciona los municipios con más de dos fuentes.
SELECT m.nombre , count(category_id) 
FROM ubicaciones u 
INNER JOIN municipios m 
	ON m.municipio_id = u.municipio_id 
WHERE category_id IN (
	SELECT category_id
	FROM categorias c
	WHERE category = 'Fountain')
GROUP BY m.municipio_id 
HAVING count(category_id) > 2 ;
-- Sorprendentemente solo Madrid tiene más de dos fuentes


-- 4.7. Localiza la dirección de todos los estudios de cine que estén abiertos en el municipio de "Madrid".
SELECT "name" AS nombre, address AS direccion
FROM ubicaciones u 
WHERE municipio_id IN (
	SELECT municipio_id 
	FROM municipios m 
	WHERE "nombre" = 'Madrid'
)
AND closed_bucket_id IN (
	SELECT closed_bucket_id 
	FROM closed_bucket cb 
	WHERE closed_bucket != 'Unsure'
) ;


-- 4.8. Encuentra la máxima temperatura para cada tipo de cielo.
SELECT estado AS TipoCielo, maximo
FROM (
	SELECT cielo_id , max(temperatura_c) AS maximo
	FROM tiempo t 
	GROUP BY cielo_id) AS tablamax
INNER JOIN cielo c
	ON c.cielo_id = tablamax.cielo_id
ORDER BY c.cielo_id ;


-- 4.9. Muestra el número de locales por categoría que muy probablemente se encuentren abiertos.
SELECT category_id AS categoria , count(fsq_id) AS numero_locales
FROM ubicaciones u 
WHERE closed_bucket_id IN (
	SELECT closed_bucket_id
	FROM closed_bucket cb
	WHERE closed_bucket = 'VeryLikelyOpen')
GROUP BY category_id ;


-- BONUS. 4.10. Encuentra los municipios que tengan más de 3 parques, los cuales se encuentren a una distancia menor de las coordenadas de su municipio correspondiente que la del Parque Pavia. Además, el cielo debe estar despejado a las 12.
SELECT m.nombre , count(category_id) 
FROM ubicaciones u 
INNER JOIN municipios m 
	ON m.municipio_id = u.municipio_id 
WHERE category_id IN (
	SELECT category_id
	FROM categorias c
	WHERE category = 'Park')
	AND u.distance < (
		SELECT distance 
		FROM ubicaciones u 
		WHERE "name" = 'Parque Pavia')
GROUP BY m.municipio_id 
HAVING count(category_id) > 3 ;
-- No tengo la hora en fechas así que esa parte no la puedo hacer