
-- Ejercicio  1: Encuentra el nombre de las pistas y los títulos de los álbumes a los que pertenecen.
select t."Name" as "Track", a."Title" as "Album"
from "Track" t 
inner join "Album" a ON a."AlbumId" = t."AlbumId" ;

-- Ejercicio 2: Lista los nombres de los artistas y los títulos de sus álbumes. Ordena los resultados por artista.
select a."Name" as "Artist", a2."Title" as "Album"
from "Artist" a 
inner join "Album" a2 ON a."ArtistId" = a2."ArtistId"
order by a."Name" ;

-- Ejercicio 3: Encuentra los nombres de los clientes y los totales de sus facturas. Extrae solo los 5 clientes con mayor total. 
select c."FirstName" as "Name", i."Total" as "Total"
from "Customer" c 
inner join "Invoice" i on i."CustomerId" = c."CustomerId" 
order by i."Total" desc
limit 5 ;

-- Ejercicio 4: Lista los nombres de los empleados y los nombres de los clientes que les han sido asignados.
select e."FirstName" as "Employee", c."FirstName" as "Customer"
from "Employee" e 
inner join "Customer" c on c."SupportRepId" = e."EmployeeId" ;

-- Ejercicio 5: Muestra los ID de las facturas y los nombres de las pistas incluidas en esas facturas.
select il."InvoiceId" , t."Name" 
from "InvoiceLine" il 
inner join "Track" t on il."TrackId" = t."TrackId" ;

-- Ejercicio 6: Encuentra los nombres de los artistas y los géneros de sus pistas.
select  g."Name" as "Genre", 
		t."Name" as "Track", 
		a."Name" as "Artista"
from "Track" t 
inner join "Genre" g on t."GenreId" = g."GenreId"
inner join "Album" al on al."AlbumId" = t."AlbumId" 
inner join "Artist" a on a."ArtistId" = al."ArtistId" ;

--  Ejercicio 7: Muestra los nombres de las pistas y el tipo de medio en el que están disponibles.
select t."Name" as "Track", mt."Name" as "Media Type"
from "MediaType" mt 
inner join "Track" t on t."MediaTypeId" = mt."MediaTypeId" ;

-- Ejercicio 8: Encuentra todas las pistas y, si existen, muestra los nombres de los géneros a los que pertenecen. Incluye también las pistas que no tienen un género asignado.**
select t."Name" as "Track", g."Name" as "Genre"
from "Track" t
left join "Genre" g on g."GenreId" = t."GenreId" ;

-- Ejercicio 9: Muestra todos los clientes y, si existen, muestra las facturas que han realizado. Incluye también los clientes que no tienen ninguna factura.
select c."FirstName" as "Name", i."InvoiceId" as "Invoice"
from "Customer" c 
left join "Invoice" i on c."CustomerId" = i."CustomerId"
order by c."FirstName" ;

-- Ejercico 10: Encuentra todos los álbumes y, si existen, muestra los nombres de los artistas que los crearon. Incluye también los álbumes que no tienen un artista asignado (aunque en este caso en la base de datos de Chinook, todos los álbumes tienen un artista asignado).
select a."Title" as "Album", a2."Name" as "Artist"
from "Album" a 
left join "Artist" a2 on a."ArtistId" = a2."ArtistId" 

-- Ejercicio 11: Cuenta cuántas pistas hay en cada género. Ordena los generos en función del número de canciones de mayor a menor. 
select g."Name", count(t."Name") as "Numbre of tracks"
from "Genre" g 
left join "Track" t on g."GenreId" = t."GenreId" 
group by g."Name" 
order by count(t."Name") desc ;

--  Ejercicio 12: Muestra los títulos de los álbumes y la duración total de todas las pistas en cada álbum.
select a."Title" as "Album", sum(t."Milliseconds") as "Total"
from "Album" a 
left join "Track" t on a."AlbumId" = t."AlbumId" 
group by a."Title" 
order by sum(t."Milliseconds") desc ;

-- Ejercicio 14: Encuentra los nombres de los clientes y el total gastado por cada uno.
select c."FirstName" as "Customer", sum(i."Total") as "Total spent"
from "Customer" c 
left join "Invoice" i on c."CustomerId" = i."CustomerId" 
group by c."FirstName"
order by sum(i."Total") desc ;

-- Ejercicio 15: Encuentra todos los empleados y, si existen, muestra los nombres de los clientes que tienen asignados. Incluye también los empleados que no tienen clientes asignados.
select e."FirstName" as "Employee", c."FirstName" as "Customer"
from "Employee" e 
left join "Customer" c on e."EmployeeId" = c."SupportRepId" 
order by e."FirstName" ;