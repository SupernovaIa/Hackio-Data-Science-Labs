# Creación tablas
# Regiones
query_creacion_regiones = """
create table if not exists regiones ( 
    region VARCHAR(50) not null unique,
    region_id INT primary key
);
"""

# Provincias
query_creacion_provincias = """
create table if not exists provincias (
    provincia VARCHAR(50) not null unique,
    provincia_id INT primary key,
    region_id INT not null references regiones(region_id) on delete cascade
);
"""

# Demográfico
query_creacion_demografico = """
create table if not exists demografico (
    demografico_id SERIAL primary key,
    grupo_edad VARCHAR(100) not null, 
    origen VARCHAR(50) not null, 
    sexo VARCHAR(50) not null,
    year INT not null, 
    total FLOAT not null, 
    provincia_id INT not null references provincias(provincia_id) on delete cascade
);
"""

# Económico
query_creacion_economico = """
create table if not exists economico (
    economico_id SERIAL primary key,
    categories VARCHAR(400) not null,
    year INT not null,
    total FLOAT not null, 
    provincia_id INT not null references provincias(provincia_id) on delete cascade,
    region_id INT references regiones(region_id) on delete set null
);
"""

# Generación
query_creacion_generacion = """ 
create table if not exists generacion (
    generacion_id SERIAL primary key,
    value FLOAT not null,
    percentage FLOAT not null,
    category VARCHAR(50) not null,
    region_id INT references regiones(region_id) on delete set null,
    year INT not null,
    month INT not null
);
"""

# Demanda
query_creacion_demanda = """
create table if not exists demanda (
    demanda_id SERIAL primary key,
    value FLOAT not null,
    percentage FLOAT not null,
    region_id INT references regiones(region_id) on delete set null,
    year INT not null,
    month INT not null
);
"""

# Inserción datos
# Regiones
query_insercion_regiones = """
INSERT INTO regiones (region, region_id)
VALUES
(%s, %s)
"""

# Provincias
query_insercion_provincias = """
INSERT INTO provincias (provincia, provincia_id, region_id)
VALUES
(%s, %s, %s)
"""

# Demográfico
query_insercion_demografico = """
INSERT INTO demografico (grupo_edad, origen, sexo, year, total, provincia_id)
VALUES
(%s, %s, %s, %s, %s, %s)
"""

# Económico
query_insercion_economico = """
INSERT INTO economico (categories, year, total, provincia_id, region_id)
VALUES 
(%s, %s, %s, %s, %s)
"""

# Generación
query_insercion_generacion = """ 
INSERT INTO generacion (value, percentage, category, region_id, year, month)
VALUES
(%s, %s, %s, %s, %s, %s)
"""

# Demanda
query_insercion_demanda = """
INSERT INTO demanda (value, percentage, region_id, year, month)
VALUES
(%s, %s, %s, %s, %s)
"""