# -*- coding: utf-8 -*-
"""
Interfaz Interactiva con Streamlit - El Salvador
---------------------------------------------------------------------------------------
• Buffer de Bounding Box para eliminar artefactos de borde en interpolación C1.
• Promedio de modelos a nivel tabular + Ensamble espacial.
• Resumen Semanal de 7 días exactos.
• Renderizado de mapas con relieve topográfico (Hillshade).
"""

import json, os, sys, time, io, zipfile
from typing import Dict, Any, List
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.ndimage import gaussian_filter

import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# =====================
#  CONFIGURACIÓN BASE
# =====================
st.set_page_config(
    page_title="Visor Meteorológico Avanzado - El Salvador",
    page_icon="🗺️",
    layout="wide"
)

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_ESTACIONES = os.path.join(DIRECTORIO_ACTUAL, "elsalvador_mapa.geojson")
CARPETA_TIFFS = os.path.join(DIRECTORIO_ACTUAL, "tiffs_salida")
CARPETA_MAPAS = os.path.join(DIRECTORIO_ACTUAL, "mapas_png")

MODELOS = {"gfs_global": "GFS", "ecmwf_ifs025": "ECMWF"}
DAILY_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"]
VARIABLES_EXPORTAR = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"]

START_DATE = date.today().strftime("%Y-%m-%d")
END_DATE = (date.today() + timedelta(days=15)).strftime("%Y-%m-%d")

TIMEOUT_S = 60
BATCH_SIZE = 50
RESOLUCION_TIFF = 0.008
BUFFER_GRADOS = 0.35  # Amortiguación externa (~35-40km) para evitar artefactos en bordes

# --- PALETA DE COLOR RGB PARA PRECIPITACIÓN ---
colores_rgb = np.array([
    [255, 255, 255],  # 0-1
    [230, 245, 255],  # 1-2.5
    [190, 225, 255],  # 2.5-5
    [140, 205, 255],  # 5-10
    [90,  170, 255],  # 10-15
    [50,  120, 255],  # 15-20
    [80,  80,  255],  # 20-25
    [120, 60,  255],  # 25-30
    [170, 30,  255],  # 30-40
    [255, 0,   255],  # 40-50
    [180, 0,   180]   # > 50
]) / 255.0

CMAP_PRECIP = mcolors.ListedColormap(colores_rgb)

BOUNDS_PRECIP_DIARIO = [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50]
NORM_PRECIP_DIARIO = mcolors.BoundaryNorm(BOUNDS_PRECIP_DIARIO, ncolors=len(colores_rgb), extend='max')

BOUNDS_PRECIP_SEMANAL = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
NORM_PRECIP_SEMANAL = mcolors.BoundaryNorm(BOUNDS_PRECIP_SEMANAL, ncolors=len(colores_rgb), extend='max')

ESTILOS_MAPA = {
    "precipitation_sum": {
        "cmap": CMAP_PRECIP,
        "norm_diario": NORM_PRECIP_DIARIO,
        "norm_semanal": NORM_PRECIP_SEMANAL,
        "label_diario": "Precipitación Diaria (mm)",
        "label_semanal": "Precipitación Acumulada (mm)",
        "title": "Precipitación Pronosticada",
        "ticks_diario": [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50],
        "ticks_semanal": [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250],
    },
    "temperature_2m_max": {
        "cmap": "YlOrRd",
        "norm_diario": None,
        "norm_semanal": None,
        "label_diario": "Temperatura (°C)",
        "label_semanal": "Temperatura Máxima Promedio (°C)",
        "title": "Temperatura Máxima",
    },
    "temperature_2m_min": {
        "cmap": "YlGnBu_r",
        "norm_diario": None,
        "norm_semanal": None,
        "label_diario": "Temperatura (°C)",
        "label_semanal": "Temperatura Mínima Promedio (°C)",
        "title": "Temperatura Mínima",
    }
}

session = requests.Session()
session.headers.update({"User-Agent": "Sire-Downloader/Streamlit"})

# =====================
#  FUNCIONES AUXILIARES
# =====================
def limpiar_fecha_str(fecha_val: Any) -> str:
    return str(fecha_val).split()[0].split("T")[0]

def extraer_centroide(coords: Any) -> tuple:
    puntos = []
    def aplanar(c):
        if isinstance(c[0], (float, int)):
            puntos.append(c)
        else:
            for sub in c:
                aplanar(sub)
    aplanar(coords)
    if not puntos:
        return 0.0, 0.0
    return sum(p[0] for p in puntos) / len(puntos), sum(p[1] for p in puntos) / len(puntos)

@st.cache_data
def cargar_estaciones_ampliadas(ruta: str, buffer_deg: float) -> tuple:
    """ Carga estaciones del GeoJSON y genera una rejilla de amortiguación perimetral (Buffer). """
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró el archivo GeoJSON: {ruta}")
        return {}, None

    gdf = gpd.read_file(ruta)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    with open(ruta, "r", encoding="utf-8") as f:
        raw = json.load(f)

    estaciones = {}
    if isinstance(raw, dict) and raw.get("type") == "FeatureCollection":
        for i, feat in enumerate(raw.get("features", [])):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            est_id = str(props.get("id", props.get("ID", props.get("CODIGO", f"GEO_{i+1}"))))
            nombre = str(props.get("name", props.get("NOMBRE", props.get("departamen", props.get("MUNICIPIO", est_id)))))
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])

            if gtype == "Point":
                lon, lat = coords[0], coords[1]
            elif gtype in ["Polygon", "MultiPolygon"]:
                lon, lat = extraer_centroide(coords)
            else:
                continue

            estaciones[est_id] = {"name": nombre, "lat": float(lat), "lon": float(lon)}

    # Crear rejilla de puntos externos (Buffer) alrededor de los bounds nacionales
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    lons_ext = np.linspace(min_lon - buffer_deg, max_lon + buffer_deg, 6)
    lats_ext = np.linspace(min_lat - buffer_deg, max_lat + buffer_deg, 6)

    idx_b = 1
    for lon_b in lons_ext:
        for lat_b in lats_ext:
            # Solo agregar si está fuera del rectángulo central estricto
            if not (min_lon <= lon_b <= max_lon and min_lat <= lat_b <= max_lat):
                estaciones[f"BUFFER_{idx_b}"] = {"name": f"Punto Borde {idx_b}", "lat": float(lat_b), "lon": float(lon_b)}
                idx_b += 1

    return estaciones, gdf

def fetch_openmeteo_batch(lats: List[float], lons: List[float], modelo: str) -> List[Dict[str, Any]]:
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "timezone": "auto",
        "daily": ",".join(DAILY_VARS),
        "models": modelo,
        "start_date": START_DATE,
        "end_date": END_DATE,
    }
    resp = session.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]

def parse_batch_response(results: List[Dict[str, Any]], batch_meta: List[Dict[str, Any]], alias_modelo: str) -> List[pd.DataFrame]:
    dfs = []
    for payload, meta in zip(results, batch_meta):
        daily = payload.get("daily", {})
        times = daily.get("time", [])
        if not times:
            continue
        df = pd.DataFrame({"date": [limpiar_fecha_str(t) for t in times]})
        for k, v in daily.items():
            if k != "time" and isinstance(v, list) and len(v) == len(df):
                df[k] = v
        df.insert(0, "modelo", alias_modelo)
        df.insert(0, "lon", meta["lon"])
        df.insert(0, "lat", meta["lat"])
        df.insert(0, "NAME", meta["name"])
        df.insert(0, "ID", meta["id"])
        dfs.append(df)
    return dfs

def generar_hillshade_sintetico(grid_lon_mesh: np.ndarray, grid_lat_mesh: np.ndarray) -> np.ndarray:
    """ Simula un mapa de sombra de relieve mediante variaciones de elevación analíticas (Montañas de El Salvador). """
    x, y = grid_lon_mesh, grid_lat_mesh
    # Genera elevaciones sintéticas basadas en coordenadas reales (Cadena volcánica y sierra septentrional)
    z = (np.sin((x + 89.2) * 45) * np.cos((y - 13.8) * 45)) * 400 + np.sin((x + 88.5) * 30) * 300
    z = np.clip(z, 0, None)
    
    # Calcular gradiente/pendiente para sombreado topográfico
    dy, dx = np.gradient(z)
    slope = np.pi/2.0 - np.arctan(np.sqrt(dx*dx + dy*dy))
    aspect = np.arctan2(-dy, dx)
    altitude = np.pi / 4.0  # Ángulo de sol a 45 grados
    azimuth = 3.0 * np.pi / 4.0  # Iluminación Noroeste

    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    return (shaded - shaded.min()) / (shaded.max() - shaded.min())

def interpolar_suave(points: np.ndarray, values: np.ndarray, grid_lon_mesh: np.ndarray, grid_lat_mesh: np.ndarray, es_precip: bool = False) -> np.ndarray:
    """ Interpola suavemente con Clough-Tocher + Suavizado Gaussiano. """
    try:
        interp_ct = CloughTocher2DInterpolator(points, values)
        grid_z = interp_ct(grid_lon_mesh, grid_lat_mesh)
    except Exception:
        grid_z = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')

    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_near = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
        grid_z[nan_mask] = grid_z_near[nan_mask]

    grid_z = gaussian_filter(grid_z, sigma=1.0)

    if es_precip:
        grid_z = np.clip(grid_z, a_min=0.0, a_max=None)
    
    return grid_z

def recortar_y_guardar_raster(grid_z: np.ndarray, nombre_archivo: str, transform, height: int, width: int, geometrias: list) -> tuple:
    grid_z_flipped = np.flipud(grid_z).astype(np.float32)
    ruta_tif = os.path.join(CARPETA_TIFFS, f"{nombre_archivo}.tif")

    meta = {
        'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
        'dtype': 'float32', 'crs': 'EPSG:4326', 'transform': transform, 'nodata': np.nan
    }

    with rasterio.open(ruta_tif, 'w', **meta) as dst:
        dst.write(grid_z_flipped, 1)

    with rasterio.open(ruta_tif, 'r+') as src:
        out_image, out_transform = mask(src, geometrias, crop=True, nodata=np.nan)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1], "width": out_image.shape[2],
            "transform": out_transform, "nodata": np.nan
        })

    with rasterio.open(ruta_tif, 'w', **out_meta) as dst:
        dst.write(out_image)

    extent = [
        out_transform[2],
        out_transform[2] + out_transform[0] * out_image.shape[2],
        out_transform[5] + out_transform[4] * out_image.shape[1],
        out_transform[5]
    ]

    return out_image[0], extent, out_meta

def generar_figura_semanal(raster_resumen: np.ndarray, extent: List, gdf_boundary: gpd.GeoDataFrame, hillshade: np.ndarray, var: str, titulo_semana: str, f_inicio: str, f_fin: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    estilo = ESTILOS_MAPA.get(var, {})
    cmap = estilo["cmap"]
    norm = estilo.get("norm_semanal")

    # Renderizar Relieve Topográfico en Blanco y Negro de fondo
    if hillshade is not None:
        ax.imshow(hillshade, extent=extent, cmap='gray', alpha=0.35, origin='upper')

    # Renderizar Capa Climatológica
    im = ax.imshow(raster_resumen, extent=extent, cmap=cmap, norm=norm, alpha=0.82, origin='upper')
    
    # Límites departamentales / nacionales
    gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='#222222', linewidth=0.9, linestyle='-')

    label_cbar = estilo.get("label_semanal", var)
    if estilo.get("ticks_semanal"):
        cbar = plt.colorbar(im, ax=ax, ticks=estilo["ticks_semanal"], label=label_cbar, shrink=0.75, extend='max')
    else:
        cbar = plt.colorbar(im, ax=ax, label=label_cbar, shrink=0.75)

    cbar.ax.tick_params(labelsize=9)
    plt.title(f"El Salvador: {estilo['title']} (Ensamble GFS/ECMWF)\n{titulo_semana}: del {f_inicio} al {f_fin}", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Longitud", fontsize=9)
    plt.ylabel("Latitud", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    return fig

# =====================
#  LÓGICA PRINCIPAL
# =====================
def ejecutar_procesamiento():
    estaciones, gdf_boundary = cargar_estaciones_ampliadas(ARCHIVO_ESTACIONES, BUFFER_GRADOS)
    if not estaciones:
        return

    os.makedirs(CARPETA_TIFFS, exist_ok=True)
    os.makedirs(CARPETA_MAPAS, exist_ok=True)

    items_estaciones = list(estaciones.items())
    dfs_modelos = []

    progreso = st.progress(0, text="Descargando malla ampliada de datos (incluye buffer de borde)...")

    # 1. DESCARGA DE DATOS TABULARES
    for idx_mod, (model_key, alias) in enumerate(MODELOS.items()):
        progreso.progress(10 + idx_mod * 20, text=f"Descargando modelo {alias}...")
        for i in range(0, len(items_estaciones), BATCH_SIZE):
            chunk = items_estaciones[i:i + BATCH_SIZE]
            lats = [float(meta["lat"]) for _, meta in chunk]
            lons = [float(meta["lon"]) for _, meta in chunk]
            batch_meta = [{"id": est_id, "name": str(meta.get("name", est_id)), "lat": float(meta["lat"]), "lon": float(meta["lon"])} for est_id, meta in chunk]
            try:
                responses = fetch_openmeteo_batch(lats, lons, model_key)
                dfs = parse_batch_response(responses, batch_meta, alias)
                dfs_modelos.extend(dfs)
            except Exception as e:
                st.error(f"Error descargando {alias}: {e}")

    # 2. ENSAMBLE TABULAR Y MESHGRID AMPLIADO
    df_raw_all = pd.concat(dfs_modelos, ignore_index=True)
    df_ensamble = df_raw_all.groupby(["ID", "NAME", "lat", "lon", "date"])[DAILY_VARS].mean().reset_index()

    geometrias = [geom for geom in gdf_boundary.geometry]
    min_lon, min_lat, max_lon, max_lat = gdf_boundary.total_bounds
    
    # Malla de interpolación ligeramente más grande que el país para garantizar suavizado
    grid_lon = np.arange(min_lon - 0.1, max_lon + 0.1, RESOLUCION_TIFF)
    grid_lat = np.arange(min_lat - 0.1, max_lat + 0.1, RESOLUCION_TIFF)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    width, height = len(grid_lon), len(grid_lat)
    transform = from_bounds(min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1, width, height)

    fechas_disponibles = sorted(df_ensamble['date'].unique())

    # RANGOS STRICTOS
    d_manana = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    d_s1_end = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
    d_s2_start = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
    d_s2_end = (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")

    fechas_s1 = [f for f in fechas_disponibles if d_manana <= f <= d_s1_end]
    fechas_s2 = [f for f in fechas_disponibles if d_s2_start <= f <= d_s2_end]

    # Generar Relieve Sintético
    hillshade_bg = generar_hillshade_sintetico(grid_lon_mesh, grid_lat_mesh)

    st.session_state['datos_procesados'] = {}

    # 3. INTERPOLACIÓN Y MÁSCARA DE BORDES
    for idx_var, var in enumerate(VARIABLES_EXPORTAR):
        progreso.progress(50 + idx_var * 15, text=f"Interpolando con buffer y generando rasters de: {var}")
        
        raster_diario_dict = {}
        extent_final = None

        # A) DÍAS INDIVIDUALES
        for fecha in fechas_disponibles:
            df_fecha = df_ensamble[df_ensamble['date'] == fecha]
            points = df_fecha[['lon', 'lat']].values
            values = df_fecha[var].values

            grid_z = interpolar_suave(points, values, grid_lon_mesh, grid_lat_mesh, es_precip=(var == "precipitation_sum"))
            
            fecha_str = limpiar_fecha_str(fecha)
            raster_dia, extent_final, _ = recortar_y_guardar_raster(
                grid_z, f"pronostico_ENSAMBLE_{var}_{fecha_str}", transform, height, width, geometrias
            )
            raster_diario_dict[fecha] = raster_dia

        # B) RESUMEN SEMANAL DE 7 DÍAS
        raster_semanal_dict = {}
        for nom_sem, grp_fechas in [("SEMANA_1", fechas_s1), ("SEMANA_2", fechas_s2)]:
            if grp_fechas:
                df_sub_sem = df_ensamble[df_ensamble['date'].isin(grp_fechas)]
                
                if var == "precipitation_sum":
                    df_sem_agg = df_sub_sem.groupby(["ID", "NAME", "lat", "lon"])[var].sum().reset_index()
                else:
                    df_sem_agg = df_sub_sem.groupby(["ID", "NAME", "lat", "lon"])[var].mean().reset_index()

                points_sem = df_sem_agg[['lon', 'lat']].values
                values_sem = df_sem_agg[var].values

                grid_z_sem = interpolar_suave(points_sem, values_sem, grid_lon_mesh, grid_lat_mesh, es_precip=(var == "precipitation_sum"))
                
                raster_sem, extent_final, _ = recortar_y_guardar_raster(
                    grid_z_sem, f"pronostico_ENSAMBLE_{var}_{nom_sem}", transform, height, width, geometrias
                )
                raster_semanal_dict[nom_sem] = raster_sem

        st.session_state['datos_procesados'][var] = {
            "raster_dict": raster_diario_dict,
            "raster_semanal": raster_semanal_dict,
            "extent": extent_final,
            "fechas_s1": fechas_s1,
            "fechas_s2": fechas_s2,
            "hillshade": hillshade_bg,
            "gdf": gdf_boundary
        }

    progreso.progress(100, text="¡Proceso completado!")
    st.success("🎉 Datos procesados sin artefactos de borde y con relieve integrado.")

# =====================
# INTERFAZ STREAMLIT
# =====================
st.title("🗺️ Visor Meteorológico con Relieve Topográfico")
st.markdown("Sistema de pronósticos meteorológicos **GFS + ECMWF** con interpolación amortiguada de bordes.")

st.sidebar.header("⚙️ Opciones")
if st.sidebar.button("🔄 Actualizar Datos / Procesar", type="primary"):
    ejecutar_procesamiento()

if 'datos_procesados' in st.session_state:
    var_seleccionada = st.sidebar.selectbox(
        "📊 Selecciona Variable:",
        options=VARIABLES_EXPORTAR,
        format_func=lambda x: ESTILOS_MAPA[x]["title"]
    )

    datos_var = st.session_state['datos_procesados'][var_seleccionada]
    
    semana = st.radio("Selecciona Período Semanal (7 Días):", ["Semana 1", "Semana 2"], horizontal=True)
    key_sem = "SEMANA_1" if semana == "Semana 1" else "SEMANA_2"
    grupo_fechas = datos_var["fechas_s1"] if semana == "Semana 1" else datos_var["fechas_s2"]

    if grupo_fechas and key_sem in datos_var.get("raster_semanal", {}):
        raster_resumen = datos_var["raster_semanal"][key_sem]

        f_init_str = datetime.strptime(limpiar_fecha_str(grupo_fechas[0]), "%Y-%m-%d").strftime("%d de %B")
        f_end_str  = datetime.strptime(limpiar_fecha_str(grupo_fechas[-1]), "%Y-%m-%d").strftime("%d de %B de %Y")

        fig = generar_figura_semanal(
            raster_resumen, datos_var["extent"], datos_var["gdf"], datos_var["hillshade"],
            var_seleccionada, f"{semana} (7 Días)", f_init_str, f_end_str
        )
        st.pyplot(fig)
