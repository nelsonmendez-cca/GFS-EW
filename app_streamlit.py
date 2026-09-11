# -*- coding: utf-8 -*-
"""
Interfaz Interactiva con Streamlit - El Salvador
---------------------------------------------------------------------------------------
• Máxima resolución espacial sin fugas de memoria (plt.close + gc.collect).
• Corrección de alineación del Hillshade (sombras calzadas con la frontera).
• Fechas 100% en español sin dependencia de locale.
• Perfil térmico ajustado (#5593ff en T. Mínima) y tabla de T. Media por estación.
"""

import json, os, sys, time, io, zipfile, datetime, gc
from typing import Dict, Any, List
from datetime import date, timedelta

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
from matplotlib.colors import LightSource
import plotly.graph_objects as go
import streamlit as st

# =====================
#  CONFIGURACIÓN BASE
# =====================
st.set_page_config(
    page_title="Visor Meteorológico - El Salvador",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_ESTACIONES = os.path.join(DIRECTORIO_ACTUAL, "elsalvador_mapa.geojson")
CARPETA_TIFFS = os.path.join(DIRECTORIO_ACTUAL, "tiffs_salida")

MODELOS = {"gfs_global": "GFS", "ecmwf_ifs025": "ECMWF"}
DAILY_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"]
VARIABLES_EXPORTAR = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"]

START_DATE = date.today().strftime("%Y-%m-%d")
END_DATE = (date.today() + timedelta(days=15)).strftime("%Y-%m-%d")

TIMEOUT_S = 60
BATCH_SIZE = 50
RESOLUCION_TIFF = 0.005  # Definición máxima mantenida
BUFFER_GRADOS = 0.35 

MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}

ESTACIONES_JSON = {
  "stations": [
    {"id": "A15", "name": "GUIJA", "lat": 14.2283888888889, "lon": -89.4690833333333, "elevation": 430},
    {"id": "A18", "name": "FINCA LOS ANDES", "lat": 13.87475, "lon": -89.6283055555556, "elevation": 1750},
    {"id": "A27", "name": "C. DE LA FRONTERA", "lat": 14.1193055555556, "lon": -89.6558611111111, "elevation": 750},
    {"id": "A31", "name": "PLANES DE MONTECRISTO", "lat": 14.3988888888889, "lon": -89.3605555555556, "elevation": 1850},
    {"id": "A37", "name": "SANTA ANA-UNICAES", "lat": 13.9826666666667, "lon": -89.54925, "elevation": 665},
    {"id": "B01", "name": "CH. DEL GUAYABO", "lat": 13.98775, "lon": -88.7559444444444, "elevation": 200},
    {"id": "B06", "name": "SENSUNTEPEQUE", "lat": 13.8713055555556, "lon": -88.64475, "elevation": 730},
    {"id": "B10", "name": "CERRON GRANDE", "lat": 13.9342222222222, "lon": -88.8979166666667, "elevation": 240},
    {"id": "C09", "name": "COJUTEPEQUE SM", "lat": 13.7205833333333, "lon": -88.92625, "elevation": 850},
    {"id": "G03", "name": "NUEVA CONCEPCION", "lat": 14.1254444444444, "lon": -89.2883055555556, "elevation": 325},
    {"id": "G04", "name": "LA PALMA", "lat": 14.2782777777778, "lon": -89.1591388888889, "elevation": 1000},
    {"id": "G13", "name": "LAS PILAS", "lat": 14.3725277777778, "lon": -89.0964444444444, "elevation": 2000},
    {"id": "H08", "name": "AHUACHAPAN SM", "lat": 13.94311, "lon": -89.860083, "elevation": 780},
    {"id": "H14", "name": "LA HACHADURA", "lat": 13.8596666666667, "lon": -90.0859722222222, "elevation": 30},
    {"id": "L04", "name": "SAN ANDRES", "lat": 13.8064722222222, "lon": -89.4037777777778, "elevation": 460},
    {"id": "L27", "name": "CHILTIUPAN", "lat": 13.5924444444444, "lon": -89.4799166666667, "elevation": 800},
    {"id": "M24", "name": "S. MIGUEL UES", "lat": 13.4389166666667, "lon": -88.1590833333333, "elevation": 110},
    {"id": "N02", "name": "La Union/CPI", "lat": 13.3249444444444, "lon": -87.8147222222222, "elevation": 15},
    {"id": "S10", "name": "A. ILOPANGO", "lat": 13.697416, "lon": -89.117, "elevation": 615},
    {"id": "T06", "name": "ACAJUTLA, PTO NUEVO", "lat": 13.5764, "lon": -89.8335, "elevation": 15},
    {"id": "T24", "name": "LOS NARANJOS", "lat": 13.8749444444444, "lon": -89.6741111111111, "elevation": 1450},
    {"id": "U06", "name": "SANTIAGO DE MARIA", "lat": 13.4796111111111, "lon": -88.4715555555556, "elevation": 900},
    {"id": "V09", "name": "PUENTE CUSCATLAN", "lat": 13.5983888888889, "lon": -88.5943055555556, "elevation": 60},
    {"id": "Z02", "name": "SAN FCO. GOTERA", "lat": 13.69225, "lon": -88.1085, "elevation": 250},
    {"id": "Z03", "name": "PERQUIN", "lat": 13.9610277777778, "lon": -88.1583888888889, "elevation": 1200}
  ]
}

colores_precip_rgb = np.array([
    [255, 255, 255], [230, 245, 255], [190, 225, 255], [140, 205, 255],
    [90,  170, 255], [50,  120, 255], [80,  80,  255], [120, 60,  255],
    [170, 30,  255], [255, 0,   255], [180, 0,   180]
]) / 255.0

CMAP_PRECIP = mcolors.ListedColormap(colores_precip_rgb)
BOUNDS_PRECIP_DIARIO = [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50]
NORM_PRECIP_DIARIO = mcolors.BoundaryNorm(BOUNDS_PRECIP_DIARIO, ncolors=len(colores_precip_rgb), extend='max')

BOUNDS_PRECIP_SEMANAL = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
NORM_PRECIP_SEMANAL = mcolors.BoundaryNorm(BOUNDS_PRECIP_SEMANAL, ncolors=len(colores_precip_rgb), extend='max')

STEPS_TEMP = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
COLORES_TEMP = [
    "#1922FB", "#3B56FC", "#4780FC", "#43A5FD", "#28CCFE", "#00F0FE",
    "#55FCE7", "#96FCC8", "#BDFDA4", "#DAFD7E", "#F0FD57", "#FFF32F",
    "#FFD228", "#FFB220", "#FF8D18", "#FF6610", "#FF2B06"
]

CMAP_TEMP = mcolors.ListedColormap(COLORES_TEMP)
NORM_TEMP = mcolors.BoundaryNorm(STEPS_TEMP, ncolors=len(COLORES_TEMP), extend='neither')

ESTILOS_MAPA = {
    "precipitation_sum": {
        "cmap": CMAP_PRECIP, "norm_diario": NORM_PRECIP_DIARIO, "norm_semanal": NORM_PRECIP_SEMANAL,
        "label_diario": "Precipitación (mm)", "label_semanal": "Precipitación Acumulada (mm)",
        "title": "Precipitación Pronosticada", "ticks_diario": [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50],
        "ticks_semanal": [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250], "extend": "max"
    },
    "temperature_2m_max": {
        "cmap": CMAP_TEMP, "norm_diario": NORM_TEMP, "norm_semanal": NORM_TEMP,
        "label_diario": "Temperatura (°C)", "label_semanal": "Temp. Máx Promedio (°C)",
        "title": "Temperatura Máxima", "ticks_diario": STEPS_TEMP, "ticks_semanal": STEPS_TEMP, "extend": "neither"
    },
    "temperature_2m_min": {
        "cmap": CMAP_TEMP, "norm_diario": NORM_TEMP, "norm_semanal": NORM_TEMP,
        "label_diario": "Temperatura (°C)", "label_semanal": "Temp. Mín Promedio (°C)",
        "title": "Temperatura Mínima", "ticks_diario": STEPS_TEMP, "ticks_semanal": STEPS_TEMP, "extend": "neither"
    }
}

session = requests.Session()
session.headers.update({"User-Agent": "Sire-Downloader/Streamlit"})

# =====================
#  FUNCIONES AUXILIARES
# =====================
def fecha_a_espanol(fecha_str: str, con_ano: bool = True) -> str:
    """ ConvierteYYYY-MM-DD a formato español exacto """
    dt = datetime.datetime.strptime(str(fecha_str).split()[0], "%Y-%m-%d")
    mes_nombre = MESES_ES[dt.month]
    if con_ano:
        return f"{dt.day} de {mes_nombre} de {dt.year}"
    return f"{dt.day} de {mes_nombre}"

def cargar_estaciones_y_border(ruta_geojson: str, buffer_deg: float) -> tuple:
    if not os.path.exists(ruta_geojson):
        st.error(f"❌ No se encontró el archivo GeoJSON: {ruta_geojson}")
        return {}, None

    gdf = gpd.read_file(ruta_geojson)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    estaciones = {}
    for st_item in ESTACIONES_JSON["stations"]:
        est_id = str(st_item["id"])
        estaciones[est_id] = {
            "name": str(st_item["name"]),
            "lat": float(st_item["lat"]),
            "lon": float(st_item["lon"]),
            "elevation": float(st_item.get("elevation", 100))
        }

    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    lons_ext = np.linspace(min_lon - buffer_deg, max_lon + buffer_deg, 6)
    lats_ext = np.linspace(min_lat - buffer_deg, max_lat + buffer_deg, 6)

    idx_b = 1
    for lon_b in lons_ext:
        for lat_b in lats_ext:
            if not (min_lon <= lon_b <= max_lon and min_lat <= lat_b <= max_lat):
                estaciones[f"BUFFER_{idx_b}"] = {"name": f"Punto Borde {idx_b}", "lat": float(lat_b), "lon": float(lon_b), "elevation": 0}
                idx_b += 1

    return estaciones, gdf

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

    # Cálculo correcto y consistente del Bounding Box (Extent) de la imagen recortada
    xmin = out_transform[2]
    ymax = out_transform[5]
    xmax = xmin + out_transform[0] * out_image.shape[2]
    ymin = ymax + out_transform[4] * out_image.shape[1]
    extent = [xmin, xmax, ymin, ymax]

    return out_image[0], extent, out_meta

def generar_dem_hillshade(estaciones: Dict[str, Any], grid_lon_mesh: np.ndarray, grid_lat_mesh: np.ndarray, gdf_boundary: gpd.GeoDataFrame, transform, height: int, width: int) -> tuple:
    """ Genera el sombreado topográfico alineado perfectamente con el recorte vectorial. """
    points = np.array([[meta["lon"], meta["lat"]] for meta in estaciones.values()])
    elevations = np.array([meta["elevation"] for meta in estaciones.values()])

    dem_grid = griddata(points, elevations, (grid_lon_mesh, grid_lat_mesh), method='cubic')
    dem_grid = gaussian_filter(np.nan_to_num(dem_grid, nan=0.0), sigma=2.0)

    ls = LightSource(azdeg=315, altdeg=45)
    hillshade_raw = ls.hillshade(dem_grid, vert_exag=3.0)

    geometrias = [geom for geom in gdf_boundary.geometry]
    
    # Se pasa por el mismo recorte exacto para alinear coordenadas
    hillshade_crop, extent_hs, _ = recortar_y_guardar_raster(
        hillshade_raw, "dem_hillshade_base", transform, height, width, geometrias
    )
    
    return hillshade_crop, extent_hs

def fetch_openmeteo_batch(lats: List[float], lons: List[float], modelo: str) -> List[Dict[str, Any]]:
    params = {
        "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
        "timezone": "auto", "daily": ",".join(DAILY_VARS), "models": modelo,
        "start_date": START_DATE, "end_date": END_DATE,
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
        if not times: continue
        df = pd.DataFrame({"date": [str(t).split("T")[0] for t in times]})
        for k, v in daily.items():
            if k != "time" and isinstance(v, list) and len(v) == len(df): df[k] = v
        df.insert(0, "modelo", alias_modelo)
        df.insert(0, "lon", meta["lon"])
        df.insert(0, "lat", meta["lat"])
        df.insert(0, "NAME", meta["name"])
        df.insert(0, "ID", meta["id"])
        dfs.append(df)
    return dfs

def interpolar_suave(points: np.ndarray, values: np.ndarray, grid_lon_mesh: np.ndarray, grid_lat_mesh: np.ndarray, es_precip: bool = False) -> np.ndarray:
    try:
        interp_ct = CloughTocher2DInterpolator(points, values)
        grid_z = interp_ct(grid_lon_mesh, grid_lat_mesh)
    except Exception:
        grid_z = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')

    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_near = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
        grid_z[nan_mask] = grid_z_near[nan_mask]

    grid_z = gaussian_filter(grid_z, sigma=1.2)
    if es_precip: grid_z = np.clip(grid_z, a_min=0.0, a_max=None)
    return grid_z

def generar_figura_semanal(raster_resumen: np.ndarray, hillshade: np.ndarray, extent: List, gdf_boundary: gpd.GeoDataFrame, var: str, titulo_semana: str, f_inicio: str, f_fin: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.set_facecolor('#d9ebf9')

    estilo = ESTILOS_MAPA.get(var, {})
    cmap, norm = estilo["cmap"], estilo.get("norm_semanal")

    if hillshade is not None:
        ax.imshow(hillshade, extent=extent, cmap='gray', origin='upper', alpha=0.5, zorder=1)

    im = ax.imshow(raster_resumen, extent=extent, cmap=cmap, norm=norm, origin='upper', alpha=0.68, zorder=2)
    gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='#111111', linewidth=0.8, zorder=3)

    label_cbar = estilo.get("label_semanal", var)
    ext_val = estilo.get("extend", "neither")

    if estilo.get("ticks_semanal"):
        cbar = plt.colorbar(im, ax=ax, ticks=estilo["ticks_semanal"], label=label_cbar, shrink=0.75, extend=ext_val)
    else:
        cbar = plt.colorbar(im, ax=ax, label=label_cbar, shrink=0.75)

    cbar.ax.tick_params(labelsize=9)
    
    # Fechas formateadas estrictamente a Español
    f_init_es = fecha_a_espanol(f_inicio, con_ano=False)
    f_fin_es = fecha_a_espanol(f_fin, con_ano=True)

    plt.title(f"{estilo['title']}\n{titulo_semana}: del {f_init_es} al {f_fin_es}", fontsize=11, fontweight='bold', pad=10)
    plt.xlabel("Longitud", fontsize=9)
    plt.ylabel("Latitud", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.tight_layout()
    return fig

def generar_collage_16_dias(raster_dict: Dict[str, np.ndarray], hillshade: np.ndarray, extent: List, gdf_boundary: gpd.GeoDataFrame, var: str) -> plt.Figure:
    fechas = sorted(list(raster_dict.keys()))[:16]
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), dpi=150)
    axes = axes.flatten()

    estilo = ESTILOS_MAPA.get(var, {})
    cmap, norm = estilo["cmap"], estilo.get("norm_diario")

    last_im = None
    for idx, fecha in enumerate(fechas):
        ax = axes[idx]
        ax.set_facecolor('#d9ebf9')
        
        raster = raster_dict[fecha]
        
        if hillshade is not None:
            ax.imshow(hillshade, extent=extent, cmap='gray', origin='upper', alpha=0.5, zorder=1)

        last_im = ax.imshow(raster, extent=extent, cmap=cmap, norm=norm, origin='upper', alpha=0.68, zorder=2)
        gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='#111111', linewidth=0.45, zorder=3)

        dt_fecha = datetime.datetime.strptime(str(fecha).split()[0], "%Y-%m-%d")
        ax.set_title(dt_fecha.strftime("%d/%m/%Y"), fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(fechas), 16):
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    if last_im:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        ext_val = estilo.get("extend", "neither")
        if estilo.get("ticks_diario"):
            fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal', ticks=estilo["ticks_diario"], label=estilo.get("label_diario", var), extend=ext_val)
        else:
            fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal', label=estilo.get("label_diario", var))

    fig.suptitle(f"Evolución Diaria (16 Días) - El Salvador: {estilo['title']}", fontsize=13, fontweight='bold', y=0.98)
    return fig

def crear_zip_tiffs(var_seleccionada: str) -> bytes:
    buffer_zip = io.BytesIO()
    with zipfile.ZipFile(buffer_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for archivo in os.listdir(CARPETA_TIFFS):
            if archivo.endswith('.tif') and var_seleccionada in archivo:
                ruta_completa = os.path.join(CARPETA_TIFFS, archivo)
                zip_file.write(ruta_completa, arcname=archivo)
    buffer_zip.seek(0)
    return buffer_zip.getvalue()

def construir_matriz_estaciones(df_raw: pd.DataFrame, variable: str, es_acumulado: bool = True) -> pd.DataFrame:
    df_ens = df_raw.groupby(["ID", "NAME", "date"])[variable].mean().reset_index()
    piv = df_ens.pivot(index=["ID", "NAME"], columns="date", values=variable).reset_index()
    
    fechas_cols = [c for c in piv.columns if c not in ["ID", "NAME"]]
    
    if es_acumulado:
        piv["TOTAL"] = piv[fechas_cols].sum(axis=1)
    else:
        piv["TOTAL"] = piv[fechas_cols].mean(axis=1)
        
    piv = piv.round(1)

    row_prom = {"ID": "", "NAME": "prom. Nac.diario"}
    for f in fechas_cols:
        row_prom[f] = round(piv[f].mean(), 1)
    
    row_prom["TOTAL"] = round(piv["TOTAL"].mean(), 1)

    df_final = pd.concat([piv, pd.DataFrame([row_prom])], ignore_index=True)
    df_final = df_final.rename(columns={"ID": "Estacion", "NAME": "Estacion_Nombre"})
    return df_final

# =====================
#  LÓGICA PRINCIPAL
# =====================
def ejecutar_procesamiento():
    estaciones, gdf_boundary = cargar_estaciones_y_border(ARCHIVO_ESTACIONES, BUFFER_GRADOS)
    if not estaciones: return

    os.makedirs(CARPETA_TIFFS, exist_ok=True)
    items_estaciones = list(estaciones.items())
    dfs_modelos = []

    progreso = st.sidebar.progress(0, text="Descargando datos por estación...")

    for idx_mod, (model_key, alias) in enumerate(MODELOS.items()):
        progreso.progress(10 + idx_mod * 20, text=f"Descargando {alias}...")
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

    df_raw_all = pd.concat(dfs_modelos, ignore_index=True)
    df_raw_all = df_raw_all.sort_values(["ID", "modelo", "date"]).ffill().bfill()

    df_puntos_reales = df_raw_all[~df_raw_all["ID"].astype(str).str.startswith("BUFFER_")].copy()
    st.session_state['df_raw_all'] = df_puntos_reales

    df_ensamble = df_raw_all.groupby(["ID", "NAME", "lat", "lon", "date"])[DAILY_VARS].mean().reset_index()

    geometrias = [geom for geom in gdf_boundary.geometry]
    min_lon, min_lat, max_lon, max_lat = gdf_boundary.total_bounds
    
    grid_lon = np.arange(min_lon - 0.1, max_lon + 0.1, RESOLUCION_TIFF)
    grid_lat = np.arange(min_lat - 0.1, max_lat + 0.1, RESOLUCION_TIFF)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    width, height = len(grid_lon), len(grid_lat)
    transform = from_bounds(min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1, width, height)

    # Hillshade perfectamente alineado
    hillshade_dem, extent_hs = generar_dem_hillshade(estaciones, grid_lon_mesh, grid_lat_mesh, gdf_boundary, transform, height, width)

    fechas_disponibles = sorted(df_ensamble['date'].unique())[:16]

    d_manana = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    d_s1_end = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
    d_s2_start = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
    d_s2_end = (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")

    fechas_s1 = [f for f in fechas_disponibles if d_manana <= f <= d_s1_end]
    fechas_s2 = [f for f in fechas_disponibles if d_s2_start <= f <= d_s2_end]

    st.session_state['datos_procesados'] = {}

    for idx_var, var in enumerate(VARIABLES_EXPORTAR):
        progreso.progress(50 + idx_var * 15, text=f"Generando rasters de: {var}")
        
        raster_diario_dict = {}

        for fecha in fechas_disponibles:
            df_fecha = df_ensamble[df_ensamble['date'] == fecha]
            
            if df_fecha[var].dropna().empty:
                fechas_prev = [f for f in fechas_disponibles if f < fecha]
                if fechas_prev:
                    df_fecha = df_ensamble[df_ensamble['date'] == fechas_prev[-1]]

            points, values = df_fecha[['lon', 'lat']].values, df_fecha[var].values

            grid_z = interpolar_suave(points, values, grid_lon_mesh, grid_lat_mesh, es_precip=(var == "precipitation_sum"))
            fecha_str = str(fecha).split("T")[0]
            raster_dia, _, _ = recortar_y_guardar_raster(
                grid_z, f"pronostico_ENSAMBLE_{var}_{fecha_str}", transform, height, width, geometrias
            )
            raster_diario_dict[fecha] = raster_dia

        raster_semanal_dict = {}
        for nom_sem, grp_fechas in [("SEMANA_1", fechas_s1), ("SEMANA_2", fechas_s2)]:
            if grp_fechas:
                df_sub_sem = df_ensamble[df_ensamble['date'].isin(grp_fechas)]
                df_sem_agg = df_sub_sem.groupby(["ID", "NAME", "lat", "lon"])[var].sum().reset_index() if var == "precipitation_sum" else df_sub_sem.groupby(["ID", "NAME", "lat", "lon"])[var].mean().reset_index()

                points_sem, values_sem = df_sem_agg[['lon', 'lat']].values, df_sem_agg[var].values
                grid_z_sem = interpolar_suave(points_sem, values_sem, grid_lon_mesh, grid_lat_mesh, es_precip=(var == "precipitation_sum"))
                
                raster_sem, _, _ = recortar_y_guardar_raster(
                    grid_z_sem, f"pronostico_ENSAMBLE_{var}_{nom_sem}", transform, height, width, geometrias
                )
                raster_semanal_dict[nom_sem] = raster_sem

        st.session_state['datos_procesados'][var] = {
            "raster_dict": raster_diario_dict,
            "raster_semanal": raster_semanal_dict,
            "hillshade": hillshade_dem,
            "extent": extent_hs,
            "fechas_s1": fechas_s1,
            "fechas_s2": fechas_s2,
            "gdf": gdf_boundary
        }

    progreso.progress(100, text="¡Completado!")
    st.sidebar.success("🎉 Datos cargados exitosamente.")

def render_graficos_promedio():
    st.subheader("📊 Promedio Nacional y Datos por Estación")
    
    if 'df_raw_all' not in st.session_state:
        st.info("👈 Haga clic en **'🔄 Cargar / Actualizar Datos'** en el panel izquierdo para generar los gráficos.")
        return

    df_raw = st.session_state['df_raw_all'].copy()

    df_prom_diario = df_raw.groupby(["modelo", "date"])[DAILY_VARS].mean().reset_index()
    df_prom_diario["temperature_2m_mean"] = (df_prom_diario["temperature_2m_max"] + df_prom_diario["temperature_2m_min"]) / 2.0

    piv_rain = df_prom_diario.pivot(index="date", columns="modelo", values="precipitation_sum").reset_index()
    piv_tmax = df_prom_diario.pivot(index="date", columns="modelo", values="temperature_2m_max").reset_index()
    piv_tmin = df_prom_diario.pivot(index="date", columns="modelo", values="temperature_2m_min").reset_index()
    piv_tmean = df_prom_diario.pivot(index="date", columns="modelo", values="temperature_2m_mean").reset_index()

    cols_modelos = [m for m in ["GFS", "ECMWF"] if m in piv_rain.columns]
    piv_rain["Ensamble"] = piv_rain[cols_modelos].mean(axis=1)
    piv_tmax["Ensamble"] = piv_tmax[cols_modelos].mean(axis=1)
    piv_tmin["Ensamble"] = piv_tmin[cols_modelos].mean(axis=1)
    piv_tmean["Ensamble"] = piv_tmean[cols_modelos].mean(axis=1)

    # 1. Precipitación Promedio
    st.markdown("#### 🌧️ Precipitación Promedio Diaria (mm)")
    fig_rain = go.Figure()
    if "ECMWF" in piv_rain.columns:
        fig_rain.add_trace(go.Bar(x=piv_rain["date"], y=piv_rain["ECMWF"], name="ECMWF (Europeo)", marker_color="#1f77b4"))
    if "GFS" in piv_rain.columns:
        fig_rain.add_trace(go.Bar(x=piv_rain["date"], y=piv_rain["GFS"], name="GFS (EE.UU.)", marker_color="#ff7f0e"))
    fig_rain.add_trace(go.Scatter(x=piv_rain["date"], y=piv_rain["Ensamble"], name="Ensamble Consolidado", mode="lines+markers", line=dict(color="#2ca02c", width=3, dash="dot")))
    fig_rain.update_layout(barmode="group", xaxis_title="Fecha", yaxis_title="Precipitación (mm)", hovermode="x unified", height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_rain, use_container_width=True)

    # 2. Perfil Térmico (Azul #5593ff en T. Mínima)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Modelo Europeo (ECMWF)")
        fig_eur = go.Figure()
        if "ECMWF" in piv_tmax.columns:
            fig_eur.add_trace(go.Scatter(x=piv_tmax["date"], y=piv_tmax["ECMWF"], name="T. Máxima", mode="lines+markers", line=dict(color="#f57046", width=2.5)))
            fig_eur.add_trace(go.Scatter(x=piv_tmean["date"], y=piv_tmean["ECMWF"], name="T. Media", mode="lines+markers", line=dict(color="#b7ee40", width=2.5, dash="dash")))
            fig_eur.add_trace(go.Scatter(x=piv_tmin["date"], y=piv_tmin["ECMWF"], name="T. Mínima", mode="lines+markers", line=dict(color="#5593ff", width=2.5)))
        fig_eur.update_layout(xaxis_title="Fecha", yaxis_title="Temperatura (°C)", hovermode="x unified", height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_eur, use_container_width=True)

    with col_g2:
        st.markdown("#### Modelo GFS")
        fig_gfs = go.Figure()
        if "GFS" in piv_tmax.columns:
            fig_gfs.add_trace(go.Scatter(x=piv_tmax["date"], y=piv_tmax["GFS"], name="T. Máxima", mode="lines+markers", line=dict(color="#f57046", width=2.5)))
            fig_gfs.add_trace(go.Scatter(x=piv_tmean["date"], y=piv_tmean["GFS"], name="T. Media", mode="lines+markers", line=dict(color="#b7ee40", width=2.5, dash="dash")))
            fig_gfs.add_trace(go.Scatter(x=piv_tmin["date"], y=piv_tmin["GFS"], name="T. Mínima", mode="lines+markers", line=dict(color="#5593ff", width=2.5)))
        fig_gfs.update_layout(xaxis_title="Fecha", yaxis_title="Temperatura (°C)", hovermode="x unified", height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gfs, use_container_width=True)

    df_raw["temperature_2m_mean"] = (df_raw["temperature_2m_max"] + df_raw["temperature_2m_min"]) / 2.0

    df_matriz_precip = construir_matriz_estaciones(df_raw, "precipitation_sum", es_acumulado=True)
    df_matriz_tmax = construir_matriz_estaciones(df_raw, "temperature_2m_max", es_acumulado=False)
    df_matriz_tmean = construir_matriz_estaciones(df_raw, "temperature_2m_mean", es_acumulado=False)
    df_matriz_tmin = construir_matriz_estaciones(df_raw, "temperature_2m_min", es_acumulado=False)

    with st.expander("📋 Ver Matrices por Estaciones Convencionales", expanded=True):
        st.markdown("##### 🌧️ Precipitación Diaria por Estación (mm)")
        st.dataframe(df_matriz_precip, use_container_width=True, height=220)

        st.markdown("##### 🌡️ Temperatura Máxima por Estación (°C)")
        st.dataframe(df_matriz_tmax, use_container_width=True, height=220)

        st.markdown("##### 🌤️ Temperatura Media por Estación (°C)")
        st.dataframe(df_matriz_tmean, use_container_width=True, height=220)

        st.markdown("##### ❄️ Temperatura Mínima por Estación (°C)")
        st.dataframe(df_matriz_tmin, use_container_width=True, height=220)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_matriz_precip.to_excel(writer, sheet_name='Precipitacion_Estaciones', index=False)
        df_matriz_tmax.to_excel(writer, sheet_name='TMax_Estaciones', index=False)
        df_matriz_tmean.to_excel(writer, sheet_name='TMedia_Estaciones', index=False)
        df_matriz_tmin.to_excel(writer, sheet_name='TMin_Estaciones', index=False)

    st.session_state['excel_buffer'] = buffer.getvalue()

# =====================
#  BARRA LATERAL
# =====================
st.sidebar.title("⚙️ Panel de Control")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Cargar / Actualizar Datos", type="primary", use_container_width=True):
    ejecutar_procesamiento()

var_seleccionada = None
if 'datos_procesados' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Capa Visualizada")
    var_seleccionada = st.sidebar.selectbox(
        "Variable Meteorológica:",
        options=VARIABLES_EXPORTAR,
        format_func=lambda x: ESTILOS_MAPA[x]["title"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Exportar GIS")
    bytes_zip = crear_zip_tiffs(var_seleccionada)
    st.sidebar.download_button(
        label=f"⬇️ GeoTIFFs (.ZIP)\n{ESTILOS_MAPA[var_seleccionada]['title']}",
        data=bytes_zip,
        file_name=f"capas_raster_{var_seleccionada}.zip",
        mime="application/zip",
        use_container_width=True
    )

if 'excel_buffer' in st.session_state:
    st.sidebar.download_button(
        label="📥 Matriz Estaciones (.XLSX)",
        data=st.session_state['excel_buffer'],
        file_name=f"Reporte_Estaciones_SV_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.caption("📌 **El Salvador Weather Visor**\nFuente: Open-Meteo API ")

# =====================
#  ÁREA PRINCIPAL
# =====================
st.title("🗺️ Visor de Pronóstico Meteorológico - El Salvador")

tab1, tab2, tab3 = st.tabs(["📅 Resumen Semanal", "🗓️ Collage 16 Días", "📊 Gráficos & Estaciones"])

if 'datos_procesados' in st.session_state and var_seleccionada:
    datos_var = st.session_state['datos_procesados'][var_seleccionada]

    with tab1:
        col_rad, col_space = st.columns([1, 2])
        with col_rad:
            semana = st.radio("Período de Análisis:", ["Semana 1 ", "Semana 2 "], horizontal=True)
        
        key_sem = "SEMANA_1" if "Semana 1" in semana else "SEMANA_2"
        grupo_fechas = datos_var["fechas_s1"] if "Semana 1" in semana else datos_var["fechas_s2"]

        if grupo_fechas and key_sem in datos_var.get("raster_semanal", {}):
            raster_resumen = datos_var["raster_semanal"][key_sem]
            f_init_str = grupo_fechas[0]
            f_end_str  = grupo_fechas[-1]

            fig = generar_figura_semanal(
                raster_resumen, datos_var["hillshade"], datos_var["extent"], datos_var["gdf"],
                var_seleccionada, f"{semana.split(' ')[0]} {semana.split(' ')[1]}", f_init_str, f_end_str
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)  # Liberación explícita de RAM
            gc.collect()

    with tab2:
        st.subheader("Pronóstico Diario")
        fig_collage = generar_collage_16_dias(
            datos_var["raster_dict"], datos_var["hillshade"], datos_var["extent"], datos_var["gdf"], var_seleccionada
        )
        st.pyplot(fig_collage, use_container_width=True)
        plt.close(fig_collage)  # Liberación explícita de RAM
        gc.collect()

else:
    with tab1:
        st.info("👈 Presione **'🔄 Cargar / Actualizar Datos'** en el panel lateral para iniciar la descarga y visualización de mapas.")
    with tab2:
        st.info("👈 Presione **'🔄 Cargar / Actualizar Datos'** en el panel lateral para generar el collage de 16 días.")

with tab3:
    render_graficos_promedio()
