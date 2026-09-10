# -*- coding: utf-8 -*-
"""
Interfaz Interactiva con Streamlit - El Salvador
---------------------------------------------------------------------------------------
• Buffer de Bounding Box para eliminar artefactos en bordes.
• Elevación y Relieve Real (DEM) limitado al país.
• Generación de Collage de 16 Días (4x4).
• Descarga de capas GeoTIFF empacadas en ZIP.
• Módulo de Gráficos Promedio Nacional (Precipitación y Temperatura GFS vs ECMWF vs Ensamble) y descarga en Excel.
"""

import json, os, sys, time, io, zipfile, datetime
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
import plotly.graph_objects as go
import streamlit as st

# =====================
#  CONFIGURACIÓN BASE
# =====================
st.set_page_config(
    page_title="Visor Meteorológico - El Salvador",
    page_icon="🗺️",
    layout="wide"
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
RESOLUCION_TIFF = 0.008
BUFFER_GRADOS = 0.35 

# --- PALETA DE COLOR RGB ---
colores_rgb = np.array([
    [255, 255, 255], [230, 245, 255], [190, 225, 255], [140, 205, 255],
    [90,  170, 255], [50,  120, 255], [80,  80,  255], [120, 60,  255],
    [170, 30,  255], [255, 0,   255], [180, 0,   180]
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
        "label_diario": "Precipitación (mm)",
        "label_semanal": "Precipitación Acumulada (mm)",
        "title": "Precipitación Pronosticada",
        "ticks_diario": [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50],
        "ticks_semanal": [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250],
    },
    "temperature_2m_max": {
        "cmap": "YlOrRd", "norm_diario": None, "norm_semanal": None,
        "label_diario": "Temperatura (°C)", "label_semanal": "Temp. Máx Promedio (°C)",
        "title": "Temperatura Máxima",
    },
    "temperature_2m_min": {
        "cmap": "YlGnBu_r", "norm_diario": None, "norm_semanal": None,
        "label_diario": "Temperatura (°C)", "label_semanal": "Temp. Mín Promedio (°C)",
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
        if isinstance(c[0], (float, int)): puntos.append(c)
        else:
            for sub in c: aplanar(sub)
    aplanar(coords)
    return (sum(p[0] for p in puntos)/len(puntos), sum(p[1] for p in puntos)/len(puntos)) if puntos else (0.0, 0.0)

@st.cache_data
def cargar_estaciones_ampliadas(ruta: str, buffer_deg: float) -> tuple:
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
            props, geom = feat.get("properties", {}), feat.get("geometry", {})
            est_id = str(props.get("id", props.get("ID", f"GEO_{i+1}")))
            nombre = str(props.get("name", props.get("NOMBRE", est_id)))
            gtype, coords = geom.get("type", ""), geom.get("coordinates", [])

            if gtype == "Point": lon, lat = coords[0], coords[1]
            elif gtype in ["Polygon", "MultiPolygon"]: lon, lat = extraer_centroide(coords)
            else: continue

            estaciones[est_id] = {"name": nombre, "lat": float(lat), "lon": float(lon)}

    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    lons_ext = np.linspace(min_lon - buffer_deg, max_lon + buffer_deg, 6)
    lats_ext = np.linspace(min_lat - buffer_deg, max_lat + buffer_deg, 6)

    idx_b = 1
    for lon_b in lons_ext:
        for lat_b in lats_ext:
            if not (min_lon <= lon_b <= max_lon and min_lat <= lat_b <= max_lat):
                estaciones[f"BUFFER_{idx_b}"] = {"name": f"Punto Borde {idx_b}", "lat": float(lat_b), "lon": float(lon_b)}
                idx_b += 1

    return estaciones, gdf

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
        df = pd.DataFrame({"date": [limpiar_fecha_str(t) for t in times]})
        for k, v in daily.items():
            if k != "time" and isinstance(v, list) and len(v) == len(df): df[k] = v
        df.insert(0, "modelo", alias_modelo)
        df.insert(0, "lon", meta["lon"])
        df.insert(0, "lat", meta["lat"])
        df.insert(0, "NAME", meta["name"])
        df.insert(0, "ID", meta["id"])
        dfs.append(df)
    return dfs

def obtener_hillshade_real_recortado(grid_lon_mesh: np.ndarray, grid_lat_mesh: np.ndarray, gdf_boundary: gpd.GeoDataFrame, transform, height: int, width: int) -> np.ndarray:
    try:
        lons, lats = grid_lon_mesh.flatten()[::15], grid_lat_mesh.flatten()[::15]
        url = "https://api.open-elevation.com/api/v1/lookup"
        locations = [{"latitude": round(lat, 4), "longitude": round(lon, 4)} for lat, lon in zip(lats, lons)]
        
        elevations = []
        for i in range(0, len(locations), 100):
            res = requests.post(url, json={"locations": locations[i:i+100]}, timeout=15)
            if res.status_code == 200: elevations.extend([r["elevation"] for r in res.json()["results"]])
            else: break

        if len(elevations) == len(lons):
            points = np.column_stack((lons, lats))
            z_grid = griddata(points, elevations, (grid_lon_mesh, grid_lat_mesh), method='cubic')
            z_grid = gaussian_filter(np.nan_to_num(z_grid, nan=0.0), sigma=1.5)
            
            dy, dx = np.gradient(z_grid)
            slope = np.pi/2.0 - np.arctan(np.sqrt(dx*dx + dy*dy))
            aspect = np.arctan2(-dy, dx)
            shaded = np.sin(np.pi/4.0) * np.sin(slope) + np.cos(np.pi/4.0) * np.cos(slope) * np.cos(3.0*np.pi/4.0 - aspect)
            shaded_norm = (shaded - shaded.min()) / (shaded.max() - shaded.min() + 1e-5)
            
            geometrias = [geom for geom in gdf_boundary.geometry]
            hill_recortado, _, _ = recortar_y_guardar_raster(shaded_norm, "hillshade_temp", transform, height, width, geometrias)
            return hill_recortado
    except Exception:
        pass
    return None

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

    grid_z = gaussian_filter(grid_z, sigma=1.0)
    if es_precip: grid_z = np.clip(grid_z, a_min=0.0, a_max=None)
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
    ax.set_facecolor('white')
    
    estilo = ESTILOS_MAPA.get(var, {})
    cmap, norm = estilo["cmap"], estilo.get("norm_semanal")

    im = ax.imshow(raster_resumen, extent=extent, cmap=cmap, norm=norm, origin='upper', zorder=2)

    if hillshade is not None:
        ax.imshow(hillshade, extent=extent, cmap='gray', alpha=0.25, origin='upper', zorder=3)
    
    gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='#111111', linewidth=0.8, linestyle='-', zorder=4)

    label_cbar = estilo.get("label_semanal", var)
    if estilo.get("ticks_semanal"):
        cbar = plt.colorbar(im, ax=ax, ticks=estilo["ticks_semanal"], label=label_cbar, shrink=0.75, extend='max')
    else:
        cbar = plt.colorbar(im, ax=ax, label=label_cbar, shrink=0.75)

    cbar.ax.tick_params(labelsize=9)
    plt.title(f"{estilo['title']}\n{titulo_semana}: del {f_inicio} al {f_fin}", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Longitud", fontsize=9)
    plt.ylabel("Latitud", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    return fig

def generar_collage_16_dias(raster_dict: Dict[str, np.ndarray], extent: List, gdf_boundary: gpd.GeoDataFrame, hillshade: np.ndarray, var: str) -> plt.Figure:
    fechas = sorted(list(raster_dict.keys()))[:16]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=150)
    axes = axes.flatten()

    estilo = ESTILOS_MAPA.get(var, {})
    cmap, norm = estilo["cmap"], estilo.get("norm_diario")

    last_im = None
    for idx, fecha in enumerate(fechas):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        raster = raster_dict[fecha]
        last_im = ax.imshow(raster, extent=extent, cmap=cmap, norm=norm, origin='upper', zorder=2)
        
        if hillshade is not None:
            ax.imshow(hillshade, extent=extent, cmap='gray', alpha=0.2, origin='upper', zorder=3)

        gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='#222222', linewidth=0.4, zorder=4)

        f_obj = datetime.datetime.strptime(limpiar_fecha_str(fecha), "%Y-%m-%d")
        ax.set_title(f_obj.strftime("%d/%m/%Y"), fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(fechas), 16):
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    if last_im:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        if estilo.get("ticks_diario"):
            fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal', ticks=estilo["ticks_diario"], label=estilo.get("label_diario", var))
        else:
            fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal', label=estilo.get("label_diario", var))

    fig.suptitle(f"Evolución Diaria (16 Días) - El Salvador: {estilo['title']}", fontsize=14, fontweight='bold', y=0.98)
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

# =====================
#  LÓGICA PRINCIPAL
# =====================
def ejecutar_procesamiento():
    estaciones, gdf_boundary = cargar_estaciones_ampliadas(ARCHIVO_ESTACIONES, BUFFER_GRADOS)
    if not estaciones: return

    os.makedirs(CARPETA_TIFFS, exist_ok=True)
    items_estaciones = list(estaciones.items())
    dfs_modelos = []

    progreso = st.progress(0, text="Descargando malla de datos...")

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

    df_raw_all = pd.concat(dfs_modelos, ignore_index=True)
    
    # Filtrar únicamente puntos geográficos reales excluyendo buffer
    df_puntos_reales = df_raw_all[~df_raw_all["ID"].astype(str).str.startswith("BUFFER_")]
    st.session_state['df_raw_all'] = df_puntos_reales

    # Ensamble promedio
    df_ensamble = df_raw_all.groupby(["ID", "NAME", "lat", "lon", "date"])[DAILY_VARS].mean().reset_index()

    geometrias = [geom for geom in gdf_boundary.geometry]
    min_lon, min_lat, max_lon, max_lat = gdf_boundary.total_bounds
    
    grid_lon = np.arange(min_lon - 0.1, max_lon + 0.1, RESOLUCION_TIFF)
    grid_lat = np.arange(min_lat - 0.1, max_lat + 0.1, RESOLUCION_TIFF)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    width, height = len(grid_lon), len(grid_lat)
    transform = from_bounds(min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1, width, height)

    fechas_disponibles = sorted(df_ensamble['date'].unique())

    d_manana = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    d_s1_end = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
    d_s2_start = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
    d_s2_end = (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")

    fechas_s1 = [f for f in fechas_disponibles if d_manana <= f <= d_s1_end]
    fechas_s2 = [f for f in fechas_disponibles if d_s2_start <= f <= d_s2_end]

    hillshade_real = obtener_hillshade_real_recortado(grid_lon_mesh, grid_lat_mesh, gdf_boundary, transform, height, width)

    st.session_state['datos_procesados'] = {}

    for idx_var, var in enumerate(VARIABLES_EXPORTAR):
        progreso.progress(50 + idx_var * 15, text=f"Generando rasters de: {var}")
        
        raster_diario_dict = {}
        extent_final = None

        for fecha in fechas_disponibles:
            df_fecha = df_ensamble[df_ensamble['date'] == fecha]
            points, values = df_fecha[['lon', 'lat']].values, df_fecha[var].values

            grid_z = interpolar_suave(points, values, grid_lon_mesh, grid_lat_mesh, es_precip=(var == "precipitation_sum"))
            fecha_str = limpiar_fecha_str(fecha)
            raster_dia, extent_final, _ = recortar_y_guardar_raster(
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
            "hillshade": hillshade_real,
            "gdf": gdf_boundary
        }

    progreso.progress(100, text="¡Proceso completado!")
    st.success("🎉 Datos procesados. Ya puedes ver los mapas, gráficos promedio y descargar los reportes.")

def render_graficos_promedio():
    st.header("📊 Promedio Nacional Diario (Puntos de Control)")
    
    if 'df_raw_all' not in st.session_state:
        st.warning("⚠️ Por favor haga clic en '🔄 Actualizar Datos / Procesar' en la barra lateral para generar la información.")
        return

    df_raw = st.session_state['df_raw_all'].copy()

    # Calcular promedios a nivel nacional por modelo y fecha
    df_prom_diario = df_raw.groupby(["modelo", "date"])[DAILY_VARS].mean().reset_index()
    df_prom_diario["temperature_2m_mean"] = (df_prom_diario["temperature_2m_max"] + df_prom_diario["temperature_2m_min"]) / 2.0

    # DataFrames Pivotados
    piv_rain = df_prom_diario.pivot(index="date", columns="modelo", values="precipitation_sum").reset_index()
    piv_tmax = df_prom_diario.pivot(index="date", columns="modelo", values="temperature_2m_max").reset_index()
    piv_tmin = df_prom_diario.pivot(index="date", columns="modelo", values="temperature_2m_min").reset_index()
    piv_tmean = df_prom_diario.pivot(index="date", columns="modelo", values="temperature_2m_mean").reset_index()

    piv_rain["Ensamble"] = (piv_rain["GFS"] + piv_rain["ECMWF"]) / 2.0
    piv_tmax["Ensamble"] = (piv_tmax["GFS"] + piv_tmax["ECMWF"]) / 2.0
    piv_tmin["Ensamble"] = (piv_tmin["GFS"] + piv_tmin["ECMWF"]) / 2.0
    piv_tmean["Ensamble"] = (piv_tmean["GFS"] + piv_tmean["ECMWF"]) / 2.0

    # 1. Gráfico de Precipitación Promedio
    st.subheader("🌧️ Precipitación Promedio Diaria (mm)")
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(x=piv_rain["date"], y=piv_rain["ECMWF"], name="ECMWF (Europeo)", marker_color="#1f77b4"))
    fig_rain.add_trace(go.Bar(x=piv_rain["date"], y=piv_rain["GFS"], name="GFS (EE.UU.)", marker_color="#ff7f0e"))
    fig_rain.add_trace(go.Scatter(x=piv_rain["date"], y=piv_rain["Ensamble"], name="Ensamble Consolidado", mode="lines+markers", line=dict(color="#2ca02c", width=3, dash="dot")))
    fig_rain.update_layout(barmode="group", xaxis_title="Fecha", yaxis_title="Precipitación Promedio (mm)", hovermode="x unified")
    st.plotly_chart(fig_rain, use_container_width=True)

    # 2. Gráfico del Modelo Europeo
    st.subheader("🇪🇺 Perfil Térmico - Modelo Europeo (ECMWF)")
    fig_eur = go.Figure()
    fig_eur.add_trace(go.Scatter(x=piv_tmax["date"], y=piv_tmax["ECMWF"], name="T. Máxima Promedio", mode="lines+markers", line=dict(color="#d62728", width=2)))
    fig_eur.add_trace(go.Scatter(x=piv_tmean["date"], y=piv_tmean["ECMWF"], name="T. Media Promedio", mode="lines+markers", line=dict(color="#2ca02c", width=2, dash="dash")))
    fig_eur.add_trace(go.Scatter(x=piv_tmin["date"], y=piv_tmin["ECMWF"], name="T. Mínima Promedio", mode="lines+markers", line=dict(color="#17becf", width=2)))
    fig_eur.update_layout(xaxis_title="Fecha", yaxis_title="Temperatura (°C)", hovermode="x unified")
    st.plotly_chart(fig_eur, use_container_width=True)

    # 3. Gráfico del Modelo GFS
    st.subheader("🇺🇸 Perfil Térmico - Modelo GFS")
    fig_gfs = go.Figure()
    fig_gfs.add_trace(go.Scatter(x=piv_tmax["date"], y=piv_tmax["GFS"], name="T. Máxima Promedio", mode="lines+markers", line=dict(color="#e377c2", width=2)))
    fig_gfs.add_trace(go.Scatter(x=piv_tmean["date"], y=piv_tmean["GFS"], name="T. Media Promedio", mode="lines+markers", line=dict(color="#bcbd22", width=2, dash="dash")))
    fig_gfs.add_trace(go.Scatter(x=piv_tmin["date"], y=piv_tmin["GFS"], name="T. Mínima Promedio", mode="lines+markers", line=dict(color="#8c564b", width=2)))
    fig_gfs.update_layout(xaxis_title="Fecha", yaxis_title="Temperatura (°C)", hovermode="x unified")
    st.plotly_chart(fig_gfs, use_container_width=True)

    # 4. Gráfico del Modelo Consolidado / Ensamble
    st.subheader("🌐 Perfil Térmico - Modelo Consolidado (Ensamble)")
    fig_ens = go.Figure()
    fig_ens.add_trace(go.Scatter(x=piv_tmax["date"], y=piv_tmax["Ensamble"], name="T. Máxima Ensamble", mode="lines+markers", line=dict(color="#d62728", width=3)))
    fig_ens.add_trace(go.Scatter(x=piv_tmean["date"], y=piv_tmean["Ensamble"], name="T. Media Ensamble", mode="lines+markers", line=dict(color="#7f7f7f", width=3, dash="dash")))
    fig_ens.add_trace(go.Scatter(x=piv_tmin["date"], y=piv_tmin["Ensamble"], name="T. Mínima Ensamble", mode="lines+markers", line=dict(color="#1f77b4", width=3)))
    fig_ens.update_layout(xaxis_title="Fecha", yaxis_title="Temperatura (°C)", hovermode="x unified")
    st.plotly_chart(fig_ens, use_container_width=True)

    # 5. Tablas y Exportación a Excel
    st.subheader("📋 Tablas de Datos y Exportación Excel")

    df_rain_export = piv_rain.rename(columns={"ECMWF": "Lluvia ECMWF (mm)", "GFS": "Lluvia GFS (mm)", "Ensamble": "Lluvia Ensamble (mm)"})
    
    df_temp_export = pd.DataFrame({
        "Fecha": piv_tmax["date"],
        "TMax ECMWF (°C)": piv_tmax["ECMWF"],
        "TMin ECMWF (°C)": piv_tmin["ECMWF"],
        "TMedia ECMWF (°C)": piv_tmean["ECMWF"],
        "TMax GFS (°C)": piv_tmax["GFS"],
        "TMin GFS (°C)": piv_tmin["GFS"],
        "TMedia GFS (°C)": piv_tmean["GFS"],
        "TMax Ensamble (°C)": piv_tmax["Ensamble"],
        "TMin Ensamble (°C)": piv_tmin["Ensamble"],
        "TMedia Ensamble (°C)": piv_tmean["Ensamble"]
    })

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Precipitación Promedio Nacional**")
        st.dataframe(df_rain_export, use_container_width=True)
    with col_t2:
        st.write("**Temperaturas Promedio Nacional**")
        st.dataframe(df_temp_export, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_rain_export.to_excel(writer, sheet_name='Precipitacion_Promedio', index=False)
        df_temp_export.to_excel(writer, sheet_name='Temperaturas_Promedio', index=False)

    st.download_button(
        label="📥 Descargar Promedios Nacionales (.xlsx)",
        data=buffer.getvalue(),
        file_name=f"Promedios_Meteorologicos_SV_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================
# INTERFAZ STREAMLIT
# =====================
st.title("🗺️ Visor de Pronóstico Meteorológico - El Salvador")

st.sidebar.header("⚙️ Control")
if st.sidebar.button("🔄 Actualizar Datos / Procesar", type="primary"):
    ejecutar_procesamiento()

tab1, tab2, tab3 = st.tabs(["📅 Resumen Semanal", "🗓️ Collage 16 Días", "📊 Gráficos Promedio Nacional"])

if 'datos_procesados' in st.session_state:
    var_seleccionada = st.sidebar.selectbox(
        "📊 Selecciona Variable para Mapas:",
        options=VARIABLES_EXPORTAR,
        format_func=lambda x: ESTILOS_MAPA[x]["title"]
    )

    datos_var = st.session_state['datos_procesados'][var_seleccionada]

    with tab1:
        semana = st.radio("Selecciona Período Semanal (7 Días):", ["Semana 1", "Semana 2"], horizontal=True)
        key_sem = "SEMANA_1" if semana == "Semana 1" else "SEMANA_2"
        grupo_fechas = datos_var["fechas_s1"] if semana == "Semana 1" else datos_var["fechas_s2"]

        if grupo_fechas and key_sem in datos_var.get("raster_semanal", {}):
            raster_resumen = datos_var["raster_semanal"][key_sem]
            f_init_str = datetime.datetime.strptime(limpiar_fecha_str(grupo_fechas[0]), "%Y-%m-%d").strftime("%d de %B")
            f_end_str  = datetime.datetime.strptime(limpiar_fecha_str(grupo_fechas[-1]), "%Y-%m-%d").strftime("%d de %B de %Y")

            fig = generar_figura_semanal(
                raster_resumen, datos_var["extent"], datos_var["gdf"], datos_var["hillshade"],
                var_seleccionada, f"{semana} (7 Días)", f_init_str, f_end_str
            )
            st.pyplot(fig)

    with tab2:
        st.subheader("Pronóstico de los próximos 16 días")
        fig_collage = generar_collage_16_dias(
            datos_var["raster_dict"], datos_var["extent"], datos_var["gdf"], datos_var["hillshade"], var_seleccionada
        )
        st.pyplot(fig_collage)

    # BOTÓN DE DESCARGA DE RASTERS (ZIP)
    st.markdown("---")
    st.subheader("💾 Exportación de Datos GIS")
    bytes_zip = crear_zip_tiffs(var_seleccionada)
    st.download_button(
        label=f"⬇️ Descargar Paquete GeoTIFFs (.ZIP) - {ESTILOS_MAPA[var_seleccionada]['title']}",
        data=bytes_zip,
        file_name=f"capas_raster_{var_seleccionada}.zip",
        mime="application/zip",
        type="secondary"
    )

with tab3:
    render_graficos_promedio()
