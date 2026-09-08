# -*- coding: utf-8 -*-
"""
Interfaz Interactiva con Streamlit - El Salvador
---------------------------------------------------------------------------------------
• Interpolación suave C1 (Clough-Tocher + Suavizado Gaussiano) preservando extremos.
• Suavizado continuo en acumulados y promedios semanales para precipitación y temperaturas.
• Guardado y descarga de GeoTIFFs diarios y semanales (Acumulados/Promedios).
• Generación de gráficos y mapas interactivos.
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
    page_title="Visor de Pronósticos - El Salvador",
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

# --- PALETA DE COLOR PERSONALIZADA RGB PARA PRECIPITACIÓN ---
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
    [180, 0,   180]   # Extensión > 50 (magenta oscuro)
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
def cargar_estaciones_geojson(ruta: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró el archivo GeoJSON: {ruta}")
        return {}

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
        return estaciones
    return {}

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

def parse_batch_response(results: List[Dict[str, Any]], batch_meta: List[Dict[str, Any]], alias_modelo: str) -> tuple:
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

def interpolar_suave(points: np.ndarray, values: np.ndarray, grid_lon_mesh: np.ndarray, grid_lat_mesh: np.ndarray, es_precip: bool = False, sigma_smooth: float = 1.2) -> np.ndarray:
    """ Interpola suavemente con Clough-Tocher + Filtro Gaussiano continuo. """
    try:
        interp_ct = CloughTocher2DInterpolator(points, values)
        grid_z = interp_ct(grid_lon_mesh, grid_lat_mesh)
    except Exception:
        grid_z = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')

    # Rellenar zonas vacías/bordes
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_near = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
        grid_z[nan_mask] = grid_z_near[nan_mask]

    # Suavizado de bordes y polígonos
    grid_z = gaussian_filter(grid_z, sigma=sigma_smooth)

    # Conservar picos/valores en los puntos originales
    min_orig = values.min()

    if es_precip:
        grid_z = np.clip(grid_z, a_min=max(0.0, min_orig), a_max=None)
    
    return grid_z

def suavizar_raster_acumulado(raster: np.ndarray, es_precip: bool = False, sigma: float = 1.5) -> np.ndarray:
    """ Aplica suavizado continuo a la matriz de acumulados/promedios semanales conservando la máscara. """
    val_mask = ~np.isnan(raster)
    if not val_mask.any():
        return raster
    
    raster_filled = raster.copy()
    raster_filled[~val_mask] = np.nanmean(raster)
    
    raster_smooth = gaussian_filter(raster_filled, sigma=sigma)
    raster_smooth[~val_mask] = np.nan
    
    if es_precip:
        raster_smooth = np.clip(raster_smooth, a_min=0.0, a_max=None)
        
    return raster_smooth

def generar_rasters_modelo(df: pd.DataFrame, gdf_boundary: gpd.GeoDataFrame, var: str, alias_modelo: str) -> tuple:
    geometrias = [geom for geom in gdf_boundary.geometry]
    min_lon, min_lat, max_lon, max_lat = gdf_boundary.total_bounds
    grid_lon = np.arange(min_lon, max_lon, RESOLUCION_TIFF)
    grid_lat = np.arange(min_lat, max_lat, RESOLUCION_TIFF)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    width, height = len(grid_lon), len(grid_lat)
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    fechas = sorted(df['date'].unique())
    raster_dict = {}
    extent_final = None

    for fecha in fechas:
        df_fecha = df[df['date'] == fecha]
        points = df_fecha[['lon', 'lat']].values
        values = df_fecha[var].values

        grid_z = interpolar_suave(points, values, grid_lon_mesh, grid_lat_mesh, es_precip=(var == "precipitation_sum"), sigma_smooth=1.2)
        grid_z_flipped = np.flipud(grid_z).astype(np.float32)

        fecha_str = limpiar_fecha_str(fecha)
        nombre_base = f"pronostico_{alias_modelo}_{var}_{fecha_str}"
        ruta_tif = os.path.join(CARPETA_TIFFS, f"{nombre_base}.tif")

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

        raster_dict[fecha] = out_image[0]

        if extent_final is None:
            extent_final = [
                out_transform[2],
                out_transform[2] + out_transform[0] * out_image.shape[2],
                out_transform[5] + out_transform[4] * out_image.shape[1],
                out_transform[5]
            ]

    return raster_dict, extent_final, fechas, out_meta, df

def generar_figura_semanal(raster_resumen: np.ndarray, extent: List, gdf_boundary: gpd.GeoDataFrame, var: str, titulo_semana: str, f_inicio: str, f_fin: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    estilo = ESTILOS_MAPA.get(var, {})
    cmap = estilo["cmap"]
    norm = estilo.get("norm_semanal")

    im = ax.imshow(raster_resumen, extent=extent, cmap=cmap, norm=norm, origin='upper')
    gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)

    label_cbar = estilo.get("label_semanal", var)
    if estilo.get("ticks_semanal"):
        cbar = plt.colorbar(im, ax=ax, ticks=estilo["ticks_semanal"], label=label_cbar, shrink=0.75, extend='max')
    else:
        cbar = plt.colorbar(im, ax=ax, label=label_cbar, shrink=0.75)

    cbar.ax.tick_params(labelsize=9)
    plt.title(f"El Salvador: {estilo['title']} (Pronóstico Modelado)\n{titulo_semana}: del {f_inicio} al {f_fin}", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Longitud", fontsize=9)
    plt.ylabel("Latitud", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def generar_figura_collage(fechas_grupo: List, raster_dict: Dict, extent: List, gdf_boundary: gpd.GeoDataFrame, var: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12.8), dpi=150)
    axes_flat = axes.flatten()
    estilo = ESTILOS_MAPA.get(var, {})
    cmap = estilo["cmap"]
    norm = estilo.get("norm_diario")
    im_ref = None

    for idx, fecha in enumerate(fechas_grupo):
        ax = axes_flat[idx]
        fecha_str = limpiar_fecha_str(fecha)

        if fecha in raster_dict:
            raster_data = raster_dict[fecha]
            im_ref = ax.imshow(raster_data, extent=extent, cmap=cmap, norm=norm, origin='upper')

        gdf_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6)
        ax.set_title(fecha_str, fontsize=9, fontweight='bold', pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(fechas_grupo), len(axes_flat)):
        axes_flat[idx].axis('off')

    f_inicio = datetime.strptime(limpiar_fecha_str(fechas_grupo[0]), "%Y-%m-%d").strftime("%d de %B de %Y")
    f_fin = datetime.strptime(limpiar_fecha_str(fechas_grupo[-1]), "%Y-%m-%d").strftime("%d de %B de %Y")
    
    plt.suptitle(f"{estilo['title']} (Pronóstico Modelado)\nPronóstico Diario de 16 Días: del {f_inicio} al {f_fin}", fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.1)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    
    if estilo.get("ticks_diario"):
        cbar = fig.colorbar(im_ref, cax=cbar_ax, ticks=estilo["ticks_diario"], extend='max')
    else:
        cbar = fig.colorbar(im_ref, cax=cbar_ax)

    cbar.set_label(estilo["label_diario"], fontsize=10, fontweight='bold')
    return fig

# =====================
#  LÓGICA PRINCIPAL
# =====================
def ejecutar_procesamiento():
    estaciones = cargar_estaciones_geojson(ARCHIVO_ESTACIONES)
    gdf_boundary = gpd.read_file(ARCHIVO_ESTACIONES)
    if gdf_boundary.crs is None or gdf_boundary.crs.to_epsg() != 4326:
        gdf_boundary = gdf_boundary.set_crs(epsg=4326, allow_override=True)

    os.makedirs(CARPETA_TIFFS, exist_ok=True)
    os.makedirs(CARPETA_MAPAS, exist_ok=True)

    items_estaciones = list(estaciones.items())
    dfs_modelos = {}

    progreso = st.progress(0, text="Iniciando descarga de datos...")

    # 1. Descarga de modelos
    for idx_mod, (model_key, alias) in enumerate(MODELOS.items()):
        progreso.progress(10 + idx_mod * 20, text=f"Descargando modelo {alias}...")
        df_lista = []
        for i in range(0, len(items_estaciones), BATCH_SIZE):
            chunk = items_estaciones[i:i + BATCH_SIZE]
            lats = [float(meta["lat"]) for _, meta in chunk]
            lons = [float(meta["lon"]) for _, meta in chunk]
            batch_meta = [{"id": est_id, "name": str(meta.get("name", est_id)), "lat": float(meta["lat"]), "lon": float(meta["lon"])} for est_id, meta in chunk]
            try:
                responses = fetch_openmeteo_batch(lats, lons, model_key)
                dfs = parse_batch_response(responses, batch_meta, alias)
                df_lista.extend(dfs)
            except Exception as e:
                st.error(f"Error descargando {alias}: {e}")
        if df_lista:
            dfs_modelos[alias] = pd.concat(df_lista, ignore_index=True)

    # 2. Generación del Ensamble y GeoTIFFs
    progreso.progress(50, text="Generando rasters e interpolación de acumulados y promedios semanales...")
    
    st.session_state['datos_procesados'] = {}

    for idx_var, var in enumerate(VARIABLES_EXPORTAR):
        rasters_modelos = {}
        extent_final = None
        fechas_comunes = None

        for alias, df_mod in dfs_modelos.items():
            r_dict, extent, fechas, meta_tif, _ = generar_rasters_modelo(df_mod, gdf_boundary, var, alias)
            rasters_modelos[alias] = r_dict
            extent_final = extent
            fechas_comunes = fechas if fechas_comunes is None else list(set(fechas_comunes).intersection(fechas))

        fechas_comunes = sorted(fechas_comunes)
        raster_ensamble_dict = {}

        for fecha in fechas_comunes:
            r_gfs = rasters_modelos["GFS"][fecha]
            r_ecm = rasters_modelos["ECMWF"][fecha]
            raster_promedio = np.nanmean(np.stack([r_gfs, r_ecm], axis=0), axis=0)
            raster_ensamble_dict[fecha] = raster_promedio

            # Guardar GeoTIFF Diario
            fecha_str = limpiar_fecha_str(fecha)
            ruta_tif_ens = os.path.join(CARPETA_TIFFS, f"pronostico_ENSAMBLE_{var}_{fecha_str}.tif")
            with rasterio.open(ruta_tif_ens, 'w', **meta_tif) as dst:
                dst.write(np.expand_dims(raster_promedio, axis=0))

        # --- RE-INTERPOLACIÓN / SUAVIZADO CONTINUO DE ACUMULADOS / PROMEDIOS SEMANALES ---
        fechas_s1 = fechas_comunes[:8]
        fechas_s2 = fechas_comunes[8:16]

        raster_semanal_dict = {}

        for nom_sem, grp_f in [("SEMANA_1", fechas_s1), ("SEMANA_2", fechas_s2)]:
            if grp_f:
                rasters_grupo = [raster_ensamble_dict[f] for f in grp_f if f in raster_ensamble_dict]
                stack = np.stack(rasters_grupo, axis=0)
                
                # Suma para precipitación, promedio para temperaturas
                raster_sem_raw = np.nansum(stack, axis=0) if var == "precipitation_sum" else np.nanmean(stack, axis=0)
                
                # Aplica suavizado continuo tanto a lluvia como a temperaturas semanales
                raster_sem_suave = suavizar_raster_acumulado(raster_sem_raw, es_precip=(var == "precipitation_sum"), sigma=1.5)
                raster_semanal_dict[nom_sem] = raster_sem_suave

                # Guardar GeoTIFF Semanal Suave
                ruta_tif_sem = os.path.join(CARPETA_TIFFS, f"pronostico_ENSAMBLE_{var}_{nom_sem}.tif")
                with rasterio.open(ruta_tif_sem, 'w', **meta_tif) as dst:
                    dst.write(np.expand_dims(raster_sem_suave, axis=0))

        # Guardar en Session State para renderizado rápido
        st.session_state['datos_procesados'][var] = {
            "raster_dict": raster_ensamble_dict,
            "raster_semanal": raster_semanal_dict,
            "extent": extent_final,
            "fechas": fechas_comunes,
            "gdf": gdf_boundary
        }
        progreso.progress(60 + idx_var * 13, text=f"Procesada variable: {var}")

    progreso.progress(100, text="¡Proceso completado con éxito!")
    st.success("🎉 Datos, GeoTIFFs Diarios y Semanales actualizados e interpolados correctamente.")

# =====================
# INTERFAZ STREAMLIT
# =====================
st.title("🗺️ Visor Meteorológico de El Salvador")
st.markdown("Sistema interactivo de visualización de ensamble de pronósticos **GFS + ECMWF**.")

# Botón de actualización en el sidebar
st.sidebar.header("⚙️ Configuración")
if st.sidebar.button("🔄 Actualizar Datos / Procesar", type="primary"):
    ejecutar_procesamiento()

# Verificar si existen datos cargados
if 'datos_procesados' not in st.session_state:
    st.info("👋 Haz clic en **'Actualizar Datos / Procesar'** en la barra lateral para calcular el ensamble y generar los productos.")
else:
    var_seleccionada = st.sidebar.selectbox(
        "📊 Selecciona la Variable:",
        options=VARIABLES_EXPORTAR,
        format_func=lambda x: ESTILOS_MAPA[x]["title"]
    )

    tipo_mapa = st.sidebar.radio(
        "🖼️ Tipo de Vista:",
        ["Resumen Semanal (Semana 1 / Semana 2)", "Collage Diario (16 Días)"]
    )

    datos_var = st.session_state['datos_procesados'][var_seleccionada]
    
    st.subheader(f"Vista: {ESTILOS_MAPA[var_seleccionada]['title']}")

    if tipo_mapa == "Resumen Semanal (Semana 1 / Semana 2)":
        semana = st.radio("Selecciona Semana:", ["Semana 1", "Semana 2"], horizontal=True)
        key_sem = "SEMANA_1" if semana == "Semana 1" else "SEMANA_2"
        grupo_fechas = datos_var["fechas"][:8] if semana == "Semana 1" else datos_var["fechas"][8:16]

        if grupo_fechas and key_sem in datos_var.get("raster_semanal", {}):
            raster_resumen = datos_var["raster_semanal"][key_sem]

            f_init_str = datetime.strptime(limpiar_fecha_str(grupo_fechas[0]), "%Y-%m-%d").strftime("%d de %B")
            f_end_str  = datetime.strptime(limpiar_fecha_str(grupo_fechas[-1]), "%Y-%m-%d").strftime("%d de %B de %Y")

            fig = generar_figura_semanal(
                raster_resumen, datos_var["extent"], datos_var["gdf"], 
                var_seleccionada, semana, f_init_str, f_end_str
            )
            st.pyplot(fig)

            fn_png = f"MAPA_{semana.upper().replace(' ', '_')}_{var_seleccionada}.png"
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches='tight')
            st.download_button("📥 Descargar este Mapa (PNG)", data=buf.getvalue(), file_name=fn_png, mime="image/png")

    else:
        fig = generar_figura_collage(
            datos_var["fechas"][:16], datos_var["raster_dict"], 
            datos_var["extent"], datos_var["gdf"], var_seleccionada
        )
        st.pyplot(fig)

        fn_png = f"COLLAGE_16DIAS_{var_seleccionada}.png"
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches='tight')
        st.download_button("📥 Descargar Collage (PNG)", data=buf.getvalue(), file_name=fn_png, mime="image/png")

    # =====================
    # SECCIÓN DE DESCARGA TIFFS
    # =====================
    st.markdown("---")
    st.subheader("📦 Descarga de Capas Raster (GeoTIFFs)")
    
    tiffs_disponibles = [f for f in os.listdir(CARPETA_TIFFS) if f.startswith(f"pronostico_ENSAMBLE_{var_seleccionada}")]
    
    if tiffs_disponibles:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for fname in tiffs_disponibles:
                fpath = os.path.join(CARPETA_TIFFS, fname)
                zip_file.write(fpath, fname)

        st.download_button(
            label=f"⬇️ Descargar todos los GeoTIFFs (Diarios + Semanales) de {ESTILOS_MAPA[var_seleccionada]['title']} (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"GeoTIFFs_Ensamble_{var_seleccionada}.zip",
            mime="application/zip"
        )
