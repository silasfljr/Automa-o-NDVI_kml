import streamlit as st
import ee
import geemap
import geopandas as gpd
import pandas as pd
import tempfile
import os
import json
import shapely.wkb
from datetime import datetime, timedelta

# --- AUTENTICAÇÃO EARTH ENGINE ---
def authenticate_ee():
    try:
        if 'EE_KEYS' in st.secrets:
            # Autenticação para Produção (Streamlit Cloud)
            ee_key_dict = json.loads(st.secrets['EE_KEYS'])
            credentials = ee.ServiceAccountCredentials(
                ee_key_dict['client_email'], 
                key_data=st.secrets['EE_KEYS']
            )
            ee.Initialize(credentials)
        else:
            # Autenticação para uso Local
            ee.Initialize()
    except Exception as e:
        st.error(f"Erro na autenticação do Earth Engine: {e}")
        st.stop()

authenticate_ee()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="🌿 NDVI Mapper", 
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Função sem @st.cache_data para evitar erro UnhashableParamError com KML
def force_2d_geometry(geom):
    """Remove coordenada Z das geometrias"""
    if getattr(geom, "has_z", False):
        return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geom

def processar_ndvi(kml_file, data_inicio, data_fim, limite_nuvens):
    # Salvar KML temporariamente para leitura
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_file.getvalue())
        tmp_path = tmp.name

    # Ler KML
    kml = gpd.read_file(tmp_path)
    os.unlink(tmp_path)
    
    # Garantir WGS84
    if kml.crs and kml.crs.to_epsg() != 4326:
        kml = kml.to_crs(4326)
    
    kml['geometry'] = kml['geometry'].apply(force_2d_geometry)
    kml_ee = geemap.geopandas_to_ee(kml)

    # Filtrar Sentinel-2
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(kml_ee)
              .filterDate(data_inicio.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', limite_nuvens))
              .sort('system:time_start', False))

    img_count = s2_col.size().getInfo()
    if img_count == 0:
        return None, None, None, None, 0

    recent_image = s2_col.first().clip(kml_ee)
    ndvi = recent_image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    ndvi_stats = ndvi.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        ), 
        geometry=kml_ee,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    return kml_ee, ndvi, recent_image, ndvi_stats, img_count

# --- INTERFACE ---
st.markdown('<h1 class="main-header">🌿 NDVI Mapper</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("📁 Upload KML", type="kml")
col_d1, col_d2 = st.sidebar.columns(2)
data_fim = col_d2.date_input("📅 Fim", value=datetime.now().date())
data_inicio = col_d1.date_input("📅 Início", value=datetime.now().date() - timedelta(days=60))
limite_nuvens = st.sidebar.slider("☁️ Limite Nuvens (%)", 0, 100, 20)

if st.sidebar.button("🚀 GERAR MAPA NDVI", type="primary", use_container_width=True):
    if uploaded_file:
        with st.spinner("🔄 Processando imagens Sentinel-2..."):
            kml_ee, ndvi, recent_image, ndvi_stats, img_count = processar_ndvi(
                uploaded_file, data_inicio, data_fim, limite_nuvens
            )
            
            if kml_ee:
                # --- MÉTRICAS ---
                c1, c2, c3 = st.columns(3)
                c1.metric("Imagens", img_count)
                c2.metric("NDVI Médio", round(ndvi_stats.get('NDVI_mean', 0), 3))
                c3.metric("NDVI Máx", round(ndvi_stats.get('NDVI_max', 0), 3))

                st.divider() # Adiciona uma linha divisória

                # --- CONFIGURAÇÃO DO MAPA ---
                Map = geemap.Map()
                
                # Centralizamos na geometria do KML
                Map.centerObject(kml_ee, 14)
                
                # Adicionamos as camadas
                Map.addLayer(recent_image, {'bands': ['B4', 'B3', 'B2'], 'max': 3000}, 'RGB (Natural)')
                Map.addLayer(ndvi, {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDVI')
                Map.addLayer(kml_ee, {'color': '0000FF'}, 'Área KML')
                
                # Exibição robusta para Streamlit
                Map.to_streamlit(height=600, scrolling=True)
            else:
                st.error("Nenhuma imagem sem nuvens encontrada no período.")
    else:
        st.warning("Aguardando arquivo KML...")
