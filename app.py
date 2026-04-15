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
import folium
from streamlit_folium import st_folium
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="AgroIntel Pro", page_icon="🌱", layout="wide")

# --- ESTILO CSS MINIMALISTA ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 16px;
    }
    div.stButton > button:first-child {
        background-color: #1b5e20;
        color: white;
        border-radius: 5px;
        border: none;
        height: 45px;
        width: 100%;
    }
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES DE NÚCLEO (EARTH ENGINE) ---
def authenticate_ee():
    try:
        if 'EE_KEYS' in st.secrets:
            ee_key_dict = json.loads(st.secrets['EE_KEYS'])
            credentials = ee.ServiceAccountCredentials(ee_key_dict['client_email'], key_data=st.secrets['EE_KEYS'])
            ee.Initialize(credentials)
        else:
            ee.Initialize()
    except Exception as e:
        st.error(f"Erro de conexão: {e}"); st.stop()

authenticate_ee()

def processar_dados(kml_file, d_inicio, d_fim, nuvens):
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_file.getvalue()); t_path = tmp.name
    kml = gpd.read_file(t_path); os.unlink(t_path)
    if kml.crs and kml.crs.to_epsg() != 4326: kml = kml.to_crs(4326)
    kml_ee = geemap.geopandas_to_ee(kml)
    
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(kml_ee)
          .filterDate(d_inicio.strftime('%Y-%m-%d'), d_fim.strftime('%Y-%m-%d'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', nuvens))
          .sort('system:time_start', False))
    
    if s2.size().getInfo() == 0: return None
    
    img = s2.first().clip(kml_ee)
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    stats = ndvi.reduceRegion(ee.Reducer.mean(), kml_ee, 10).getInfo()
    area = kml_ee.geometry().area().getInfo() / 10000
    
    return {'ee_poly': kml_ee, 'ndvi': ndvi, 'raw': img, 'mean': stats.get('NDVI',0), 'area': area, 'coll': s2}

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("🌱 AgroIntel")
    st.markdown("---")
    file = st.file_uploader("1. Carregar Talhão (KML)", type="kml")
    
    st.markdown("### 2. Parâmetros de Data")
    c_data1, c_data2 = st.columns(2)
    f_date = c_data2.date_input("Fim", value=datetime.now().date())
    i_date = c_data1.date_input("Início", value=f_date - timedelta(days=90))
    
    st.markdown("### 3. Visualização")
    map_type = st.selectbox("Camada de Satélite", ["NDVI (Vigor)", "EVI (Densidade)", "RGB (Real)"])
    
    st.markdown("---")
    btn = st.button("GERAR RELATÓRIO")

# --- CONTEÚDO PRINCIPAL ---
if 'analise' not in st.session_state and not btn:
    st.info("👋 Bem-vindo! Carregue um arquivo KML na barra lateral e clique em 'Gerar Relatório' para iniciar a análise.")

if btn and file:
    with st.spinner("Analisando dados orbitais..."):
        res = processar_dados(file, i_date, f_date, 30)
        if res: st.session_state['analise'] = res
        else: st.error("Nenhuma imagem de satélite encontrada para o período.")

if 'analise' in st.session_state:
    d = st.session_state['analise']
    centro = d['ee_poly'].geometry().centroid().coordinates().getInfo()

    # Título da área analisada
    st.title(f"Relatório Técnico: {file.name}")
    
    # Grid de métricas simples
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Área total", f"{d['area']:.2f} ha")
    m2.metric("NDVI Médio", f"{d['mean']:.3f}")
    m3.metric("Status", "Produtivo" if d['mean'] > 0.5 else "Atenção")
    m4.metric("Satélite", "Sentinel-2")
    
    st.markdown("---")

    t1, t2 = st.tabs(["🗺️ MAPEAMENTO ESPACIAL", "📊 EVOLUÇÃO TEMPORAL"])

    with t1:
        # Coluna única larga para o mapa ficar profissional
        m = folium.Map(location=[centro[1], centro[0]], zoom_start=15, tiles='cartodbpositron')
        folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satélite').add_to(m)
        
        # Adicionar camada NDVI
        map_id = d['ndvi'].getMapId({'min': 0, 'max': 1, 'palette': ['#d73027', '#fee08b', '#1a9850']})
        folium.raster_layers.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='EE', overlay=True).add_to(m)
        
        st_folium(m, width="100%", height=550, key="mapa_full")

    with t2:
        st.subheader("Variação do NDVI no Período")
        # Aqui entra o seu código de gráfico Plotly já existente
        st.write("Gráfico de evolução temporal carregando...")
