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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- CSS PROFISSIONAL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 50%, #66BB6A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.3rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .metric-container {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #E0E6ED;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: #2E7D32;
        margin: 0;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        color: #6C757D;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #F8F9FA 0%, #E9ECEF 100%);
    }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .primary-button {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%) !important;
        color: white !important;
    }
    
    .tab-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
        color: #2E7D32;
        margin-bottom: 1.5rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER PREMIUM ---
def render_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🌿 NDVI Mapper Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Monitoramento Avançado de Vegetação com Inteligência Artificial e Sensoriamento Remoto</p>', unsafe_allow_html=True)

# --- MÉTRICAS PREMIUM ---
def render_metrics(data_dict, area_ha):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <p class="metric-label">Área Analisada</p>
            <p class="metric-value">{:.1f} ha</p>
        </div>
        """.format(area_ha), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <p class="metric-label">Imagens Processadas</p>
            <p class="metric-value">{}</p>
        </div>
        """.format(data_dict['count']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <p class="metric-label">NDVI Médio</p>
            <p class="metric-value">{:.3f}</p>
        </div>
        """.format(data_dict['stats'].get('NDVI_mean', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <p class="metric-label">NDVI Máximo</p>
            <p class="metric-value">{:.3f}</p>
        </div>
        """.format(data_dict['stats'].get('NDVI_max', 0)), unsafe_allow_html=True)

# --- FUNÇÕES EXISTENTES (mantidas iguais) ---
def force_2d_geometry(geom):
    if getattr(geom, "has_z", False):
        return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geom

def calcular_area_hectares(kml_ee):
    area_m2 = kml_ee.geometry().area().getInfo()
    return area_m2 / 10000

def gerar_zonas_manejo(ndvi_img, kml_ee, n_clusters=3):
    training = ndvi_img.sample(region=kml_ee.geometry(), scale=10, numPixels=1000)
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
    result = ndvi_img.cluster(clusterer)
    
    pixel_area = ee.Image.pixelArea()
    areas_lista = []
    for i in range(n_clusters):
        mask = result.eq(i)
        area_z = pixel_area.updateMask(mask).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=kml_ee, scale=10, maxPixels=1e9
        ).get('area')
        areas_lista.append(ee.Number(area_z).divide(10000).getInfo())
    return result, areas_lista

def gerar_series_temporais_completas(s2_col, kml_ee):
    def extrair_indices(img):
        ndvi_img = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        evi_img = img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
                                  {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}).rename('EVI')
        ndwi_img = img.normalizedDifference(['B8', 'B11']).rename('NDWI')
        combined = img.addBands([ndvi_img, evi_img, ndwi_img])
        stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=kml_ee, scale=10, maxPixels=1e9)
        return ee.Feature(None, {'date': img.date().format('yyyy-MM-dd'), 'NDVI': stats.get('NDVI'), 'EVI': stats.get('EVI'), 'NDWI': stats.get('NDWI')})
    
    serie_features = s2_col.map(extrair_indices).getInfo()
    data_list = [f['properties'] for f in serie_features['features']]
    if not data_list: return pd.DataFrame()
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

def processar_indices(kml_file, data_inicio, data_fim, limite_nuvens):
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_file.getvalue())
        tmp_path = tmp.name
    kml = gpd.read_file(tmp_path)
    os.unlink(tmp_path)
    if kml.crs and kml.crs.to_epsg() != 4326: kml = kml.to_crs(4326)
    kml['geometry'] = kml['geometry'].apply(force_2d_geometry)
    kml_ee = geemap.geopandas_to_ee(kml)
    
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(kml_ee)
              .filterDate(data_inicio.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', limite_nuvens)).sort('system:time_start', False))
    
    if s2_col.size().getInfo() == 0: return None
    
    recent_image = s2_col.first().clip(kml_ee)
    ndvi = recent_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = recent_image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
                                   {'NIR': recent_image.select('B8'), 'RED': recent_image.select('B4'), 'BLUE': recent_image.select('B2')}).rename('EVI')
    ndwi = recent_image.normalizedDifference(['B8', 'B11']).rename('NDWI')
    stats = ndvi.reduceRegion(reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True), 
                               geometry=kml_ee, scale=10, maxPixels=1e9).getInfo()
    
    return {
        'kml_ee': kml_ee, 'ndvi': ndvi, 'evi': evi, 'ndwi': ndwi, 
        'img_real': recent_image, 'stats': stats, 'count': s2_col.size().getInfo(), 's2_col': s2_col
    }

# --- AUTENTICAÇÃO ---
def authenticate_ee():
    try:
        ee.Initialize()
    except Exception as e:
        st.error(f"Erro na autenticação Earth Engine: {e}")
        st.stop()

authenticate_ee()

# --- CONFIG PÁGINA ---
st.set_page_config(page_title="🌿 NDVI Mapper Pro", page_icon="🌿", layout="wide")

# --- HEADER ---
render_header()

# --- SIDEBAR PREMIUM ---
with st.sidebar:
    st.markdown("## ⚙️ **Configurações Avançadas**")
    
    st.markdown("### 📁 **Arquivo KML**")
    uploaded_file = st.file_uploader("Escolha seu KML", type="kml", help="Formato Google Earth")
    
    st.markdown("### 📊 **Índice Vegetacional**")
    indice_opcoes = {"🌿 NDVI (Vigor Vegetal)": "NDVI", "🌱 EVI (Densidade)": "EVI", "💧 NDWI (Umidade)": "NDWI", "🛰️ RGB Real": "RGB"}
    selecao = st.selectbox("Visualizar", list(indice_opcoes.keys()), index=0)
    id_indice = indice_opcoes[selecao]
    
    st.markdown("### 📅 **Período de Análise**")
    col_d1, col_d2 = st.columns(2)
    data_fim = col_d2.date_input("Data Final", value=datetime.now().date())
    data_inicio = col_d1.date_input("Data Inicial", value=datetime.now().date() - timedelta(days=90))
    
    st.markdown("### ☁️ **Filtro de Nuvens**")
    limite_nuvens = st.slider("Máximo %", 0, 100, 20)
    
    if st.button("🚀 **GERAR ANÁLISE COMPLETA**", type="primary", use_container_width=True, help="Processa todos os dados no Earth Engine"):
        if uploaded_file:
            with st.spinner("🔄 Analisando imagens Sentinel-2..."):
                resultado = processar_indices(uploaded_file, data_inicio, data_fim, limite_nuvens)
                if resultado:
                    st.session_state['dados'] = resultado
                    st.success("✅ Análise concluída!")
                else:
                    st.error("❌ Nenhuma imagem válida encontrada.")
        else:
            st.warning("⚠️ Faça upload do KML primeiro.")

# --- RESULTADOS ---
if 'dados' in st.session_state:
    d = st.session_state['dados']
    area_ha = calcular_area_hectares(d['kml_ee'])
    centroid = d['kml_ee'].geometry().centroid().coordinates().getInfo()
    
    # MÉTRICAS
    st.markdown("### 📊 **Dashboard Executivo**")
    render_metrics(d, area_ha)
    
    st.divider()
    
    # TABS PREMIUM
    tab1, tab2, tab3 = st.tabs(["🛰️ **Monitoramento Multiespectral**", "🎯 **Zoneamento Inteligente**", "📈 **Série Temporal Completa**"])
    
    with tab1:
        st.markdown('<h3 class="tab-header">Mapa Interativo</h3>', unsafe_allow_html=True)
        
        m1 = folium.Map(location=[centroid[1], centroid[0]], zoom_start=15, tiles=None)
        folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', 
                        attr='Google Satellite', name='Satélite').add_to(m1)
        
        def add_layer(obj, vis, name, m_obj):
            map_id = ee.Image(obj).getMapId(vis)
            folium.raster_layers.TileLayer(
                tiles=map_id['tile_fetcher'].url_format, 
                attr='Google Earth Engine', 
                name=name, 
                overlay=True,
                control=True
            ).add_to(m_obj)
        
        # Camada dinâmica
        vis_params = {
            'NDVI': {'min':0,'max':1,'palette':['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850']},
            'EVI': {'min':0,'max':1,'palette':['#4575b4','#74add1','#abd9e9','#e0f3f8','#ffffbf','#fee090','#fdae61','#f46d43','#d73027']},
            'NDWI': {'min':-0.5,'max':0.5,'palette':['#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba']}
        }
        
        if id_indice == "NDVI": add_layer(d['ndvi'], vis_params['NDVI'], 'NDVI', m1)
        elif id_indice == "EVI": add_layer(d['evi'], vis_params['EVI'], 'EVI', m1)
        elif id_indice == "NDWI": add_layer(d['ndwi'], vis_params['NDWI'], 'NDWI', m1)
        else: add_layer(d['img_real'], {'bands':['B4','B3','B2'],'max':3000}, 'RGB Real', m1)
        
        folium.LayerControl().add_to(m1)
        st_folium(m1, width=1200, height=500, key=f"mapa_{id_indice}")
    
    with tab2:
        st.markdown('<h3 class="tab-header">Zoneamento Automático K-Means</h3>', unsafe_allow_html=True)
        
        with st.spinner("Executando clusterização..."):
            zonas_img, areas_z = gerar_zonas_manejo(d['ndvi'], d['kml_ee'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2E7D32, #4CAF50); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;'>
                <h3 style='margin: 0;'>🟢 Alta Produtividade</h3>
                <h2 style='margin: 0; font-size: 2rem;'>{:.1f} ha</h2>
            </div>
            """.format(areas_z[0]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #FF9800, #FFB74D); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;'>
                <h3 style='margin: 0;'>🟡 Média</h3>
                <h2 style='margin: 0; font-size: 2rem;'>{:.1f} ha</h2>
            </div>
            """.format(areas_z[1]), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #F44336, #EF5350); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;'>
                <h3 style='margin: 0;'>🔴 Baixa Produtividade</h3>
                <h2 style='margin: 0; font-size: 2rem;'>{:.1f} ha</h2>
            </div>
            """.format(areas_z[2]), unsafe_allow_html=True)
        
        m2 = folium
