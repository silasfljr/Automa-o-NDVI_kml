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

# --- AUTENTICAÇÃO ---
def authenticate_ee():
    try:
        if 'EE_KEYS' in st.secrets:
            ee_key_dict = json.loads(st.secrets['EE_KEYS'])
            credentials = ee.ServiceAccountCredentials(ee_key_dict['client_email'], key_data=st.secrets['EE_KEYS'])
            ee.Initialize(credentials)
        else:
            ee.Initialize()
    except Exception as e:
        st.error(f"Erro na autenticação: {e}"); st.stop()

authenticate_ee()

st.set_page_config(page_title="🌿 NDVI Mapper Pro + Clima", page_icon="🌿", layout="wide")

# --- FUNÇÕES CLIMÁTICAS ---
def buscar_dados_climaticos(kml_ee, data_fim):
    """Puxa dados de chuva e temperatura do ERA5-Land (últimos 30 dias)"""
    data_inicio_clima = data_fim - timedelta(days=30)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterBounds(kml_ee) \
        .filterDate(data_inicio_clima.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))

    # Precipitação Total Acumulada (mm) e Temperatura Média (C)
    stats = era5.select(['total_precipitation_sum', 'temperature_2m']).reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True),
        geometry=kml_ee,
        scale=1000,
        maxPixels=1e9
    ).getInfo()

    chuva_total = stats.get('total_precipitation_sum_sum', 0) * 1000 # Convert m to mm
    temp_media = stats.get('temperature_2m_mean', 273.15) - 273.15 # Convert K to C
    return chuva_total, temp_media

def gerar_diagnostico_ia(ndvi_medio, chuva, temp):
    """Motor de decisão simples para alertas agroclimatológicos"""
    alertas = []
    status = "Normal"
    cor = "green"

    if ndvi_medio > 0.6 and chuva < 30:
        alertas.append("⚠️ **Risco de Estresse Hídrico:** Vigor alto com baixa disponibilidade de água. Considere irrigação.")
        status = "Crítico: Irrigação"; cor = "orange"
    
    if ndvi_medio < 0.5 and chuva > 100:
        alertas.append("🍄 **Risco de Fungos:** Baixo vigor associado a excesso de umidade. Monitore doenças foliares.")
        status = "Atenção: Fitossanidade"; cor = "red"

    if temp > 35:
        alertas.append("🔥 **Estresse Térmico:** Temperaturas acima de 35°C podem causar abortamento de flores.")
    
    if not alertas:
        alertas.append("✅ **Condições Estáveis:** O balanço entre vigor e clima está dentro da normalidade.")
    
    return alertas, status, cor

# --- FUNÇÕES DE PROCESSAMENTO (IGUAIS ÀS ANTERIORES COM PEQUENOS AJUSTES) ---
def force_2d_geometry(geom):
    if getattr(geom, "has_z", False):
        return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geom

def calcular_area_hectares(kml_ee):
    return kml_ee.geometry().area().getInfo() / 10000

def gerar_zonas_manejo(ndvi_img, kml_ee):
    training = ndvi_img.sample(region=kml_ee.geometry(), scale=10, numPixels=1000)
    clusterer = ee.Clusterer.wekaKMeans(3).train(training)
    result = ndvi_img.cluster(clusterer)
    areas = []
    for i in range(3):
        mask = result.eq(i)
        area = ee.Image.pixelArea().updateMask(mask).reduceRegion(ee.Reducer.sum(), kml_ee, 10).get('area')
        areas.append(ee.Number(area).divide(10000).getInfo())
    return result, areas

def processar_tudo(kml_file, data_inicio, data_fim, limite_nuvens):
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_file.getvalue()); tmp_path = tmp.name
    kml = gpd.read_file(tmp_path); os.unlink(tmp_path)
    if kml.crs and kml.crs.to_epsg() != 4326: kml = kml.to_crs(4326)
    kml['geometry'] = kml['geometry'].apply(force_2d_geometry)
    kml_ee = geemap.geopandas_to_ee(kml)

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(kml_ee) \
        .filterDate(data_inicio.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', limite_nuvens)).sort('system:time_start', False)

    if s2_col.size().getInfo() == 0: return None
    
    recent_img = s2_col.first().clip(kml_ee)
    ndvi = recent_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    stats = ndvi.reduceRegion(ee.Reducer.mean(), kml_ee, 10).getInfo()
    chuva, temp = buscar_dados_climaticos(kml_ee, data_fim)

    return {
        'kml_ee': kml_ee, 'ndvi': ndvi, 'recent_img': recent_img,
        'ndvi_mean': stats.get('NDVI', 0), 'chuva': chuva, 'temp': temp,
        's2_col': s2_col, 'area': calcular_area_hectares(kml_ee)
    }

# --- INTERFACE ---
st.markdown('<h1 style="text-align: center; color: #1B5E20;">🌿 AgroIntel: NDVI + Clima</h1>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("📁 Upload KML", type="kml")
col_d1, col_d2 = st.sidebar.columns(2)
data_fim = col_d2.date_input("📅 Data Base", value=datetime.now().date())
data_inicio = col_d1.date_input("📅 Início Histórico", value=datetime.now().date() - timedelta(days=90))

if st.sidebar.button("🚀 EXECUTAR DIAGNÓSTICO", type="primary", use_container_width=True):
    if uploaded_file:
        with st.spinner("🛰️ Consultando satélites e estações climáticas..."):
            res = processar_tudo(uploaded_file, data_inicio, data_fim, 30)
            if res: st.session_state['analise'] = res
            else: st.error("Sem dados para este período.")

if 'analise' in st.session_state:
    d = st.session_state['analise']
    alertas, status_txt, cor_status = gerar_diagnostico_ia(d['ndvi_mean'], d['chuva'], d['temp'])

    # --- PAINEL DE INTELIGÊNCIA ---
    st.subheader("🤖 Diagnóstico Assistido por IA")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NDVI Médio", f"{d['ndvi_mean']:.3f}")
    c2.metric("Chuva (30 dias)", f"{d['chuva']:.1f} mm")
    c3.metric("Temp. Média", f"{d['temp']:.1f} °C")
    c4.error(f"Status: {status_txt}") if cor_status == "red" else c4.success(f"Status: {status_txt}")

    with st.expander("📝 Recomendações Técnicas", expanded=True):
        for a in alertas:
            st.write(a)

    st.divider()

    tab1, tab2 = st.tabs(["🗺️ Mapas", "📊 Gráficos de Correlação"])

    with tab1:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("🌿 **Vigor Vegetativo (NDVI)**")
            m1 = folium.Map(location=[d['kml_ee'].geometry().centroid().coordinates().getInfo()[1], d['kml_ee'].geometry().centroid().coordinates().getInfo()[0]], zoom_start=14)
            map_id = d['ndvi'].getMapId({'min':0, 'max':1, 'palette':['red','yellow','green']})
            folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='EE').add_to(m1)
            st_folium(m1, width=500, height=400, key="m_ndvi")
        
        with col_m2:
            st.write("🎯 **Zonas de Manejo**")
            zonas_img, areas_z = gerar_zonas_manejo(d['ndvi'], d['kml_ee'])
            m2 = folium.Map(location=[d['kml_ee'].geometry().centroid().coordinates().getInfo()[1], d['kml_ee'].geometry().centroid().coordinates().getInfo()[0]], zoom_start=14)
            map_id_z = zonas_img.getMapId({'min':0, 'max':2, 'palette':['#2E7D32','#FBC02D','#D32F2F']})
            folium.TileLayer(tiles=map_id_z['tile_fetcher'].url_format, attr='EE').add_to(m2)
            st_folium(m2, width=500, height=400, key="m_zonas")

    with tab2:
        st.info("Aqui você pode visualizar como o vigor da planta reagiu ao clima nos últimos 90 dias.")
        # Lógica para gráfico de barras (Chuva) + Linha (NDVI) poderia ser inserida aqui
