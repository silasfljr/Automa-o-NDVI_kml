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

# --- AUTENTICAÇÃO EARTH ENGINE ---
def authenticate_ee():
    try:
        if 'EE_KEYS' in st.secrets:
            ee_key_dict = json.loads(st.secrets['EE_KEYS'])
            credentials = ee.ServiceAccountCredentials(
                ee_key_dict['client_email'], 
                key_data=st.secrets['EE_KEYS']
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize()
    except Exception as e:
        st.error(f"Erro na autenticação do Earth Engine: {e}")
        st.stop()

authenticate_ee()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="🌿 NDVI Mapper Pro", page_icon="🌿", layout="wide")

def force_2d_geometry(geom):
    if getattr(geom, "has_z", False):
        return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geom

def calcular_area_hectares(kml_ee):
    area_m2 = kml_ee.geometry().area().getInfo()
    return area_m2 / 10000

def gerar_serie_temporal(s2_col, kml_ee):
    def extrair_media(img):
        ndvi_img = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        stats = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=kml_ee,
            scale=10,
            maxPixels=1e9
        )
        return ee.Feature(None, {
            'date': img.date().format('yyyy-MM-dd'),
            'NDVI': stats.get('NDVI')
        })
    serie_features = s2_col.map(extrair_media).getInfo()
    data_list = [f['properties'] for f in serie_features['features'] if f['properties']['NDVI'] is not None]
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
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(kml_ee)
              .filterDate(data_inicio.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', limite_nuvens))
              .sort('system:time_start', False))
    if s2_col.size().getInfo() == 0: return None, None, None, None, None, None, 0, None
    recent_image = s2_col.first().clip(kml_ee)
    ndvi = recent_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = recent_image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
                                   {'NIR': recent_image.select('B8'), 'RED': recent_image.select('B4'), 'BLUE': recent_image.select('B2')}).rename('EVI')
    ndwi = recent_image.normalizedDifference(['B8', 'B11']).rename('NDWI')
    stats = ndvi.reduceRegion(reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True), 
                               geometry=kml_ee, scale=10, maxPixels=1e9).getInfo()
    return kml_ee, ndvi, evi, ndwi, recent_image, stats, s2_col.size().getInfo(), s2_col

# --- INTERFACE ---
st.markdown('<h1 style="text-align: center; color: #2E7D32;">🌿 NDVI Mapper Pro</h1>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("📁 Upload KML", type="kml")

# NOVA OPÇÃO: Seleção do Índice
indice_selecionado = st.sidebar.selectbox(
    "📊 Selecione o Índice para o Mapa",
    ["NDVI (Vigor Geral)", "EVI (Cultura Densa)", "NDWI (Umidade/Água)", "RGB (Foto Real)"]
)

col_d1, col_d2 = st.sidebar.columns(2)
data_fim = col_d2.date_input("📅 Fim", value=datetime.now().date())
data_inicio = col_d1.date_input("📅 Início", value=datetime.now().date() - timedelta(days=90))
limite_nuvens = st.sidebar.slider("☁️ Limite Nuvens (%)", 0, 100, 36)

if st.sidebar.button("🚀 GERAR ANÁLISE COMPLETA", type="primary", use_container_width=True):
    if uploaded_file:
        with st.spinner("🔄 Analisando dados..."):
            kml_ee, ndvi, evi, ndwi, img_real, stats, count, s2_col = processar_indices(
                uploaded_file, data_inicio, data_fim, limite_nuvens
            )
            
            if kml_ee:
                area_ha = calcular_area_hectares(kml_ee)
                
                # --- MÉTRICAS ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Área Total (ha)", f"{area_ha:.2f}")
                c2.metric("Imagens", count)
                c3.metric("NDVI Médio", round(stats.get('NDVI_mean', 0), 3))
                c4.metric("NDVI Máximo", round(stats.get('NDVI_max', 0), 3))

                st.divider()

                # --- MAPA DINÂMICO ---
                st.subheader(f"🗺️ Visualizando: {indice_selecionado}")
                centroid = kml_ee.geometry().centroid().coordinates().getInfo()
                m = folium.Map(location=[centroid[1], centroid[0]], zoom_start=14)
                folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)

                def add_ee_layer(ee_object, vis_params, name):
                    map_id_dict = ee.Image(ee_object).getMapId(vis_params)
                    folium.raster_layers.TileLayer(tiles=map_id_dict['tile_fetcher'].url_format, attr='Google Earth Engine', name=name, overlay=True).add_to(m)

                # Lógica de exibição baseada na escolha do usuário
                if indice_selecionado == "NDVI (Vigor Geral)":
                    add_ee_layer(ndvi, {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDVI')
                elif indice_selecionado == "EVI (Cultura Densa)":
                    add_ee_layer(evi, {'min': 0, 'max': 1, 'palette': ['blue', 'yellow', 'green']}, 'EVI')
                elif indice_selecionado == "NDWI (Umidade/Água)":
                    add_ee_layer(ndwi, {'min': -1, 'max': 1, 'palette': ['brown', 'white', 'blue']}, 'NDWI')
                else:
                    add_ee_layer(img_real, {'bands': ['B4', 'B3', 'B2'], 'max': 3000}, 'RGB Real')
                
                st_folium(m, width=1100, height=500, returned_objects=[])

                st.divider()

                # --- GRÁFICO ---
                st.subheader("📈 Histórico de Vigor (NDVI)")
                df_hist = gerar_serie_temporal(s2_col, kml_ee)
                if not df_hist.empty:
                    fig = px.line(df_hist, x='date', y='NDVI', markers=True, template="plotly_white", color_discrete_sequence=['#2E7D32'])
                    fig.update_layout(yaxis_range=[0, 1], hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Nenhuma imagem encontrada.")
    else:
        st.warning("Aguardando arquivo KML...")
