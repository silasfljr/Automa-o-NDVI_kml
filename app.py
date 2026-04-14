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

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E7D32; text-align: center; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

def force_2d_geometry(geom):
    if getattr(geom, "has_z", False):
        return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geom

def calcular_area_hectares(kml_ee):
    """Calcula a área da geometria em hectares"""
    # O Earth Engine calcula a área em metros quadrados por padrão
    area_m2 = kml_ee.geometry().area().getInfo()
    area_ha = area_m2 / 10000  # Convertendo para hectares
    return area_ha

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
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def processar_ndvi(kml_file, data_inicio, data_fim, limite_nuvens):
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_file.getvalue())
        tmp_path = tmp.name

    kml = gpd.read_file(tmp_path)
    os.unlink(tmp_path)
    
    if kml.crs and kml.crs.to_epsg() != 4326:
        kml = kml.to_crs(4326)
    
    kml['geometry'] = kml['geometry'].apply(force_2d_geometry)
    kml_ee = geemap.geopandas_to_ee(kml)

    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(kml_ee)
              .filterDate(data_inicio.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', limite_nuvens))
              .sort('system:time_start', False))

    img_count = s2_col.size().getInfo()
    if img_count == 0:
        return None, None, None, None, 0, None

    recent_image = s2_col.first().clip(kml_ee)
    ndvi = recent_image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    ndvi_stats = ndvi.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True), 
        geometry=kml_ee, scale=10, maxPixels=1e9
    ).getInfo()

    return kml_ee, ndvi, recent_image, ndvi_stats, img_count, s2_col

# --- INTERFACE ---
st.markdown('<h1 class="main-header">🌿 NDVI Mapper Pro</h1>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("📁 Upload KML", type="kml")
col_d1, col_d2 = st.sidebar.columns(2)
data_fim = col_d2.date_input("📅 Fim", value=datetime.now().date())
data_inicio = col_d1.date_input("📅 Início", value=datetime.now().date() - timedelta(days=90))
limite_nuvens = st.sidebar.slider("☁️ Limite Nuvens (%)", 0, 100, 36)

if st.sidebar.button("🚀 GERAR ANÁLISE COMPLETA", type="primary", use_container_width=True):
    if uploaded_file:
        with st.spinner("🔄 Processando dados satelitais e área..."):
            kml_ee, ndvi, recent_image, ndvi_stats, img_count, s2_col = processar_ndvi(
                uploaded_file, data_inicio, data_fim, limite_nuvens
            )
            
            if kml_ee:
                # Cálculo da área total
                area_total = calcular_area_hectares(kml_ee)

                # --- MÉTRICAS (Agora com 4 colunas) ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Área Total (ha)", f"{area_total:.2f}")
                c2.metric("Imagens Analisadas", img_count)
                c3.metric("NDVI Médio (Atual)", round(ndvi_stats.get('NDVI_mean', 0), 3))
                c4.metric("NDVI Máx (Atual)", round(ndvi_stats.get('NDVI_max', 0), 3))

                st.divider()

                # --- MAPA (FOLIUM) ---
                st.subheader("🗺️ Mapa de Vigor Vegetativo")
                centroid = kml_ee.geometry().centroid().coordinates().getInfo()
                m = folium.Map(location=[centroid[1], centroid[0]], zoom_start=14)
                folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
                                 attr='Google', name='Google Satellite').add_to(m)

                def add_ee_layer(ee_object, vis_params, name):
                    map_id_dict = ee.Image(ee_object).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine', name=name, overlay=True, control=True
                    ).add_to(m)

                add_ee_layer(recent_image, {'bands': ['B4', 'B3', 'B2'], 'max': 3000}, 'RGB Natural')
                add_ee_layer(ndvi, {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDVI')
                
                st_folium(m, width=1100, height=500, returned_objects=[])

                st.divider()

                # --- GRÁFICO DE SÉRIE TEMPORAL ---
                st.subheader("📈 Evolução do NDVI no Período")
                with st.spinner("📊 Calculando série histórica..."):
                    df_historico = gerar_serie_temporal(s2_col, kml_ee)
                    
                    if not df_historico.empty:
                        fig = px.line(df_historico, x='date', y='NDVI', 
                                      markers=True,
                                      template="plotly_white",
                                      color_discrete_sequence=['#2E7D32'])
                        fig.update_layout(
                            hovermode="x unified",
                            xaxis_title="Data da Captura",
                            yaxis_title="NDVI Médio",
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Não foi possível extrair dados para o gráfico histórico.")
                
            else:
                st.error("Nenhuma imagem encontrada.")
    else:
        st.warning("Por favor, faça o upload de um arquivo KML.")
