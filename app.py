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

# --- CONFIGURAÇÃO DA PÁGINA (Deve ser o primeiro comando) ---
st.set_page_config(
    page_title="NDVI Mapper Pro", 
    page_icon="🌿", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS CUSTOMIZADA ---
st.markdown("""
    <style>
    /* Fundo principal e fontes */
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Estilo dos Cards de Métricas */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #2E7D32;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }

    /* Títulos e Headers */
    h1 {
        font-weight: 800 !important;
        color: #1B5E20 !important;
        letter-spacing: -1px;
    }
    
    /* Botão Primário */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #2E7D32 !important;
        border: none;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)

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
        st.error(f"Erro na autenticação: {e}")
        st.stop()

authenticate_ee()

# --- FUNÇÕES DE PROCESSAMENTO ---
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

# --- BARRA LATERAL (SideBar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2950/2950130.png", width=80)
    st.title("Painel de Controle")
    st.markdown("Configure os parâmetros para análise satelital.")
    
    uploaded_file = st.file_uploader("📁 Arquivo KML da Área", type="kml")
    
    st.subheader("Visualização")
    indice_opcoes = {"NDVI (Vigor)": "NDVI", "EVI (Densidade)": "EVI", "NDWI (Umidade)": "NDWI", "RGB (Real)": "RGB"}
    selecao = st.selectbox("Selecione o Índice", list(indice_opcoes.keys()))
    id_indice = indice_opcoes[selecao]
    
    st.subheader("Período e Filtros")
    data_fim = st.date_input("Data Final", value=datetime.now().date())
    data_inicio = st.date_input("Data Inicial", value=datetime.now().date() - timedelta(days=90))
    limite_nuvens = st.slider("Filtro de Nuvens (%)", 0, 100, 30)
    
    btn_gerar = st.button("🚀 PROCESSAR DADOS", use_container_width=True)

# --- CONTEÚDO PRINCIPAL ---
st.title("🌿 NDVI Mapper Pro")
st.markdown("Monitoramento inteligente de safras e zoneamento de manejo via Sentinel-2.")

if btn_gerar:
    if uploaded_file:
        with st.spinner("🛰️ Conectando ao Earth Engine..."):
            resultado = processar_indices(uploaded_file, data_inicio, data_fim, limite_nuvens)
            if resultado:
                st.session_state['dados'] = resultado
            else:
                st.error("Nenhuma imagem clara encontrada no período selecionado.")
    else:
        st.warning("Aguardando o upload do arquivo KML.")

if 'dados' in st.session_state:
    d = st.session_state['dados']
    area_ha = calcular_area_hectares(d['kml_ee'])
    centroid = d['kml_ee'].geometry().centroid().coordinates().getInfo()

    # MÉTRICAS COM ESTILO DE CARD
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Área Total", f"{area_ha:.2f} ha")
    with m2: st.metric("Imagens Processadas", d['count'])
    with m3: st.metric("NDVI Médio", round(d['stats'].get('NDVI_mean', 0), 3))
    with m4: st.metric("NDVI Máximo", round(d['stats'].get('NDVI_max', 0), 3))

    st.markdown("<br>", unsafe_allow_html=True) # Espaçador

    tab1, tab2 = st.tabs(["📊 ANÁLISE ESPACIAL", "🎯 ZONAS DE MANEJO"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown(f"### Mapa: {selecao}")
            m1 = folium.Map(location=[centroid[1], centroid[0]], zoom_start=14, tiles="cartodbpositron")
            folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satélite').add_to(m1)

            def add_layer(obj, vis, name, m_obj):
                map_id = ee.Image(obj).getMapId(vis)
                folium.raster_layers.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='EE', name=name, overlay=True).add_to(m_obj)

            if id_indice == "NDVI": add_layer(d['ndvi'], {'min':0,'max':1,'palette':['#d73027','#fee08b','#1a9850']}, 'NDVI', m1)
            elif id_indice == "EVI": add_layer(d['evi'], {'min':0,'max':1,'palette':['#4575b4','#e0f3f8','#1a9850']}, 'EVI', m1)
            elif id_indice == "NDWI": add_layer(d['ndwi'], {'min':-1,'max':1,'palette':['#8c510a','#f5f5f5','#01665e']}, 'NDWI', m1)
            else: add_layer(d['img_real'], {'bands':['B4','B3','B2'],'max':3000}, 'RGB Real', m1)
            
            st_folium(m1, width="100%", height=500, key=f"map_p_{id_indice}")

        with c2:
            st.markdown("### Tendência")
            df_hist = gerar_series_temporais_completas(d['s2_col'], d['kml_ee'])
            if not df_hist.empty and id_indice != "RGB":
                fig = px.line(df_hist, x='date', y=id_indice, markers=True, template="simple_white")
                fig.update_traces(line_color='#2E7D32', line_width=3)
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=470)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Classificação Automática de Produtividade")
        with st.spinner("Calculando clusters..."):
            zonas_img, areas_z = gerar_zonas_manejo(d['ndvi'], d['kml_ee'])
            
            z1, z2, z3 = st.columns(3)
            z1.markdown(f"**🟢 Zona de Alta:** {areas_z[0]:.2f} ha")
            z2.markdown(f"**🟡 Zona de Média:** {areas_z[1]:.2f} ha")
            z3.markdown(f"**🔴 Zona de Baixa:** {areas_z[2]:.2f} ha")

            m2 = folium.Map(location=[centroid[1], centroid[0]], zoom_start=14)
            folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satélite').add_to(m2)
            add_layer(zonas_img, {'min':0,'max':2,'palette':['#2E7D32','#FBC02D','#D32F2F']}, 'Zonas', m2)
            st_folium(m2, width="100%", height=500, key="map_z")
