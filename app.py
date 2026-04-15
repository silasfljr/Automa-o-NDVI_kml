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
        st.error(f"Erro na autenticação: {e}")
        st.stop()

authenticate_ee()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="AgroIndex", page_icon="🌿", layout="wide")

# --- FUNÇÕES DE PROCESSAMENTO ---
def force_2d_geometry(geom):
    if getattr(geom, "has_z", False):
        return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geom

def calcular_area_hectares(kml_ee):
    area_m2 = kml_ee.geometry().area().getInfo()
    return area_m2 / 10000

def gerar_zonas_manejo(ndvi_img, kml_ee, n_clusters=3):
    """Cria zonas de manejo usando o algoritmo K-Means"""
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
    """Extrai médias de todos os índices para o gráfico"""
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
    """Faz a busca pesada no Earth Engine e retorna um dicionário com tudo"""
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

# --- INTERFACE ---
st.markdown('<h1 style="text-align: center; color: #2E7D32;">🌿 AgroIndex</h1>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("📁 Upload KML", type="kml")
indice_opcoes = {"NDVI (Vigor)": "NDVI", "EVI (Densidade)": "EVI", "NDWI (Umidade)": "NDWI", "RGB (Real)": "RGB"}
selecao = st.sidebar.selectbox("📊 Selecione o Índice para Visualizar", list(indice_opcoes.keys()))
id_indice = indice_opcoes[selecao]

col_d1, col_d2 = st.sidebar.columns(2)
data_fim = col_d2.date_input("📅 Fim", value=datetime.now().date())
data_inicio = col_d1.date_input("📅 Início", value=datetime.now().date() - timedelta(days=90))
limite_nuvens = st.sidebar.slider("☁️ Limite Nuvens (%)", 0, 100, 36)

if st.sidebar.button("🚀 GERAR ANÁLISE COMPLETA", type="primary", use_container_width=True):
    if uploaded_file:
        with st.spinner("🔄 Processando dados multiespectrais..."):
            resultado = processar_indices(uploaded_file, data_inicio, data_fim, limite_nuvens)
            if resultado:
                st.session_state['dados'] = resultado
            else:
                st.error("Nenhuma imagem encontrada.")
    else:
        st.warning("Faça o upload do arquivo KML.")

# --- EXIBIÇÃO DOS RESULTADOS (Sempre que houver dados na sessão) ---
if 'dados' in st.session_state:
    d = st.session_state['dados']
    area_ha = calcular_area_hectares(d['kml_ee'])
    centroid = d['kml_ee'].geometry().centroid().coordinates().getInfo()

    # MÉTRICAS RÁPIDAS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Área (ha)", f"{area_ha:.2f}")
    c2.metric("Imagens", d['count'])
    c3.metric("NDVI Médio", round(d['stats'].get('NDVI_mean', 0), 3))
    c4.metric("NDVI Máximo", round(d['stats'].get('NDVI_max', 0), 3))

    st.divider()

    tab1, tab2 = st.tabs(["🔍 Monitoramento Multiespectral", "🎯 Zoneamento de Manejo"])

    with tab1:
        st.subheader(f"Exibindo Mapa de {selecao}")
        m1 = folium.Map(location=[centroid[1], centroid[0]], zoom_start=14)
        folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite').add_to(m1)

        def add_layer(obj, vis, name, m_obj):
            map_id = ee.Image(obj).getMapId(vis)
            folium.raster_layers.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='EE', name=name, overlay=True).add_to(m_obj)

        # Camada do mapa dinâmica conforme a barra lateral
        if id_indice == "NDVI": add_layer(d['ndvi'], {'min':0,'max':1,'palette':['red','yellow','green']}, 'NDVI', m1)
        elif id_indice == "EVI": add_layer(d['evi'], {'min':0,'max':1,'palette':['blue','yellow','green']}, 'EVI', m1)
        elif id_indice == "NDWI": add_layer(d['ndwi'], {'min':-1,'max':1,'palette':['brown','white','blue']}, 'NDWI', m1)
        else: add_layer(d['img_real'], {'bands':['B4','B3','B2'],'max':3000}, 'RGB Real', m1)
        
        # A key dinâmica "mapa_" + id_indice resolve o bug de não atualizar
        st_folium(m1, width=1100, height=450, key=f"mapa_principal_{id_indice}")

        st.subheader("📈 Histórico do Talhão")
        df_hist = gerar_series_temporais_completas(d['s2_col'], d['kml_ee'])
        if not df_hist.empty and id_indice != "RGB":
            # Gráfico muda conforme o índice selecionado
            fig = px.line(df_hist, x='date', y=id_indice, markers=True, template="plotly_white", title=f"Evolução de {id_indice}")
            fig.update_layout(yaxis_range=[-1,1] if id_indice=="NDWI" else [0,1])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🎯 Zoneamento para Taxa Variável")
        with st.spinner("Calculando agrupamentos..."):
            zonas_img, areas_z = gerar_zonas_manejo(d['ndvi'], d['kml_ee'])
            
            cz1, cz2, cz3 = st.columns(3)
            cz1.success(f"🟢 Zona Alta: {areas_z[0]:.2f} ha")
            cz2.warning(f"🟡 Zona Média: {areas_z[1]:.2f} ha")
            cz3.error(f"🔴 Zona Baixa: {areas_z[2]:.2f} ha")

            m2 = folium.Map(location=[centroid[1], centroid[0]], zoom_start=14)
            folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite').add_to(m2)
            add_layer(zonas_img, {'min':0,'max':2,'palette':['#2E7D32','#FBC02D','#D32F2F']}, 'Zonas de Manejo', m2)
            st_folium(m2, width=1100, height=450, key="mapa_zonas_manejo_fixo")
