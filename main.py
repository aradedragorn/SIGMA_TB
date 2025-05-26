import os
os.environ['GDAL_DATA'] = r'C:/Users/ASUS/anaconda3/envs/geo_env/Library/share/gdal'

import streamlit as st
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap, linear
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# Konfigurasi Streamlit
# ================================
st.set_page_config(layout="wide")

logo, title = st.columns([2, 9])
with logo:
    st.image("images/logo.png")

with title:
    st.title("SIGMA-TB : SISTEM INFORMASI GEOGRAFIS UNTUK PEMODELAN MULTIMETODE RISIKO TUBERCULOSIS (STUDI KASUS : KOTA SURABAYA)")

gdf_choro = gpd.read_file("Choroplet/Distribusi_Penyebaran_TBC_Tahun 2024.shp")
if gdf_choro.crs is None:
    gdf_choro.set_crs(epsg=4326, inplace=True)

gdf_heat = gpd.read_file("Heatmap/Centroid_Kecamatan_SBY.shp")
if gdf_heat.crs is None:
    gdf_heat.set_crs(epsg=4326, inplace=True)

df = pd.read_excel("Statistik/Hasil_Prediksi_TBC_Lengkap.xlsx")

# ================================
# Hitung Pusat Peta
# ================================
gdf_proj = gdf_choro.to_crs(epsg=32748)
centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
center = [centroid.y.mean(), centroid.x.mean() + 0.3]

model_label_map_heatmap = {
    "Aktual": "Aktual",
    "Model Negative Binomial": "NB_pred",
    "Model Random Forest": "RF_pred",
    "Model XGBoost": "XGB_pred"
}

klasifikasi_label_map = {
    "Aktual": ("Ak_Klas", "Aktual"),
    "Model Negative Binomial": ("NB_Klas", "NB_pred"),
    "Model Random Forest": ("RF_Klas", "RF_pred"),
    "Model XGBoost": ("XGB_Klas", "XGB_pred")
}

tabs = st.tabs(["🗺️ Peta Interaktif", "📊 Statistik Model", "📋 Data Lengkap"])

# ================================
# Tab 1: Peta Interaktif
# ================================
with tabs[0]:
    def map1(klasifikasi_field, prediksi_field):
        values = pd.to_numeric(gdf_choro[prediksi_field], errors='coerce')
        unique_vals = 10
        vmin, vmax = values.min(), values.max()

        colormap = LinearColormap(
            colors=['#4B0055', '#813080', '#B266A3', '#D98CB3', '#F2C6D9'],
            vmin=vmin,
            vmax=vmax
        ).to_step(unique_vals)

        m1 = folium.Map(location=center, zoom_start=10, tiles="Esri NatGeoWorldMap")

        def style_function(feature):
            kecamatan = feature["properties"]["NAMOBJ"]
            raw_value = gdf_choro.loc[gdf_choro["NAMOBJ"] == kecamatan, prediksi_field].values[0]
            value = float(raw_value)
            color = colormap(value)

            return {
                "fillColor": color,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7,
            }

        tooltip = folium.GeoJsonTooltip(
            fields=["NAMOBJ", prediksi_field, klasifikasi_field],
            aliases=["Kecamatan", "Jumlah Kasus", "Klasifikasi"],
            localize=True,
            sticky=True,
            labels=True,
        )

        folium.GeoJson(
            gdf_choro,
            style_function=style_function,
            tooltip=tooltip
        ).add_to(m1)

        return m1, colormap

    def map2(heatmap_model):
        m2 = folium.Map(location=center, zoom_start=10, tiles="Esri NatGeoWorldMap")

        heat_data = [
            [point.y, point.x, weight]
            for point, weight in zip(gdf_heat.geometry, gdf_heat[heatmap_model])
            if not pd.isna(weight)
        ]
        HeatMap(heat_data, radius=35, blur=15, max_zoom=2).add_to(m2)

        folium.GeoJson(
            gdf_choro,
            style_function=lambda feature: {
                "fillColor": "#ffffff",
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.01
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAMOBJ"],
                aliases=["Kecamatan:"],
                localize=True,
                sticky=True,
                labels=True,
            )
        ).add_to(m2)
        
        return m2
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            col1, _ = st.columns([8,1])
            with col1:
                st.subheader("🟡 Peta Choropleth TBC per Kecamatan")
                choropleth_options = list(klasifikasi_label_map.keys())
                choropleth_label = st.selectbox("Model Prediksi untuk Choropleth:", choropleth_options)
                klasifikasi_field, prediksi_field = klasifikasi_label_map[choropleth_label]
                choropleth_map, choropleth_colormap = map1(klasifikasi_field, prediksi_field)
                st_folium(choropleth_map, width=900, height=700)
            with _:
                values = pd.to_numeric(gdf_choro[prediksi_field], errors='coerce')
                vmin, vmax = values.min(), values.max()

                colors = ['#4B0055', '#813080', '#B266A3', '#D98CB3', '#F2C6D9']
                colormap = LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(10)
                colormap.caption = f"Prediksi TBC ({prediksi_field})"
                colormap_html = colormap._repr_html_()
                st.markdown(
                    f"""
                    <div style='margin-top: 200px; margin-left: -30px; transform: rotate(90deg); transform-origin: left top;'>
                        {colormap_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            col2_map, _ = st.columns([8, 1])

            with col2_map:
                st.subheader("🟡 Peta Heatmap TBC per Kecamatan")
                heatmap_options = list(model_label_map_heatmap.keys())
                heatmap_label = st.selectbox("Model Prediksi untuk Heatmap:", heatmap_options)
                heatmap_model = model_label_map_heatmap[heatmap_label]

                st_folium(map2(heatmap_model), width=900, height=700)

            with _:
                values = pd.to_numeric(gdf_heat[heatmap_model], errors='coerce')
                vmin, vmax = values.min(), values.max()

                step = (vmax - vmin) / 5
                index = [vmin + i * step for i in range(6)]
                colors = ['blue', 'cyan', 'lime', 'yellow', 'orange', 'red']

                from branca.colormap import StepColormap
                colormap = StepColormap(
                    colors=colors,
                    index=index,
                    vmin=vmin,
                    vmax=vmax,
                    caption=f"Heatmap TBC : ({heatmap_model})"
                )
                colormap_html = colormap._repr_html_()
                st.markdown(
                    f"""
                    <div style='margin-top: 200px; margin-left: -30px; transform: rotate(90deg); transform-origin: left top;'>
                        {colormap_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ================================
# ================================
# Tab 2: Statistik Model
# ================================
with tabs[1]:
    def model_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2 Score": r2_score(y_true, y_pred)
        }

    models = ["NB", "RF", "XGB"]
    model_names = {"NB": "Negative Binomial", "RF": "Random Forest", "XGB": "XGBoost"}
    colors = {"Aktual": "#6A5ACD", "Pred": "#00BFFF"}
    metrics = {model: model_metrics(df["Aktual"], df[f"{model}_Pred"]) for model in models}

    st.markdown("### 📈 Akurasi Model")
    metrics_df = pd.DataFrame(metrics).T.rename(index=model_names)

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE Terendah", metrics_df["MAE"].idxmin(), f"{metrics_df['MAE'].min():.2f}")
    col2.metric("🔁 RMSE Terendah", metrics_df["RMSE"].idxmin(), f"{metrics_df['RMSE'].min():.2f}")
    col3.metric("📈 R² Tertinggi", metrics_df["R2 Score"].idxmax(), f"{metrics_df['R2 Score'].max():.2f}")

    with st.expander("🔍 Lihat Tabel Evaluasi Lengkap"):
        st.dataframe(metrics_df.style.format("{:.2f}"))

    # Grafik batang dalam expander
    st.markdown("### 📊 Visualisasi Nilai Aktual vs Prediksi per Kecamatan")
    with st.expander("🔍 Klik untuk menampilkan grafik batang per model"):
        col1, col2, col3 = st.columns(3)
        for model, col in zip(models, [col1, col2, col3]):
            fig, ax = plt.subplots(figsize=(4.5, 4), facecolor='black')
            ax.set_facecolor('black')

            df_long = pd.DataFrame({
                "Kecamatan": list(df["Kecamatan"]) * 2,
                "Tipe": ["Aktual"] * len(df) + ["Prediksi"] * len(df),
                "Jumlah Kasus": list(df["Aktual"]) + list(df[f"{model}_Pred"])
            })

            sns.barplot(
                data=df_long,
                x="Kecamatan",
                y="Jumlah Kasus",
                hue="Tipe",
                palette={"Aktual": colors["Aktual"], "Prediksi": colors["Pred"]},
                ax=ax
            )

            ax.set_title(f"{model_names[model]}", fontsize=12, color='white')
            ax.tick_params(axis='x', rotation=90, labelsize=7, colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_xlabel("", color='white')
            ax.set_ylabel("Jumlah Kasus TBC", color='white')

            legend = ax.legend(title="")
            plt.setp(legend.get_texts(), color='white')
            plt.setp(legend.get_title(), color='white')

            for spine in ax.spines.values():
                spine.set_edgecolor('white')

            col.pyplot(fig)

    # Scatter plot dalam expander
    st.subheader("📈 Visualisasi Scatter Plot Tiap Model")
    with st.expander("🔍 Klik untuk menampilkan scatter plot per model"):
        col1, col2, col3 = st.columns(3)
        for model, col in zip(models, [col1, col2, col3]):
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='black')
            ax.set_facecolor('black')

            sns.regplot(
                x=df["Aktual"],
                y=df[f"{model}_Pred"],
                scatter_kws={'color': colors["Aktual"]},
                line_kws={'color': colors["Pred"]},
                ax=ax
            )

            ax.set_title(f"{model_names[model]} - Sebaran Data", color='white')
            ax.set_xlabel("Aktual", color='white')
            ax.set_ylabel("Prediksi", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

            col.pyplot(fig)


# ================================
# Tab 3: Data Lengkap
# ================================
with tabs[2]:
    st.markdown("### 📋 Data Lengkap Prediksi dan Error")

    # Ubah nama kolom agar lebih deskriptif
    df_renamed = df.rename(columns={
        "NB_Pred": "Prediksi RBN",
        "RF_Pred": "Prediksi Random Forest",
        "XGB_Pred": "Prediksi XGBoost"
    })

    # Fungsi highlight berdasarkan nama kolom baru
    def highlight_predictions(s):
        color_map = {
            'Prediksi RBN': 'background-color: #fc8d59',          # oranye-merah
            'Prediksi Random Forest': 'background-color: #91bfdb',# biru langit
            'Prediksi XGBoost': 'background-color: #d05ce3'       # ungu cerah
        }
        return [color_map.get(col, '') for col in s.index]

    with st.expander("📁 Klik untuk menampilkan seluruh data"):
        st.dataframe(df_renamed.style.apply(highlight_predictions, axis=1))


