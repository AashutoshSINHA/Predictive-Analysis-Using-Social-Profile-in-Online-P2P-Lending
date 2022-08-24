# Smart dustbin
import streamlit as st
from streamlit_folium import st_folium
import folium

# center on Liberty Bell, add marker
m = folium.Map(location=[27.66436936716696, 85.32659444366197], zoom_start=16)
folium.Marker(
    [27.66436936716696, 85.32659444366197], 
    popup="Dustbin: RFID-8907-0980", 
    tooltip="Dustbin: RFID-8907-0980"
).add_to(m)

# call to render Folium map in Streamlit
st_data = st_folium(m, width = 725)

    # Color negative values red; color significant p-value green and not significant red
    table2 = ncol1.write(
        metrics.style.format(
            formatter={("p-value", "z-score"): "{:.3g}", ("uplift"): "{:.3g}%"}
        )
        .applymap(style_negative, props="color:red;")
        .apply(style_p_value, props="color:red;", axis=1, subset=["p-value"])
    )
