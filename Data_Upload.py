import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from functools import partial
from geopy import Photon
from datetime import datetime, date
import requests
from geopy import Yandex
import numpy as np
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
# from st_pages import Page, show_pages, add_page_title


# st.session_state['visited'] = True
# show_pages(
#     [
#         Page("diploma_dashboard/Data_Upload.py", "base", "🏠"),
#         Page("diploma_dashboard/pages/Result.py", "result", ":books:"),
#     ]
# )


# Optional -- adds the title and icon to the current page
# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be

# add_page_title()

# if st.checkbox("Use url", value=True):
#     add_logo("http://placekitten.com/120/120")
# else:
# add_logo("./mog_logo_ru_LB.jpeg")
# st.write("👈 Check out the cat in the nav-bar!")
# image_path = 'mog_logo_ru_LB.jpeg'
# st.image(image_path, width=128)
# st.set_page_config(
#         page_title="Ввод данных",
# )
# if st.checkbox("Use url", value=True):
add_logo("https://mosoblgaz.ru/local/html/assets/images/n-logo-p.svg")

# def get_coordinates(names: list[str]) -> list[tuple[str, float, float]]:
#     result = []
#     for element in names:
#         # try:
#         #     result.append((element, float(geocode(element).latitude), float(geocode(element).longitude)))
#         # except:
#         r = requests.get('https://geocode-maps.yandex.ru/1.x',
#                          params={'geocode': element, 'apikey': '2486ab02-2c39-4e68-8d2a-50c7deec2a70', 'format': 'json',
#                                  'bbox': '35.497,56.999~40.32,54.188'})
#         lon, lat = list(map(float,
#                             r.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point'][
#                                 'pos'].split()))
#         result.append((element, lat, lon))
#     return result


def get_coordinates_row(row):
    # rewrite with geopy
    r = requests.get('https://geocode-maps.yandex.ru/1.x',
                     params={'geocode': row, 'apikey': '2486ab02-2c39-4e68-8d2a-50c7deec2a70', 'format': 'json',
                             'bbox': '35.497,56.999~40.32,54.188'})
    lon, lat = list(map(float, r.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos'].split()))
    return lat, lon


def get_file():
    return pd.read_csv(uploaded_file,
                       dtype={17: str, 18: str, 19: str, 21: str, 22: str, 24: str, 26: str, 14: str, 15: str,
                              29: str, 31: str, 32: str, 33: str, 15:float, 16:float}, parse_dates=True, encoding='utf8')


def get_brigades():
    return pd.read_csv(brigades,
                       parse_dates=True, encoding='utf8')


# st.session_state['mother_base'] = st.text_input('Введите адрес')
brigades = st.file_uploader("Choose a file brigades", key ='brigades')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None and brigades is not None:
    data_csv = get_file()
    brigades = get_brigades()
    department = st.selectbox(
        'Выберите департамент',
        sorted(data_csv.loc[:, 'department'].unique().tolist()))
    if department is not None:
        with st.spinner('Wait for it...'):
            data_csv.loc[:, 'date_start'] = pd.to_datetime(data_csv.loc[:, 'date_start'], format='%Y-%m-%d')
            data_csv.loc[:, 'date_end'] = pd.to_datetime(data_csv.loc[:, 'date_end'], format='%Y-%m-%d')
            data_csv.loc[:, 'time_norm'] = data_csv.loc[:, 'time_norm'].astype(float) / 6
            # FIXME выдает ошибку если мало дат и много работ в изначальном решении

            st.session_state['uploaded_data'] = data_csv
            data_start = st.date_input(
                "Начальная дата",
                date(2023, 10, 1))
            data_finish = st.date_input(
                "Конечная дата",
                date(2023, 10, 31))

            geolocator = Nominatim(user_agent="diploma")
            geocode = partial(geolocator.geocode, language="ru", country_codes="RU")
            filtered_data = data_csv[(data_csv['date_start'] >= pd.Timestamp(data_start)) & (
                    data_csv['date_end'] <= pd.Timestamp(data_finish)) & (
                                                data_csv['department'] == department)]
            data_csv = filtered_data
            edited_df = st.data_editor(data_csv, num_rows="dynamic", hide_index=True)
            brr = st.data_editor(brigades.loc[brigades['brigada'].str.startswith(department, na=False)], num_rows="dynamic", hide_index=True, key='gfgdf')
            edited_df = edited_df.loc[(edited_df.addr_lat > 54.111) & (edited_df.addr_lat < 57.083) & (edited_df.addr_lon > 34.958) & (edited_df.addr_lon < 40.463)]
            if st.button('Готово', key='data'):
                edited_df['addr_lat'].replace('', np.nan, inplace=True)
                edited_df['addr_lon'].replace('', np.nan, inplace=True)
                edited_df= edited_df.dropna(subset=['addr_lat'])
                edited_df= edited_df.dropna(subset=['addr_lon'])
                st.session_state['all_brigades'] = []
                for index, row in brr.iterrows():
                    st.write(row['address'])
                    addr = pd.DataFrame({
                        'date_start': pd.Timestamp(data_start),
                        'date_end': pd.Timestamp(data_finish),
                        'department': department,
                        # 'address': st.session_state['mother_base'],
                        'address': row['address'],
                        'time_norm': 0
                    }, index=[0])
                    st.session_state['all_brigades'].append(row['brigada'])
                    edited_df = pd.concat([addr, edited_df[:]])
                    edited_df.loc[0, 'addr_lat'], edited_df.loc[0, 'addr_lon'] = get_coordinates_row(row['address'])
                # edited_df.apply(lambda row: get_coordinates_row(row['address']),
                #                                             axis='columns',
                #                                             result_type='expand')
                # edited_df[['lat', 'lon']] = edited_df.apply(lambda row: get_coordinates_row(row['address']),
                #                                             axis='columns',
                #                                             result_type='expand')
                st.session_state['key'] = edited_df
                # st.write(len(st.session_state['all_brigades']))
                switch_page("result")
