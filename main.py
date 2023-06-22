import streamlit as st
import pandas as pd
import osmnx as ox
import sklearn
import numpy as np
import geopandas as gpd
import taxicab as tc
from geopy.geocoders import Nominatim
from functools import partial
from geopy import Photon
import networkx as nx
from datetime import datetime, date
import requests
from geopy import Yandex
from matplotlib import pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from joblib import Parallel, delayed
from streamlit_extras.switch_page_button import switch_page


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
    r = requests.get('https://geocode-maps.yandex.ru/1.x',
                     params={'geocode': row, 'apikey': '2486ab02-2c39-4e68-8d2a-50c7deec2a70', 'format': 'json',
                             'bbox': '35.497,56.999~40.32,54.188'})
    lon, lat = list(map(float,
                        r.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point'][
                            'pos'].split()))
    return lat, lon


def get_file():
    return pd.read_csv(uploaded_file, encoding='cp1251',
                       dtype={17: str, 18: str, 19: str, 21: str, 22: str, 24: str, 26: str, 14: str, 15: str,
                              29: str, 31: str, 32: str, 33: str}, parse_dates=True)


st.session_state['mother_base'] = st.text_input('Введите адрес')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with st.spinner('Wait for it...'):
        data_csv = get_file()
        data_csv.loc[:, 'date_start'] = pd.to_datetime(data_csv.loc[:, 'date_start'], format='%d.%m.%y')
        data_csv.loc[:, 'date_end'] = pd.to_datetime(data_csv.loc[:, 'date_end'], format='%d.%m.%y')
        st.session_state['uploaded_data'] = data_csv
        data_start = st.date_input(
            "Начальная дата",
            date(2023, 5, 1))
        data_finish = st.date_input(
            "Конечная дата",
            date(2023, 5, 31))

        geolocator = Nominatim(user_agent="diploma")
        geocode = partial(geolocator.geocode, language="ru", country_codes="RU")
        filtered_data = data_csv[
            (data_csv['department'] == 'Ступинская') & (data_csv['date_start'] >= pd.Timestamp(data_start)) & (
                    data_csv['date_end'] <= pd.Timestamp(data_finish)) & (
                    data_csv['brigada'] == 'Ступинская-Горголя-1')]
        data_csv = filtered_data
        edited_df = st.data_editor(data_csv, num_rows="dynamic", hide_index=True)

        if st.button('Готово', key='data'):
            # data.insert(0, 'ул. Горького, 18, Ступино, Московская обл., 142802')
            # coords = get_coordinates(list(edited_df['address']))
            addr = pd.DataFrame({
                'date_start': pd.Timestamp(data_start),
                'date_end': pd.Timestamp(data_finish),
                'department': 'Ступинская',
                'brigada': 'Ступинская-Горголя-1',
                'address': st.session_state['mother_base'],
                'time_norm': 0
            }, index=[0])
            edited_df = pd.concat([addr, edited_df[:]])

            edited_df[['lat', 'lon']] = edited_df.apply(lambda row: get_coordinates_row(row['address']), axis='columns',
                                                        result_type='expand')
            # df = pd.DataFrame(coords)
            # edited_df = st.data_editor(edited_df, num_rows="dynamic")
            st.session_state['key'] = edited_df
            switch_page("page1")
