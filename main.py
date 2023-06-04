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


def get_coordinates(names: list[str]) -> list[tuple[str, float, float]]:
    result = []
    for element in names:
        # try:
        #     result.append((element, float(geocode(element).latitude), float(geocode(element).longitude)))
        # except:
        r = requests.get('https://geocode-maps.yandex.ru/1.x',
                         params={'geocode': element, 'apikey': '2486ab02-2c39-4e68-8d2a-50c7deec2a70', 'format': 'json',
                                 'bbox': '35.497,56.999~40.32,54.188'})
        lon, lat = list(map(float,
                            r.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point'][
                                'pos'].split()))
        result.append((element, lat, lon))
    return result

# title = st.text_input('Column name')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with st.spinner('Wait for it...'):
        # Can be used wherever a "file-like" object is accepted:
        data_csv = pd.read_csv(uploaded_file, encoding='cp1251',
                               dtype={17: str, 18: str, 19: str, 21: str, 22: str, 24: str, 26: str}, parse_dates=True)
        data_csv.loc[:, 'date_start'] = pd.to_datetime(data_csv.loc[:, 'date_start'], format='%d.%m.%Y')
        data_csv.loc[:, 'date_end'] = pd.to_datetime(data_csv.loc[:, 'date_end'], format='%d.%m.%Y')
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
        data_csv = filtered_data['address']

        edited_df = st.data_editor(data_csv, num_rows="dynamic")

        if st.button('Готово', key='data'):
            data = list(edited_df)
            # data.insert(0, 'ул. Горького, 18, Ступино, Московская обл., 142802')
            coords = get_coordinates(data)
            df = pd.DataFrame(coords)
            edited_df = st.data_editor(df, num_rows="dynamic")
            st.session_state['key'] = edited_df
            switch_page("page1")
        # if st.button('Готово', key='coords'):
        #     coords = edited_df.values.tolist()
        #     print(coords)
        #     coords_i = [(i, coords[i][1], coords[i][2]) for i in range(len(coords))]
        #     print(coords_i)
        #     st.text(str(coords_i))
        #     G = ox.graph_from_place('Московская область', network_type='drive')
        #     G_speed = ox.speed.add_edge_speeds(G)
        #     G_travel_time = ox.speed.add_edge_travel_times(G_speed)
        #     ox.io.save_graph_geopackage(G_travel_time)
        #     ox.io.save_graphml(G_travel_time)
        #     result = Parallel(n_jobs=-1)(delayed(calculate_time_list)(coords_i, i) for i in range(len(coords_i)))
        #     time_matrix = [result[i][1] for i in range(len(result))]
        #     minute_matrix = [[time_matrix[i][j] // 60 for j in range(len(time_matrix[0]))] for i in range(len(time_matrix))]
        #     service_time = [90 for i in range(len(time_matrix))]
        #     service_time[0] = 0
        #     for i in range(len(minute_matrix)):
        #         for j in range(len(minute_matrix)):
        #             minute_matrix[i][j] += service_time[j]
        #         minute_matrix[i][i] = 0
        #
        #
        #     def create_data_model():
        #         """Stores the data for the problem."""
        #         data = {}
        #         data['time_matrix'] = minute_matrix
        #         data['num_vehicles'] = 12
        #         data['depot'] = 0
        #         return data
        #
        #
        #     def print_solution(data, manager, routing, solution):
        #         """Prints solution on console."""
        #         times = []
        #         x = []
        #         i = 0
        #         print(f'Objective: {solution.ObjectiveValue()}')
        #         time_dimension = routing.GetDimensionOrDie('Time')
        #         total_time = 0
        #         for vehicle_id in range(data['num_vehicles']):
        #             index = routing.Start(vehicle_id)
        #             plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        #             while not routing.IsEnd(index):
        #                 time_var = time_dimension.CumulVar(index)
        #                 plan_output += '{0} Time({1},{2}) -> '.format(
        #                     manager.IndexToNode(index), solution.Min(time_var),
        #                     solution.Max(time_var))
        #                 index = solution.Value(routing.NextVar(index))
        #             time_var = time_dimension.CumulVar(index)
        #             plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
        #                                                         solution.Min(time_var),
        #                                                         solution.Max(time_var))
        #             plan_output += 'Time of the route: {}min\n'.format(
        #                 solution.Min(time_var))
        #             times.append(solution.Min(time_var))
        #             i += 1
        #             x.append(i)
        #             st.text(plan_output)
        #             total_time += solution.Min(time_var)
        #         # print(times)
        #         # print(x)
        #         fig, ax = plt.subplots()
        #
        #         ax.plot(x, times, linewidth=2.0)
        #         st.text('Total time of all routes: {}min'.format(total_time))
        #
        #
        #     def main():
        #         """Solve the VRP with time windows."""
        #         # Instantiate the data problem.
        #         data = create_data_model()
        #
        #         # Create the routing index manager.
        #         manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
        #                                                data['num_vehicles'], data['depot'])
        #
        #         # Create Routing Model.
        #         routing = pywrapcp.RoutingModel(manager)
        #
        #         # Create and register a transit callback.
        #         def time_callback(from_index, to_index):
        #             """Returns the travel time between the two nodes."""
        #             # Convert from routing variable Index to time matrix NodeIndex.
        #             from_node = manager.IndexToNode(from_index)
        #             to_node = manager.IndexToNode(to_index)
        #             return data['time_matrix'][from_node][to_node]
        #
        #         transit_callback_index = routing.RegisterTransitCallback(time_callback)
        #
        #         # Define cost of each arc.
        #         routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        #
        #         # Add Time Windows constraint.
        #         time = 'Time'
        #         routing.AddDimension(
        #             transit_callback_index,
        #             0,  # allow waiting time
        #             480,  # maximum time per vehicle
        #             False,  # Don't force start cumul to zero.
        #             time)
        #         time_dimension = routing.GetDimensionOrDie(time)
        #         # Add time window constraints for each location except depot.
        #         # for location_idx, time_window in enumerate(data['time_windows']):
        #         #     if location_idx == data['depot']:
        #         #         continue
        #         #     index = manager.NodeToIndex(location_idx)
        #         #     time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        #         # # Add time window constraints for each vehicle start node.
        #         # depot_idx = data['depot']
        #         # for vehicle_id in range(data['num_vehicles']):
        #         #     index = routing.Start(vehicle_id)
        #         #     time_dimension.CumulVar(index).SetRange(
        #         #         data['time_windows'][depot_idx][0],
        #         #         data['time_windows'][depot_idx][1])
        #
        #         # Instantiate route start and end times to produce feasible times.
        #         for i in range(data['num_vehicles']):
        #             routing.AddVariableMinimizedByFinalizer(
        #                 time_dimension.CumulVar(routing.Start(i)))
        #             routing.AddVariableMinimizedByFinalizer(
        #                 time_dimension.CumulVar(routing.End(i)))
        #
        #         # Setting first solution heuristic.
        #         search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        #         search_parameters.first_solution_strategy = (
        #             routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        #         # search_parameters.time_limit.seconds = 30
        #
        #         # Solve the problem.
        #         solution = routing.SolveWithParameters(search_parameters)
        #
        #         # Print solution on console.
        #         if solution:
        #             print_solution(data, manager, routing, solution)
        #
        #
        #     main()
