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
from streamlit_custom_notification_box import custom_notification_box

styles = {'material-icons': {'color': 'blue'},
          'text-icon-link-close-container': {'box-shadow': '#3896de 0px 4px'},
          'notification-text': {'': ''},
          'close-button': {'': ''},
          'link': {'': ''}}


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


def calculate_time_list(places: list[tuple[int, float, float]], i: int):
    matrix = []
    for j in range(len(places)):
        matrix.append(int(nx.shortest_path_length(G_travel_time, source=ox.distance.nearest_nodes(
            G_travel_time, places[i][2], places[i][1], return_dist=False),
                                                  target=ox.distance.nearest_nodes(
                                                      G_travel_time, places[j][2], places[j][1],
                                                      return_dist=False), weight='travel_time')))
    return i, matrix


num_vehicles = int(st.number_input('Введите число бригадодней', value=15))
service_time_avg = st.number_input('Введите среднее время одной работы в минутах', value=90)
edited_df = st.session_state['key']
edited_df = st.data_editor(edited_df, num_rows="dynamic", hide_index=True)
if st.button('Готово', key='coords'):
    with st.spinner('Идет составление расписания, пожалуйста подождите'):
        count_df = edited_df.groupby(['date_start']).size().values.tolist()
        print(count_df)
        coords = edited_df.values.tolist()
        # print(coords)
        coords_i = [(i, coords[i][-2], coords[i][-1]) for i in range(len(coords))]
        # print(coords_i)
        st.text(str(coords_i))
        # G = ox.graph_from_place('Московская область', network_type='drive')
        # G_speed = ox.speed.add_edge_speeds(G)
        # G_travel_time = ox.speed.add_edge_travel_times(G_speed)
        if 'map' not in st.session_state:
            st.session_state['map'] = ox.io.load_graphml('data/graph.graphml')
        G_travel_time = st.session_state['map']
        custom_notification_box(icon='info', textDisplay='Загрузили карту Московской области',
                                externalLink='', url='#', styles=styles, key="map_ready")
        # ox.io.save_graph_geopackage(G_travel_time)
        # ox.io.save_graphml(G_travel_time)
        custom_notification_box(icon='info', textDisplay='Приступаем к матрице смежности',
                                externalLink='shortest_path_length',
                                url='https://networkx.org/documentation/stable/reference/algorithms/generated'
                                    '/networkx.algorithms.shortest_paths.generic.shortest_path_length.html',
                                styles=styles, key="matrix_start")
        result = Parallel(n_jobs=-1)(delayed(calculate_time_list)(coords_i, i) for i in range(len(coords_i)))
        custom_notification_box(icon='info', textDisplay='Закончили с матрицей смежности',
                                externalLink='', url='#', styles=styles, key="matrix_end")
        time_matrix = [result[i][1] for i in range(len(result))]
        minute_matrix = [[time_matrix[i][j] // 60 for j in range(len(time_matrix[0]))] for i in range(len(time_matrix))]
        service_time = [service_time_avg for i in range(len(time_matrix))]
        service_time[0] = 0

        for i in range(len(minute_matrix)):
            for j in range(len(minute_matrix)):
                minute_matrix[i][j] += service_time[j]
            minute_matrix[i][i] = 0


        def create_data_model():
            """Stores the data for the problem."""
            data = {}
            data['time_matrix'] = minute_matrix
            data['num_vehicles'] = num_vehicles
            data['depot'] = 0
            return data


        def print_solution(data, manager, routing, solution):
            """Prints solution on console."""
            # times = []
            # x = []
            i = 0
            indexes = []
            print(f'Objective: {solution.ObjectiveValue()}')
            time_dimension = routing.GetDimensionOrDie('Time')
            total_time = 0
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                indexes.append([])
                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    plan_output += '{0} Time({1},{2}) -> '.format(
                        manager.IndexToNode(index), solution.Min(time_var),
                        solution.Max(time_var))
                    index = solution.Value(routing.NextVar(index))
                    indexes[i].append(manager.IndexToNode(index))
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                            solution.Min(time_var),
                                                            solution.Max(time_var))
                plan_output += 'Time of the route: {}min\n'.format(
                    solution.Min(time_var))
                # times.append(solution.Min(time_var))
                i += 1
                # x.append(i)
                st.text(plan_output)
                total_time += solution.Min(time_var)
            # print(indexes)
            # print(x)
            # fig, ax = plt.subplots()
            my_works = [len(indexes[i]) - 1 for i in range(len(indexes))]
            my_works = list(filter(lambda num: num != 0, my_works))
            work_time = sum(my_works)
            day_work = len(my_works)
            old_work_time = sum(count_df)
            old_day_work = len(count_df)
            print(count_df)
            print(my_works)
            if len(my_works) < len(count_df):
                my_works.extend([0] * (len(count_df) - len(my_works)))
            elif len(my_works) > len(count_df):
                count_df.extend([0] * (len(my_works) - len(count_df)))
            # ax.plot(x, times, linewidth=2.0)
            st.text('Total time of all routes: {}min'.format(total_time))
            df = {'Предложенное решение': my_works, 'Изначальное решение': count_df}
            st.line_chart(df)
            col1, col2, col3 = st.columns(3)

            col1.metric(label="Среднее число задач в день", value=str(work_time / day_work),
                        delta=str(work_time / day_work - old_work_time / old_day_work))
            col2.metric(label="% рабочего и путевого времени",
                        value=str(round(total_time / day_work / 8 / 60 * 100, 2)) + '%')
            col3.metric(label="% рабочего времени",
                        value=str(round((work_time * 90 / day_work / 8 / 60) * 100, 2)) + '%',
                        delta=str(round(((work_time / day_work) - (old_work_time / old_day_work)) * 100 * 90 / 8 / 60))
                              + '%')


        def main():
            """Solve the VRP with time windows."""
            # Instantiate the data problem.
            data = create_data_model()

            # Create the routing index manager.
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                                   data['num_vehicles'], data['depot'])

            # Create Routing Model.
            routing = pywrapcp.RoutingModel(manager)

            # Create and register a transit callback.
            def time_callback(from_index, to_index):
                """Returns the travel time between the two nodes."""
                # Convert from routing variable Index to time matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['time_matrix'][from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(time_callback)

            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Add Time Windows constraint.
            time = 'Time'
            routing.AddDimension(
                transit_callback_index,
                0,  # allow waiting time
                480,  # maximum time per vehicle
                False,  # Don't force start cumul to zero.
                time)
            time_dimension = routing.GetDimensionOrDie(time)

            # Instantiate route start and end times to produce feasible times.
            for i in range(data['num_vehicles']):
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.Start(i)))
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.End(i)))

            # Setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
            # search_parameters.time_limit.seconds = 30
            search_parameters.use_full_propagation = False
            # search_parameters.time_limit.seconds = 180
            search_parameters.log_search = True
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)
            # Print solution on console.
            if solution:
                print_solution(data, manager, routing, solution)


        main()
