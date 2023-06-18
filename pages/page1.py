import math
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
                                 'bbox': '35.497,54.188~40.32,56.999'})
        lon, lat = list(map(float,
                            r.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point'][
                                'pos'].split()))
        result.append((element, lat, lon))
    return result


def split_big_work(df, time_matrix, work_times, working_day=480):
    print(df.shape)
    time_matrix = np.array(time_matrix)
    n = df.shape[0]
    df['time_norm'] = df['time_norm'].astype(float)
    for i in range(1, n):
        first_work_time = work_times[i]
        # print(work_times[i])
        # print(time_matrix[0, i], time_matrix[i, 0])
        # делим работу, пока она не укладывается в один день
        while time_matrix[0, i] + time_matrix[i, 0] > working_day:
            # print(work_times[i] + time_matrix[0, i] + time_matrix[i, 0])
            delta = working_day - (time_matrix[0, i] + time_matrix[i, 0] - work_times[i])
            work_times.append(delta)
            work_times[i] -= delta

            # далее делаем магию с time_matrix
            # вычитать дельту!!!!!!!!!!!!!!
            # вставляем столбец
            X = np.copy(time_matrix[:, i])
            time_matrix = np.append(time_matrix, np.expand_dims(X, axis=1), axis=1)

            # вставляем строку
            X = np.copy(time_matrix[i, :])
            # X = np.append(X, 0)
            # print(np.expand_dims(X, axis=0))
            # print(type(X))
            # print(type(time_matrix))
            # print(time_matrix)
            time_matrix = np.append(time_matrix, np.expand_dims(X, axis=0), axis=0)
            print(time_matrix)
            # df.reset_index(inplace=True)
            new_row = df.iloc[i].copy()
            # new_row[]
            # df.loc[len(df)] = new_row
            # df = df.append(new_row)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            # print(new_row)
            # print(type(new_row))
            # print(df)
            df.loc[-1, 'time_norm'] = delta / 60
            df.loc[i, 'time_norm'] -= delta / 60


def export_to_csv():
    pass


def working_days(start_date, finish_date):
    pass


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
        if 'map' not in st.session_state:
            st.session_state['map'] = ox.io.load_graphml('data/graph.graphml')
        G_travel_time = st.session_state['map']
        custom_notification_box(icon='info', textDisplay='Загрузили карту Московской области',
                                externalLink='', url='#', styles=styles, key="map_ready")
        edited_df.reset_index(inplace=True)
        for i, row in edited_df.iterrows():
            print(i)
            row['time_norm'] = float(row['time_norm'])
            if i == 0:
                base_lat = row['lat']
                base_lon = row['lon']
            time_norm = math.ceil(float(row['time_norm']) * 60)
            time_from = math.ceil(int(nx.shortest_path_length(G_travel_time, source=ox.distance.nearest_nodes(
                G_travel_time, base_lon, base_lat, return_dist=False),
                                                              target=ox.distance.nearest_nodes(
                                                                  G_travel_time, row['lon'], row['lat'],
                                                                  return_dist=False), weight='travel_time')) / 60)
            time_to = math.ceil(int(nx.shortest_path_length(G_travel_time, source=ox.distance.nearest_nodes(
                G_travel_time, row['lon'], row['lat'], return_dist=False),
                                                            target=ox.distance.nearest_nodes(
                                                                G_travel_time, base_lon, base_lat,
                                                                return_dist=False), weight='travel_time')) / 60)
            while time_norm + time_from + time_to > 480:
                delta = 480 - time_from - time_to
                edited_df.loc[i, 'time_norm'] = float(edited_df.loc[i, 'time_norm']) - delta / 60
                row['time_norm'] -= delta / 60
                time_norm -= delta
                print(row['time_norm'])

                #         добавить строку копию
                new_row = row.copy()
                new_row['time_norm'] = delta / 60
                # new_row[]
                # df.loc[len(df)] = new_row
                # df = df.append(new_row)
                edited_df = pd.concat([edited_df, new_row.to_frame().T], ignore_index=True)
        edited_df = st.data_editor(edited_df, num_rows="dynamic", hide_index=True)

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
        # print(time_matrix)
        minute_matrix = [[math.ceil(time_matrix[i][j] / 60) for j in range(len(time_matrix[0]))] for i in
                         range(len(time_matrix))]
        # service_time = [service_time_avg for i in range(len(time_matrix))]
        # print(minute_matrix)
        service_time = edited_df['time_norm'].astype(float).tolist()
        service_time = [math.ceil(service_time[i] * 60) for i in range(len(service_time))]
        service_time[0] = 0
        print(service_time)
        for i in range(len(minute_matrix)):
            for j in range(len(minute_matrix)):
                minute_matrix[i][j] += service_time[j]
            minute_matrix[i][i] = 0
        print(minute_matrix)


        # делим большие работы на кусочки
        # split_big_work(edited_df, minute_matrix, service_time)
        # time_matrix = np.array(minute_matrix)
        # print(time_matrix)
        # work_times = minute_matrix
        # working_day = 480
        # i = 1
        # delta = working_day - (time_matrix[0, i] + time_matrix[i, 0] - work_times[i])
        # work_times.append(delta)
        # work_times[i] -= delta
        # # нужно запоминать номера уже добавленных строк из этой строки
        # # далее делаем магию с time_matrix
        # # вычитать дельту!!!!!!!!!!!!!!
        # # вставляем столбец
        # X = np.copy(time_matrix[:, i])
        # time_matrix = np.append(time_matrix, np.expand_dims(X, axis=1), axis=1)
        #
        # # вставляем строку
        # X = np.copy(time_matrix[i, :])
        # # X = np.append(X, 0)
        # # print(np.expand_dims(X, axis=0))
        # # print(type(X))
        # # print(type(time_matrix))
        # # print(time_matrix)
        # time_matrix = np.append(time_matrix, np.expand_dims(X, axis=0), axis=0)
        #
        # # дальше перебрать две колонки
        # # вычитаем уменьшение работы
        # time_matrix[:, i] = time_matrix[:, i] - delta
        # time_matrix[i, i] = 0
        # # уже вычли дельту
        # time_matrix[-1, i] = work_times[i]
        # # переходим к колонке -1
        # # это все можно сделать сразу после получения df!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(time_matrix)
        # # df.reset_index(inplace=True)
        # new_row = edited_df.iloc[i].copy()
        # # new_row[]
        # # df.loc[len(df)] = new_row
        # # df = df.append(new_row)
        # df = pd.concat([edited_df, new_row.to_frame().T], ignore_index=True)
        # # print(new_row)
        # # print(type(new_row))
        # # print(df)
        # df.loc[-1, 'time_norm'] = delta / 60
        # df.loc[i, 'time_norm'] -= delta / 60
        # print(df)

        def create_data_model():
            """Stores the data for the problem."""
            data = {}
            data['time_matrix'] = minute_matrix
            data['num_vehicles'] = num_vehicles
            data['depot'] = 0
            # data['time_windows'] = [
            #     (0, 480) for i in range(len(minute_matrix))  # 16
            # ]
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
            day_time = []
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
                day_time.append(solution.Min(time_var))
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
            new_work_time = [[service_time[j] for j in indexes[i]] for i in range(len(indexes))]
            print(new_work_time)
            new_work_time = [sum(new_work_time[i]) for i in range(len(new_work_time))]
            print(new_work_time)
            # work_time = sum(my_works)
            work_time = sum(new_work_time)
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
            # df = {'Предложенное решение': my_works, 'Изначальное решение': count_df}
            df1 = {'Время работы': new_work_time[::-1], 'Время работы и пути': day_time[::-1]}
            st.bar_chart(df1)
            # st.line_chart(df)
            col1, col2, col3 = st.columns(3)

            col1.metric(label="Среднее число задач в день", value=str(sum(my_works) / day_work),
                        delta=str(sum(my_works) / day_work - old_work_time / old_day_work))
            col2.metric(label="% рабочего и путевого времени",
                        value=str(round(sum(day_time) / day_work / 8 / 60 * 100, 2)) + '%')
            col3.metric(label="% рабочего времени",
                        value=str(round((sum(service_time) / day_work / 8 / 60) * 100, 2)) + '%',
                        delta=str(round(((sum(service_time) / day_work)
                                         - (sum(service_time) / old_day_work)) * 100 / 8 / 60)) + '%')


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
                30,  # allow waiting time
                480,  # maximum time per vehicle
                False,  # Don't force start cumul to zero.
                time)
            time_dimension = routing.GetDimensionOrDie(time)
            # for location_idx, time_window in enumerate(data['time_windows']):
            #     if location_idx == data['depot']:
            #         continue
            #     index = manager.NodeToIndex(location_idx)
            #     time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
            # depot_idx = data['depot']
            # for vehicle_id in range(data['num_vehicles']):
            #     index = routing.Start(vehicle_id)
            #     time_dimension.CumulVar(index).SetRange(
            #         data['time_windows'][depot_idx][0],
            #         data['time_windows'][depot_idx][1])
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
