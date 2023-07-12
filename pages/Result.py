import math
import streamlit as st
import pandas as pd
import osmnx as ox
import sklearn
import numpy as np
import geopandas as gpd
# import taxicab as tc
from functools import partial
import networkx as nx
from datetime import datetime, date, timedelta
import requests
from matplotlib import pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from joblib import Parallel, delayed
from streamlit_custom_notification_box import custom_notification_box
from streamlit_extras.app_logo import add_logo
from st_pages import Page, show_pages, add_page_title

# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be


# st.set_page_config(
#         page_title="Результат2222",
# )

styles = {'material-icons': {'color': 'blue'},
          'text-icon-link-close-container': {'box-shadow': '#3896de 0px 4px'},
          'notification-text': {'': ''},
          'close-button': {'': ''},
          'link': {'': ''}}

# if st.checkbox("Use url", value=True):
#     add_logo("http://placekitten.com/120/120")
# else:
# add_logo("./mog_logo_ru_LB.jpeg")
# st.write("👈 Check out the cat in the nav-bar!")
# image_path = 'mog_logo_ru_LB.jpeg'
# st.image(image_path, width=128)
# if st.checkbox("Use url", value=True):
add_logo("https://mosoblgaz.ru/local/html/assets/images/n-logo-p.svg")


# show_pages(
#     [
#         Page("diploma-dashboard/Data_Upload.py", "Ввод данных", "🏠"),
#         Page("pages/Result.py", "Результат", ":books:"),
#     ]
# )


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
    pass


def export_to_csv():
    pass


def working_days(start_date, finish_date):
    pass


def calculate_time_list(places: list[int], i: int):
    lenghts = nx.single_source_dijkstra_path_length(G_travel_time, places[i], cutoff=28800, weight='travel_time')
    matrix = {place: lenghts[place] for place in places}
    return matrix


edited_df = st.session_state['key']
edited_df = st.data_editor(edited_df, num_rows="dynamic", hide_index=True)
if st.button('Готово', key='coords'):
    with st.spinner('Идет составление расписания, пожалуйста подождите'):
        st.text(edited_df['time_norm'].astype(float).sum() * 60)
        st.text(len(edited_df))
        edited_df['time_norm'] = edited_df['time_norm'].astype(float)
        classic_work = edited_df.iloc[1:].groupby('date_start')['time_norm'].sum()
        days = edited_df.iloc[1:].groupby('date_start').groups.keys()
        print(classic_work)
        classic_work = classic_work.tolist()
        classic_work = [work * 60 for work in classic_work]
        print(classic_work)

        df1 = {'Время работы в минутах': classic_work, 'Дни': days}
        st.bar_chart(df1, x='Дни', y='Время работы в минутах')
        st.text('Количество рабочих дней: ' + str(len(days)))
        if 'map' not in st.session_state:
            st.session_state['map'] = ox.io.load_graphml('data/graph.graphml')
        G_travel_time = st.session_state['map']
        custom_notification_box(icon='info', textDisplay='Загрузили карту Московской области',
                                externalLink='', url='#', styles=styles, key="map_ready")
        edited_df.reset_index(inplace=True)
        basic_len = len(edited_df)
        works_num = dict()
        for i, row in edited_df.iterrows():
            print(i)
            works_num[i] = ox.distance.nearest_nodes(
                G_travel_time, row['lon'], row['lat'], return_dist=False)
            row['time_norm'] = float(row['time_norm'])

            time_norm = math.ceil(float(row['time_norm']) * 60)
            time_from = math.ceil(int(nx.shortest_path_length(G_travel_time, source=works_num[0],
                                                              target=works_num[i], weight='travel_time')) / 60)
            time_to = math.ceil(int(nx.shortest_path_length(G_travel_time, source=works_num[i],
                                                            target=works_num[0], weight='travel_time')) / 60)
            while time_norm + time_from + time_to > 480:
                delta = 480 - time_from - time_to - 3
                edited_df.loc[i, 'time_norm'] = float(edited_df.loc[i, 'time_norm']) - delta / 60
                row['time_norm'] -= delta / 60
                time_norm -= delta
                print(row['time_norm'])
                new_row = row.copy()
                new_row['time_norm'] = delta / 60
                edited_df = pd.concat([edited_df, new_row.to_frame().T], ignore_index=True)
                works_num[len(edited_df) - 1] = works_num[i]
        edited_df = st.data_editor(edited_df, num_rows="dynamic", hide_index=True)

        count_df = edited_df.groupby(['date_start']).size().values.tolist()
        coords = edited_df.values.tolist()
        coords_i = [(i, coords[i][-2], coords[i][-1]) for i in range(len(coords))]
        # i : node number

        # remove ununique node numbers
        works_unique = list(set(works_num.values()))
        dicts_number = {works_unique[i]: i for i in range(len(works_unique))}
        st.text(str(coords_i))
        st.text(len(edited_df))
        custom_notification_box(icon='info', textDisplay='Приступаем к матрице смежности',
                                externalLink='shortest_path_length',
                                url='https://networkx.org/documentation/stable/reference/algorithms/generated'
                                    '/networkx.algorithms.shortest_paths.generic.shortest_path_length.html',
                                styles=styles, key="matrix_start")
        result = Parallel(n_jobs=-1)(delayed(calculate_time_list)(works_unique, i) for i in range(len(works_unique)))
        # i [distance to [0] [1]] node number
        # time_matrix = [[works_unique[j] for j in range(len(coords_i))]]
        # [i [distance to 0 1]] node number
        #     from coords_i i to others
        time_matrix = [[result[dicts_number[works_num[i]]][works_num[j]]
                        for j in range(len(coords_i))] for i in range(len(coords_i))]

        custom_notification_box(icon='info', textDisplay='Закончили с матрицей смежности',
                                externalLink='', url='#', styles=styles, key="matrix_end")

        minute_matrix = [[math.ceil(time_matrix[i][j] / 60) for j in range(len(time_matrix[0]))] for i in
                         range(len(time_matrix))]

        service_time = edited_df['time_norm'].astype(float).tolist()
        service_time = [math.ceil(service_time[i] * 60) for i in range(len(service_time))]
        service_time[0] = 0

        for i in range(len(minute_matrix)):
            for j in range(len(minute_matrix)):
                minute_matrix[i][j] += service_time[j]
            minute_matrix[i][i] = 0

        num_vehicles = len(edited_df)


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
            j = 1
            plan_outputs = []
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)

                plan_output = 'Маршрут для бригады {}:\n'.format(j)
                indexes.append([])
                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    plan_output += '{0} Время({1},{2}) -> '.format(
                        manager.IndexToNode(index), solution.Min(time_var),
                        solution.Max(time_var))
                    index = solution.Value(routing.NextVar(index))
                    indexes[i].append(manager.IndexToNode(index))
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Время({1},{2})\n'.format(manager.IndexToNode(index),
                                                            solution.Min(time_var),
                                                            solution.Max(time_var))
                plan_output += 'Время маршрута: {}min\n'.format(
                    solution.Min(time_var))
                day_time.append(solution.Min(time_var))
                # times.append(solution.Min(time_var))
                i += 1
                plan_outputs.append((plan_output, solution.Min(time_var)))
                # x.append(i)
                # if solution.Min(time_var) > 0:
                #     j += 1
                #     st.text(plan_output)
                total_time += solution.Min(time_var)
            # print(indexes)
            plan_outputs = sorted(plan_outputs, key=lambda x: x[1], reverse=True)
            for route, time in plan_outputs:
                if time > 0:
                    st.text(route)
            # Сохраняем таблицу
            global edited_df
            inds = list(filter(lambda i: i != [0], indexes))

            df = edited_df.reset_index(drop=True)
            print(334, df.columns.values)
            df.drop(columns=['lat', 'lon'], index=0, inplace=True)
            dates = sorted(df.date_start.unique())

            for date_num, route in enumerate(inds):
                for count, value in enumerate(route):
                    if value == 0:
                        continue
                    df.loc[value, 'date_number'] = date_num
                    df.loc[value, 'date_start'] = df.loc[value, 'date_end'] = dates[date_num]
                    df.loc[value, 'visiting_order'] = count
            df = df.astype({"date_number": int, "visiting_order": int})

            cols = list(df.columns.values)
            cols = ['date_number', 'visiting_order'] + cols[:-2]
            df = df.reindex(columns=cols)

            df.sort_values(by=['date_number', 'visiting_order'], inplace=True)
            csv = df.to_csv(index=False)

            my_works = [len(indexes[i]) - 1 for i in range(len(indexes))]
            my_works = list(filter(lambda num: num != 0, my_works))
            new_work_time = [[service_time[j] for j in indexes[i]] for i in range(len(indexes))]
            new_work_time = [sum(new_work_time[i]) for i in range(len(new_work_time))]
            work_time = sum(new_work_time)
            day_work = len(my_works)
            old_work_time = sum(count_df)
            old_day_work = len(count_df)

            if len(my_works) < len(count_df):
                my_works.extend([0] * (len(count_df) - len(my_works)))
            elif len(my_works) > len(count_df):
                count_df.extend([0] * (len(my_works) - len(count_df)))
            st.text('Время всех маршрутов: {}min'.format(total_time))
            new_work_time, day_time = zip(*sorted(zip(new_work_time, day_time), key=lambda x: x[1], reverse=True))
            way_time = [day_time[i] - new_work_time[i] for i in range(len(day_time))]
            days = edited_df.iloc[1:].groupby('date_start').groups.keys()
            df1 = {'Рабочие дни': list(days)[:day_work], 'Время работы в минутах': new_work_time[:day_work],
                   'Время пути в минутах': way_time[:day_work]}
            st.bar_chart(df1, x='Рабочие дни', y=('Время работы в минутах', 'Время пути в минутах'))

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(label="Среднее число задач в день", value=str(basic_len / day_work),
                        delta=str(basic_len / day_work - basic_len / len(days)))
            col2.metric(label="% рабочего и путевого времени",
                        value=str(round(sum(day_time) / day_work / 8 / 60 * 100, 2)) + '%')
            col3.metric(label="% рабочего времени",
                        value=str(round((sum(service_time) / day_work / 8 / 60) * 100, 2)) + '%',
                        delta=str(round(((sum(service_time) / day_work)
                                         - (sum(service_time) / len(days))) * 100 / 8 / 60)) + '%'
                        )
            col4.metric(label="Число рабочих дней", value=str(day_work),
                        delta=str(day_work - len(days)), delta_color="inverse")

            st.download_button(
                label="Скчать расписание в формате CSV",
                data=csv,
                file_name='result.csv',
                mime='text/csv',
            )


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

            # Instantiate route start and end times to produce feasible times.
            for i in range(data['num_vehicles']):
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.Start(i)))
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.End(i)))

            # Setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            # search_parameters.time_limit.seconds = 30
            search_parameters.use_full_propagation = False
            search_parameters.time_limit.seconds = 60
            search_parameters.log_search = True
            search_parameters.use_full_propagation = True
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)
            # Print solution on console.
            if solution:
                print_solution(data, manager, routing, solution)
            else:
                print('no solution')
        #     можно переписать на комбинацию всех методов и выбор лучшего

        main()
