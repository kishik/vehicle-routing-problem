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
import os.path
import sqlite3



class Saver:


    @staticmethod
    def read_distnces_from_point_exist(point, points):
        con = sqlite3.connect("test.db")
        cursor = con.cursor()
        cursor.execute(f"SELECT target_point, distance FROM distances WHERE source_point={point}")
        data = cursor.fetchall()
        data = Saver.convert(data)
        result = []
        error = False
        print(data)
        for key in points:
            if key not in points:
                result = None
                error = True
            else:
                result.append(data[key])
        return error, result


    @staticmethod
    def save_distnces_from_point_exist(point, points, distances):
        con = sqlite3.connect("test.db")
        cursor = con.cursor()
        query = '''INSERT INTO distances(source_point, target_point, distance) VALUES'''
        for i in range(len(distances)):
            query += f'''
            ({point}, {points[i]}, {distances[i]}),
            ({points[i]}, {point}, {distances[i]}),'''
        query = query[:-1] + ';'
        cursor.execute(query)
        con.commit()


    @staticmethod
    def load_ids(names):
        con = sqlite3.connect("test.db")
        cursor = con.cursor()
        result = []
        for name in names:
            cursor.execute(f"SELECT Id FROM addr WHERE Name='{name}'")
            data = cursor.fetchall()
            result.append(data[0][0])
        return result


    @staticmethod
    def convert(tup):
        return dict(tup)

# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be


# st.set_page_config(
#         page_title="Результат2222",
# )


class Keeper:

    def load_file(self, source):
        self.data = pd.read_csv(source)
        return self.data
    

    def save_file(self, target):
        self.data.to_csv(target)
    

    def extract_subset(self, values):
        return self.data.loc[values, [f'{value}' for value in values]]
    

    def is_enough_data(self, values):
        return not self.extract_subset(values).isnull().values.any()
    

    def add_data(self, new_data):
        new_cols = set(new_data.columns) - set(self.data.columns)
        old_cols = set(new_data.columns).intersection(set(self.data.columns))
        no_cols = new_cols | old_cols
        # no_cols.remove('index')
        # self.data.loc[[int(new_col) for new_col in no_cols], [f'{new_col}' for new_col in no_cols]] = new_data.loc[[int(new_col) for new_col in no_cols], [f'{new_col}' for new_col in no_cols]]
        # for new_col in new_cols:
        #     self.data.loc[int(new_col), f'{new_col}'] = 0
            # fill data
        # for index, row in new_data.iterrows():
        #     if index in self.data.index:
        #         for new_col in new_cols:
        #             self.data.loc[int(index), f'{new_col}'] = new_data.loc[int(index), f'{new_col}']
        return self.data
                


    
    @staticmethod
    def numpy_to_pandas(matrix, indexes):
        panda = pd.DataFrame(matrix, columns=indexes)
        idxs = [f'{index}' for index in indexes]
        panda['index'] = idxs
        panda.set_index('index', inplace=True)
        return panda


    def pandas_to_numpy(self):
        return self.data.to_numpy(), self.data.index



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
    # numpy.busdaycalendar.holidays добавлять праздники
    # print('start date')
    # print(start_date)
    # print(f'start_date {start_date} {np.datetime64(start_date)}')
    # print(finish_date)
    return np.busday_count(np.datetime64(pd.to_datetime((start_date))).astype('datetime64[D]'), np.datetime64(pd.to_datetime(finish_date)).astype('datetime64[D]') + np.timedelta64(1,'D'))


def calculate_time_list(places: list[int], i: int):
    lenghts = nx.single_source_dijkstra_path_length(G_travel_time, places[i], cutoff=28800, weight='travel_time')
    res = {places[i]: lenghts}
    res[places[i]][places[i]] = 0
    return res



edited_df = st.session_state['key']
edited_df = st.data_editor(edited_df, num_rows="dynamic", hide_index=True)
st.write(len(st.session_state['all_brigades']))
if st.button('Готово', key='coords'):
    edited_df['date_start'] = pd.to_datetime(edited_df['date_start']).dt.date
    edited_df['date_end'] = pd.to_datetime(edited_df['date_end']).dt.date
    with st.spinner('Идет составление расписания, пожалуйста подождите'):
        st.text(edited_df['time_norm'].astype(float).sum() * 60)
        st.text(len(edited_df))
        edited_df['time_norm'] = edited_df['time_norm'].astype(float)
        classic_work = edited_df.iloc[1:].groupby('date_start')['time_norm'].sum()
        days = edited_df.iloc[1:].groupby('date_start').groups.keys()
        # print(classic_work)
        classic_work = classic_work.tolist()
        classic_work = [work * 60 for work in classic_work]
        # print(classic_work)
        # brigades_num = len(edited_df['brigada'].unique())
        brigades_num = len(st.session_state['all_brigades'])
        print(f'brigades num {brigades_num}')
        df1 = {'Время работы в минутах': classic_work, 'Дни': days}
        st.bar_chart(df1, x='Дни', y='Время работы в минутах')
        edited_df.reset_index(inplace=True)
        st.text('Количество рабочих дней: ' + str(int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end']))))
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
                G_travel_time, row['addr_lon'], row['addr_lat'], return_dist=False)
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
        coords_i = [(i, coords[i][18], coords[i][19]) for i in range(len(coords))]
        # i : node number

        # remove ununique node numbers
        works_unique = list(set(works_num.values()))
        dicts_number = {works_unique[i]: i for i in range(len(works_unique))}
        st.text(str(coords_i))
        st.text(len(edited_df))

        loaded = []
        loaded_flag = []
        for work in works_unique:
            line = Saver.read_distnces_from_point_exist(work, works_unique)
            loaded.append(line[1])
            loaded_flag.append(line[0])
        
     
        custom_notification_box(icon='info', textDisplay='Приступаем к матрице смежности',
                            externalLink='shortest_path_length',
                            url='https://networkx.org/documentation/stable/reference/algorithms/generated'
                                '/networkx.algorithms.shortest_paths.generic.shortest_path_length.html',
                            styles=styles, key="matrix_start")
        # visited_points = st.session_state['cached']
        visited_points = []
        # points_to_check = list(set(works_num.values()) - set(visited_points.keys()))
        points_to_check = list(set([works_unique[i] for i in range(len(works_unique))]))
        lenghts = Parallel(n_jobs=-1)(delayed(calculate_time_list)(points_to_check, i) for i in range(len(points_to_check)))
        print(lenghts)
        for i in range(len(points_to_check)):
            visited_points[points_to_check[i]] = lenghts[i]
        # print(visited_points[list(visited_points.keys())[0]])
        # print(visited_points.keys())
        result = [{place: visited_points[source][source].get(place, 9999999) for place in works_unique} for source in works_unique]

        # i [distance to [0] [1]] node number
        # time_matrix = [[works_unique[j] for j in range(len(coords_i))]]
        # [i [distance to 0 1]] node number
        #     from coords_i i to others
        time_matrix = [[result[dicts_number[works_num[i]]][works_num[j]]
                        for j in range(len(coords_i))] for i in range(len(coords_i))]

        # custom_notification_box(icon='info', textDisplay='Закончили с матрицей смежности',
        #                         externalLink='', url='#', styles=styles, key="matrix_end")

        minute_matrix = [[math.ceil(time_matrix[i][j] / 60) for j in range(len(time_matrix[0]))] for i in
                            range(len(time_matrix))]
            # test.data = test.numpy_to_pandas(minute_matrix, list(range(len(minute_matrix))))
        
        # else:
        #     custom_notification_box(icon='info', textDisplay='Загружаем из файла',
        #                         externalLink='', url='',
        #                         styles=styles, key="matrix_load")
        #     print('WOW')
        #     test.load_file('test.csv')
        #     minute_matrix = test.pandas_to_numpy()[0]
        # print(list(range(len(minute_matrix))))
        # print(test.data)
        # test.save_file('test.csv')
        service_time = edited_df['time_norm'].astype(float).tolist()
        service_time = [math.ceil(service_time[i] * 60) for i in range(len(service_time))]
        service_time[0] = 0

        for i in range(len(minute_matrix)):
            for j in range(len(minute_matrix)):
                minute_matrix[i][j] += service_time[j]
            minute_matrix[i][i] = 0

        days_for_brigade = working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])
        print(f"days {days_for_brigade}")
        
        num_vehicles = brigades_num * days_for_brigade
        print(f"num {num_vehicles}")


        def flatten(a):
            return [c for b in a for c in flatten(b)] if hasattr(a, '__iter__') else [a]


        def create_data_model():
            """Stores the data for the problem."""
            data = {}
            data['time_matrix'] = minute_matrix
            data['num_vehicles'] = int(num_vehicles)
            # data['depot'] = 0

            data["starts"] = flatten([[i] * days_for_brigade for i in range(int(brigades_num))])
            data["ends"] = data["starts"].copy()
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
            global edited_df
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)

                plan_output = ''
                indexes.append([])
                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    plan_output += '{0} Время({1},{2}) -> '.format(
                        edited_df.iloc[manager.IndexToNode(index)]['address'], solution.Min(time_var),
                        solution.Max(time_var))
                    index = solution.Value(routing.NextVar(index))
                    indexes[i].append(manager.IndexToNode(index))
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Время({1},{2})\n'.format(edited_df.iloc[manager.IndexToNode(index)]['address'],
                                                            solution.Min(time_var),
                                                            solution.Max(time_var))
                plan_output += 'Время маршрута: {}min\n'.format(
                    solution.Min(time_var))
                day_time.append(solution.Min(time_var))
                # times.append(solution.Min(time_var))
                i += 1
                plan_outputs.append((plan_output, solution.Min(time_var)))
                # x.append(i)
                if solution.Min(time_var) > 0:
                    j += 1
                #     st.text(plan_output)
                total_time += solution.Min(time_var)
            # print(indexes)
            plan_outputs = sorted(plan_outputs, key=lambda x: x[1], reverse=True)
            j = 1
            for route, time in plan_outputs:
                if time > 0:
                    text = 'Маршрут в {} день:\n'.format(j) + route
                    j += 1
                    st.text(text)
            # Сохраняем таблицу

            inds = list(filter(lambda i: i != [0], indexes))

            df = edited_df.reset_index(drop=True)
            print(334, df.columns.values)
            df.drop(columns=['addr_lat', 'addr_lon'], index=0, inplace=True)
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


        def new_print(data, manager, routing, solution):
            i = 0
            indexes = []
            # print(f'Objective: {solution.ObjectiveValue()}')
            time_dimension = routing.GetDimensionOrDie('Time')
            total_time = 0
            day_time = []
            j = 1
            plan_outputs = []
            global edited_df
            brigade_total_work = []
            brigades_days_time = [[], []]
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                l = 0
                plan_output = f"Номер бригады: {vehicle_id // int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) + 1}\n"
                plan_output += f"Номер рабочего дня: {vehicle_id % int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) + 1}\n"
                indexes.append([])
                while not routing.IsEnd(index):
                    l += 1
                    time_var = time_dimension.CumulVar(index)
                    plan_output += '{0} Время({1},{2}) -> '.format(
                        edited_df.iloc[manager.IndexToNode(index)]['address'], solution.Min(time_var),
                        solution.Max(time_var))
                    index = solution.Value(routing.NextVar(index))
                    indexes[i].append(manager.IndexToNode(index))
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Время({1},{2})\n'.format(edited_df.iloc[manager.IndexToNode(index)]['address'],
                                                            solution.Min(time_var),
                                                            solution.Max(time_var))
                plan_output += 'Время маршрута: {}min\n'.format(
                    solution.Min(time_var))
                plan_output += f"Количество задач: {l - 1}"
                brigades_days_time[0].append(
                    pd.to_datetime(np.busday_offset(np.datetime64(edited_df.loc[0, 'date_start']), vehicle_id % int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) + 1, roll='forward')))
                brigades_days_time[1].append(solution.Min(time_var))
                if solution.Min(time_var) > 0:
                    st.text(plan_output)
                day_time.append(solution.Min(time_var))
                # times.append(solution.Min(time_var))
                i += 1
                plan_outputs.append((plan_output, solution.Min(time_var)))
                # x.append(i)
                if solution.Min(time_var) > 0:
                    j += 1
                #     st.text(plan_output)
                total_time += solution.Min(time_var)
                if (vehicle_id % int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end']))) == (int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) - 1):
                    brigade_total_work.append(total_time)
                    
                    if total_time > 0:
                        st.text(f"Общее время: {total_time}")
                        # print(brigades_days_time[:][0])
                        # print(brigades_days_time[:][1])
                        df1 = {'Рабочие дни': np.array(brigades_days_time[0]),
                               'Общее время в минутах': np.array(brigades_days_time[1])}
                        st.bar_chart(df1, x='Рабочие дни', y=('Общее время в минутах'))
                        total_time = 0
                        brigades_days_time = [[], []]
            if brigades_days_time != [] and total_time > 0:
                st.text(f"Общее время: {total_time}")
                df1 = {'Рабочие дни': np.array(brigades_days_time[0]),
                               'Общее время в минутах': np.array(brigades_days_time[1])}
                st.bar_chart(df1, x='Рабочие дни', y=('Общее время в минутах'))



        def fixing_task(manager, routing, work_id, date):
            index = manager.NodeToIndex(work_id)
            brigades_ids = [-1]
            print(f'id {work_id}')
            for j in range(brigades_num):
                # print(f"num days {int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end']))}")
                # print(f"num days 2 {int(working_days(edited_df.loc[0, 'date_start'], date))}")
                # print(f"daybrigade {j * int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) + int(working_days(edited_df.loc[0, 'date_start'], date))}")
                brigades_ids.append(j * int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) + int(working_days(edited_df.loc[0, 'date_start'], date)))
            routing.VehicleVar(index).SetValues(brigades_ids)


        def main():
            """Solve the VRP with time windows."""
            # Instantiate the data problem.
            data = create_data_model()
            # print(data)
            # Create the routing index manager.
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                                   int(num_vehicles), data["starts"], data["ends"])

            # Create Routing Model.
            routing = pywrapcp.RoutingModel(manager)
            reset_df = edited_df.reset_index(drop=True)
            frozen_tasks = reset_df[reset_df['work_type'] == "OTS"]
            for i, row in frozen_tasks.iterrows():
                fixing_task(manager, routing, i, row['date_start'])

            # Create and register a transit callback.
            def time_callback(from_index, to_index):
                """Returns the travel time between the two nodes."""
                # Convert from routing variable Index to time matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['time_matrix'][from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(time_callback)
            for vehicle_id in range(num_vehicles):
                routing.SetFixedCostOfVehicle(
                    vehicle_id * 10000, vehicle_id)
                    # 1000 * vehicle_id**3 // int(working_days(edited_df.loc[0, 'date_start'], edited_df.loc[0, 'date_end'])) + vehicle_id ** 2, vehicle_id)
            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            # index = manager.NodeToIndex(1)
            # routing.VehicleVar(index).SetValues([-1, 0])
            # index = manager.NodeToIndex(22)
            # routing.VehicleVar(index).SetValues([-1, 0])
            # Add Time Windows constraint.
            time = 'Time'
            routing.AddDimension(
                transit_callback_index,
                0,  # allow waiting time
                480,  # maximum time per vehicle
                True,  # force start cumul to zero.
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
            search_parameters.log_search = False
            search_parameters.use_full_propagation = True
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)
            # Print solution on console.
            if solution:
                new_print(data, manager, routing, solution)
            else:
                print('no solution')
        # #     можно переписать на комбинацию всех методов и выбор лучшего
        #     for i, row in frozen_tasks.iterrows():
        #         fixing_task(manager, routing, i, row['date_start'])

        main()
