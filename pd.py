import pandas as pd
import numpy as np
import sqlite3

#    0 1 2 3
#    _______
# 0 |0 1 2 4
# 1 |1 0 1 3
# 2 |1 1 0 8
# 3 |1 1 7 0


#    0 2 4
#    _____
# 0 |0 5 6
# 2 |8 0 7
# 4 |9 4 0

save = pd.DataFrame(np.array([[0, 0, 1, 2, 4], [1, 1, 0, 1, 3], [2, 1, 1, 0, 8], [3, 1, 1, 7, 0]]),
                   columns=['index', '0', '1', '2', '3'])
save.set_index('index', inplace=True)
save.to_csv('save.csv')
# print(save)
loaded = pd.read_csv('save.csv')
# loaded['index'] = loaded['index'].astype('int')
added = pd.DataFrame(np.array([[0, 0, 5, 6], [2, 8, 0, 7], [4, 9, 4, 0]]),
                   columns=['index', '0', '2', '4'])
# print(added.columns)
# find new columns

# print(save.loc[[0,1,3],['0', '1', '3']])


# print(set(added.columns))
# print(set(loaded.columns))
# print(set(added.columns) - set(loaded.columns))
new_cols = set(added.columns) - set(loaded.columns)
inter_cols = set(added.columns).intersection(set(loaded.columns))
# for new column in new columns
added["4"] = np.nan
added.loc[4, '4'] = 0

# fill info about inter cols in main

# print(added)
# print(loaded.set_index('index'))




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



con = sqlite3.connect("test.db")
cursor = con.cursor()
cursor.execute(f"SELECT dest_place_id, travel_time FROM fsma_cross_time WHERE source_place_id={270831237};")
data = cursor.fetchall()
print(Saver.convert(data))


# a = Keeper()
# a.data = save
# print(a.pandas_to_numpy())
# print(list(a.pandas_to_numpy()[1]))
# print(a.pandas_to_numpy()[0])
# print(a.numpy_to_pandas(*a.pandas_to_numpy()))