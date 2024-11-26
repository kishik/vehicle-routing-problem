import pandas as pd
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///test.db', echo=False)
import sqlite3


data = pd.read_csv('umb_sap_tasks_plan.csv')
data.dropna(subset=['address'], inplace=True)
data.reset_index(inplace=True)
data = data['address']
data = pd.Series(pd.unique(data))
# data.to_sql('addr', con=engine)
con = sqlite3.connect("test.db")
cursor = con.cursor()
query = '''INSERT INTO addr(name) VALUES'''
for el in list(data):
    query += f'''
            ('{el}'),'''

query = query[:-1] + ';'
cursor.execute(query)
con.commit()
# # data.dropna(inplace=True)
# # data.reset_index(inplace=True)
# print(data)