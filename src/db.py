import os
import pandas as pd
from pymongo import MongoClient


class MongoDB:
    def __init__(self, signal):
        self._host = os.getenv("HOST_MONGO", "192.168.1.116")
        self._port = os.getenv("PORT_MONGO", "27017")
        self._user = os.getenv("USER_MONGO", "admindb")
        self._pass = os.getenv("PASSWORD_MONGO", "passworddb")
        self.db = os.getenv("MONGODB", "plattom")
        self.collection = os.getenv("S02", "S02")
        self.signal = signal

    def init_client(self):
        CONNECTION_STRING = f"mongodb://{self._user}:{self._pass}@{self._host}:{self._port}/"
        client = MongoClient(CONNECTION_STRING)
        self.cl = client

    def close_client(self):
        self.cl.close()

    def get_cnc_id(self, cnc):
        return cnc, [c[:3] + "S" + c[4:] for c in cnc]

    def read_mongo(self, cnc, cnc_info):
        # Connect mongodb
        self.init_client()
        # Cnc id
        cncs, cnts = self.get_cnc_id(cnc=cnc)
        # Query
        if cnc_info:

            data = self.cl[self.db][self.collection].find({'cncid': {'$in': cncs}, 'cntid': {'$in': cnts}},
                                                          {'_id': 0, 'cncid': 1, 'cntid': 1, 'fh': 1, self.signal: 1})
        else:
            data = self.cl[self.db][self.collection].find({'cncid': {'$in': cncs}, 'cntid': {'$nin': cnts}},
                                                          {'_id': 0, 'cncid': 1, 'cntid': 1, 'fh': 1, self.signal: 1})
        # From mongodb to dataframe
        raw_df = pd.DataFrame(list(data))
        # Columns
        raw_df['timestamp'] = raw_df['fh'].dt.date
        raw_df['period'] = raw_df['fh'].dt.hour
        # Close connection
        self.close_client()
        return raw_df

    def get_data(self, cnc, cnc_info, start_dt, end_dt):
        # Get data
        data = self.read_mongo(cnc=cnc, cnc_info=cnc_info)
        # Filter data by date
        filter_df = data[(data['fh'] >= pd.to_datetime(start_dt)) &
                         (data['fh'] <= pd.to_datetime(end_dt))].reset_index(drop=True)
        return filter_df
