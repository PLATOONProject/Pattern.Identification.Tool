import json
import os
import sys
import logging
from datetime import datetime
from os import listdir
from os.path import isfile, join

from src.db import MongoDB
from src.clusterization import main_cluster
from src.balance import main_balance


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')


def get_args():
    conffile = os.getenv("CONFIG_FILE_NAME", "./src/conf/config.json")
    conf = json.load(open(conffile))

    return conf['model'], conf['granularity'], conf['start_dt'], conf['end_dt'], conf['signal'], conf['concentrator_id']


def main(db, start_dt, end_dt, model, gran, signal, cnc_id):
    # Get data
    cnt_df = db.get_data(cnc=cnc_id, cnc_info=False, start_dt=start_dt, end_dt=end_dt)
    # Choose the model
    if model == 'balance':
        # Set the granularity
        daily = True if gran == 'daily' else False
        # Get cnc data
        cnc_df = db.get_data(cnc=cnc_id, cnc_info=True, start_dt=start_dt, end_dt=end_dt)
        res = main_balance(concentrador_df=cnc_df, contador_df=cnt_df, signal=signal, daily=daily)

    elif model == 'cluster':
        res = main_cluster(data=cnt_df, signal=signal)

    else:
        res = {'code': 500, 'message': 'Error. Details: Model not implemented'}

    return res


if __name__ == '__main__':
    # Get params
    model, gran, start_dt, end_dt, sgn, cnc_id = get_args()
    # Database object
    db = MongoDB(signal=sgn)
    # Call main function
    res = main(db=db, start_dt=start_dt, end_dt=end_dt, model=model, gran=gran, signal=sgn, cnc_id=cnc_id)
    finalname = f'_{model}_{gran}_solutions.csv' if model == 'balance' else f'_{model}_solutions'
    outputfilename = f'src/solutions/{str(datetime.today().date())}' + finalname
    if res['code'] == 200:
        res['data'].to_csv(outputfilename+".csv")
    else:
        f = open(outputfilename+'_logs.csv', 'w')
        f.write(res['message'])
        f.close()
    exit()
