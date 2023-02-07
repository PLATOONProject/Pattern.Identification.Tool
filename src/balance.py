import pandas as pd


class Balance(object):

    def __init__(self, signal):
        self.signal = signal

    def pivot_data(self, data, groupby=True):
        if groupby:
            return pd.DataFrame(data.groupby('timestamp')[self.signal].sum())
        else:
            return data[self.signal].sum()

    def balance_diario(self, data_ct, data_cnc):
        cnt_grouped = self.pivot_data(data=data_ct)
        cnt_grouped.rename(columns={self.signal: self.signal + '_cnt'}, inplace=True)
        cnc_grouped = self.pivot_data(data=data_cnc)
        cnc_grouped[self.signal] = 1000 * cnc_grouped[self.signal]
        cnc_grouped.rename(columns={self.signal: self.signal + '_cnc'}, inplace=True)
        df = pd.concat([cnc_grouped, cnt_grouped], axis=1)
        df[self.signal+'_balance'] = df[self.signal+'_cnc'] - df[self.signal+'_cnt']
        return df

    def balance_completo(self, data_ct, data_cnc):
        cnc_value = 1000 * self.pivot_data(data=data_cnc, groupby=False)
        ct_value = self.pivot_data(data=data_ct, groupby=False)
        df = pd.DataFrame([], columns=[self.signal+'_cnc', self.signal+'_cnt', self.signal+'_balance'])
        df.loc[0, :] = cnc_value, ct_value, cnc_value - ct_value
        return df


def main_balance(concentrador_df, contador_df, signal, daily=True):
    try:
        balance_obj = Balance(signal=signal)
        if daily:
            res_data = balance_obj.balance_diario(data_ct=contador_df, data_cnc=concentrador_df)
        else:
            res_data = balance_obj.balance_completo(data_ct=contador_df, data_cnc=concentrador_df)
        return {'code': 200, 'message': 'OK', 'data': res_data}

    except Exception as e:
        return {'code': 500, 'message': 'Error in main balance. Details: ' + str(e)}
