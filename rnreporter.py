# -*- coding:utf-8 -*-
import os
import os.path
import pandas as pd
from collections import OrderedDict


class TradeData:
    def __init__(self, csv_path = None):
        if csv_path is None:
            self.csv_path = r"N:\projects\project_chengbo\position\position"
        else:
            self.csv_path = csv_path
        self.positions = self.read_positions()
        self.notional = self.read_notional()
        self.daily_ret = self.daily_ret()
        return

    def read_positions(self):
        positions = {}
        if os.path.isdir(self.csv_path):
            print(self.csv_path, "is dir")
        for dt_folder in os.listdir(self.csv_path):
            # print(dt_folder)
            trade_date = dt_folder
            dt_folder = os.path.join(self.csv_path, dt_folder)
            for file in os.listdir(dt_folder):
                if "posterior.csv" in file:
                    file_name = os.path.join(dt_folder, file)
                    #print(file_name)
                    df = pd.read_csv(file_name)
                    df = df[df['CNY'] > 0]
                    positions[trade_date] = df
        # noinspection PyTypeChecker
        positions = OrderedDict(sorted(positions.items()))
        return positions

    def read_notional_as_positions(self):
        positions = {}
        if os.path.isdir(self.csv_path):
            print(self.csv_path, "is dir")
        for dt_folder in os.listdir(self.csv_path):
            # print(dt_folder)
            trade_date = dt_folder
            dt_folder = os.path.join(self.csv_path, dt_folder)
            for file in os.listdir(dt_folder):
                if "notional.csv" in file:
                    file_name = os.path.join(dt_folder, file)
                    # print(file_name)
                    df = pd.read_csv(file_name)
                    positions[trade_date] = df
        # noinspection PyTypeChecker
        positions = OrderedDict(sorted(positions.items()))
        return positions

    def read_notional(self):
        if os.path.isdir(self.csv_path):
            print(self.csv_path, "is dir")
        max_trade_date = max(os.listdir(csv_path))
        for dt_folder in os.listdir(self.csv_path):
            # print(dt_folder)
            trade_date = dt_folder
            if trade_date != max_trade_date:
                continue
            dt_folder = os.path.join(self.csv_path, dt_folder)
            for file in os.listdir(dt_folder):
                if "notional.csv" in file:
                    file_name = os.path.join(dt_folder, file)
                    # print(file_name)
                    df = pd.read_csv(file_name)
        df['DATETIME'] = df['DATETIME'].str[:10]
        df = df.rename(columns={'DATETIME': 'date', 'NOTIONAL': 'notional'})
        return df


    def daily_ret_as_positions(self):
            daily_cny = []
            for k, v in self.positions.items():
                sum_cny = v['CNY'].sum()
                daily_cny.append([k, sum_cny])
            df = pd.DataFrame(daily_cny, columns=['date', 'cny'])
            notionals = []
            for k, v in self.notional.items():
                sum_notional = v['NOTIONAL'].sum()
                notionals.append([k, sum_notional])
            dfn = pd.DataFrame(notionals, columns=['date', 'notional'])
            df = df.merge(dfn, on='date', how='left')
            df['total'] = df['notional']
            df['pre_total'] = df['total'].shift(1)
            df['ret'] = (df['total'] - df['pre_total']) / df['pre_total'] * 100
            total_sta = df.iloc[0]['total']
            df['cum_ret'] = (df['total'] - total_sta) / total_sta * 100
            return df


    def daily_ret(self):
        daily_cny = []
        for k, v in self.positions.items():
            sum_cny = v['CNY'].sum()
            daily_cny.append([k, sum_cny])
        df = pd.DataFrame(daily_cny, columns=['date', 'cny'])
        dfn = self.notional
        df = df.merge(dfn, on='date', how='left')
        df['total'] = df['notional']
        df['pre_total'] = df['total'].shift(1)
        df['ret'] = (df['total'] - df['pre_total']) / df['pre_total'] * 100
        total_sta = df.iloc[0]['total']
        df['cum_ret'] = (df['total'] - total_sta) / total_sta * 100
        return df


class BTResult:
    def __init__(self, trade_data):
        self.TradeData = trade_data
        self.AvgRet = self.avgret()
        self.AvgStd = self.avgstd()
        self.AvgShp = self.avgshp()
        self.MDD, self.MddSta, self.MddEnd, self.MddPrd = self.mdd()
        self.Calmar = self.calmar()
        self.MinRet1Y = self.minret(240)
        self.MinRet1H = self.minret(120)
        self.MinRet1S = self.minret(60)
        self.VaR1D99 = None
        self.VaR1D95 = None
        self.VaR1W99 = None
        self.PortRank = None
        self.PosRetPct = self.posretpct()
        self.NegRetPct = self.negretpct()
        self.PosRetPct1W = None
        self.TurOvr = None
        self.Part95 = None
        self.Part99 = None

    def avgret(self):
        df = self.TradeData.daily_ret
        df['pd_date'] = pd.to_datetime(df['date'])
        df = df.dropna()
        df['year'] = df['pd_date'].dt.year
        yhdf = df.groupby('year').head(1)
        ytdf = df.groupby('year').tail(1)
        yhtdf = pd.concat([yhdf, ytdf])
        yhtdf = yhtdf.sort_values('date', ascending=True)
        yhtdf['p_total'] = yhtdf['total'].shift(1)
        yhtdf['year_ret'] = (yhtdf['total'] - yhtdf['p_total']) / yhtdf['p_total'] * 100
        yhtdf = yhtdf.groupby('year').tail(1)
        year_ret = yhtdf['year_ret'].sum() / yhtdf.shape[0]
        return year_ret

    def avgstd(self):
        df = self.TradeData.daily_ret
        df['pd_date'] = pd.to_datetime(df['date'])
        df = df.dropna()
        df['year'] = df['pd_date'].dt.year
        yearstd = df.groupby('year')['ret'].std()
        yearstd = yearstd.rename('std')
        yearstd = yearstd.to_frame().reset_index()
        year_std = yearstd['std'].sum() / yearstd.shape[0]
        return year_std

    def avgshp(self):
        if self.AvgStd is not None and self.AvgStd > 0:
            return self.AvgRet / self.AvgStd
        return None

    def avgshp_df(self):
        df = self.AvgStd.merge(self.AvgRet, on='year', how='left')
        df['avgshp'] = df['year_ret'] / df['std']
        return df[['year', 'avgshp']]

    def mdd(self):
        df = self.TradeData.daily_ret
        df['max_total'] = df['total'].cummax()
        df['mdd'] = (df['max_total'] - df['total']) / df['max_total'] * 100
        if df[df['mdd'] > 0].empty:
            return None, None, None, None
        max_idx = df['mdd'].idxmax()
        sta_idx = df[:max_idx]['total'].idxmax()
        max_row = df.loc[max_idx]
        mdd = max_row['mdd']
        mddend = max_row['date']
        mdf = df[:df['mdd'].idxmax()]
        mddsta = mdf.loc[sta_idx]['date']
        mddprd = df[sta_idx:max_idx].shape[0]
        return mdd, mddsta, mddend, mddprd

    def calmar(self):
        if self.MDD is not None and self.MDD > 0:
            return self.AvgRet / self.MDD
        return None

    def minret(self, days = 5):
        days = days
        df = self.TradeData.daily_ret
        df['total_sta'] = df['total'].shift(days)
        df['date_sta'] = df['date'].shift(days)
        df['ret_sta'] = (df['total'] - df['total_sta']) / df['total'] * 100
        cls = ['date', 'total', 'date_sta', 'total_sta', 'ret_sta']
        min_row = df.loc[df['ret_sta'].idxmin()][cls]
        return min_row['ret_sta']

    def posretpct(self):
        w5 = 0.0005
        df = self.TradeData.daily_ret
        number = df.shape[0]
        num_w5 = df[df['ret'] > w5].shape[0]
        return num_w5 / number

    def negretpct(self):
        w5 = -0.0005
        df = self.TradeData.daily_ret
        number = df.shape[0]
        num_w5 = df[df['ret'] > w5].shape[0]
        return num_w5 / number


import xlsxwriter


class ExcelWriter:
    def __init__(self, result):
        self.result = result

    def write(self, excel_name="report.xlsx"):
        if self.result is None:
            return
        workbook = xlsxwriter.Workbook(excel_name)
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True})
        worksheet.write('A1', '指标名称', bold)
        worksheet.write('B1', '指标数值', bold)
        names = ['AvgRet', 'AvgStd', 'AvgShp', 'MDD', 'MddSta', 'MddEnd', 'MddPrd', 'Calmar', 'MinRet1Y', 'MinRet1H',
                 'MinRet1S', 'VaR1D99', 'VaR1D95', 'VaR1W99', 'PortRank', 'PosRetPct', 'NegRetPct', 'PosRetPct1W',
                 'TurOvr', 'Part95', 'Part99']
        # print(names)
        datas = {}
        for name in names:
            datas[name] = None
        if self.result is None:
            return
        datas['AvgRet'] = self.result.AvgRet
        datas['AvgStd'] = self.result.AvgStd
        datas['AvgShp'] = self.result.AvgShp
        datas['MDD'] = self.result.MDD
        datas['MddSta'] = self.result.MddSta
        datas['MddEnd'] = self.result.MddEnd
        datas['MddPrd'] = self.result.MddPrd
        datas['Calmar'] = self.result.Calmar
        datas['MinRet1Y'] = self.result.MinRet1Y
        datas['MinRet1H'] = self.result.MinRet1H
        datas['MinRet1S'] = self.result.MinRet1S
        datas['VaR1D99'] = self.result.VaR1D99
        datas['VaR1D95'] = self.result.VaR1D95
        datas['VaR1W99'] = self.result.VaR1W99
        datas['PortRank'] = self.result.PortRank
        datas['PosRetPct'] = self.result.PosRetPct
        datas['NegRetPct'] = self.result.NegRetPct
        datas['PosRetPct1W'] = self.result.PosRetPct1W
        datas['TurOvr'] = self.result.TurOvr
        datas['Part95'] = self.result.Part95
        datas['Part99'] = self.result.Part99
        row = 1
        col = 0
        for k, v in datas.items():
            worksheet.write(row, col, k)  # 带默认格式写入
            worksheet.write(row, col + 1, v)  # 带自定义money格式写入
            row += 1

        # chart
        df = self.result.TradeData.daily_ret

        worksheet2 = workbook.add_worksheet()
        # worksheet2.write('A1', '指标名称2', bold)
        # df.to_excel(workbook, sheet_name='Sheet2')
        df = df.dropna()
        headings = ['date', 'cum_ret']
        worksheet2.write_row('A1', headings, bold)
        worksheet2.write_column('A2', df['date'])
        worksheet2.write_column('B2', df['cum_ret'])
        # here we create a line chart object .
        chart1 = workbook.add_chart({'type': 'line'})
        ret_len = df.shape[0]
        chart1.add_series({
            'name': ['Sheet2', 0, 1],
            'categories': ['Sheet2', 1, 0, ret_len, 0],
            'values': ['Sheet2', 1, 1, ret_len, 1],
        })

        chart1.set_title({'name': '净值成长曲线'})
        chart1.set_x_axis({'name': 'date'})
        chart1.set_y_axis({'name': 'return (%)'})

        # # Set an Excel chart style.
        # chart1.set_style(11)

        # # add chart to the worksheet with given
        # # offset values at the top-left corner of
        # # a chart is anchored to cell D2 .
        worksheet.insert_chart('D2', chart1, {'x_offset': 25, 'y_offset': 10})

        mean = df['ret'].mean()
        skew = df['ret'].skew()
        mean = round(mean, 4)
        skew = round(skew, 4)
        df['cat'] = pd.qcut(df['ret'], 10)
        # mean, skew
        cat_df = df['cat'].value_counts()
        cat_right = cat_df.index.categories.right.tolist()
        cat_count = cat_df.values.tolist()

        headings = ['cat_right', 'cat_count']
        worksheet2.write_row('C1', headings, bold)
        worksheet2.write_column('C2', cat_right)
        worksheet2.write_column('D2', cat_count)
        cat_len = cat_df.shape[0]
        chart2 = workbook.add_chart({'type': 'column'})
        chart2.add_series({
            'name': ['Sheet2', 0, 3],
            'categories': ['Sheet2', 1, 2, cat_len, 2],
            'values': ['Sheet2', 1, 3, cat_len, 3],
        })

        chart21 = workbook.add_chart({'type': 'line'})
        chart21.add_series({
            'name': ['Sheet2', 0, 2],
            'categories': ['Sheet2', 1, 2, cat_len, 2],
            'values': ['Sheet2', 1, 3, cat_len, 3],
        })
        chart2.combine(chart21)

        chart2_name = '单日收益率 ' + ' mean=' + str(mean) + ' skew=' + str(skew)
        chart2.set_title({'name': chart2_name})
        chart2.set_x_axis({'name': 'Right Index'})
        chart2.set_y_axis({'name': 'Count (mm)'})

        worksheet.insert_chart('D20', chart2, {'x_offset': 25, 'y_offset': 10})

        # 每年净值成长图概览
        df['year'] = df['pd_date'].dt.year
        year_ends = df.groupby('year').tail(1)['date']
        year_end_len = year_ends.shape[0]
        year_end_max = []
        for y in year_ends.tolist():
            for i in range(2):
                year_end_max.append(df['ret'].max())

        headings = ['date', 'ret', 'year_end', 'year_end_data']
        worksheet2.write_row('E1', headings, bold)
        worksheet2.write_column('E2', df['date'])
        worksheet2.write_column('F2', df['ret'])
        worksheet2.write_column('G2', year_ends)
        worksheet2.write_column('H2', year_end_max)

        chart3 = workbook.add_chart({'type': 'line'})
        ret_len = df.shape[0]
        chart3.add_series({
            'name': ['Sheet2', 0, 5],
            'categories': ['Sheet2', 1, 4, ret_len, 4],
            'values': ['Sheet2', 1, 5, ret_len, 5],
        })

        for i in range(1, year_end_len + 1):
            chart31 = workbook.add_chart({'type': 'scatter'})
            chart31.add_series({
                'name': ['Sheet2', 0, 5],
                'categories': ['Sheet2', 1, 6, year_end_len, 6],
                'values': ['Sheet2', 1, 7, year_end_len, 7],
                'y_error_bars': {
                    'type': 'percentage',
                    'value': 100,
                },
            })
            chart3.combine(chart31)

        chart3.set_title({'name': '每年净值成长图概览'})
        chart3.set_x_axis({'name': 'date'})
        chart3.set_y_axis({'name': 'return (%)'})
        worksheet.insert_chart('D40', chart3, {'x_offset': 25, 'y_offset': 10, 'x_scale': 2, 'y_scale': 1})

        # df['month'] = df['pd_date'].dt.month
        df['month'] = df['date'].str[:-3]
        mhdf = df.groupby('month').head(1)
        mtdf = df.groupby('month').tail(1)
        mdf = pd.concat([mhdf, mtdf])
        mdf.sort_values('date')
        # mdf['total_ms'] =
        # mdf['month_ret'] =

        # month_ret
        df['month'] = df['date'].str[:-3]
        mhdf = df.groupby('month').head(1)
        mtdf = df.groupby('month').tail(1)
        mdf = pd.concat([mhdf, mtdf])
        mdf = mdf.sort_values('date')
        mdf['bm_total'] = mdf['total'].shift(1)
        mdf['month_ret'] = (mdf['total'] - mdf['bm_total']) / mdf['total'] * 100
        mdf = mdf.groupby('month').tail(1)
        # chart
        month_len = mdf.shape[0]
        headings = ['month', 'month_ret']
        worksheet2.write_row('I1', headings, bold)
        worksheet2.write_column('I2', mdf['month'])
        worksheet2.write_column('J2', mdf['month_ret'])
        chart4 = workbook.add_chart({'type': 'column'})
        chart4.add_series({
            'name': ['Sheet2', 0, 9],
            'categories': ['Sheet2', 1, 8, month_len, 8],
            'values': ['Sheet2', 1, 9, month_len, 9],
        })

        chart41 = workbook.add_chart({'type': 'line'})
        chart41.add_series({
            'name': ['Sheet2', 0, 9],
            'categories': ['Sheet2', 1, 8, month_len, 8],
            'values': ['Sheet2', 1, 9, month_len, 9],
        })
        chart4.combine(chart41)

        chart4.set_title({'name': '月度收益率'})
        chart4.set_x_axis({'name': '时间(月)'})
        chart4.set_y_axis({'name': '收益率(%)'})

        worksheet.insert_chart('M3', chart4, {'x_offset': 25, 'y_offset': 10})

        workbook.close()


def run_reporter():
    return


if __name__ == "__main__":
    print("in")
    csv_path = r"N:\projects\project_chengbo\position\position"
    td = TradeData(csv_path)
    print(len(td.positions))
    print(td.daily_ret.head())
    btr = BTResult(td)
    print(btr)

    ew = ExcelWriter(btr)
    ew.write()

    exit()
    d = {2: 3, 1: 89, 4: 5, 3: 0}
    od = OrderedDict(sorted(d.items()))
    # od[2] = 3
    # od[1] = 89
    # od[4] = 5
    print(od)
    # print(td.positions)
