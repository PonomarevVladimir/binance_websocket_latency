import websocket
import json
import time
import re
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
from multiprocessing import Pool
import scipy.stats as stats

class SocketConnection:
    def __init__(self, num, wss):
        self.num = num
        self.wss = wss
        self.msg_list = list()
        self.result_list = list()

    def current_ms(self):
        return round(time.time() * 1000)
    
    def on_message(self, _wsa, data):
        # сохраняем время получения сообщения и само сообщение as is
        local_time = self.current_ms()
        self.msg_list.append([local_time, data])
        # через минуту после старта закрываем соединение
        if self.current_ms() - self.start_time > 60000:
            _wsa.close() 

    def prepare_latency(self):
        # парсим сообщение по описанию с сайта и оставляем только нужные нам поля
        for data in self.msg_list:
            local_time = data[0]
            msg_id_list = re.findall(r'\"u\"\:\d+',data[1])
            msg_time_list = re.findall(r'\"E\"\:\d+',data[1])
            # проверяем что сообщение содержит оба поля и вычисляем задержку
            if (len(msg_id_list) > 0) & (len(msg_time_list) > 0):
                msg_id = msg_id_list[0][4:]
                msg_time = int(msg_time_list[0][4:])
                latency = local_time - msg_time
                self.result_list.append([self.num, msg_id, latency])

    def on_open(self, _wsa):
        # подписываемся на нужный стрим
        data = {"method": "SUBSCRIBE",
                "params":["btcusdt@bookTicker"],
                "id": self.num}
        _wsa.send(json.dumps(data))

    def on_error(self, _wsa, data):
        print(data)

    def connect(self):
        self.start_time = self.current_ms()
        wsa = websocket.WebSocketApp(self.wss, on_message=self.on_message, on_open=self.on_open, on_error=self.on_error)
        wsa.run_forever()


def conn_func(num, wss):
    conn = SocketConnection(num, wss)
    conn.connect()
    conn.prepare_latency()
    return conn.result_list

def stat_research(df:pd.DataFrame):
    # разбиваем на выборки по номеру соединения и оставляем сообщения, 
    # которые получили по всем соединениям (отбрасываем несколько первых и последних)
    sample = df.pivot(index = "msg_id", columns = "conn_id", values = "latency").dropna()
    # для проверки равенства дисперсий воспользуемся тестом Левена 
    # потому что он не требует нормальности и его можно применять сразу к нескольким выборкам
    v_stat, v_p_value = stats.levene(sample.iloc[:,0], sample.iloc[:,1], sample.iloc[:,2], sample.iloc[:,3], sample.iloc[:,4])
    v_quant = stats.f.ppf(v_p_value, 4, len(sample) - 5)
    print(f"Variances test for all connections:\n    statistic = {v_stat:.4f},\n    p-value = {v_p_value:.4f},\n    f-quantile = {v_quant:.4f}.")
    if v_stat>v_quant:
        print("H1\n")
    else:
        print("H0\n")
    for i in range(5):
        for j in range(i+1,5):
            # попарные тесты тоже проведём
            v_stat, v_p_value = stats.levene(sample.iloc[:,0], sample.iloc[:,1])
            v_quant = stats.f.ppf(v_p_value, 1, len(sample) - 1)
            print(f"Variances test for {i+1} and {j+1} connections:\n    statistic = {v_stat:.4f},\n    p-value = {v_p_value:.4f},\n    f-quantile = {v_quant:.4f}.")
            if v_stat>v_quant:
                print("H1\n")
            else:
                print("H0\n")
            # для матожиданий поспользуемся t-критерием Стьюдента, по-хорошему для этого необходимо проверить нормальность матожиданий
            # но мы сделаем вид, что можем применить ЦПТ, так как других тестов на равенство матожиданий ненормальных величин,
            # а у нас точно не нормальные величины, у нас и нет. Они все требуют дополнительных условий или проверяют не совсем ту гипотезу,
            # которая нам нужна
            m_stat, m_p_value = stats.ttest_ind(sample.iloc[:,i], sample.iloc[:,j], equal_var = False)
            m_quant = stats.t.ppf(m_p_value, len(sample))
            print(f"Means test for {i+1} and {j+1} connections:\n    statistic = {m_stat:.4f},\n    p-value = {m_p_value:.4f},\n    t-quantile = {m_quant:.4f}.")
            if m_stat>m_quant:
                print("H1\n")
            else:
                print("H0\n")



if __name__ == '__main__':
    print("start\n")
    # так как в задании один стрим, то я захардкодил его адрес и название, 
    # но можно сделать покрасивее и либо передавать их через стандартный ввод, 
    # либо через переменные оркестратора, если такой используется
    stream_name = 'btcusdt@bookTicker'
    wss = f'wss://fstream.binance.com/ws/{stream_name}'
    result = list()
    # создаём 5 параллельных соединений 
    # тут возникнут проблемы, если соединений окажется больше чем вычислительных ядер процессора
    print("openning connections, wait for them, please\n")
    with Pool(5) as p:
        result.extend(p.starmap(conn_func, [(1,wss),(2,wss),(3,wss),(4,wss),(5,wss)]))

    print("connections closed\n")
    result = sum(result,[])
    # записываем результат в pandas DataFrame (хотя spark мне нравится больше, но для такой задачи он будет перебором)
    df = pd.DataFrame(result, columns = ["conn_id", "msg_id", "latency"])
    # для каждого сообщения находим наименьшую задержку
    df["min_lat"] = df[["msg_id", "latency"]].groupby("msg_id", as_index=False).transform(lambda x: x.min())
    # ставим 1, если задержка равна наименьшей, и 0, если она больше
    df["min_lat_flg"] = pd.Series(0, index=df.index).mask(df["min_lat"] == df["latency"], 1)
    # сумма значений этого поля будет равна количеству сообщений, которые данное соединение получило первым
    df["fast_cnt"] = df[["conn_id", "min_lat_flg"]].groupby("conn_id", as_index=False).transform(lambda x: x.sum())
    # вычисляем количество уникальных сообщений
    msg_num = len(df["msg_id"].drop_duplicates())
    # находим доли "быстрых" апдейтов (их сумма будет больше 1, так как наименьшую задержку может иметь сразу несколько соединений)
    df["fast_ratio"] = df["fast_cnt"] / msg_num
    # убираем промежуточные столбцы
    df = df.drop(columns = ["min_lat", "min_lat_flg", "fast_cnt"])
    # выводим доли "быстрых" апдейтов
    df_fast = df[["conn_id", "fast_ratio"]].drop_duplicates().reindex()
    print("fast updates\n",df_fast,"\n")
    # проводим исследования на совпадения выборочных матожиданий и дисперсий. H0 - гипотеза о равенстве, H1 - её отрицание
    # стандартное отклонение - это корень из дисперсии, значит нам не важно равенство чего проверять
    stat_research(df[["conn_id", "msg_id", "latency"]])
    # строим графики функций распределения задержек
    sb.displot(df, x = "latency", hue = "conn_id", kind="ecdf", palette=sb.color_palette("tab10"))
    matplotlib.pyplot.show()

