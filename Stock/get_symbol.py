"""
get symbol from execl
"""
import os , csv
import pandas as pd
import numpy as np

data_list = ("kospi" , "kosdaq")
def get_data():
    path_dir = os.path.join('Stock_symbol/')
    temp = None
    for i in data_list:
        df = pd.read_csv( path_dir + i + '.csv',error_bad_lines=False)
        i = df[["기업명","종목코드"]]
        i = i.rename(columns={'기업명': 'name', '종목코드': 'code'})

        if temp is None:
            j = i
            temp = i 

    i.to_csv(path_dir + "data/kospi_s.csv", encoding='utf-8',index=False)
    j.to_csv(path_dir + "data/kosdaq_s.csv", encoding='utf-8',index=False)

def main():
    get_data()



if __name__ == '__main__':

    main()