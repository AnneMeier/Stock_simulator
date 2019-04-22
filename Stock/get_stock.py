"""
Stock crawling

https://github.com/kh-kim/korea_stock_market_crawler #stock code

19/4/2019
"""
import os,csv,glob
import pandas as pd
import numpy as np
from os import listdir



class Dataset():
    #name , code
    def __init__(self):
        self.path_dir = os.path.join('Stock_symbol/data/')
        self.refile_name , self.file_name , self.num_of_file = self.get_file_names()
        self.result = self.read_df()

    def get_file_names(self):
        files = listdir(self.path_dir)
        
        file_name = []
        refile_name = [] 
        temp = 0
        for fname in files:
            file_name.append(fname)
            fname = fname.rstrip('_s.csv')
            refile_name.append(fname)
            temp = temp + 1

        return refile_name, file_name , temp

    def read_df(self):
        list_name = self.file_name

        df = pd.read_csv( self.path_dir + list_name[0] ,error_bad_lines=False)
        left = pd.DataFrame(df)
        left.rename( columns={"name": "name_" + str(self.refile_name[0]) , "code": "code_" + str(self.refile_name[0])} ,inplace=True)

        for s_name in list_name[1:]:
            df = pd.read_csv( self.path_dir + s_name ,error_bad_lines=False)
            right = pd.DataFrame(df)

            s_name = s_name.rstrip('_s.csv')
            right.rename( columns={"name": "name_" + str(s_name) , "code": "code_" + str(s_name)} ,inplace=True)
            result = pd.merge(left, right, left_index=True, right_index=True, how='outer')

        return result

    def div_col_df(self,i):
        df = self.result.iloc[:,0+(2*i):2+(2*i)]
        df = df.dropna()
        df.rename(columns={ df.columns[0]: "name", df.columns[1]: "code" },inplace=True)

        return df

    def Naver_par_url(self):
        num = self.num_of_file

        print ("112")

        for j in range(1,num):

            
            div_col_df = self.div_col_df(j)
            url = []
            print ("\n/////start " + self.refile_name[j] + "_file/////" )

            for i in range(0,len(div_col_df.index)):            
                item_name = div_col_df.name[i]
                code = div_col_df.code[i]
                code = int(code)

                #if len(code) is not 6:
                if len(str(code)) is not 6:
                    if len(str(code)) is 1:
                        code = "00000"+ (str(code))
                    elif len(str(code)) is 2:
                        code = "0000"+ (str(code))
                    elif len(str(code)) is 3:
                        code = "000"+ (str(code))
                    elif len(str(code)) is 4:
                        code = "00"+ (str(code))
                    elif len(str(code)) is 5:
                        code = "0"+ (str(code))                 
                url.append('http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code))
            
            df = self.get_Naver_data(url,div_col_df)

            print ("/////end " + self.refile_name[j] + "_file/////" )
            
            file_name = 'Stock_symbol/data/' + self.refile_name[j] + '_dataframe.csv'
            df.to_csv(file_name, index = True)
            print ("/////save " + self.refile_name[j] + "_dataframe file/////\n" )
 



    
    def get_Naver_data(self, N_url, div_col_df):   
        df_ = pd.DataFrame()
        index_name = div_col_df
        index_name.rename(columns={ index_name.columns[0]: "name", index_name.columns[1]: "code" },inplace=True)
        index_name = index_name['name'].values.tolist()

        count = 0
        
        for url in N_url:  
            page = 0
            df = pd.DataFrame()
            temp = pd.DataFrame()
            temp2 = pd.DataFrame()
            done = True           
            print ("parsing : " + url)

            while done is True:
                if page is 0:
                    ex_pg_url = '{url}&page={page}'.format(url=url, page=1)
                else:
                    ex_pg_url = '{url}&page={page}'.format(url=url, page=page)

                page = page + 1
                pg_url = '{url}&page={page}'.format(url=url, page=page)
                temp = pd.read_html(ex_pg_url, header=0)[0]
                temp = temp.dropna()
                temp = temp['날짜'].values.tolist()
                temp= temp[-1]

                temp2 = pd.read_html(pg_url, header=0)[0]
                temp2 = temp2.dropna()
                temp2 = temp2['날짜'].values.tolist()
                temp2= temp2[-1]



                if (page == 1) or (temp != temp2):
                    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
                    #print ("Page : " + str(page) + " is Done")
                else:
                    print ("End this url")
                    done = False
                    #print ("Page : " + str(page) + " is not Done")

            df.rename( columns={"날짜": "날짜_" + str(index_name[count]) , "종가": "종가_" + str(index_name[count])
            ,"전일비": "전일비_" + str(index_name[count]) , "시가": "시가_" + str(index_name[count])
            ,"고가": "고가_" + str(index_name[count]) , "저가": "저가_" + str(index_name[count])
            ,"거래량": "거래량_" + str(index_name[count])} ,inplace=True)
            df = df.dropna()
            df_ = pd.merge(df_, df, left_index=True, right_index=True, how='outer')
            count = count + 1

        return df_

def main():
    data_set = Dataset()
    data_set.Naver_par_url()

    

if __name__ == '__main__':

    main()






