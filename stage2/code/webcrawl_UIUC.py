#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 12 21:03:12 2018

@author: Mingren Shen

This python program get data of all "Data Science" books form libraray of UIUC


"""

#import needed package
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
#import pickle

def collectBookURL(webpage,result_dict):
    for a in webpage.findAll("a", {"class": "title"}):
        # key is URL and content is title
        # in case Book with same title
        result_dict[a['href']] = a.get_text().replace('/',"")
        
def parseBookDetailPage(pageURL,pandasDF,rowIndex):
    tmp_r = requests.get(pageURL)
    tmp_soup =  bs(tmp_r.text,"lxml")
    tableList = tmp_soup.find_all('table')
    #print(len(tableList))
    Info_table = tableList[2]
    ISBN_table = tableList[3]
    InfoSummary = dictify(Info_table)
    # Processing general Infomation Table
    for key,val in InfoSummary.items():
        if ( key in pandasDF.columns):
            pandasDF.at[rowIndex, key]= val
        else:
            print("===Error Uncaught record field in Table 1: " + key)
            pandasDF.at[rowIndex, key]= val
    # Processing ISBN Table
    ISBNSummary = dictify(ISBN_table)
    for key,val in ISBNSummary.items():
        if ( key in pandasDF.columns):
            pandasDF.at[rowIndex, key]= val
        else:
            print("===Error Uncaught record field in Table 2 : " + key)
            pandasDF.at[rowIndex, key]= val


# dictify all the elements of HTML Table
def dictify(tableHTML):
    result_dict = dict()
    for table_rows in tableHTML.select("tr"):
        content = table_rows.text.split(':')
        result_dict[content[0].strip()] =  content[1].strip()
    return result_dict

    

if __name__ == "__main__":
    """
    method description :
    1.get the search result from UW-Madison's library
    2.get basic data
    3.collect all URL for the books

    """
    print("================The Start=========================")

    # 20 Books per page, so 500 pages give 10000 Books.
    #print("We only need first 10000 Books in the searching results.")
    # dictory to store URL for the details pages of Book
    dict_BookDetailsURL = dict()
    # Rotate to get all URL of Books
    preURL = 'https://vufind.carli.illinois.edu/vf-uiu/Search/Home?type%5B%5D=&lookfor%5B%5D=Data%20Science&bool%5B%5D=AND&type%5B%5D=title&lookfor%5B%5D=&bool%5B%5D=AND&type%5B%5D=author&lookfor%5B%5D=&start_over=1&specDate=&version=any&gPub=&page='
    #postURL = '&q=Data+Science'
    # 10 Books per page, so 1000 pages give 10000 Books.
    print("We only need first 10000 Books in the searching results.")
    MAX_PAGES = 501
    for i in range(1,MAX_PAGES):
        if ( i % 50 == 0):
            print("Processing Page " + str(i))
        tmp_URL = preURL + str(i)
        tmp_r = requests.get(tmp_URL)
        tmp_soup =  bs(tmp_r.text,"lxml")
        collectBookURL(tmp_soup,dict_BookDetailsURL)

    # set up DataFrame to store the data
    df = pd.DataFrame(columns=['ID','TITLE','Author','Other Names','Published','Topics','Genres','Tags','ISBN'])
    
    # Processing the detailed page of each book
    keyID = 0
    for key,value in dict_BookDetailsURL.items():
        itemURL = 'https://vufind.carli.illinois.edu' + key + '/Description'
        parseBookDetailPage(itemURL,df,keyID)
        df.at[keyID,'ID'] = keyID
        df.at[keyID,'TITLE'] = value
        if (keyID % 500 == 0):
            print ("Now processing Book " + str(keyID))
        keyID = keyID + 1
        #print(itemURL)
    df.to_csv("BOOKS_UIUC.csv",sep = '\t',index = False, encoding='utf-8')
    print("================The End===========================")
    