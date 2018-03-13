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
    return 0
    """
    tmp_r = requests.get(pageURL)
    tmp_soup =  bs(tmp_r.text, "html.parser")
    pandasDF.at[rowIndex,'ID'] = rowIndex
    #pandasDF.at[rowIndex,'TITLE'] = tmp_soup.find('div',{'class':'titles flex_row'}).get_text().split('\n')[2]
    # processing information in Publication Details
    publicationDetails = tmp_soup.find('ul',{'class': "publication expand_list"})
    # keep the level of <li></li> elements
    dictFields = dictify(publicationDetails)

    # loop all key,value pairs of dictFields

    for key,val in dictFields.items():
        if (key.upper() in pandasDF.columns):
            pandasDF.at[rowIndex, key.upper()]= val
        else:
            print("===Error Uncaught record field : " + key)
            pandasDF.at[rowIndex, key.upper()]= val
    """

if __name__ == "__main__":
    """
    method description :
    1.get the search result from UW-Madison's library
    2.get basic data
    3.collect all URL for the books

    """
    print("================The Start=========================")

    
    # 20 Books per page, so 500 pages give 10000 Books.
    print("We only need first 10000 Books in the searching results.")
    # dictory to store URL for the details pages of Book
    dict_BookDetailsURL = dict()
    # Rotate to get all URL of Books
    preURL = 'https://vufind.carli.illinois.edu/vf-uiu/Search/Home?type%5B%5D=&lookfor%5B%5D=Data%20Science&bool%5B%5D=AND&type%5B%5D=title&lookfor%5B%5D=&bool%5B%5D=AND&type%5B%5D=author&lookfor%5B%5D=&start_over=1&specDate=&version=any&gPub=&page='
    #postURL = '&q=Data+Science'
    # 10 Books per page, so 1000 pages give 10000 Books.
    print("We only need first 10000 Books in the searching results.")
    MAX_PAGES = 5#01
    for i in range(1,MAX_PAGES):
        print("Processing Page" + str(i))
        tmp_URL = preURL + str(i)
        tmp_r = requests.get(tmp_URL)
        tmp_soup =  bs(tmp_r.text,"lxml")
        collectBookURL(tmp_soup,dict_BookDetailsURL)

    # set up DataFrame to store the data
    df = pd.DataFrame(columns=['ID','TITLE','CREATOR','FORMAT','PUBLICATION DATES','CONTRIBUTORS','PUBLICATION','PHYSICAL DETAILS','ISBNS','OCLC'])
    
    # Processing the detailed page of each book
    keyID = 0
    for key,value in dict_BookDetailsURL.items():
        itemURL = 'https://vufind.carli.illinois.edu' + key
        parseBookDetailPage(itemURL,df,keyID)
        df.at[keyID,'ID'] = keyID
        df.at[keyID,'TITLE'] = value
        keyID = keyID + 1
        #print(itemURL)
    df.to_csv("BOOKS_UIUC.csv",sep = '\t',index = False, encoding='utf-8')
    print("================The End===========================")
    