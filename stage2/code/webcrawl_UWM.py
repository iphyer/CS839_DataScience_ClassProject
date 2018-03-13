#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Mar 11 10:45:12 2018

@author: Mingren Shen

This python program get data of all "Data Science" books form libraray of UW-Madison

UW Library using static URL to link all resource which makes the programe easy

"""

#import needed package
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests

def collectBookURL(webpage,result_dict):
    for a in webpage.findAll("a", {"class": "item_path"}):
        # key is URL and content is title
        # in case Book with same title
        result_dict[a['href']] = a.string


def parseBookDetailPage(pageURL,pandasDF,rowIndex):
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

# helper function to deal with nested list of HTML page
def dictify(ul):
    result = {}
    for li in ul.find_all("li", recursive=False):
        key = next(li.stripped_strings)
        ul = li.find("ul")
        #print(li)
        if ul:
            tmpstr = ""
            for item in li.findAll('a'):
                tmpstr += item.get_text() + ","
            #result[key] = dictify(ul)
            #print(li)

            result[key] = tmpstr
        else:
            tmp_list = list()
            for span in li.findAll('span'):
                tmp_list.append(span.get_text())
            result[key] = tmp_list[1].strip()#li.get_text().split()[1]
    return result


if __name__ == "__main__":
    """
    method description :
    1.get the search result from UW-Madison's library
    2.get basic data
    3.collect all URL for the books

    """
    print("================The Start=========================")
    #Assign URL for search
    URL = 'https://search.library.wisc.edu/search/system?q=Data+Science'
    # request the web page
    r = requests.get(URL, "lxml")
    # format webpage with beautifulSoup
    soup =  bs(r.text, "lxml")
    #print(soup.prettify())

    # dictory to store URL for the details pages of Book
    dict_BookDetailsURL = dict()

    """
    Get Basic Data of the web page
    """
    number_books_results = soup.find("span", {"class": "num_results"}).string.strip().split()[0]
    print("Totoal results is " + number_books_results + " Books")
    # turn unicode to int
    tmp = str(number_books_results).replace(',','')
    number_books_results = int(tmp)
    # find last page of results
    tmp = str(soup.find("span", {"class": "last"}).a)
    number_result_pages = tmp[tmp.index('page=') + 5 : tmp.index('&amp;')]
    print("Totoal Pages of the searching result is " + number_result_pages)
    # collect Book Details URL for Page 1
    collectBookURL(soup,dict_BookDetailsURL)


    # Rotate to get all URL of Books
    preURL = 'https://search.library.wisc.edu/search/system?page='
    postURL = '&q=Data+Science'
    # 10 Books per page, so 1000 pages give 10000 Books.
    print("We only need first 10000 Books in the searching results.")
    MAX_PAGES = 1001
    for i in range(2,MAX_PAGES):
        print("Processing Page" + str(i))
        tmp_URL = preURL + str(i) + postURL
        tmp_r = requests.get(tmp_URL)
        tmp_soup =  bs(tmp_r.text,"lxml")
        collectBookURL(tmp_soup,dict_BookDetailsURL)

    # test whether the loop precoess all pages correctly
    # assert( len (dict_BookDetailsURL) == number_books_results)
    # saved dict as pickle
    # with open('BookURL.pickle', 'wb') as handle:
    #    pickle.dump(dict_BookDetailsURL, handle)


    # set up DataFrame to store the data
    df = pd.DataFrame(columns=['ID','TITLE','CREATOR','FORMAT','PUBLICATION DATES','CONTRIBUTORS','PUBLICATION','PHYSICAL DETAILS','ISBNS','OCLC'])

    # Processing the detailed page of each book
    keyID = 0
    for key,value in dict_BookDetailsURL.items():
        itemURL = 'https://search.library.wisc.edu' + key
        parseBookDetailPage(itemURL,df,keyID)
        df.at[keyID,'TITLE'] = value
        keyID = keyID + 1
        #print(itemURL)
    df.to_csv("BOOKS_UWM.csv",sep = '\t',index = False)
    print("================The End===========================")

    #exit(0)
