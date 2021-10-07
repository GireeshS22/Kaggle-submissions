# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 21:26:30 2021

@author: Gireesh Sundaram
"""


import requests
from bs4 import BeautifulSoup

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", 
           "Accept-Encoding":"gzip, deflate", 
           "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", 
           "DNT":"1","Connection":"close", 
           "Upgrade-Insecure-Requests":"1"}

URL = "https://www.amazon.com/Pokemon-Power-Eternatus-Tin-Multicolor/dp/B08H4G17FB/ref=sr_1_2?dchild=1&keywords=mint+pokemon+v+power+tin&qid=1633623311&sr=8-2"
r = requests.get(URL, headers=headers)

soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib

#%%
alls = []
for d in soup.findAll('div', attrs = {'class':'a-box mbc-offer-row pa_mbc_on_amazon_offer'}):
    seller = d.find('span', attrs={'class':'a-size-small mbcMerchantName'})
    if seller is not None:
        print(seller.text.strip())

#%%

bb = soup.find("div", attrs = {'class': 'a-box-inner'})
seller = bb.find('div', attrs={"id": 'merchant-info'})
print(seller.text.strip())

#%%

urrllit = ["https://www.amazon.com/Ancaixin-Children-Birthday-Christmas-Thanksgiving/dp/B0716KG4SC/ref=sr_1_5?crid=38AAM7SI1PXJU&dchild=1&keywords=baby+balance+bikes+10-24+month+children+walker&qid=1633624039&sprefix=baby+balance+bike+10-24+month+%2Caps%2C377&sr=8-5",
           "https://www.amazon.com/Pokemon-Power-Eternatus-Tin-Multicolor/dp/B08H4G17FB/ref=sr_1_2?dchild=1&keywords=mint+pokemon+v+power+tin&qid=1633623311&sr=8-2"]
