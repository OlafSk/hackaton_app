# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:39:45 2018

@author: Szymon S
"""
import requests
import pandas as pd


class DataWise(object):
   
   def __init__(self):
      self.api_key = "maraton0n895gbsgc72bbksa042mad03"
      
   
   def query_poi(self, x, y, radius):
      url = """http://locit.eu/webservice/spatial-query-poi/v1.0.0/{x}/{y}/{radius}/*%3A*?format=&charset=&key={key}"""
      payload = dict(x=x,
                     y=y,
                     radius=radius,
                     key=self.api_key
                     )
      r = requests.get(url.format(**payload))
      response = r.json()
      return pd.DataFrame(response['data']['listing'])
   
   def get_data(self, uuid):
      url = """http://api.locit.pl/webservice/uuid-info/v1.0.0/{uuid}?key={key}"""
      payload = dict(uuid=uuid, key=self.api_key)
      r = requests.get(url.format(**payload))
      response = r.json()['data']
      data = dict(response['stats_msw'], **response['stats_address'])
      data = dict(data, **response['stats_income'])
      return data
      
      
      
      
      
#      stats_msw = pd.DataFrame.from_dict(response['stats_msw'],orient='index').T
#      stats_address = pd.DataFrame.from_dict(response['stats_address'],orient='index').T
#      stats_income = pd.DataFrame.from_dict(response['stats_income'],orient='index').T
#      data = pd.concat([stats_msw, stats_address, stats_income], axis=1)
#      data['uuid'] = uuid
#      return data
#   
   
   
## poczatek scrapowania
