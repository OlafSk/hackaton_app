# -*- coding: utf-8 -*-
"""
Created on Sun May 27 03:57:33 2018

@author: Szymon S
"""
import pandas as pd


grid_centers = pd.read_csv('grid_centers.csv', sep=';')

def get_grid_center(grid):
   return grid_centers[grid_centers['eurogrid_0250']==grid].values[0]

pred = pd.read_csv('Sklep.csv')
pred['grid_x'] = pred['eurogrid_0250_1'].apply(lambda x: get_grid_center(x)[2])
pred['grid_y'] = pred['eurogrid_0250_1'].apply(lambda x: get_grid_center(x)[1])
pred['digits'] = np.digitize(pred['pred'], [np.percentile(pred.pred.values, q) for q in np.linspace(20,80,3)])

pred.to_csv('Sklep_xy.csv')
