# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:35:38 2017

@author: Simon
"""

import numpy as np
from matplotlib import pyplot as plt

class Scan:
    def __init__(self,name_of_scan,type_of_scan,number_of_scan):
        file_address = name_of_scan + '_' + type_of_scan + '_' + number_of_scan + '.dat'
        data_file = open(file_address,'rb')
        self.initial_time = data_file.readline().split('\t')[1]       
        self.final_time = data_file.readline().split('\t')[1]
        data = np.loadtxt(file_address)
        
        if type_of_scan is 'QE':
            self.time = 
            self.QE = 
            self.pressure =
            self.laser_power = 

        elif type_of_scan is 'XANES':
            self.energy = data[:,0]
            self.photocurrent = data[:,1].normalize()
            
    def fit(self):
        return 0
        
    def normalize(self):
        self.normalized_photocurrent = 
        
class Set_of_scans:
    def __init__(self,name_of_scan,type_of_scan):
        
    def superimpose(self):
        
        
        
def plot_XANES(set_of_scans, format_of_figure):
    figure = plt.plot()
    if format_of_figure is 'big':
        
    elif format_of_figure is 'small':
        
    print 'Name, type, parameters of the scan'
    return figure
    
#def plot_QE():
#    
#def plot_correlation_QE_XANES_P():
#    
#def plot_correlation_CS_I():
#        
#def plot_all_sets_of_scans():
#
#def plot_fit():
