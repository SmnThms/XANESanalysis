# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:35:38 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

class Scan:
    def __init__(self,name_of_scan,type_of_scan,number_of_scan):
        file_address = '.\\Data\\' + name_of_scan + '_' + type_of_scan + '_' + str(number_of_scan) + '.txt'
        data_file = open(file_address,'rb')
        self.initial_time = data_file.readline().split('\t')[1]       
        self.final_time = data_file.readline().split('\t')[1]
        data = np.loadtxt(file_address)
        
        if type_of_scan is 'QE':
            self.time = data[:,0] - data[0,0]
            self.QE = data[:,1]
            self.pressure = data[:,2]
            self.laser_power = data[:,3]

        elif type_of_scan is 'XANES':
            self.energy = data[:,0]
            self.topup = data[:,2]
            self.I0 = data[:,3]
            self.photocurrent = data[:,1]
            self.normalize()
            
    def fit(self):
        return 0
        
    def normalize(self):
        self.photocurrent *= 1/self.topup
        N = 10
        self.photocurrent -= np.polyval(np.polyfit(self.energy[:N],self.photocurrent[:N],1),self.energy)
        
        
class Set_of:
    def __init__(self,name_of_scan,type_of_scan):
        nb = {'Al':5,'Al2':5}
        self.scans = [Scan(name_of_scan,type_of_scan,number_of_scan) for number_of_scan in range(nb[name_of_scan])]
        
    def scaling(self):
        for scan in self.scans:
            reference = 0
            coefs = np.arange(10000)*2/10000
            diff = [np.sum(abs(coef*scan.photocurrent - self.scans[reference].photocurrent)) for coef in coefs]
            scan.photocurrent *= coefs[np.argmin(diff)]
        
#def plot_XANES(set_of_scans, format_of_figure):
#    figure = plt.plot()
#    if format_of_figure is 'big':
#        
#    elif format_of_figure is 'small':
#        
#    print 'Name, type, parameters of the scan'
#    return figure
    
##def plot_QE():
##    
##def plot_correlation_QE_XANES_P():
##    
##def plot_correlation_CS_I():
##        
##def plot_all_sets_of_scans():
##
##def plot_fit():

s = Scan('AlLirp','XANES',0)
#plt.plot(s.time,s.QE)
plt.plot(s.energy,s.photocurrent)