# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:35:38 2017

@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as col


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
        nb = {'Al':6,'Alrp':9}
        self.name = name_of_scan
        self.length = nb[name_of_scan]
        self.scans = [Scan(name_of_scan,type_of_scan,number_of_scan) for number_of_scan in range(self.length)]
        if type_of_scan is 'XANES':
            self.scaling()
        
    def scaling(self):
        for scan in self.scans:
            reference, coefs = 0, np.arange(10000)*2/10000
            diff = [np.sum(abs(coef*scan.photocurrent - self.scans[reference].photocurrent)) for coef in coefs]
            scan.photocurrent *= coefs[np.argmin(diff)]
        
        
def plot_XANES(set_of_scans, format_of_figure):
    figure = plt.figure(1)
    if format_of_figure is 'big':
        for scan in set_of_scans.scans:
            inf = np.argmin(abs(scan.energy - 1550))
            plt.plot(scan.energy[inf:],scan.photocurrent[inf:])
#            plt.tick_params(axis='y',)
            plt.gca().get_yaxis().set_ticks([])
            plt.xlim(1551,1600)
            plt.ylim(-2E-4,25E-4)
#            plt.savefig('.\\Figures\\' + set_of_scans.name + '_XANES.jpg')
    elif format_of_figure is 'small':
#        selection = {'Al':(0,1,2,3,4,5)}
        colors = plt.cm.jet(np.linspace(0,1,set_of_scans.length))
        plt.gca().set_color_cycle(colors)
        for scan in set_of_scans.scans:#[selection[set_of_scans.name]]:
            inf,sup = np.argmin(abs(scan.energy - 1566)),np.argmin(abs(scan.energy - 1575))
            plt.plot(scan.energy[inf:sup],scan.photocurrent[inf:sup])
#            colors = plt.cm.jet(np.linspace(1,256,set_of_scans.length))
#            plt.gca().set_color_cycle(colors)
#            plt.gca().set_prop_cycle(cycler('color', colors))
            plt.xlim(1566,1575)
#            plt.gca().get_yaxis().set_ticks([])

    print 'Name, type, parameters of the scan'
    return figure
    
def plot_QE(set_of_scans,format_of_figure):
    figure = plt.figure(2)
    if format_of_figure is 'big':
        print 'BGG'
    elif format_of_figure is 'small':
        print 'Kirikou'
        for scan in set_of_scans.scans:
            plt.plot(scan.time,scan.QE)
            plt.set_cmap('jet')

##def plot_correlation_QE_XANES_P():
##    
##def plot_correlation_CS_I():
##        
##def plot_all_sets_of_scans():
##
##def plot_fit():

S = Set_of('Al','XANES')
plot_XANES(S,'big')
