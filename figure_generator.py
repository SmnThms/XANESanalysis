# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:35:38 2017
@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colorbar as clb
from matplotlib.patches import Rectangle

class Scan:
    def __init__(self,name_of_scan,type_of_scan,number_of_scan):
        file_address = '.\\Data\\' + name_of_scan + '_' + type_of_scan + '_' \
                       + str(number_of_scan) + '.txt'
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
        self.photocurrent /= self.topup
        N = 10
        self.photocurrent -= np.polyval(np.polyfit(self.energy[:N],
                             self.photocurrent[:N],1),self.energy)
        
    def arg(self,value):
        return np.argmin(abs(self.energy-value))
        
class Set:
    def __init__(self,name_of_scan,type_of_scan):
        self.type = type_of_scan
        self.name = name_of_scan
        self.length = nb[name_of_scan]
        self.scans = [Scan(name_of_scan,type_of_scan,number_of_scan) \
                      for number_of_scan in range(self.length)]
        if type_of_scan is 'XANES':
            self.scaling()
        
    def scaling(self):
        for scan in self.scans:
            reference = ref_scan[self.name]
            coefs = np.arange(10000)*2/10000
            limit1, limit2, limit3 = 1567, 1574, 1599
            diff = [np.sum(abs(coef*scan.photocurrent[:scan.arg(limit1)] - self.scans[reference].photocurrent[:scan.arg(limit1)])) \
                    + np.sum(abs(coef*scan.photocurrent[scan.arg(limit2):scan.arg(limit3)] - self.scans[reference].photocurrent[scan.arg(limit2):scan.arg(limit3)])) \
                    for coef in coefs]
            scan.photocurrent *= coefs[np.argmin(diff)]

def plot(set_of_scans, fig=1, cmap='jet_r'):
    figure = plt.figure(fig,figsize=(6,6))
    colormap = getattr(plt.cm,cmap)
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, set_of_scans.length)])
    for scan in set_of_scans.scans:
        if set_of_scans.type is 'XANES':
            inf = np.argmin(abs(scan.energy - 1550))
            plt.xlim(1551,1600)
            plt.xlabel('Energy (eV)')
            plt.ylabel('Photocurrent (arb. units)')
            plt.plot(scan.energy[inf:],scan.photocurrent[inf:])
        if set_of_scans.type is 'QE':
            inf = 1    
            plt.xlabel('Time (s)')
            plt.ylabel('QE (arb. units)')
            plt.plot(scan.time[inf:],scan.QE[inf:],'.')
    plt.gca().get_yaxis().set_ticks([])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.gca().add_patch(Rectangle((0.63,0.8),0.325,0.16,fill=False,transform=plt.gca().transAxes))
    ax_cb = figure.add_axes([0.64,0.78,0.2,0.02])
    cb = clb.Colorbar(ax_cb,sm,orientation='horizontal')
    cb.set_ticks(np.arange(set_of_scans.length)/(set_of_scans.length-1))
    cb.set_ticklabels(np.arange(set_of_scans.length)+1)
    ax_cb.text(0,1.6,'Scan number')
#        plt.savefig('.\\Figures\\' + set_of_scans.name + '_XANES.jpg')
    print 'Figure',fig,':',set_of_scans.name,set_of_scans.type
    return
            
nb = {'Al':6,'Alrp':9}
ref_scan = {'Al':4,'Alrp':4}

plt.close('all')
plot(Set('Al','XANES'))
plot(Set('Al','QE'),fig=2,cmap='viridis_r')