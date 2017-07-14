# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:35:38 2017
@author: Simon
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as col
from matplotlib import colorbar as clb
from matplotlib import patches as patches


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
        self.photocurrent /= self.topup
        N = 10
        self.photocurrent -= np.polyval(np.polyfit(self.energy[:N],self.photocurrent[:N],1),self.energy)
        
    def arg(self,value):
        return np.argmin(abs(self.energy-value))
        
        
class Set:
    def __init__(self,name_of_scan,type_of_scan):
        self.name = name_of_scan
        self.length = nb[name_of_scan]
        self.scans = [Scan(name_of_scan,type_of_scan,number_of_scan) for number_of_scan in range(self.length)]
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
        
def cmap(nb_plots):
    colormap = plt.cm.jet
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, nb_plots)])

def plot_XANES(set_of_scans, format_of_figure, colours='jet'):
    fig = plt.figure(1,figsize=(6,6))
    if format_of_figure is 'big':
        if colours is 'b&w':
            colormap = plt.cm.Greys
            plt.gca().set_color_cycle([colormap(0.9-i) for i in np.linspace(0, 0.7, set_of_scans.length)])
        if colours is 'jet':
            colormap = plt.cm.jet
            plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, set_of_scans.length)])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
#        mymap = col.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
#        cmap(len(set_of_scans.scans))
#        Z = [[0,0],[0,0]]
#        cbmin,cbmax,cbsteps = 1,set_of_scans.length,
#        levels = range(set_of_scans.length)
#        CS3 = plt.contourf(Z, levels, cmap=mymap)
#        plt.clf()
        for scan in set_of_scans.scans:
            inf = np.argmin(abs(scan.energy - 1550))
            plt.plot(scan.energy[inf:],scan.photocurrent[inf:])
#            plt.tick_params(axis='y',)
            plt.gca().get_yaxis().set_ticks([])
            plt.xlim(1551,1600)
#            plt.ylim(-2E-4,25E-4)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Photocurrent (arb. units)')
        plt.gca().add_patch(patches.Rectangle((0.6,0.1),0.3,0.03,fill=False))
        plt.show()
        ax_cb = fig.add_axes([0.63,0.2,0.2,0.02])
        cb= clb.Colorbar(ax_cb,sm,orientation='horizontal')
#        cb = plt.colorbar(sm,cax=ax_cb,orientation='horizontal',pad=0.2)
        cb.set_ticks(np.arange(set_of_scans.length)/(set_of_scans.length-1))
        cb.set_ticklabels(np.arange(set_of_scans.length)+1)
#        cb.set_ticks()
        ax_cb.text(0,1.6,'Scan number')
#        cb.set_label('Scan number')
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
    return fig
    
def plot_QE(set_of_scans,format_of_figure,colours='bw'):
    fig = plt.figure(2,figsize=(6,6))
    if format_of_figure is 'big':
        if colours is 'bw':
            colormap = plt.cm.Greys
            plt.gca().set_color_cycle([colormap(0.9-i) for i in np.linspace(0, 0.7, set_of_scans.length)])
        if colours is 'jet':
            colormap = plt.cm.jet
            plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, set_of_scans.length)])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        for scan in set_of_scans.scans:
            inf = 3
#            inf = np.argmin(abs(scan.energy - 1550))
            plt.plot(scan.time[inf:],scan.QE[inf:],'.')
#            plt.tick_params(axis='y',)
            plt.gca().get_yaxis().set_ticks([])
#            plt.xlim(1551,1600)
#            plt.ylim(-2E-4,25E-4)
        plt.xlabel('Time (s)')
        plt.ylabel('QE (arb. units)')
        plt.gca().add_patch(patches.Rectangle((0.6,0.1),0.3,0.03,fill=False))
        plt.show()
        ax_cb = fig.add_axes([0.63,0.78,0.2,0.02])
        cb= clb.Colorbar(ax_cb,sm,orientation='horizontal')
#        cb = plt.colorbar(sm,cax=ax_cb,orientation='horizontal',pad=0.2)
        cb.set_ticks(np.arange(set_of_scans.length)/(set_of_scans.length-1))
        cb.set_ticklabels(np.arange(set_of_scans.length)+1)
#        cb.set_ticks()
        ax_cb.text(0,1.6,'Scan number')
        
    elif format_of_figure is 'small':
        print 'Kirikou'
        for scan in set_of_scans.scans:
            plt.plot(scan.time,scan.QE)
            plt.set_cmap('jet')
            
nb = {'Al':6,'Alrp':9}
ref_scan = {'Al':4,'Alrp':4}
#nonscaled_zone = {'Al':(0,)}


##def plot_correlation_QE_XANES_P():
##    
##def plot_correlation_CS_I():
##        
##def plot_all_sets_of_scans():
##
##def plot_fit():

plt.close('all')
#S = Set('Al','XANES')
#plot_XANES(S,'big')

S = Set('Alrp','QE')
plot_QE(S,'big')