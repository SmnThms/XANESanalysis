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
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d

def power():
    return np.polyfit([1.8,1.9,2.0,2.1,2.2,2.3],[3,8,18,35,61,98],2)
    

class Scan:
    def __init__(self,name_of_scan,number_of_scan):
        for type_of_scan in ['QE','XANES']:
            file_address = '.\\Data\\' + name_of_scan + '_' + type_of_scan \
                           + '_' + str(number_of_scan) + '.txt'
            data_file = open(file_address,'rb')
            self.initial_time = data_file.readline().split('\t')[1]       
            self.final_time = data_file.readline().split('\t')[1]
            data = np.loadtxt(file_address)
            if type_of_scan is 'QE':
                self.time = data[:,0] - data[0,0]
                self.pressure = data[:,2]
                self.laser_power = np.polyval(power(),data[:,3]) # in mW ?
                self.temperature = data[:,4]
                self.QE = data[:,1]/self.laser_power
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
        self.photocurrent /= self.photocurrent[self.arg(1566)]
    def arg(self,value):
        return np.argmin(abs(self.energy-value))
        
class Set:
    def __init__(self,name_of_scan):
        self.name = name_of_scan
        self.scans = [Scan(name_of_scan,number_of_scan) \
                      for number_of_scan in scan_range[name_of_scan]] #range(self.length)]
        self.length = len(self.scans)    
        for scan in self.scans:
            reference = ref_scan[self.name]
            coefs = np.arange(10000)*2/10000
            limit1, limit2, limit3 = 1567, 1574, 1599
            diff = [np.sum(abs(coef*scan.photocurrent[:scan.arg(limit1)] \
                    - self.scans[reference].photocurrent[:scan.arg(limit1)])) \
                    + np.sum(abs(coef*scan.photocurrent[scan.arg(limit2): \
                    scan.arg(limit3)] - self.scans[reference].photocurrent[ \
                    scan.arg(limit2):scan.arg(limit3)])) \
                    for coef in coefs]
            scan.photocurrent *= coefs[np.argmin(diff)]
            
def plot(set_of_scans, type_of_scan, cmap='jet_r'):
    fig = plt.figure(figsize=(6,6))
    colormap = getattr(plt.cm,cmap)
    cmap_lim = {'QE':0.1,'XANES':0,'P':0,'T':0}
    plt.gca().set_color_cycle([colormap(i) \
      for i in np.linspace(cmap_lim[type_of_scan],1,set_of_scans.length)])
    for scan in set_of_scans.scans:
        if type_of_scan is 'XANES':
            inf = scan.arg(1550)
            plt.xlim(1551,1600)
            plt.xlabel('Energy (eV)',fontsize=14)
            plt.ylabel('Photocurrent (arb. units)',fontsize=14)
            plt.xticks(fontsize=11)
            plt.plot(scan.energy[inf:],scan.photocurrent[inf:])
        elif type_of_scan is 'QE':
            inf = 1    
            plt.xlabel('Time (s)',fontsize=14)
            plt.ylabel('QE (arb. units)',fontsize=14)
            plt.xticks(fontsize=11)
            plt.plot(scan.time[inf:],scan.QE[inf:],'.')
        elif type_of_scan is 'P':
            plt.plot(scan.time,scan.pressure)
        elif type_of_scan is 'T':
            plt.plot(scan.time,scan.temperature)
#    plt.gca().get_yaxis().set_ticks([])
    sm = plt.cm.ScalarMappable(cmap=colormap,norm=plt.Normalize(vmin=0,vmax=1))
    sm._A = []
    plt.gca().add_patch(Rectangle((0.63,0.8),0.325,0.16,
                        fill=False,transform=plt.gca().transAxes))
    ax_cb = fig.add_axes([0.64,0.78,0.2,0.02])
    cb = clb.Colorbar(ax_cb,sm,orientation='horizontal')
    cb.set_ticks(np.arange(set_of_scans.length)/(set_of_scans.length-1))
    cb.set_ticklabels(np.arange(set_of_scans.length)+1)
    ax_cb.text(0,1.6,'Scan number')
    plt.savefig('.\\Figures\\'+set_of_scans.name+'_'+type_of_scan+'.jpg')
    print 'Figure',fig.number,':',set_of_scans.name,type_of_scan
    return

def plotall(type_of_scan,cmap='jet_r'):
    fig = plt.figure(figsize=(5,8))
    colormap = getattr(plt.cm,cmap)
    sets_of_scans = [Set('Al'),Set('Al2'),Set('Alrp'),Set('Alrp2')]#,Set('AlLi'),Set('AlLirp')]
    for n,set_of_scans in enumerate(sets_of_scans):
        plt.gca().set_color_cycle([colormap(i) \
            for i in np.linspace(0,1,set_of_scans.length)])
        for scan in set_of_scans.scans:
            if type_of_scan is 'XANES':
                inf = scan.arg(1550)
                plt.plot(scan.energy[inf:],scan.photocurrent[inf:]+n*0)
            elif type_of_scan is 'QE':
                inf = 1
                plt.plot(scan.time[inf:],scan.QE[inf:]+n*0.1,'.')
    if type_of_scan is 'XANES':
        plt.xlim(1551,1600)
        plt.xlabel('Energy (eV)',fontsize=14)
        plt.ylabel('Photocurrent (arb. units)',fontsize=14)
    elif type_of_scan is 'QE':
        plt.xlabel('Time (s)',fontsize=14)
        plt.ylabel('QE (arb. units)',fontsize=14)
    plt.xticks(fontsize=11)
    plt.gca().get_yaxis().set_ticks([])
    print 'Figure',fig.number,': all',type_of_scan
    return

def correlation_pressure(nom):
    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
#    fig, (ax1,ax2) = plt.subplots(2,sharex=True, sharey=True)
#    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11,6))
    print 'Figure',fig.number,': Pressure correlation'
    pressure,QE,XANESheight = [],[],[]
    point = {'Alrp':'+','Al':'*','AlLi':'x','AlLirp':'^','Al2':'.','Alrp2':'.'}
    for set_of_scans in [Set(nom)]:#,Set('Alrp'),Set('Al'),Set('AlLirp')]:
        for scan in set_of_scans.scans:  
            pressure.append(scan.pressure[-1]*1e6)
            QE.append(scan.QE[-1])
            XANESheight.append(scan.photocurrent[scan.arg(1570)])
#            if set_of_scans.name is 'AlLi':
#                XANESheight[-1] *= 0.9
#            pressure_init.append(scan.pressure[1])
#            pressure_final.append(scan.pressure[-1])
#            QE_init.append(scan.QE[1])
#            QE_final.append(scan.QE[-1])
#            inf1, sup1  = scan.arg(1564),scan.arg(1568)
#            first_peak  = interp1d(np.linspace(0,1,sup1-inf1),
#                          scan.photocurrent[inf1:sup1], kind='cubic')
#            inf2, sup2  = scan.arg(1567),scan.arg(1571)
#            dip         = interp1d(np.linspace(0,1,sup2-inf2),
#                          scan.photocurrent[inf2:sup2], kind='cubic')
#            inf3, sup3  = scan.arg(1568),scan.arg(1574)
#            second_peak = interp1d(np.linspace(0,1,sup3-inf3),
#                          scan.photocurrent[inf3:sup3], kind='cubic')
#            peakheight.append(np.max(second_peak(np.linspace(0,1))))
##            norm_peakheight.append(np.max(scan.photocurrent[inf3:sup3]))
#            norm_peakheight.append(np.max(second_peak(np.linspace(0,1))) \
#                                   /np.max(first_peak(np.linspace(0,1))))
#            dipheight.append(np.min(dip(np.linspace(0,1))))
##            norm_dipheight.append(np.min(scan.photocurrent[inf2:sup2]))
#            norm_dipheight.append(np.min(dip(np.linspace(0,1)) \
#                                   /np.max(first_peak(np.linspace(0,1)))))
#            interm.append(scan.photocurrent[scan.arg(1570)])
#    plt.figure(1)
#    plt.plot(pressure_final,QE_final,'o')
#    print np.corrcoef(pressure_final,QE_final)[1,0]
##    print np.corrcoef(np.concatenate((pressure_final,QE_final)))
#    plt.figure(2)
#    plt.plot(pressure_final,peakheight,'o')
#    print np.corrcoef(pressure_final,peakheight)[1,0]    
#    plt.figure(3)
#    plt.plot(pressure_final,norm_peakheight,'o')
#    print np.corrcoef(pressure_final,norm_peakheight)[1,0]
#    plt.figure(4)
#    plt.plot(pressure_final,dipheight,'o')
#    print np.corrcoef(pressure_final,dipheight)[1,0]
#    plt.figure(5)
#    plt.plot(pressure_final,norm_dipheight,'o')
#    print np.corrcoef(pressure_final,norm_dipheight)[1,0]
#    plt.figure(6)
#    plt.plot(pressure_final,interm,'o')
#    print np.corrcoef(pressure_final,interm)[1,0]
    ax1.plot(pressure,QE,'x',color='k')
    print set_of_scans.name,'QE : r^2 =',np.corrcoef(pressure,QE)[1,0]
#    ax2.plot(pressure,XANESheight,point[set_of_scans.name],color='k')
    ax2.plot(pressure,XANESheight,'x',color='k')
#    ax2.set_ylim([0.9,1])
    print set_of_scans.name,'XANES : r^2 =',np.corrcoef(pressure,XANESheight)[1,0]
    fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
#    plt.show()
#    ax1.set_xlabel('Pressure (x$10^{-6}$ mbar)',fontsize=14)
#    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.plot(pressure,np.polyval(np.polyfit(pressure,QE,1),pressure),'midnightblue',alpha=0.4)
    ax2.plot(pressure,np.polyval(np.polyfit(pressure,XANESheight,1),pressure),'midnightblue',alpha=0.4)
#    ax1.set_xlabel('Pressure (x$10^{-6}$ mbar)',fontsize=14)
    ax1.set_ylabel('QE level',fontsize=14)
    ax1.set_title(nom)
    ax1.get_yaxis().set_ticks([])
    ax2.set_xlabel('Pressure',fontsize=14)
    ax2.set_ylabel('XANES variation',fontsize=14)
    ax2.get_yaxis().set_ticks([])
    ax1.yaxis.set_ticks_position("right")
    ax2.yaxis.set_ticks_position("right")
    ax1.text(0.7,0.1,'R='+str('%.3f' % np.corrcoef(pressure,QE)[1,0]),transform=ax1.transAxes)
    ax2.text(0.7,0.1,'R='+str('%.3f' % np.corrcoef(pressure,XANESheight)[1,0]),transform=ax2.transAxes)
    plt.savefig('.\\Figures\\'+nom+'_PressureCorrelation'+'.jpg')
    plt.show()
    return

#def correlation_laser():
            
#nb = {'Al':6,'Alrp':9}
scan_range = {'Alrp':np.arange(0,9),'Al':[0,1,2,3,4,5,6,7,8,9],'AlLirp':[1,2,3,4,5,6],
              'AlLi':[1,2,3,4,5,6,7,8,9],'Al2':[0,1,2,3,4],'Alrp2':[0,1,2,3,4]}
ref_scan = {'Al':2,'Alrp':4,'AlLirp':4,'AlLi':2,'Al2':1,'Alrp2':1}

plt.close('all')
#plotall('XANES')
#plotall('QE')
#plot(Set('Alrp'),'XANES',fig=8)
#plot(Set('Alrp'),'QE',fig=2,cmap='bone_r')
correlation_pressure('Alrp2')
correlation_pressure('Alrp')
correlation_pressure('AlLi')
correlation_pressure('AlLirp')
#plot(Set('AlLi'),'P')
#plot(Set('AlLi'),'T')