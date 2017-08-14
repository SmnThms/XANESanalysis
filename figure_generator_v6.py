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
from scipy.optimize import curve_fit

########################### STRUCTURE OF THE PROGRAM ##########################
# 1  Classes 'Scan' and 'Set' to create easy-to-handle python objects.
# 2. Function 'plot' to plot or the QE, the XANES, the pressure of the tempera-
#     ture for one set of scans. The function 'plotall' does the same with dif-
#     ferent sets of scans alltogether.
# 3. Functions 'correlation_pressure' and 'correlation_laser' for the Al-repol-
#     ished and AlLi sets.
# 4. Functions and dictionnaries providing the range of relevant scans for each
#     set, converting the laser current into laser power, etc.
# 5. Call of the desired plotting functions.

save_plots = True # if False, the figures are plotted, not saved.

###################################### 1 ######################################
class Scan:
    def __init__(self,name_of_scan,number_of_scan):
        for type_of_scan in ['QE','XANES']:
            file_address = '.\\Data\\' + name_of_scan + '_' + type_of_scan \
                           + '_' + str(number_of_scan) + '.txt'
            data = np.loadtxt(file_address)
            if type_of_scan is 'QE':
                self.time = data[:,0] - data[0,0]
                self.pressure = data[:,2]
                self.laser_power = np.polyval(power(),data[:,3])
                self.temperature = data[:,4]
                self.QE = data[:,1]*10**data[:,-1]/self.laser_power
            elif type_of_scan is 'XANES':
                self.energy = data[:,0]
                self.topup = data[:,2]
                self.I0 = data[:,3]
                self.photocurrent = data[:,1]
                self.normalize()  
            self.name = name_of_scan
    def normalize(self):
        self.photocurrent /= self.topup
        N = 10
        self.photocurrent -= np.polyval(np.polyfit(self.energy[:N],
                             self.photocurrent[:N],1),self.energy)  
        self.photocurrent /= self.photocurrent[self.arg(1566)]
    def arg(self,value):
        return np.argmin(abs(self.energy-value))
    def fit(self,inf=3):
        p0 = [self.QE[-1], self.QE[inf]-self.QE[-1], 0.01, 
              self.QE[100]-self.QE[-1], 0.1]
        if self.name=='AlLirp':
            p0 = [70, -170, 0.4, 20, 0.05]
            inf = 1
        param,cov = curve_fit(f,self.time[inf:],self.QE[inf:],p0=p0) 
        return param,cov
        
class Set:
    def __init__(self,name_of_scan):
        self.name = name_of_scan
        self.scans = [Scan(name_of_scan,number_of_scan) \
                      for number_of_scan in scan_range[name_of_scan]]
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
            
            
###################################### 2 ######################################
def plot(set_of_scans, type_of_scan,with_fit=False):
    fig = plt.figure(figsize=(6,6))
    if type_of_scan=='QE':
        colormap = getattr(plt.cm,'bone_r')
    else:
        colormap = getattr(plt.cm,'jet_r')
    cmap_lim = {'QE':0.1,'XANES':0,'P':0,'T':0}
    name_fit= ''
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
            if with_fit:
                plt.plot(scan.time[inf:],f(scan.time[inf:],*scan.fit()[0]),
                '--',color='r',alpha=0.6)
                name_fit = '_fitted'
        elif type_of_scan is 'P':
            plt.plot(scan.time,scan.pressure)
        elif type_of_scan is 'T':
            plt.plot(scan.time,scan.temperature)
    plt.gca().get_yaxis().set_ticks([])
    if set_of_scans.name=='Alrp':
        plt.title('Al',fontsize=14)
    elif set_of_scans.name=='AlLi':
        plt.title('Al $-$ Li',fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=colormap,norm=plt.Normalize(vmin=0,vmax=1))
    sm._A = []
    plt.gca().add_patch(Rectangle((0.63,0.8),0.325,0.16,
                        fill=False,transform=plt.gca().transAxes))
    ax_cb = fig.add_axes([0.64,0.78,0.2,0.02])
    cb = clb.Colorbar(ax_cb,sm,orientation='horizontal')
    cb.set_ticks(np.arange(set_of_scans.length)/(set_of_scans.length-1))
    cb.set_ticklabels(np.arange(set_of_scans.length)+1)
    ax_cb.text(0,1.6,'Scan number')
    if save_plots:
        plt.savefig('.\\Figures\\'+set_of_scans.name \
                    +'_'+type_of_scan+name_fit+'.jpg')
    print 'Figure',fig.number,':',set_of_scans.name,type_of_scan
    return

def plotall(type_of_scan,cmap='jet_r'):
    fig = plt.figure(figsize=(5,8))
    colormap = getattr(plt.cm,cmap)
    sets_of_scans = [Set('Al'),Set('Al2'),Set('Alrp'),Set('Alrp2'),
                     Set('AlLi'),Set('AlLirp')]
    for n,set_of_scans in enumerate(sets_of_scans):
        plt.gca().set_color_cycle([colormap(i) \
            for i in np.linspace(0,1,set_of_scans.length)])
        for scan in set_of_scans.scans:
            if type_of_scan is 'XANES':
                inf = scan.arg(1550)
                plt.plot(scan.energy[inf:],scan.photocurrent[inf:]+n*0.3)
            elif type_of_scan is 'QE':
                inf = 1
                plt.plot(scan.time[inf:],scan.QE[inf:]+n*1e5,'.')
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


###################################### 3 ######################################
def correlation_pressure(set1,set2):
    fig = plt.figure(figsize=(5,5))
    ax1, ax2 = fig.add_subplot(221), fig.add_subplot(223)
    ax3, ax4 = fig.add_subplot(222), fig.add_subplot(224)
    fig.subplots_adjust(hspace=0)
    print 'Figure',fig.number,': Pressure correlation'
    subplot_correlation_pressure(set1,ax1,ax2)
    subplot_correlation_pressure(set2,ax3,ax4)
    ax4.set_xticks([0.98,1.01,1.04])
    ax1.set_title('Al') # if set1 is 'Alrp'
    ax3.set_title('Al $-$ Li') # if set2 is 'AlLi'
    ax1.set_ylabel('QE level',fontsize=13)
    ax2.set_ylabel('XANES variation',fontsize=13)
    ax2.text(1.1,-0.24,'Pressure $(x10^{-6}mbar)$',fontsize=13,
             transform=ax2.transAxes,ha='center')
    if save_plots:
        plt.savefig('.\\Figures\\'+'Pressure_Correlation'+'.jpg')
    return
    
def subplot_correlation_pressure(set_of_scans,ax1,ax2):
    pressure,QE,XANESheight = [],[],[]
    for scan in Set(set_of_scans).scans:  
        pressure.append(scan.pressure[-1]*1e6)
        QE.append(scan.QE[-1])
        XANESheight.append(scan.photocurrent[scan.arg(1570)])
    ax1.plot(pressure,np.polyval(np.polyfit(pressure,QE,1),pressure),
             color='0.7')
    ax2.plot(pressure,np.polyval(np.polyfit(pressure,XANESheight,1),pressure),
             color='0.7')
    ax1.plot(pressure,QE,'x',color='k')
    ax2.plot(pressure,XANESheight,'x',color='k')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.get_yaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax1.yaxis.set_ticks_position("right")
    ax2.yaxis.set_ticks_position("right")
    ax1.text(0.6,0.06,'R='+str('%.3f' % np.corrcoef(pressure,QE)[1,0]),
             transform=ax1.transAxes)
    ax2.text(0.6,0.06,'R='+str('%.3f' % np.corrcoef(pressure,XANESheight)[1,0]),
             transform=ax2.transAxes)
    return
    
def correlation_laser(set1,set2,set3):
    fig = plt.figure(figsize=(5,5))
    ax1, ax2 = fig.add_subplot(222), fig.add_subplot(224)
    ax3, ax4 = fig.add_subplot(221), fig.add_subplot(223)
    fig.subplots_adjust(hspace=0)
    print 'Figure',fig.number,': Laser power correlation'
    subplot_correlation_laser(set3,ax1,ax2)
    subplot_correlation_laser(set1,ax3,ax4)
    subplot_correlation_laser(set2,ax3,ax4)
    ax1.set_xlim([20,115])
    ax2.set_xlim([20,115])
    ax3.set_xlim([-11,50])
    ax4.set_xlim([-11,50])
    ax3.set_ylabel('$a_1$ $(x10^{-3}s^{-1})$',fontsize=14)
    ax4.set_ylabel('$a_2$ $(x10^{-2}s^{-1})$',fontsize=14)
    ax1.set_title('Al $-$ Li')
    ax3.set_title('Al')
    ax1.set_yticks([0.005,0.008,0.011])
    ax1.set_yticklabels(['5','8','11'])
    ax2.set_xticks([35,61,98])
    ax2.set_yticks([0.05,0.10,0.15])
    ax2.set_yticklabels(['5','10','15'])
    ax3.set_yticks([0.003,0.004,0.005])
    ax3.set_yticklabels(['3','4','5'])
    ax4.set_xticks([3,35])
    ax4.set_yticks([0.04,0.06,0.08])
    ax4.set_yticklabels(['4','6','8'])
    ax4.text(1.1,-0.24,'Laser power $(mW)$',
             fontsize=13,transform=ax4.transAxes,ha='center')
    if save_plots:
        plt.savefig('.\\Figures\\'+'Laser_Power_Correlation'+'.jpg')
    return

def subplot_correlation_laser(set_of_scans,ax1,ax2):
    I,s1,s2,yerr1,yerr2 = [],[],[],[],[]
    for scan in Set(set_of_scans).scans:  
        param,cov = scan.fit()
        I.append(scan.laser_power[0])
        s1.append(param[2])
        yerr1.append(cov[2,2])
        s2.append(param[4])
        yerr2.append(cov[4,4])
    ax1.errorbar(I,s1,yerr=3*np.sqrt(yerr1),
        marker='x',markersize=6,color='k',ls='None',ecolor='0.7')
    ax2.errorbar(I,s2,yerr=3*np.sqrt(yerr2),
        marker='x',markersize=6,color='k',ls='None',ecolor='0.7')
    # factor 3 for representativity of overlapping errorbars
    plt.setp(ax1.get_xticklabels(),visible=False)
    ax1.yaxis.set_ticks_position("right")
    ax2.yaxis.set_ticks_position("right")
    return
    

###################################### 4 ######################################
# Exclusion of the shaky scans
scan_range = {'Alrp':np.arange(0,9),'Al':[0,1,2,3,7,9,11,12],
              'AlLirp':[1,2,3,4,5,6],'AlLi':[1,2,3,4,5,6,7,8,9],
              'Al2':[0,1,2,3,4],'Alrp2':[0,1,2,3,4]}
# 
ref_scan = {'Al':2,'Alrp':4,'AlLirp':4,'AlLi':2,'Al2':1,'Alrp2':1}
# Conversion between the data value and the actual laser power
def power(): 
    return np.polyfit([1.8,1.9,2.0,2.1,2.2,2.3],[3,8,18,35,61,98],2)
# Fitting function
def f(t,a0,a1,s1,a2,s2): 
    return a0 + a1*np.exp(-s1*t) + a2*np.exp(-s2*t)
    

###################################### 5 ######################################
plt.close('all')
plot(Set('Alrp'),'XANES')
plot(Set('Alrp'),'QE')
plot(Set('AlLi'),'XANES')
plot(Set('AlLi'),'QE',with_fit=True)
correlation_pressure('Alrp','AlLi')
correlation_laser('Alrp','Alrp2','AlLi')

#plotall('XANES')
#plot(Set('AlLi'),'P')
#plot(Set('AlLi'),'T')
#plot(Set('AlLirp'),'QE',with_fit=True)