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
from scipy.optimize import curve_fit
#from matplotlib import rcParams
#rcParams['text.usetex'] = True

def power():
    return np.polyfit([1.8,1.9,2.0,2.1,2.2,2.3],[3,8,18,35,61,98],2)
#plt.plot([1.8,1.9,2.0,2.1,2.2,2.3],[3,8,18,35,61,98],'o')
#plt.plot(np.linspace(1.8,2.3),np.polyval(power(),np.linspace(1.8,2.3)))
    
class Scan:
    def __init__(self,name_of_scan,number_of_scan):
        for type_of_scan in ['QE','XANES']:
            file_address = '.\\Data\\' + name_of_scan + '_' + type_of_scan \
                           + '_' + str(number_of_scan) + '.txt'
            data = np.loadtxt(file_address)
            if type_of_scan is 'QE':
                self.time = data[:,0] - data[0,0]
                self.pressure = data[:,2]
                self.laser_power = np.polyval(power(),data[:,3]) # in mW ?
                self.temperature = data[:,4]
                self.QE = data[:,1]*10**data[:,-1]#/self.laser_power/self.pressure*self.pressure[-1]
            elif type_of_scan is 'XANES':
                self.energy = data[:,0]
                self.topup = data[:,2]
                self.I0 = data[:,3]
                self.photocurrent = data[:,1]
                self.normalize()            
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
        param,cov = curve_fit(f,self.time[inf:],self.QE[inf:],p0=p0) 
#        rchi2 = 
#        print 'Reduced chi2 =',
        return param,cov#,rchi2
    
def f(t,a0,a1,s1,a2,s2):
    return a0 + a1*np.exp(-s1*t) + a2*np.exp(-s2*t)
        
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
            
def plot(set_of_scans, type_of_scan, cmap='jet_r',with_fit=True):
    fig = plt.figure(figsize=(6,6))
    colormap = getattr(plt.cm,cmap)
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
                plt.plot(scan.time[inf:],f(scan.time[inf:],*scan.fit()[0]),'--',color='r',alpha=0.6)
                name_fit = '_fitted'
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
#    plt.savefig('.\\Figures\\'+set_of_scans.name+'_'+type_of_scan+name_fit+'.jpg')
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

def correlation_pressure(name):
    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    print 'Figure',fig.number,': Pressure correlation',name
    pressure,QE,XANESheight = [],[],[]
#    point = {'Alrp':'+','Al':'*','AlLi':'x','AlLirp':'^','Al2':'.','Alrp2':'.'}
    for set_of_scans in [Set(name)]:
        for scan in set_of_scans.scans:  
            pressure.append(scan.pressure[-1]*1e6)
            QE.append(scan.QE[-1])
            XANESheight.append(scan.photocurrent[scan.arg(1570)])
    ax1.plot(pressure,QE,'x',color='k')
    print set_of_scans.name,'QE : r^2 =',np.corrcoef(pressure,QE)[1,0]
    ax2.plot(pressure,XANESheight,'x',color='k')
    print set_of_scans.name,'XANES : r^2 =',np.corrcoef(pressure,XANESheight)[1,0]
    fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.plot(pressure,np.polyval(np.polyfit(pressure,QE,1),pressure),'midnightblue',alpha=0.4)
    ax2.plot(pressure,np.polyval(np.polyfit(pressure,XANESheight,1),pressure),'midnightblue',alpha=0.4)
#    ax1.set_xlabel('Pressure (x$10^{-6}$ mbar)',fontsize=14)
    ax1.set_ylabel('QE level',fontsize=14)
    ax1.set_title(name)
    ax1.get_yaxis().set_ticks([])
    ax2.set_xlabel('Pressure',fontsize=14)
    ax2.set_ylabel('XANES variation',fontsize=14)
    ax2.get_yaxis().set_ticks([])
    ax1.yaxis.set_ticks_position("right")
    ax2.yaxis.set_ticks_position("right")
    ax1.text(0.7,0.1,'R='+str('%.3f' % np.corrcoef(pressure,QE)[1,0]),transform=ax1.transAxes)
    ax2.text(0.7,0.1,'R='+str('%.3f' % np.corrcoef(pressure,XANESheight)[1,0]),transform=ax2.transAxes)
    plt.savefig('.\\Figures\\'+name+'_PressureCorrelation'+'.jpg')
    return

def correlation_laser():
    fig = plt.figure(figsize=(5,5))
    ax1, ax2 = fig.add_subplot(222), fig.add_subplot(224)
    ax3, ax4 = fig.add_subplot(221), fig.add_subplot(223)
    fig.subplots_adjust(hspace=0)
    print 'Figure',fig.number,': Laser power correlation'
    subfunction_corr_laser('AlLi',ax1,ax2)
    subfunction_corr_laser('Alrp',ax3,ax4)
    subfunction_corr_laser('Alrp2',ax3,ax4)
    ax1.set_xlim([20,115])
    ax2.set_xlim([20,115])
    ax3.set_xlim([-11,50])
    ax4.set_xlim([-11,50])
    plt.setp(ax1.get_xticklabels(),visible=False)
    plt.setp(ax3.get_xticklabels(),visible=False)
    ax3.set_ylabel('$a_1$ ${\tiny (x10^{-3}s^{-1}}$)',fontsize=14)
    ax4.set_ylabel('$a_2$ (x$10^{-2}$s$^{-1}$)',fontsize=14)
    #r'{\fontsize{30pt}{3em}\selectfont{}{bla\r}{\fontsize{18pt}{3em}\selectfont{}bla}')
    ax1.set_title('Al $-$ Li')
    ax3.set_title('Al')
    ax1.yaxis.set_ticks_position("right")
    ax2.yaxis.set_ticks_position("right")
    ax3.yaxis.set_ticks_position("right")
    ax4.yaxis.set_ticks_position("right")
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

#    fig.title('Laser power (mW)',fontsize=14)
#    plt.savefig('.\\Figures\\'+'_LaserPowerCorrelation'+'.jpg')
    return

def subfunction_corr_laser(set_of_scans,ax1,ax2):
    I,s1,s2,yerr1,yerr2 = [],[],[],[],[]
#    I,s1,s2,yerr1,yerr2,s1_mean,s1_mean_var,s2_mean,s2_mean_var = ([],)*9
    for scan in Set(set_of_scans).scans:  
        param,cov = scan.fit()
        I.append(scan.laser_power[0])
        s1.append(param[2])
        yerr1.append(cov[2,2])
        s2.append(param[4])
        yerr2.append(cov[4,4])
    ax1.errorbar(I,s1,yerr=np.sqrt(yerr1),
        marker='x',markersize=6,color='k',ls='None',ecolor='0.7')#capsize=0,capthick=0.5,elinewidth=1)
    ax2.errorbar(I,s2,yerr=np.sqrt(yerr2),
        marker='x',markersize=6,color='k',ls='None',ecolor='0.7')#elinewidth=1,capsize=0,capthick=0.5
#    for i,x in enumerate(np.unique(I)):
#        s1_x = np.array(s1)[np.array(I)==x]
#        s1_mean[i] = np.mean(s1_x)
#        s1_mean_var[i] = np.mean((s1_x-np.mean(s1_x))**2)/(len(s1_x)-1) \
#                       + np.mean(np.array(yerr1)[np.array(I)==x])
#        s2_x = np.array(s2)[np.array(I)==x]
#        s2_mean[i] = np.mean(s2_x)
#        s2_mean_var[i] = np.mean((s2_x-np.mean(s2_x))**2)/(len(s2_x)-1) \
#                       + np.mean(np.array(yerr2)[np.array(I)==x])
#    return np.unique(I),s1_mean,s1_mean_var,s2_mean,s2_mean_var
    return
    
    
#    marker = {'Alrp':'+','Al':'^','AlLi':'.','AlLirp':'*','Al2':'.','Alrp2':'.'}
##    for set_of_scans in [Set('AlLi'),Set('Alrp')]:
##    for set_of_scans in [Set('Al'),Set('Al2'),Set('Alrp2')]:
#    for set_of_scans in list_of_sets:
#        I,s1,s2,yerr1,yerr2 = ([],)*5
#        for scan in set_of_scans.scans:  
#            param,cov = scan.fit()
#            I.append(scan.laser_power[0])
#            s1.append(param[2])
#            yerr1.append(cov[2,2])
#            s2.append(param[4])
#            yerr2.append(cov[4,4])
#        ax1.errorbar(I,s1,yerr=np.sqrt(yerr1),marker=marker[set_of_scans.name],
#                     color='k',ls='None',capsize=3,capthick=0.5,elinewidth=0.7)
##        ax1.plot(I,s1,marker=marker[set_of_scans.name],color='k',ls='None')
##        print set_of_scans.name,'s1 : r^2 =',np.corrcoef(I,s1)[1,0]
#        ax2.errorbar(I,s2,yerr=np.sqrt(yerr2),marker=marker[set_of_scans.name],
#                     color='k',ls='None',capsize=3,capthick=0.5,elinewidth=0.7)        
##        ax2.plot(I,s2,marker=marker[set_of_scans.name],color='k',ls='None')
##        print set_of_scans.name,'s2 : r^2 =',np.corrcoef(I,s2)[1,0]
#        I,s1,yerr1,s2,yerr2 = np.array(I),np.array(s1),np.array(yerr1),np.array(s2),np.array(yerr2)
#        Im = np.unique(I)
#        s1m,s1m2,vars1m2 = np.zeros(len(Im)),np.zeros(len(Im)),np.zeros(len(Im))
#        for i,x in enumerate(Im):
#            s1m[i] = np.sum(s1[I==x]/yerr1[I==x])/np.sum(1/yerr1[I==x])
#            s1m2[i] = np.sum(s1[I==x])/len(s1[I==x])
#            vars1m2[i] = np.sum((s1[I==x]-s1m2[i])**2)/(len(s1[I==x])*(len(s1[I==x])-1))+np.mean(yerr1[I==x])
##        ax1.plot(Im,s1m,'or')
#        ax1.errorbar(Im,s1m2,np.sqrt(vars1m2),marker='o',
#                     color='r',ls='None',capsize=4,capthick=0.5,elinewidth=1.2) 
#    plt.setp(ax1.get_xticklabels(), visible=False)
##    ax1.plot(I,np.polyval(np.polyfit(I,s1,1),I),'midnightblue',alpha=0.4)
##    ax2.plot(I,np.polyval(np.polyfit(I,s2,1),I),'midnightblue',alpha=0.4)
##    ax1.set_xlabel('Pressure (x$10^{-6}$ mbar)',fontsize=14)
#    ax1.set_ylabel('$\sigma_1$',fontsize=14)
##    ax1.set_title()
##    ax1.get_yaxis().set_ticks([])
#    ax2.set_xlabel('Laser power (mW)',fontsize=14)
#    ax2.set_ylabel('$\sigma_2$',fontsize=14)
##    ax2.get_yaxis().set_ticks([])
##    ax1.yaxis.set_ticks_position("right")
##    ax2.yaxis.set_ticks_position("right")
##    ax1.text(0.7,0.1,'R='+str('%.3f' % np.corrcoef(I,s1)[1,0]),transform=ax1.transAxes)
##    ax2.text(0.7,0.1,'R='+str('%.3f' % np.corrcoef(I,s2)[1,0]),transform=ax2.transAxes)
#    plt.savefig('.\\Figures\\'+'_LaserPowerCorrelation'+'.jpg')
#    return
            
scan_range = {'Alrp':np.arange(0,9),'Al':[0,1,2,3,7,9,11,12],'AlLirp':[1,2,3,4,5,6],
              'AlLi':[1,2,3,4,5,6,7,8,9],'Al2':[0,1,2,3,4],'Alrp2':[0,1,2,3,4]}
ref_scan = {'Al':2,'Alrp':4,'AlLirp':4,'AlLi':2,'Al2':1,'Alrp2':1}

plt.close('all')
#plotall('XANES')
#plotall('QE')
#plot(Set('Alrp'),'XANES',fig=8)
#plot(Set('Alrp'),'QE',cmap='bone_r')
#plot(Set('AlLi'),'QE',cmap='bone_r')
#plot(Set('Al'),'QE',cmap='bone_r')
#plot(Set('Al2'),'QE',cmap='bone_r')
#plot(Set('Alrp2'),'QE',cmap='bone_r')
#correlation_pressure('Alrp2')
#correlation_pressure('Alrp')
#correlation_pressure('AlLi')
#correlation_pressure('AlLirp')
#correlation_pressure('Al')
#correlation_pressure('Al2')
#correlation_laser([Set('Al'),Set('Al2')])
#correlation_laser([Set('Alrp'),Set('Alrp2')])
correlation_laser()
#correlation_laser(3)
#correlation_laser(4)
#correlation_laser(5)
#correlation_laser(6)
#plot(Set('AlLi'),'P')
#plot(Set('AlLi'),'T')