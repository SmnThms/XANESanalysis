# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:55:44 2017

@author: Simon
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import xlrd

source_folder = '.\\Raw_data\\'
destination_folder = '.\\Data\\'
type_of_set = 'QE'

# For each set of scans : (start time of first scan, start time of last scan,)
C = {'Al':(1,6303)}#13600)}
C.update({'AlLi':(13647,18050)})#(26220,35556)})
C.update({'Al2':(18948,20960)})#(36405,40905)})
C.update({'Alrp':(3605,7900)})#(6925,15730)})
C.update({'Alrp2':(9636,11500)})#(18514,22819)})
C.update({'AlLirp':(17492,21000)})#(33600,41300)})

if type_of_set is 'QE':
    for filename, names_of_sets in zip(['Monitor_20120926_20h32','monitor2809_07h48'],[['Al','AlLi','Al2'],['Alrp','Alrp2','AlLirp']]):
        wb = xlrd.open_workbook(source_folder+filename+'.xls')
        sheet = wb.sheet_by_index(0)
        data = np.zeros((sheet.nrows,sheet.ncols))
        for row_nb in range(sheet.nrows-2): # Copy of the xls file in an array
            data[row_nb,:] = np.array([sheet.row_values(row_nb+2)])
        l1 = [str(title) for title in sheet.row_values(0)] # Titles of the columns
        for name_of_set in names_of_sets:
            scan_nb = 0            
            row_nb = C[name_of_set][0]
            while row_nb < C[name_of_set][1]:
                exported_data = []
                while not data[row_nb,l1.index('LASER_ON')]:
                    row_nb += 1
                while data[row_nb,l1.index('LASER_ON')]:
                    exported_data.append([data[row_nb,l1.index(column)] for column in ['time','I_KEITH2','P_ES','TEMP']])
                    row_nb += 1
                parameters_names = ['Date','Hour','Laser_intensity','Photocurrent_unit','','Time\tPhotocurrent(unit)\tPressure(unit)\tTemperature(unit)']
                parameters_values = ['','']
                header = '\n'.join(['\t'.join((i,str(j))) for i,j in zip(parameters_names,parameters_values)])
                destination = destination_folder + name_of_set + '_' + type_of_set + '_' + str(scan_nb) + '.txt'
                np.savetxt(destination,np.array(exported_data),header=header)
                scan_nb += 1
                print name_of_set, row_nb
    

elif type_of_set is 'XANES':
    for name_of_set in ['Al','Al2','Alrp','Alrp2','AlLi','AlLirp']:
        for scan_nb,source_file in enumerate(os.listdir(source_folder+name_of_set)):
            source_directory = source_folder + name_of_set + '\\' + source_file
            l1 = open(source_directory).readline().split('\t')
            data = np.loadtxt(source_directory)
            parameters_names = ['Date','Hour','','Energy(eV)\tPhotocurrent(unit)\tTopUp\tI0']
            parameters_values = [source_file.split('_')[1],source_file.split('_')[2],'','']
            header = '\n'.join(['\t'.join((i,str(j))) for i,j in zip(parameters_names,parameters_values)])
            exported_data = np.zeros((len(data),4))
            exported_data[:,0] = data[:,l1.index('Energy')]
            exported_data[:,1] = data[:,l1.index('I1_KEITHLEY2')]
            exported_data[:,2] = data[:,l1.index('I_SLS')] #TopUp
            exported_data[:,3] = data[:,l1.index('I0_KEITHLEY1')] #I0 for normalization, only valid before the repolishing
            destination = destination_folder + name_of_set + '_' + type_of_set + '_' + str(scan_nb) + '.txt'
            np.savetxt(destination,exported_data,header=header)

def plot(column):
    plt.plot(data[:,l1.index(column)])