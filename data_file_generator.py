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
type_of_set = 'QE'

# For each scan : (start time, end time,)
C = {'Al':((1,13600),(36405,40905))}
C.update({'Al2':((36405,40905))})
C.update({'AlLi':((26220,35556))})
C.update({'Alrp':((6925,15730),(18514,22819))})
C.update({'AlLirp':((33600,41300))})

if type_of_set is 'QE':
    for filename, names_of_sets in zip(['Monitor_20120926_20h32','monitor2809_07h48'],[['Al','Al2','AlLi']['Alrp','AlLirp']]):
        wb = xlrd.open_workbook(source_folder+filename+'.xls')
        sheet = wb.sheet_by_index(0)
        data = np.zeros((sheet.nrows,sheet.ncols))
        for row_nb in range(sheet.nrows-2): 
            data[row_nb,:] = np.array([sheet.row_values(row_nb+2)])
        l1 = [sheet.row_values(0)]
        for name_of_set in names_of_sets:
            row_nb = C[name_of_set]
            for scan_nb in range(len(row_nb)):
                parameters_names = ['Date','Hour','Laser_intensity','Photocurrent_unit','','Time\tPhotocurrent(unit)\tPressure(unit)\tTemperature(unit)']
                parameters_values = ['','']
                header = '\n'.join(['\t'.join((i,str(j))) for i,j in zip(parameters_names,parameters_values)])
                exported_data = np.zeros(row_nb[1]-row_nb[0])
                exported_data[:,0] = data[:,l1.index('time')]
                exported_data[:,1] = data[:,l1.index('I_KEITH2')]
                exported_data[:,2] = data[:,l1.index('P_ES')]
                exported_data[:,3] = data[:,l1.index('TEMP')]
                destination = '.\\Data\\' + name_of_set + '_' + type_of_set + '_' + str(scan_nb) + '.txt'
                np.savetxt(destination,exported_data,header=header)
    

elif type_of_set is 'XANES':
    for name_of_set in ['Al','Al2','Alrp','AlLi','AlLirp']:
        for scan_nb,source_file in enumerate(os.listdir(source_folder+name_of_set)):
            source_directory = source_folder + name_of_set + '\\' + source_file
            l1 = open(source_directory).readline().split('\t')
            data = np.loadtxt(source_directory)
            parameters_names = ['Date','Hour','','Energy(eV)\tPhotocurrent(unit)\tTopUp\tMonochromator']
            parameters_values = [source_file.split('_')[1],source_file.split('_')[2],'','']
            header = '\n'.join(['\t'.join((i,str(j))) for i,j in zip(parameters_names,parameters_values)])
            exported_data = np.zeros((len(data),4))
            exported_data[:,0] = data[:,l1.index('Energy')]
            exported_data[:,1] = data[:,l1.index('I1_KEITHLEY2')]
            exported_data[:,2] = data[:,l1.index('I_SLS')] #TopUp
            exported_data[:,3] = data[:,l1.index('I0_KEITHLEY1')] #Monochromator, only valid before the repolishing
            destination = '.\\Data\\' + name_of_set + '_' + type_of_set + '_' + str(scan_nb) + '.txt'
            np.savetxt(destination,exported_data,header=header)

def plot(column):
    plt.plot(data[:,l1.index(column)])