# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:55:44 2017

@author: Simon
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import xlrd

name_of_set = 'Al'
type_of_set = 'XANES' #'QE'

if type_of_set is 'QE':
    filename = ''
    wb = xlrd.open_workbook(filename)
    sh1 = wb.sheet_by_index(0)
    data = []
    for rownum in range(sh1.nrows): 
        data += [sh1.row_values(rownum)]

if type_of_set is 'XANES':
    for scan_number,source_file in enumerate(os.listdir('.\\Raw_data\\'+name_of_set)):
        source_directory = '.\\Raw_data\\' + name_of_set + '\\' + source_file
        l1 = open(source_directory).readline().split('\t')
        data = np.loadtxt(source_directory)
        parameters_names = ['date','hour','','Energy(eV)']
        parameters_values = [source_file.split('_')[1],source_file.split('_')[2],'','Photocurrent(mA)\tTopUp\tMonochromator']
        header = '\n'.join(['\t'.join((i,str(j))) for i,j in zip(parameters_names,parameters_values)])
        exported_data = np.zeros((len(data),4))
        exported_data[:,0] = data[:,l1.index('Energy')]
        exported_data[:,1] = data[:,l1.index('I1_KEITHLEY2')]
        exported_data[:,2] = data[:,l1.index('I_SLS')] #TopUp
        exported_data[:,3] = data[:,l1.index('I0_KEITHLEY1')] #Monochromator, only valid before the repolishing
        destination = '.\\Data\\' + name_of_set + '_' + type_of_set + '_' + str(scan_number) + '.txt'
        np.savetxt(destination,exported_data,header=header)

def plot(column):
    plt.plot(data[:,l1.index(column)])
#final_data.tofile