import os
import sys
import math
import time
import numpy as np
import xarray as xr
from giverny.turbulence_gizmos.constants import *
from giverny.turbulence_gizmos.basic_gizmos import *
import colorama
from colorama import Fore

def getVelocity_process_data(X,Y,Z,dx,node_x,node_y,node_z,cube,
                           axes_ranges, var, timepoint):
    # calculate how much time it takes to run the code.
    start_time_cutout_onevoxel = time.perf_counter()

    # data constants.
    # -----
    c = get_constants()

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)

    # used for determining the indices in the output array for each x, y, z datapoint.
    axes_min = axes_ranges[:, 0]
    
    # used for creating the 3-D output array using numpy.
    axes_lengths = axes_ranges[:, 1] - axes_ranges[:, 0] + 1
    
    # begin processing of data    
    # -----
    # get a map of the database files where all the data points are in.
#    print('\nStep 1: Determining which database files the user-specified box is found in...\n' + '-' * 25)
    sys.stdout.flush()
    
    
    user_single_db_boxes = cube.identify_single_database_file_sub_boxes(axes_ranges, var, timepoint)
       
    # -----
    # recursively break down each single file box into sub-boxes, each of which is exactly one of the sub-divided cubes of the database file.
#    print('\nStep 2: Recursively breaking down the portion of the user-specified box in each database file into voxels...\n' + '-' * 25)
    sys.stdout.flush()
    
    
    # iterates over the database files to figure out how many different hard disk drives these database files are stored on. if the number of disks
    # is greater than 1, then processing of the data will be distributed across several python processes using dask to speed up the processing 
    # time. if all of the database files are stored on 1 hard disk drive, then the data will be processed sequentially using base python.
    database_file_disks = set([])
    sub_db_boxes = {}
    for db_file in sorted(user_single_db_boxes, key = lambda x: os.path.basename(x)):
        # the parent folder for the database file corresponds to the hard disk drive that the file is stored on.
        database_file_disk = db_file.split(os.sep)[c['database_file_disk_index']]
        
        # add the folder to the set of folders already identified. this will be used to determine if dask is needed for processing.
        database_file_disks.add(database_file_disk)
        
        # create a new dictionary for all of the database files that are stored on this disk.
        if database_file_disk not in sub_db_boxes:
            sub_db_boxes[database_file_disk] = {}
            # keeping track of the original user-specified box ranges in case the user specifies a box outside of the dataset cube.
            sub_db_boxes[database_file_disk][db_file] = {}
        elif db_file not in sub_db_boxes[database_file_disk]:
            sub_db_boxes[database_file_disk][db_file] = {}
        
        for user_db_box_data in user_single_db_boxes[db_file]:
            user_db_box = user_db_box_data[0]
            user_db_box_minLim = user_db_box_data[1]
            # convert the user_db_box list of lists to a tuple of tuples so that it can be used as a key, along with user_db_box_minLim 
            # in the sub_db_boxes dictionary.
            user_db_box_key = (tuple([tuple(user_db_box_range) for user_db_box_range in user_db_box]), user_db_box_minLim)

            morton_voxels_to_read = cube.identify_sub_boxes_in_file(user_db_box, var, timepoint, c['voxel_side_length'])

            # update sub_db_boxes with the information for reading in the database files.
            sub_db_boxes[database_file_disk][db_file][user_db_box_key] = morton_voxels_to_read
    

    
#    print('Successfully completed.\n' + '-' * 5)
    sys.stdout.flush()

        
    # pre-fill the output data 3-d array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
    # so that the array is filled with the missing placeholder values (-999.9). 
    output_data = np.full((axes_lengths[2], axes_lengths[1], axes_lengths[0], num_values_per_datapoint),
                           fill_value = c['missing_value_placeholder'], dtype = 'f')
    
    # determines if the database files will be read sequentially with base python, or in parallel with dask. No dask!!!
    sys.stdout.flush()
        
    result_output_data = cube.read_database_files_sequentially(sub_db_boxes,
                                                                   axes_ranges,
                                                                   num_values_per_datapoint, c['bytes_per_datapoint'], c['voxel_side_length'],c['missing_value_placeholder'])

    
    # iterate over the results to fill output_data.
    for result in result_output_data:
        output_data[result[1][2] - axes_min[2] : result[2][2] - axes_min[2] + 1,
                    result[1][1] - axes_min[1] : result[2][1] - axes_min[1] + 1,
                    result[1][0] - axes_min[0] : result[2][0] - axes_min[0] + 1] = result[0]    
    
    
#    print(Fore.RED + 'output_data')
#    print(Fore.BLACK + '')
#    print(output_data) 
    # calculate how much time it takes to run the code.
    end_time_cutout_onevoxel = time.perf_counter()
    
    
    # Assuming we have 8*8*8 now intepolation test from here
#    node_x = int(math.floor(X/dx))
#    node_y = int(math.floor(Y/dx))
#    node_z = int(math.floor(Z/dx))
    
#    print(Fore.RED + 'node_x,node_y,node_z')
#    print(Fore.BLACK + '')
#    print(node_x,node_y,node_z) 

    start_time_Lag4 = time.perf_counter()
    
    #dx',dy',dz'
    dx_prime = X/dx - node_x
    dy_prime = Y/dx - node_y
    dz_prime = Z/dx - node_z
    
    print(Fore.RED + 'dxyz_prime = XYZ/dxyz - node_x,y,z')
    print(Fore.BLACK + '')
    print(dx_prime, dy_prime, dz_prime)   
    
#    print(output_data[node_x,0,:])
    
#    print('output_box lengh',len(output_data[0,0,:]))
    lagInt_x = np.zeros((4,1), dtype=np.float32)
    lagInt_y = np.zeros((4,1), dtype=np.float32)
    lagInt_z = np.zeros((4,1), dtype=np.float32)
    
    #lag coefficient in X direction
    lagInt_x[0] = (-2 * dx_prime + 3 * dx_prime**2 - dx_prime**3) / 6
    lagInt_x[1] = (2 - dx_prime - 2 * dx_prime**2 + dx_prime**3) / 2
    lagInt_x[2] = (2 * dx_prime + dx_prime**2 - dx_prime**3)/2
    lagInt_x[3] = (-dx_prime + dx_prime**3)/6
    
    #lag coefficient in Y direction
    lagInt_y[0] = (-2 * dy_prime + 3 * dy_prime**2 - dy_prime**3) / 6
    lagInt_y[1] = (2 - dy_prime - 2 * dy_prime**2 + dy_prime**3) / 2
    lagInt_y[2] = (2 * dy_prime + dy_prime**2 - dy_prime**3) / 2
    lagInt_y[3] = (-dy_prime + dy_prime**3) / 6
    
    #lag coefficient in Z direction
    lagInt_z[0] = (-2 * dz_prime + 3 * dz_prime**2 - dz_prime**3) / 6
    lagInt_z[1] = (2 - dz_prime - 2 * dz_prime**2 + dz_prime**3) / 2
    lagInt_z[2] = (2 * dz_prime + dz_prime**2 - dz_prime**3) / 2
    lagInt_z[3] = (-dz_prime + dz_prime**3) / 6
    
#    print(Fore.RED + 'lagInt_x')
#    print(Fore.BLACK + '')
#    print(lagInt_x)
    
#    print(Fore.RED + 'lagInt_y')
#    print(Fore.BLACK + '')
#    print(lagInt_y)
    
#    print(Fore.RED + 'lagInt_z')
#    print(Fore.BLACK + '')
#    print(lagInt_z)    
    
#    print(Fore.RED + 'var')
#    print(Fore.BLACK + '')
#    print(var) 
    
    #Calculate inteporlation
    if var == 'pr':
        sum=0
#        print('yes go if statement pressure')
        for i in range(2,6):
            for j in range(2,6):
                for k in range(2,6):
                    sum = sum + output_data[k,j,i]*lagInt_x[i-2]*lagInt_y[j-2]*lagInt_z[k-2]  
    elif var == 'vel':
        sum_u=0
        sum_v=0
        sum_w=0
#        print('yes go if statement velocity')
        for i in range(2,6):
            for j in range(2,6):
                for k in range(2,6):
                    sum_u = sum_u + output_data[k,j,i,0]*lagInt_x[i-2]*lagInt_y[j-2]*lagInt_z[k-2]
                    sum_v = sum_v + output_data[k,j,i,1]*lagInt_x[i-2]*lagInt_y[j-2]*lagInt_z[k-2]
                    sum_w = sum_w + output_data[k,j,i,2]*lagInt_x[i-2]*lagInt_y[j-2]*lagInt_z[k-2]
        sum=[sum_u, sum_v, sum_w]
        
        
        
    end_time_Lag4 = time.perf_counter()
    
    
    # see how long the program took to run.
#    print(f'\nstep cutout one voxel took = {round(end_time_cutout_onevoxel - start_time_cutout_onevoxel, 3)} seconds ({round((end_time_cutout_onevoxel - start_time_cutout_onevoxel) / 60, 3)} minutes)')  
    
    print(f'\nstep 4th order at one point took = {round(end_time_Lag4 - start_time_Lag4, 6)} seconds ({round((end_time_cutout_onevoxel - start_time_cutout_onevoxel) / 60, 6)} minutes)')  
        
#    print(sum)
    return sum