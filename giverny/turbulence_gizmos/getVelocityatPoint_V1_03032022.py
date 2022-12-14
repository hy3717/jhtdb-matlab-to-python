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

def getVelocity_process_data(X,Y,Z,dx,cube, cube_resolution, cube_title, output_path,
                           axes_ranges, var, timepoint,
                           axes_ranges_original, strides, var_original, timepoint_original):
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()

    # data constants.
    # -----
    c = get_constants()

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)

    # used for determining the indices in the output array for each x, y, z datapoint.
    axes_min = axes_ranges[:, 0]
    
    # number of original datapoints along each axis specified by the user. used for checking that the user did not request
    # too much data.
    axes_lengths_original = axes_ranges_original[:, 1] - axes_ranges_original[:, 0] + 1
    #print(Fore.RED + 'axes_lengths_original')
    #print(Fore.BLACK + '')
    #print(axes_lengths_original)

    # used for creating the 3-D output array using numpy.
    axes_lengths = axes_ranges[:, 1] - axes_ranges[:, 0] + 1
    #print(Fore.RED + 'axes_lengths')
    #print(Fore.BLACK + '')
    #print(axes_lengths)

    # total number of datapoints, used for checking if the user requested too much data.
    num_datapoints = np.prod(axes_lengths_original)
    # total size of data, in GBs, requested by the user's box.
    requested_data_size = (num_datapoints * c['bytes_per_datapoint'] * num_values_per_datapoint) / float(1024**3)
    # maximum number of datapoints that can be read in. currently set to 3 GBs worth of datapoints.
    max_datapoints = int((c['max_data_size'] * (1024**3)) / (c['bytes_per_datapoint'] * float(num_values_per_datapoint)))
    # approximate max size of a cube representing the maximum data points. this number is rounded down.
    approx_max_cube = int(max_datapoints**(1/3))

    if requested_data_size > c['max_data_size']:
        raise ValueError(f'Please specify a box with fewer than {max_datapoints} data points. This represents an approximate cube size ' + \
                         f'of ({approx_max_cube} x {approx_max_cube} x {approx_max_cube}).')

    # begin processing of data.
    # -----
    print('Note: For larger boxes, e.g. 512-cubed and up, processing will take approximately 1 minute or more...\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # get a map of the database files where all the data points are in.
    print('\nStep 1: Determining which database files the user-specified box is found in...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    start_time_step1 = time.perf_counter()
    
    user_single_db_boxes = cube.identify_single_database_file_sub_boxes(axes_ranges, var, timepoint)

    print(f'number of database files that the user-specified box is found in:\n{len(user_single_db_boxes)}\n')
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    end_time_step1 = time.perf_counter()

    print('Successfully completed.\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # recursively break down each single file box into sub-boxes, each of which is exactly one of the sub-divided cubes of the database file.
    print('\nStep 2: Recursively breaking down the portion of the user-specified box in each database file into voxels...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 2.
    start_time_step2 = time.perf_counter()
    
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
    
    min_file_boxes = np.min([len(sub_db_boxes[database_file_disk][db_file][user_db_box_key]) 
                             for database_file_disk in sub_db_boxes 
                             for db_file in sub_db_boxes[database_file_disk] 
                             for user_db_box_key in sub_db_boxes[database_file_disk][db_file]])
    max_file_boxes = np.max([len(sub_db_boxes[database_file_disk][db_file][user_db_box_key]) 
                             for database_file_disk in sub_db_boxes 
                             for db_file in sub_db_boxes[database_file_disk] 
                             for user_db_box_key in sub_db_boxes[database_file_disk][db_file]])
    
    print('sub-box statistics for the database file(s):\n-')
    print(f'minimum number of sub-boxes to read in a database file:\n{min_file_boxes}')
    print(f'maximum number of sub-boxes to read in a database file:\n{max_file_boxes}\n')
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 2.
    end_time_step2 = time.perf_counter()
    
    print('Successfully completed.\n' + '-' * 5)
    sys.stdout.flush()

    # -----
    # read the data.
    print('\nStep 3: Reading the data from all of the database files and storing the values into a matrix...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 3.
    start_time_step3 = time.perf_counter()
    
    # pre-fill the output data 3-d array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
    # so that the array is filled with the missing placeholder values (-999.9). 
    output_data = np.full((axes_lengths[2], axes_lengths[1], axes_lengths[0], num_values_per_datapoint),
                           fill_value = c['missing_value_placeholder'], dtype = 'f')
    
    # determines if the database files will be read sequentially with base python, or in parallel with dask.
    num_db_disks = len(database_file_disks)
    if num_db_disks == 1:
        # sequential processing.
        print('Database files are being read sequentially...')
        sys.stdout.flush()
        
        result_output_data = cube.read_database_files_sequentially(sub_db_boxes,
                                                                   axes_ranges,
                                                                   num_values_per_datapoint, c['bytes_per_datapoint'], c['voxel_side_length'],
                                                                   c['missing_value_placeholder'])
    else:
        # parallel processing.
        # optimizes the number of processes that are used by dask and makes sure that the number of processes does not exceed dask_maximum_processes.
        num_processes = c['dask_maximum_processes']
        if num_db_disks < c['dask_maximum_processes']:
            num_processes = num_db_disks
        
        result_output_data = cube.read_database_files_in_parallel_with_dask(sub_db_boxes,
                                                                            axes_ranges,
                                                                            num_values_per_datapoint, c['bytes_per_datapoint'], c['voxel_side_length'],
                                                                            c['missing_value_placeholder'], num_processes)
    
    # iterate over the results to fill output_data.
    for result in result_output_data:
        output_data[result[1][2] - axes_min[2] : result[2][2] - axes_min[2] + 1,
                    result[1][1] - axes_min[1] : result[2][1] - axes_min[1] + 1,
                    result[1][0] - axes_min[0] : result[2][0] - axes_min[0] + 1] = result[0]
    
    # determines how many copies of data need to be me made along each axis when the number of datapoints the user specified
    # exceeds cube_resolution. note: no copies of the data values should be made, hence data_value_multiplier equals 1.
    axes_multipliers = np.ceil(axes_lengths_original / cube_resolution).astype(int)
    data_value_multiplier = 1
    
    # duplicates the data along the z-, y-, and x-axes of output_data if the the user asked for more datapoints than cube_resolution along any axis.
    if np.any(axes_multipliers > 1):
        output_data = np.tile(output_data, (axes_multipliers[2], axes_multipliers[1], axes_multipliers[0], data_value_multiplier))
        # truncate any extra datapoints from the duplicate data outside of the original range of the datapoints specified by the user.
        output_data = np.copy(output_data[0 : axes_lengths_original[2], 0 : axes_lengths_original[1], 0 : axes_lengths_original[0], :])
    
    #print(Fore.RED + 'output_data = np.tile(output_data, (axes_multipliers[2], axes_multipliers[1], axes_multipliers[0], data_value_multiplier))')
    #print(Fore.BLACK + '')
    #print(output_data)   
    
    
    # apply the strides to output_data.
    #output_data = output_data[::strides[2], ::strides[1], ::strides[0], :]
    
    # create axis coordinate ranges to store in the xarray metadata.
    #z_coords = range(axes_ranges_original[2][0], axes_ranges_original[2][1] + 1, strides[2])
    #y_coords = range(axes_ranges_original[1][0], axes_ranges_original[1][1] + 1, strides[1])
    #x_coords = range(axes_ranges_original[0][0], axes_ranges_original[0][1] + 1, strides[0])
    
    # retrieve the function symbol corresponding to the variable, e.g. "velocity" corresponds to function symbol "u".
    #variable_function = get_variable_function(var)
    
    # convert output_data from a numpy array to a xarray.
    #output_data = xr.DataArray(output_data, 
    #                           coords = {'z':z_coords, 'y':y_coords, 'x':x_coords, 't':timepoint_original}, 
    #                           dims = ['z', 'y', 'x', 'v'],
    #                           attrs = {'dataset':cube_title, 'function':variable_function,
    #                                    'stridex':strides[0], 'stridey':strides[1], 'stridez':strides[2],
    #                                    'xs':axes_ranges_original[0][0], 'xe':axes_ranges_original[0][1],
    #                                    'ys':axes_ranges_original[1][0], 'ye':axes_ranges_original[1][1],
    #                                    'zs':axes_ranges_original[2][0], 'ze':axes_ranges_original[2][1]})
    
    # checks to make sure that data was read in for all points.
    #if c['missing_value_placeholder'] in output_data:
    #    raise Exception(f'output_data was not filled correctly')
        
    #print(Fore.RED + 'output_data from step3')
    #print(Fore.BLACK + '')
    #print(output_data)    
    
    # calculate how much time it takes to run step 3.
    #end_time_step3 = time.perf_counter()
    
    #print('\nSuccessfully completed.\n' + '-' * 5)
    #sys.stdout.flush()
    
    #print('\nData processing pipeline has completed successfully.\n' + '-' * 5)
    #sys.stdout.flush()  
    
    # Assuming we have 8*8*8 now intepolation test from here
    node_x = int(math.floor(X/dx))-1
    node_y = int(math.floor(Y/dx))-1
    node_z = int(math.floor(Z/dx))-1
    
    print(Fore.RED + 'node_x,node_y,node_z')
    print(Fore.BLACK + '')
    print(node_x,node_y,node_z) 
    
    dx_prime = X/dx-1 - node_x
    dy_prime = Y/dx-1 - node_y
    dz_prime = Z/dx-1 - node_z
    
    print(Fore.RED + 'dxyz_prime = XYZ/dxyz - node_x,y,z')
    print(Fore.BLACK + '')
    print(dx_prime, dy_prime, dz_prime)   
    
#    print(output_data[node_x,0,:])
    
    print('output_box lengh',len(output_data[0,0,:]))
    lagInt_x = np.zeros((4,1), dtype=float)
    lagInt_y = np.zeros((4,1), dtype=float)
    lagInt_z = np.zeros((4,1), dtype=float)
    
    lagInt_x[0] = (-2 * dx_prime + 3 * dx_prime**2 - dx_prime**3) / 6;
    lagInt_x[1] = (2 - dx_prime - 2 * dx_prime**2 + dx_prime**3) / 2;
    lagInt_x[2] = (2 * dx_prime + dx_prime**2 - dx_prime**3)/2;
    lagInt_x[3] = (-dx_prime + dx_prime**3)/6;
  
    lagInt_y[0] = (-2 * dy_prime + 3 * dy_prime**2 - dy_prime**3) / 6;
    lagInt_y[1] = (2 - dy_prime - 2 * dy_prime**2 + dy_prime**3) / 2;
    lagInt_y[2] = (2 * dy_prime + dy_prime**2 - dy_prime**3) / 2;
    lagInt_y[3] = (-dy_prime + dy_prime**3) / 6;
    
    lagInt_z[0] = (-2 * dz_prime + 3 * dz_prime**2 - dz_prime**3) / 6;
    lagInt_z[1] = (2 - dz_prime - 2 * dz_prime**2 + dz_prime**3) / 2;
    lagInt_z[2] = (2 * dz_prime + dz_prime**2 - dz_prime**3) / 2;
    lagInt_z[3] = (-dz_prime + dz_prime**3) / 6;
    
#    print(Fore.RED + 'lagInt_x')
#    print(Fore.BLACK + '')
#    print(lagInt_x)
    
#    print(Fore.RED + 'lagInt_y')
#    print(Fore.BLACK + '')
#    print(lagInt_y)
    
#    print(Fore.RED + 'lagInt_z')
#    print(Fore.BLACK + '')
#    print(lagInt_z)    
    
    sum=0
    for i in range(2,6):
        for j in range(2,6):
            for k in range(2,6):
                    sum = sum + output_data[i,j,k]*lagInt_x[i-2]*lagInt_y[j-2]*lagInt_z[k-2]  
                    print('sum',sum)
 
    
    return output_data