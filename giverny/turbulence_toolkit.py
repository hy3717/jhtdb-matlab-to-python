import pathlib
import tracemalloc
from giverny.turbulence_gizmos.getCutout import *
from giverny.turbulence_gizmos.getVelocityatPoint import *
from giverny.turbulence_gizmos.getVelocity_bigbox import *
from giverny.turbulence_gizmos.basic_gizmos import *
import colorama
from colorama import Fore

"""
retrieve a cutout of the isotropic cube.
"""
def getCutout(cube, cube_title,
              output_path,
              axes_ranges_original, strides, var_original, timepoint_original,
              trace_memory = False):
    # housekeeping procedures.
    # -----
    output_path, var, axes_ranges, timepoint, cube_resolution = \
        housekeeping_procedures(cube_title, output_path, axes_ranges_original, strides, var_original, timepoint_original)
    
    # process the data.
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
        
#    print(Fore.RED + 'axes_ranges in getCutout')
#    print(Fore.BLACK + '')
#    print(axes_ranges)
    
#    print(Fore.RED + 'axes_ranges_original in getCutout')
#    print(Fore.BLACK + '')
#    print(axes_ranges_original)
                
    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = getCutout_process_data(cube, cube_resolution, cube_title, output_path,
                                         axes_ranges, var, timepoint,
                                         axes_ranges_original, strides, var_original, timepoint_original)
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return output_data

"""
complete all of the housekeeping procedures before data processing.
    - convert 1-based axes ranges to 0-based.
    - format the variable name and get the variable identifier.
    - convert 1-based timepoint to 0-based.
    - format the output path using pathlib and create the output folder directory.
"""
def housekeeping_procedures(cube_title,
                            output_path,
                            axes_ranges_original, strides, var_original, timepoint_original):
    
    #print(Fore.RED + 'start housekeeping_procedures!!!')
    #print(Fore.BLACK + '')
    #print(self.cache)
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, cube_title)
    # check that the user-input x-, y-, and z-axis ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(axes_ranges_original)
    # check that the user-input strides are all positive integers.
    check_strides(strides)
    
    # pre-processing steps.
    # -----
    # get the number of datapoints (resolution) along each axis of the isotropic cube.
    cube_resolution = get_dataset_resolution(cube_title)
    
    # converts the 1-based axes ranges above to 0-based axes ranges, and truncates the ranges if they are longer than cube_resolution since 
    # the boundaries are periodic. output_data will be filled in with the duplicate data for the truncated data points after processing
    # so that the data files are not read redundantly.
    axes_ranges = convert_to_0_based_ranges(axes_ranges_original, cube_resolution)

    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = convert_to_0_based_value(timepoint_original)
    
    # create the output_path folder if it does not already exist and make sure output_path is formatted properly.
    output_path = pathlib.Path(output_path)
    create_output_folder(output_path)
    
    return (output_path, var, axes_ranges, timepoint, cube_resolution)

def getVelocityatPoint(X,Y,Z,dn_x, dx, cube,
              var_original, timepoint,Lagorder):
    # housekeeping procedures.
    # -----
#    output_path, var, axes_ranges, timepoint, cube_resolution = \
#        housekeeping_procedures(cube_title, output_path, axes_ranges_original, var_original, timepoint_original)
    
#    axes_ranges = convert_to_0_based_ranges(axes_ranges_original, cube_resolution)

    node_x = int(math.floor(X/dx))
    node_y = int(math.floor(Y/dx))
    node_z = int(math.floor(Z/dx))

#    print(Fore.RED + 'N_x,y,z, integer index')
#    print(Fore.BLACK + '')
#    print(node_x,node_y,node_z)
    
    #from 0 to 7 for each voxel
    x_range = [int(node_x-1), int(node_x+2)]
    y_range = [int(node_y-1), int(node_y+2)]
    z_range = [int(node_z-1), int(node_z+2)]
        
    axes_ranges = assemble_axis_data([x_range, y_range, z_range]) 
    
    axes_ranges = axes_ranges
    
    timepoint = timepoint -1  #start from 0
    
    var = get_variable_identifier(var_original)
        
#    print(Fore.RED + 'axes_ranges in getVelocityatPoint')
#    print(Fore.BLACK + '')
#    print(axes_ranges)
    
                
    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = getVelocity_process_data(X,Y,Z,dx,node_x,node_y,node_z,cube,
                                           axes_ranges, var, timepoint)

#    print(Fore.RED + 'X,Y,Z,dx')
#    print(Fore.BLACK + '')
#    print(X,Y,Z,dx)
    
   
    
    return output_data

def getVelocity(points,npoints,dn_x, dx, cube,
              var_original,timepoint,Lagorder):
    
    if var_original == 'velocity':
        output_data = np.zeros((npoints,3))
    elif var_original == 'pressure':
        output_data = np.zeros((npoints,1))
        
    for i in range(0,npoints):
        output_data[i] = getVelocityatPoint(points[i,0],points[i,1],points[i,2],dn_x,dx,cube,
                                         var_original, timepoint,Lagorder)
   
    return output_data

def getVelocity_V2(points,npoints,dn_x, dx, cube,
              var_original,timepoint,Lagorder):
    
    if var_original == 'velocity':
        output_data = np.zeros((npoints,3))
    elif var_original == 'pressure':
        output_data = np.zeros((npoints,1))
       
    min_node_x = int(math.floor(min(points[:,0])/dx))
    max_node_x = int(math.floor(max(points[:,0])/dx))
    min_node_y = int(math.floor(min(points[:,1])/dx))
    max_node_y = int(math.floor(max(points[:,1])/dx))
    min_node_z = int(math.floor(min(points[:,2])/dx))
    max_node_z = int(math.floor(max(points[:,2])/dx))
   
    x_range = [int(min_node_x-1), int(max_node_x+2)]
    y_range = [int(min_node_y-1), int(max_node_y+2)]
    z_range = [int(min_node_z-1), int(max_node_z+2)]
    
    axes_ranges = assemble_axis_data([x_range, y_range, z_range]) 
    
    axes_ranges = axes_ranges
    
    timepoint = timepoint -1  #start from 0
    
    var = get_variable_identifier(var_original)
    
    print('min_xyz,max_xyz',min_node_x,max_node_x,min_node_y,max_node_y,min_node_z,max_node_z)
    print('x_range,y_range,z_range',x_range,y_range,z_range)
    #output_data = getVelocity_bigbox(points,dn_x,dx,cube,
    #                                 var_original, timepoint,Lagorder)
   
   
    output_data=getVelocity_process_bigbox(npoints,points,dx,min_node_x,min_node_y,min_node_z,max_node_x,max_node_y,max_node_z,cube,
                           axes_ranges, var, timepoint)
    
    return output_data

def getVelocity_V3(points,npoints,dn_x, dx, cube,
              var_original,timepoint,Lagorder):
    
    if var_original == 'velocity':
        output_data = np.zeros((npoints,3))
    elif var_original == 'pressure':
        output_data = np.zeros((npoints,1))
   
    print('points', points[0],points[1],points[2])

    Z_index = cube.mortoncurve.pack(points[0],points[1],points[2])
    
    print('Z_index=', Z_index)
     
        
    
    return output_data


