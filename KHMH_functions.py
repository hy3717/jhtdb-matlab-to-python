import pathlib
import tracemalloc
from itertools import chain
from giverny.turbulence_gizmos.getCutout import *
from giverny.turbulence_gizmos.getVelocityatPoint import *
from giverny.turbulence_gizmos.getVelocity_bigbox import *
from giverny.turbulence_gizmos.basic_gizmos import *
import colorama
from colorama import Fore

"""
retrieve a cutout of the isotropic cube.
"""
# define functions

# build 16*16*16 bucket
def formBucket(voxel_point_value, num_values_per_datapoint, Bucket_length = 16, ):
    group_voxel = np.zeros((num_values_per_datapoint,Bucket_length,Bucket_length,Bucket_length))
    #Change from ZYX to XYZ, using transpose .T
    group_voxel[:,0:8,0:8,0:8] = voxel_point_value[0].T
    group_voxel[:,8:16,0:8,0:8] = voxel_point_value[1].T
    group_voxel[:,0:8,8:16,0:8] = voxel_point_value[2].T
    group_voxel[:,8:16,8:16,0:8] = voxel_point_value[3].T
    group_voxel[:,0:8,0:8,8:16] = voxel_point_value[4].T
    group_voxel[:,8:16,0:8,8:16] = voxel_point_value[5].T
    group_voxel[:,0:8,8:16,8:16] = voxel_point_value[6].T
    group_voxel[:,8:16,8:16,8:16] = voxel_point_value[7].T
    
    return group_voxel

# find center coordinates of the bucket (e.g, (7.5 7.5 7.5) for the first bucket)
def findCenter(points, dx):
    center_point = np.array([points[0], points[1], points[2]])/dx%8
    for i in range(len(center_point)):
        if (center_point[i] < 3.5):
            center_point[i]=center_point[i]+8
    return center_point

def findCenter_gradient(points, dx):
    points = np.round(np.around(points/dx, decimals=1))
#    print('points',points)
    center_point = np.array([points[0], points[1], points[2]])%8
#    print('center_point',center_point)
    for i in range(len(center_point)):
        if (center_point[i] < 3.5):
            center_point[i]=center_point[i]+8
    return center_point


################################################################################
# Coefficients for Lagrange interpolation, 4,6,8th-order
################################################################################
def interpLag8L(p,u,LW,NB):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    fr = p-ix
    gx = LW[int(NB*fr[0])]
    gy = LW[int(NB*fr[1])]
    gz = LW[int(NB*fr[2])]
    #------------------------------------
    # create the 3D kernel from the 
    # outer product of the 1d kernels
    #------------------------------------
    gk = np.einsum('i,j,k',gx,gy,gz)
    #---------------------------------------
    # assemble the 8x8x8 cube and convolve
    #---------------------------------------
    d = u[:,ix[0]-3:ix[0]+5,ix[1]-3:ix[1]+5,ix[2]-3:ix[2]+5]
    #ui = np.einsum('ijk,ijk',gk,d)
    ui = np.einsum('ijk,lijk->l',gk,d)
    return ui

def interpLag8C(p,u):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    fr = p-ix
    gx = getLag8C(fr[0])
    gy = getLag8C(fr[1])
    gz = getLag8C(fr[2])
    #------------------------------------
    # create the 3D kernel from the 
    # outer product of the 1d kernels
    #------------------------------------
    gk = np.einsum('i,j,k',gx,gy,gz)
    #---------------------------------------
    # assemble the 8x8x8 cube and convolve
    #---------------------------------------
    d = u[:,ix[0]-3:ix[0]+5,ix[1]-3:ix[1]+5,ix[2]-3:ix[2]+5]
    #ui = np.einsum('ijk,ijk',gk,d)
    ui = np.einsum('ijk,lijk->l',gk,d)
    return ui

def getLag8C(fr):
    #------------------------------------------------------
    # get the 1D vectors for the 8 point Lagrange weights
    # inline the constants, and write explicit for loop
    # for the C compilation
    #------------------------------------------------------
    #cdef int n
    wN = [1.,-7.,21.,-35.,35.,-21.,7.,-1.]
    g  = np.array([0,0,0,1.,0,0,0,0])
    #----------------------------
    # calculate weights if fr>0, and insert into gg
    #----------------------------
    if (fr>0):
        s = 0
        for n in range(8):
            g[n] = wN[n]/(fr-n+3)
            s += g[n]
        for n in range(8):
            g[n] = g[n]/s
            
    return g

def getLag6C(fr):
    #------------------------------------------------------
    # get the 1D vectors for the 8 point Lagrange weights
    # inline the constants, and write explicit for loop
    # for the C compilation
    #------------------------------------------------------
    #cdef int n
    wN = [1.,-5.,10.,-10.,5.,-1.]
    g  = np.array([0,0,1.,0,0,0])
    #----------------------------
    # calculate weights if fr>0, and insert into gg
    #----------------------------
    if (fr>0):
        s = 0
        for n in range(6):
            g[n] = wN[n]/(fr-n+2)
            s += g[n]
        for n in range(6):
            g[n] = g[n]/s
            
    return g

def getLag4C(fr):
    #------------------------------------------------------
    # get the 1D vectors for the 8 point Lagrange weights
    # inline the constants, and write explicit for loop
    # for the C compilation
    #------------------------------------------------------
    #cdef int n
    wN = [1.,-3.,3.,-1.]
    g  = np.array([0,1.,0,0])
    #----------------------------
    # calculate weights if fr>0, and insert into gg
    #----------------------------
    if (fr>0):
        s = 0
        for n in range(4):
            g[n] = wN[n]/(fr-n+1)
            s += g[n]
        for n in range(4):
            g[n] = g[n]/s
            
    return g

def InterpNone(p,u):
    ix = p.astype('int')
    d = u[:,ix[0],ix[1],ix[2]]
    return d

################################################################################
# Coefficients for spline interpolation, M1Q4, M2Q8
################################################################################
def getSplineM1Q4(fr):
    poly_val = np.zeros(4)
    poly_val[0] = fr * (fr * (-1.0 / 2.0 * fr + 1) - 1.0 / 2.0)
    poly_val[1] = fr**2 * ((3.0 / 2.0) * fr - 5.0 / 2.0) + 1
    poly_val[2] = fr * (fr * (-3.0 / 2.0 * fr + 2) + 1.0 / 2.0)
    poly_val[3] = fr**2 * ((1.0 / 2.0) * fr - 1.0 / 2.0)
    return poly_val

def getSplineM2Q8(fr):
    poly_val = np.zeros(8)  
    poly_val[0] = fr * (fr * (fr * (fr * ((2.0 / 45.0) * fr - 7.0 / 60.0) + 1.0 / 12.0) + 1.0 / 180.0) - 1.0 / 60.0)
    poly_val[1] = fr * (fr * (fr * (fr * (-23.0 / 72.0 * fr + 61.0 / 72.0) - 217.0 / 360.0) - 3.0 / 40.0) + 3.0 / 20.0)
    poly_val[2] = fr * (fr * (fr * (fr * ((39.0 / 40.0) * fr - 51.0 / 20.0) + 63.0 / 40.0) + 3.0 / 4.0) - 3.0 / 4.0)
    poly_val[3] = fr**2 * (fr * (fr * (-59.0 / 36.0 * fr + 25.0 / 6.0) - 13.0 / 6.0) - 49.0 / 36.0) + 1
    poly_val[4] = fr * (fr * (fr * (fr * ((59.0 / 36.0) * fr - 145.0 / 36.0) + 17.0 / 9.0) + 3.0 / 4.0) + 3.0 / 4.0)
    poly_val[5] = fr * (fr * (fr * (fr * (-39.0 / 40.0 * fr + 93.0 / 40.0) - 9.0 / 8.0) - 3.0 / 40.0) - 3.0 / 20.0)
    poly_val[6] = fr * (fr * (fr * (fr * ((23.0 / 72.0) * fr - 3.0 / 4.0) + 49.0 / 120.0) + 1.0 / 180.0) + 1.0 / 60.0)
    poly_val[7] = fr**3 * (fr * (-2.0 / 45.0 * fr + 19.0 / 180.0) - 11.0 / 180.0)            
    return poly_val

################################################################################
# Coefficients for gradient (central differencing), 4,6,8th-order
################################################################################
def getNone_Fd4(dx): # 5 points
    CenteredFiniteDiffCoeff =  [(1.0 / 12.0 / dx, -2.0 / 3.0 / dx, 
                               0.0, 
                               2.0 / 3.0 / dx, -1.0 / 12.0 / dx)]           
    return CenteredFiniteDiffCoeff

def getNone_Fd6(dx): # 7 points
    CenteredFiniteDiffCoeff =  [(-1.0 / 60.0 / dx, 3.0 / 20.0 / dx, -3.0 / 4.0 / dx, 
                                0.0, 
                                3.0 / 4.0 / dx, -3.0 / 20.0 / dx, 1.0 / 60.0 / dx)]           
    return CenteredFiniteDiffCoeff

def getNone_Fd8(dx): # 9 points
    CenteredFiniteDiffCoeff =  [(1.0 / 280.0 / dx, -4.0 / 105.0 / dx, 1.0 / 5.0 / dx, -4.0 / 5.0 / dx, 
                                0.0, 
                                4.0 / 5.0 / dx, -1.0 / 5.0 / dx, 4.0 / 105.0 / dx, -1.0 / 280.0 / dx )]           
    return CenteredFiniteDiffCoeff

################################################################################
# Functions for gradient, 4,6,8th-order
################################################################################
def GradientNone_Fd4(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff = getNone_Fd4(dx)
    #---------------------------------------
    # assemble 5 points in each direction
    #---------------------------------------
    component_x = u[:,ix[0]-2:ix[0]+3,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-2:ix[1]+3,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-2:ix[2]+3]
    ui = np.inner(CenteredFiniteDiffCoeff,component_x)  
    uj = np.inner(CenteredFiniteDiffCoeff,component_y)
    uk = np.inner(CenteredFiniteDiffCoeff,component_z)
    return ui, uj, uk

def GradientNone_Fd6(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff = getNone_Fd6(dx)
    #---------------------------------------
     # assemble 7 points in each direction
    #---------------------------------------
    component_x = u[:,ix[0]-3:ix[0]+4,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-3:ix[1]+4,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-3:ix[2]+4]
    ui = np.inner(CenteredFiniteDiffCoeff,component_x)  
    uj = np.inner(CenteredFiniteDiffCoeff,component_y)
    uk = np.inner(CenteredFiniteDiffCoeff,component_z)
    return ui, uj, uk

def GradientNone_Fd8(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff = getNone_Fd8(dx)
    #---------------------------------------
    # assemble 9 points in each direction
    #---------------------------------------
    component_x = u[:,ix[0]-4:ix[0]+5,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-4:ix[1]+5,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-4:ix[2]+5]
    ui = np.inner(CenteredFiniteDiffCoeff,component_x)  
    uj = np.inner(CenteredFiniteDiffCoeff,component_y)
    uk = np.inner(CenteredFiniteDiffCoeff,component_z)
    return ui, uj, uk

################################################################################
# Coefficients for Hessian (central differencing), 4,6,8th-order
################################################################################
########## 4th-order
def getNone_Fd4_diagonal(dx): # 5 points
    CenteredFiniteDiffCoeff = [( -1.0 / 12.0 / dx / dx, 4.0 / 3.0 / dx / dx, -15.0 / 6.0 / dx / dx,
                                4.0 / 3.0 / dx / dx, -1.0 / 12.0 / dx / dx)]
    return CenteredFiniteDiffCoeff

def getNone_Fd4_offdiagonal(dx): # 8 points
    CenteredFiniteDiffCoeff = [(-1.0 / 48.0 / dx / dx,1.0 / 48.0 / dx / dx,-1.0 / 48.0 / dx / dx,1.0 / 48.0 / dx / dx,
                               1.0 / 3.0 / dx / dx, -1.0 / 3.0 / dx / dx,1.0 / 3.0 / dx / dx, -1.0 / 3.0 / dx / dx)]
    return CenteredFiniteDiffCoeff

########## 6th-order
def getNone_Fd6_diagonal(dx): # 7 points
    CenteredFiniteDiffCoeff = [( 1.0 / 90.0 / dx / dx, -3.0 / 20.0 / dx / dx, 3.0 / 2.0 / dx / dx,
                               -49.0 / 18.0 / dx / dx, 3.0 / 2.0 / dx / dx,-3.0 / 20.0 / dx / dx,
                               1.0 / 90.0 / dx / dx)]
    return CenteredFiniteDiffCoeff

def getNone_Fd6_offdiagonal(dx): # 12 points
    CenteredFiniteDiffCoeff = [(1.0 / 360.0 / dx / dx,-1.0 / 360.0 / dx / dx,1.0 / 360.0 / dx / dx,-1.0 / 360.0 / dx / dx,
                               -3.0 / 80.0 / dx / dx, 3.0 / 80.0 / dx / dx,-3.0 / 80.0 / dx / dx, 3.0 / 80.0 / dx / dx,
                               3.0 / 8.0 / dx / dx, -3.0 / 8.0/ dx / dx, 3.0 / 8.0/ dx / dx, -3.0 / 8.0/ dx / dx)]
    return CenteredFiniteDiffCoeff

########## 8th-order
def getNone_Fd8_diagonal(dx): # 9 points
    CenteredFiniteDiffCoeff = [( 9.0 / 3152.0 / dx / dx, -104.0 / 8865.0 / dx / dx, -207.0 / 2955.0 / dx / dx,
                               792.0 / 591.0 / dx / dx, -35777.0 / 14184.0 / dx / dx,792.0 / 591.0 / dx / dx,
                               -207.0 / 2955.0 / dx / dx, -104.0 / 8865.0 /dx /dx, 9.0 / 3152.0 /dx /dx)]
    return CenteredFiniteDiffCoeff

def getNone_Fd8_offdiagonal(dx): # 16 points
    CenteredFiniteDiffCoeff = [(-1.0 / 2240.0 / dx / dx,1.0 / 2240.0 / dx / dx,-1.0 / 2240.0 / dx / dx,1.0 / 2240.0 / dx / dx,
                               2.0 / 315.0 / dx / dx, -2.0 / 315.0 / dx / dx,2.0 / 315.0 / dx / dx, -2.0 / 315.0 / dx / dx,
                               -1.0 / 20.0/ dx / dx, 1.0 / 20.0/ dx / dx, -1.0 / 20.0/ dx / dx, 1.0 / 20.0/ dx / dx,
                               14.0 / 35.0/ dx / dx,-14.0 / 35.0/ dx / dx, 14.0 / 35.0/ dx / dx,-14.0 / 35.0/ dx / dx)]
    return CenteredFiniteDiffCoeff

################################################################################
# Functions for Hessian, 4,6,8th-order
################################################################################
def HessianNone_Fd4(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff_dia = getNone_Fd4_diagonal(dx)
    CenteredFiniteDiffCoeff_offdia = getNone_Fd4_offdiagonal(dx)
    #---------------------------------------
    # assemble the 5 points
    #---------------------------------------
    # diagnoal components
    component_x = u[:,ix[0]-2:ix[0]+3,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-2:ix[1]+3,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-2:ix[2]+3]
    uii = np.inner(CenteredFiniteDiffCoeff_dia,component_x)  
    ujj = np.inner(CenteredFiniteDiffCoeff_dia,component_y)
    ukk = np.inner(CenteredFiniteDiffCoeff_dia,component_z)
    # off-diagnoal components, 8 points for each direction
    component_xy = np.array([u[:,ix[0]+2,ix[1]+2,ix[2]],u[:,ix[0]+2,ix[1]-2,ix[2]],u[:,ix[0]-2,ix[1]-2,ix[2]],u[:,ix[0]-2,ix[1]+2,ix[2]],
                    u[:,ix[0]+1,ix[1]+1,ix[2]],u[:,ix[0]+1,ix[1]-1,ix[2]],u[:,ix[0]-1,ix[1]-1,ix[2]],u[:,ix[0]-1,ix[1]+1,ix[2]]])
    component_xz = np.array([u[:,ix[0]+2,ix[1],ix[2]+2],u[:,ix[0]+2,ix[1],ix[2]-2],u[:,ix[0]-2,ix[1],ix[2]-2],u[:,ix[0]-2,ix[1],ix[2]+2],
                    u[:,ix[0]+1,ix[1],ix[2]+1],u[:,ix[0]+1,ix[1],ix[2]-1],u[:,ix[0]-1,ix[1],ix[2]-1],u[:,ix[0]-1,ix[1],ix[2]+1]])
    component_yz = np.array([u[:,ix[0],ix[1]+2,ix[2]+2],u[:,ix[0],ix[1]+2,ix[2]-2],u[:,ix[0],ix[1]-2,ix[2]-2],u[:,ix[0],ix[1]-2,ix[2]+2],
                    u[:,ix[0],ix[1]+1,ix[2]+1],u[:,ix[0],ix[1]+1,ix[2]-1],u[:,ix[0],ix[1]-1,ix[2]-1],u[:,ix[0],ix[1]-1,ix[2]+1]])
    uij = np.inner(CenteredFiniteDiffCoeff_offdia,component_xy.T) 
    uik = np.inner(CenteredFiniteDiffCoeff_offdia,component_xz.T) 
    ujk = np.inner(CenteredFiniteDiffCoeff_offdia,component_yz.T) 
    return uii,uij,uik,ujj,ujk,ukk

def LaplacianNone_Fd4(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff_dia = getNone_Fd4_diagonal(dx)
    #---------------------------------------
    # assemble the 5 points
    #---------------------------------------
    # diagnoal components
    component_x = u[:,ix[0]-2:ix[0]+3,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-2:ix[1]+3,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-2:ix[2]+3]
    uii = np.inner(CenteredFiniteDiffCoeff_dia,component_x)  
    ujj = np.inner(CenteredFiniteDiffCoeff_dia,component_y)
    ukk = np.inner(CenteredFiniteDiffCoeff_dia,component_z)
    Total = uii + ujj + ukk
    return Total

def HessianNone_Fd6(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff_dia = getNone_Fd6_diagonal(dx)
    CenteredFiniteDiffCoeff_offdia = getNone_Fd6_offdiagonal(dx)
    #---------------------------------------
    # assemble the 7 points
    #---------------------------------------
    # diagnoal components
    component_x = u[:,ix[0]-3:ix[0]+4,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-3:ix[1]+4,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-3:ix[2]+4]
    uii = np.inner(CenteredFiniteDiffCoeff_dia,component_x)  
    ujj = np.inner(CenteredFiniteDiffCoeff_dia,component_y)
    ukk = np.inner(CenteredFiniteDiffCoeff_dia,component_z)
    # off-diagnoal components, 12 points for each direction
    component_xy = np.array([u[:,ix[0]+3,ix[1]+3,ix[2]],u[:,ix[0]+3,ix[1]-3,ix[2]],u[:,ix[0]-3,ix[1]-3,ix[2]],u[:,ix[0]-3,ix[1]+3,ix[2]],
                    u[:,ix[0]+2,ix[1]+2,ix[2]],u[:,ix[0]+2,ix[1]-2,ix[2]],u[:,ix[0]-2,ix[1]-2,ix[2]],u[:,ix[0]-2,ix[1]+2,ix[2]],
                    u[:,ix[0]+1,ix[1]+1,ix[2]],u[:,ix[0]+1,ix[1]-1,ix[2]],u[:,ix[0]-1,ix[1]-1,ix[2]],u[:,ix[0]-1,ix[1]+1,ix[2]]])
    component_xz = np.array([ u[:,ix[0]+3,ix[1],ix[2]+3],u[:,ix[0]+3,ix[1],ix[2]-3],u[:,ix[0]-3,ix[1],ix[2]-3],u[:,ix[0]-3,ix[1],ix[2]+3],
                    u[:,ix[0]+2,ix[1],ix[2]+2],u[:,ix[0]+2,ix[1],ix[2]-2],u[:,ix[0]-2,ix[1],ix[2]-2],u[:,ix[0]-2,ix[1],ix[2]+2],
                    u[:,ix[0]+1,ix[1],ix[2]+1],u[:,ix[0]+1,ix[1],ix[2]-1],u[:,ix[0]-1,ix[1],ix[2]-1],u[:,ix[0]-1,ix[1],ix[2]+1]])
    component_yz = np.array([u[:,ix[0],ix[1]+3,ix[2]+3],u[:,ix[0],ix[1]+3,ix[2]-3],u[:,ix[0],ix[1]-3,ix[2]-3],u[:,ix[0],ix[1]-3,ix[2]+3],
                    u[:,ix[0],ix[1]+2,ix[2]+2],u[:,ix[0],ix[1]+2,ix[2]-2],u[:,ix[0],ix[1]-2,ix[2]-2],u[:,ix[0],ix[1]-2,ix[2]+2],
                    u[:,ix[0],ix[1]+1,ix[2]+1],u[:,ix[0],ix[1]+1,ix[2]-1],u[:,ix[0],ix[1]-1,ix[2]-1],u[:,ix[0],ix[1]-1,ix[2]+1]])
    uij = np.inner(CenteredFiniteDiffCoeff_offdia,component_xy.T) 
    uik = np.inner(CenteredFiniteDiffCoeff_offdia,component_xz.T) 
    ujk = np.inner(CenteredFiniteDiffCoeff_offdia,component_yz.T) 
    return uii,uij,uik,ujj,ujk,ukk

def LaplacianNone_Fd6(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff_dia = getNone_Fd6_diagonal(dx)
    #---------------------------------------
    # assemble the 7 points
    #---------------------------------------
    # diagnoal components
    component_x = u[:,ix[0]-3:ix[0]+4,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-3:ix[1]+4,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-3:ix[2]+4]
    uii = np.inner(CenteredFiniteDiffCoeff_dia,component_x)  
    ujj = np.inner(CenteredFiniteDiffCoeff_dia,component_y)
    ukk = np.inner(CenteredFiniteDiffCoeff_dia,component_z)
    Total = uii + ujj + ukk
    return Total

def HessianNone_Fd8(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff_dia = getNone_Fd8_diagonal(dx)
    CenteredFiniteDiffCoeff_offdia = getNone_Fd8_offdiagonal(dx)
    #---------------------------------------
    # assemble the 9 points
    #---------------------------------------
    # diagnoal components
    component_x = u[:,ix[0]-4:ix[0]+5,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-4:ix[1]+5,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-4:ix[2]+5]
    uii = np.inner(CenteredFiniteDiffCoeff_dia,component_x)  
    ujj = np.inner(CenteredFiniteDiffCoeff_dia,component_y)
    ukk = np.inner(CenteredFiniteDiffCoeff_dia,component_z)
    # off-diagnoal components, 16 points for each direction
    component_xy = np.array([ u[:,ix[0]+4,ix[1]+4,ix[2]],u[:,ix[0]+4,ix[1]-4,ix[2]],u[:,ix[0]-4,ix[1]-4,ix[2]],u[:,ix[0]-4,ix[1]+4,ix[2]],
                    u[:,ix[0]+3,ix[1]+3,ix[2]],u[:,ix[0]+3,ix[1]-3,ix[2]],u[:,ix[0]-3,ix[1]-3,ix[2]],u[:,ix[0]-3,ix[1]+3,ix[2]],
                    u[:,ix[0]+2,ix[1]+2,ix[2]],u[:,ix[0]+2,ix[1]-2,ix[2]],u[:,ix[0]-2,ix[1]-2,ix[2]],u[:,ix[0]-2,ix[1]+2,ix[2]],
                    u[:,ix[0]+1,ix[1]+1,ix[2]],u[:,ix[0]+1,ix[1]-1,ix[2]],u[:,ix[0]-1,ix[1]-1,ix[2]],u[:,ix[0]-1,ix[1]+1,ix[2]]])
    component_xz = np.array([ u[:,ix[0]+4,ix[1],ix[2]+4],u[:,ix[0]+4,ix[1],ix[2]-4],u[:,ix[0]-4,ix[1],ix[2]-4],u[:,ix[0]-4,ix[1],ix[2]+4],
                    u[:,ix[0]+3,ix[1],ix[2]+3],u[:,ix[0]+3,ix[1],ix[2]-3],u[:,ix[0]-3,ix[1],ix[2]-3],u[:,ix[0]-3,ix[1],ix[2]+3],
                    u[:,ix[0]+2,ix[1],ix[2]+2],u[:,ix[0]+2,ix[1],ix[2]-2],u[:,ix[0]-2,ix[1],ix[2]-2],u[:,ix[0]-2,ix[1],ix[2]+2],
                    u[:,ix[0]+1,ix[1],ix[2]+1],u[:,ix[0]+1,ix[1],ix[2]-1],u[:,ix[0]-1,ix[1],ix[2]-1],u[:,ix[0]-1,ix[1],ix[2]+1]])
    component_yz = np.array([u[:,ix[0],ix[1]+4,ix[2]+4],u[:,ix[0],ix[1]+4,ix[2]-4],u[:,ix[0],ix[1]-4,ix[2]-4],u[:,ix[0],ix[1]-4,ix[2]+4],
                    u[:,ix[0],ix[1]+3,ix[2]+3],u[:,ix[0],ix[1]+3,ix[2]-3],u[:,ix[0],ix[1]-3,ix[2]-3],u[:,ix[0],ix[1]-3,ix[2]+3],
                    u[:,ix[0],ix[1]+2,ix[2]+2],u[:,ix[0],ix[1]+2,ix[2]-2],u[:,ix[0],ix[1]-2,ix[2]-2],u[:,ix[0],ix[1]-2,ix[2]+2],
                    u[:,ix[0],ix[1]+1,ix[2]+1],u[:,ix[0],ix[1]+1,ix[2]-1],u[:,ix[0],ix[1]-1,ix[2]-1],u[:,ix[0],ix[1]-1,ix[2]+1]])
    uij = np.inner(CenteredFiniteDiffCoeff_offdia,component_xy.T) 
    uik = np.inner(CenteredFiniteDiffCoeff_offdia,component_xz.T) 
    ujk = np.inner(CenteredFiniteDiffCoeff_offdia,component_yz.T) 
    return uii,uij,uik,ujj,ujk,ukk

def LaplacianNone_Fd8(p,u,dx):
    #--------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    #---------------------------------------------------------
    # get the coefficients
    #----------------------
    ix = p.astype('int')
    CenteredFiniteDiffCoeff_dia = getNone_Fd8_diagonal(dx)
    #---------------------------------------
    # assemble the 9 points
    #---------------------------------------
    # diagnoal components
    component_x = u[:,ix[0]-4:ix[0]+5,ix[1],ix[2]]
    component_y = u[:,ix[0],ix[1]-4:ix[1]+5,ix[2]]
    component_z = u[:,ix[0],ix[1],ix[2]-4:ix[2]+5]
    uii = np.inner(CenteredFiniteDiffCoeff_dia,component_x)  
    ujj = np.inner(CenteredFiniteDiffCoeff_dia,component_y)
    ukk = np.inner(CenteredFiniteDiffCoeff_dia,component_z)
    Total = uii + ujj + ukk
    return Total

# glue adjacent cornercode
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 512 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def factorspace(strt, nd, steps):
    fac = (nd/strt)**(1/(steps-1))
    result = np.zeros(steps)
    result[0] = strt
    result[-1] = nd
    for i in range(1,(steps-1)):
        result[i] = result[i-1]*fac
    return result

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])

def LandTvelocity_increments(r_vector, velocity_plus,velocity_minus,modes):
    normal_unit_vector = [0,1,0]
    r_unit_vector = np.zeros((1,3))
    r_unit_vector_vertical = np.zeros((1,3))
    r_unit_vector_perpendicular = np.zeros((1,3))
    u_l_plus = np.zeros(1)
    u_l_minus = np.zeros(1)
    u_t_plus = np.zeros(1)
    u_t_minus = np.zeros(1)
    du2_l = 0
    du2_t = 0
    for i in range(modes):
        r_unit_vector=r_vector[i]/(np.linalg.norm(r_vector[i]))
        #print('r_unit_vector[i]=',r_unit_vector[i],np.linalg.norm(r_unit_vector[i]))
        u_l_plus = np.dot(velocity_plus[i],r_unit_vector)
        u_l_minus = np.dot(velocity_minus[0],r_unit_vector)
        
        du2_l = du2_l + (u_l_plus - u_l_minus)**2/modes
        
        r_unit_vector_vertical = np.cross(r_unit_vector,normal_unit_vector)
        r_unit_vector_vertical = r_unit_vector_vertical/(np.linalg.norm(r_unit_vector_vertical))
        #print('r_unit_vector_vertical[i]=',r_unit_vector_vertical[i],np.linalg.norm(r_unit_vector_vertical[i]))
        
        r_unit_vector_perpendicular = np.cross(r_unit_vector,r_unit_vector_vertical)
        #print('r_unit_vector_perpendicular[i]=',r_unit_vector_perpendicular[i],np.linalg.norm(r_unit_vector_vertical[i]))
        
        u_t_plus  = np.dot(velocity_plus[i], r_unit_vector_vertical) 
        u_t_minus = np.dot(velocity_minus[0], r_unit_vector_vertical)
        
        du2_t = du2_t + (u_t_plus - u_t_minus)**2/modes/2
            
        u_t_plus = np.dot(velocity_plus[i], r_unit_vector_perpendicular)
        u_t_minus = np.dot(velocity_minus[0],r_unit_vector_perpendicular)

        du2_t = du2_t + (u_t_plus - u_t_minus)**2/modes/2
        
    return du2_l, du2_t

def findPath(cube, cornercode,var,timepoint):
    t = cube.cache[(cube.cache['minLim'] <= cornercode) & (cube.cache['maxLim'] >= cornercode)]
    t = t.iloc[0]
    db_minLim = t.minLim 
    db_maxLim = t.maxLim
    path = cube.filepaths[f'{t.ProductionDatabaseName}_{var}_{timepoint}']

    return db_minLim, db_maxLim, path
    
    
def GetVelocity_Noneint(cube, points, B, dx, num_values_per_datapoint, timepoint, bytes_per_datapoint,var):
    ###################################################################################
    # step one: build unique cornercodes dictionary
    ###################################################################################

    # find 8 corner points coordinates for each point
    All_eight_corner_points=[]
    for point in points:
        x_range = [int(point[0]-4), int(point[0]+4)] ##here is changed 9 points are needed for no interpolation
        y_range = [int(point[1]-4), int(point[1]+4)]
        z_range = [int(point[2]-4), int(point[2]+4)]
        axes_ranges = assemble_axis_data([x_range, y_range, z_range])    
        box = [list(axis_range) for axis_range in axes_ranges]        

        # the order here is how we build bucket later in 'def formbucket'
        full_corner_points=((box[0][0],box[1][0],box[2][0]),(box[0][1],box[1][0],box[2][0]),(box[0][0],box[1][1],box[2][0]),
                       (box[0][1],box[1][1],box[2][0]),(box[0][0],box[1][0],box[2][1]),(box[0][1],box[1][0],box[2][1]),
                       (box[0][0],box[1][1],box[2][1]),(box[0][1],box[1][1],box[2][1]))

        All_eight_corner_points.append(full_corner_points)

    All_eight_corner_points=tuple(All_eight_corner_points) 

    t2 = time.perf_counter()

    # find corresponding cornercodes of the 8 corner points for each point
    points_cornercodes_pool=[]
    points_cornercodes_temp = []
    points_cornercodes_all = []

    num_voxel = []

    for point in All_eight_corner_points:
        points_cornercodes_temp = []
        for pp in point:    
            datapoint = [p % cube.N for p in pp]
            point_cornercode, offset = cube.get_offset(datapoint)
            points_cornercodes_temp.append(point_cornercode)
        # the number of voxel is always 8 for each queried point so that we can build bucket using 'def formBucket' for any case,
        # For point only need one voxel e.g, (3.5,3.5,3.5), points_cornercodes_temp = [0,0,0,0,0,0,0,0], 
        # For point only need two voxel e.g, (7.5,3.5,3.5), points_cornercodes_temp = [0,512,0,512,0,512,0,512]
        # For point only need four voxel e.g, (7.5,7.5,3.5), points_cornercodes_temp = [0,512,1024,1536,0,512,1024,1536]
        # ...for the most of cases, it needs 8 voxels
        # points_cornercodes_temp store the order that build bucket in 'def formBucket' 
        num_voxel.append(len(points_cornercodes_temp))
        points_cornercodes_all.append(points_cornercodes_temp)

    # find unique cornercodes for the voxels pool
    points_cornercodes_pool = np.unique(list(chain.from_iterable(points_cornercodes_all)))

    t3 = time.perf_counter()

    # groupe cornercodes in a dictionary: {[path]:[db_minLim]:[cornercodes]}
    Voxels_in_pool = {}
    for i in range(len(points_cornercodes_pool)):
        cornercode = points_cornercodes_pool[i]
        if i==0:
            db_minLim, db_maxLim, path = findPath(cube, cornercode,var,timepoint)
            if path not in Voxels_in_pool:
                Voxels_in_pool[path] = {}
                if db_minLim not in Voxels_in_pool[path]:
                    Voxels_in_pool[path][db_minLim] = {}
                    Voxels_in_pool[path][db_minLim] = ([cornercode])
        else:
            if cornercode <= db_maxLim:
                Voxels_in_pool[path][db_minLim].append(cornercode)
            else:
                db_minLim, db_maxLim, path = findPath(cube, cornercode,var,timepoint)
                if path not in Voxels_in_pool:
                    Voxels_in_pool[path] = {}
                    if db_minLim not in Voxels_in_pool[path]:
                        Voxels_in_pool[path][db_minLim] = {}
                        Voxels_in_pool[path][db_minLim] = ([cornercode])

    t3_1 = time.perf_counter()

    # further groupe adjacent cornercodes in the dictionary: {[path]:[db_minLim]:([cornercodes range], number of voxels)}
    c = get_constants()
    for path in Voxels_in_pool:
        grouped_voxels_pool = []
        for db_minLim in Voxels_in_pool[path]:
            Sorted_voxels = np.sort(Voxels_in_pool[path][db_minLim])
            #glue cornercodes if they are adjacent
            group_voxels = ranges(Sorted_voxels)
            for group_voxel in group_voxels:
                # the number of glued voxels
                number_vox = (group_voxel[1] - group_voxel[0])/512 + 1
                grouped_voxels_pool.append([group_voxel, number_vox])

            #update Voxels_in_pool dictionary with glued voxels
            Voxels_in_pool[path][db_minLim]=grouped_voxels_pool    

    ###################################################################################
    # step two: reading voxel sequentially and build voxels pool
    ###################################################################################

    t3_2 = time.perf_counter()
    # Reading data sequentially
    voxel_value_pool = []
    for db_file in Voxels_in_pool:
        for db_minLim in Voxels_in_pool[db_file]:
            morton_voxels_to_read = Voxels_in_pool[db_file][db_minLim]
            for morton_data in morton_voxels_to_read:
                    morton_index_range = morton_data[0]
                    num_voxels = morton_data[1]

                    seek_distance = num_values_per_datapoint * bytes_per_datapoint * (morton_index_range[0] - db_minLim)                    

                    read_length = num_values_per_datapoint * int(num_voxels) * 512
                    
                    
#                     print('read_value',np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance))
#                     print('morton_index_range',morton_index_range)
#                     print('num_voxels',num_voxels)
                    
                    # read voxels and reshape to (number of voxels, 8, 8, 8) because we read voxels in chunk
                    voxel_value = np.reshape(np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance), 
                                             (int(num_voxels),8,8,8,num_values_per_datapoint))

                    # build voxel pools
                    voxel_value_pool.extend(voxel_value)

    
    ###################################################################################
    # step three: Building bucket for each point and calculate interpolation
    ###################################################################################    
    t4 = time.perf_counter() 
    # Calculate interpolation for each point
    ui = []
    # Loop over all queried points
    for j in range(len(points)):
        voxel_point_value = []
        Bucket = []
        center_point = []
        # num_voxel[j] is always 8, because we need 8 voxels
        for i in range(num_voxel[j]):
            # pull out correspoding 8 voxels from the voxels pool
            index = np.where(points_cornercodes_all[j][i]==points_cornercodes_pool)[0][0]
            voxel_point_value.append(voxel_value_pool[index]) 
        # build bucket for each queried point    
        Bucket = formBucket(voxel_point_value,num_values_per_datapoint,Bucket_length = 16)
        # need the center coordinate of the bucket (e.g, (7.5, 7.5, 7.5) for the first bucket) to calculate interpolation.
        center_point = findCenter_gradient(B[j], dx)
    #    ui.append(interpLag4C(center_point,Bucket))
    #    ui.append(interpSplineM2Q8(center_point,Bucket))
    #    ui.append(LaplacianNone_Fd8(center_point,Bucket,dx))
    #    ui.append(HessianNone_Fd8(center_point,Bucket,dx))
        ui.append(InterpNone(center_point,Bucket))
    #    ui.append(LaplacianNone_Fd8(center_point,Bucket,dx))
    #    ui.append(HessianNone_Fd8(center_point,Bucket,dx))
    #    ui.append(GradientNone_Fd8(center_point,Bucket,dx))

    t5 = time.perf_counter()
    # TIC = (t2-t1)
#     print('Find all 8 corners coordinates',(t2-t1),' sec')
#     print('Find unique cornercode',(t3-t2),' sec')
#     print('Load voxel pool',(t4-t3),' sec')
#     print('Load voxel pool_creat_empty_dictionary',(t3_1-t3),' sec')
#     print('Load voxel pool_creat_voxel_dictionary',(t3_2-t3_1),' sec')
#     print('Load voxel pool_read_from dictionary',(t4-t3_2),' sec')
#     print('Interpolation, Build Bucket + Calc',(t5-t4),' sec')
#     print('In total',(t5-t1),' sec')
    
    return ui
            
def GetGradient_Noneint(cube, points, B, dx, num_values_per_datapoint, timepoint, bytes_per_datapoint,var):
    ###################################################################################
    # step one: build unique cornercodes dictionary
    ###################################################################################

    # find 8 corner points coordinates for each point
    All_eight_corner_points=[]
    for point in points:
        x_range = [int(point[0]-4), int(point[0]+4)] ##here is changed 9 points are needed for no interpolation
        y_range = [int(point[1]-4), int(point[1]+4)]
        z_range = [int(point[2]-4), int(point[2]+4)]
        axes_ranges = assemble_axis_data([x_range, y_range, z_range])    
        box = [list(axis_range) for axis_range in axes_ranges]        

        # the order here is how we build bucket later in 'def formbucket'
        full_corner_points=((box[0][0],box[1][0],box[2][0]),(box[0][1],box[1][0],box[2][0]),(box[0][0],box[1][1],box[2][0]),
                       (box[0][1],box[1][1],box[2][0]),(box[0][0],box[1][0],box[2][1]),(box[0][1],box[1][0],box[2][1]),
                       (box[0][0],box[1][1],box[2][1]),(box[0][1],box[1][1],box[2][1]))

        All_eight_corner_points.append(full_corner_points)

    All_eight_corner_points=tuple(All_eight_corner_points) 

    t2 = time.perf_counter()

    # find corresponding cornercodes of the 8 corner points for each point
    points_cornercodes_pool=[]
    points_cornercodes_temp = []
    points_cornercodes_all = []

    num_voxel = []

    for point in All_eight_corner_points:
        points_cornercodes_temp = []
        for pp in point:    
            datapoint = [p % cube.N for p in pp]
            point_cornercode, offset = cube.get_offset(datapoint)
            points_cornercodes_temp.append(point_cornercode)
        # the number of voxel is always 8 for each queried point so that we can build bucket using 'def formBucket' for any case,
        # For point only need one voxel e.g, (3.5,3.5,3.5), points_cornercodes_temp = [0,0,0,0,0,0,0,0], 
        # For point only need two voxel e.g, (7.5,3.5,3.5), points_cornercodes_temp = [0,512,0,512,0,512,0,512]
        # For point only need four voxel e.g, (7.5,7.5,3.5), points_cornercodes_temp = [0,512,1024,1536,0,512,1024,1536]
        # ...for the most of cases, it needs 8 voxels
        # points_cornercodes_temp store the order that build bucket in 'def formBucket' 
        num_voxel.append(len(points_cornercodes_temp))
        points_cornercodes_all.append(points_cornercodes_temp)

    # find unique cornercodes for the voxels pool
    points_cornercodes_pool = np.unique(list(chain.from_iterable(points_cornercodes_all)))

    t3 = time.perf_counter()

    # groupe cornercodes in a dictionary: {[path]:[db_minLim]:[cornercodes]}
    Voxels_in_pool = {}
    for i in range(len(points_cornercodes_pool)):
        cornercode = points_cornercodes_pool[i]
        if i==0:
            db_minLim, db_maxLim, path = findPath(cube, cornercode,var,timepoint)
            if path not in Voxels_in_pool:
                Voxels_in_pool[path] = {}
                if db_minLim not in Voxels_in_pool[path]:
                    Voxels_in_pool[path][db_minLim] = {}
                    Voxels_in_pool[path][db_minLim] = ([cornercode])
        else:
            if cornercode <= db_maxLim:
                Voxels_in_pool[path][db_minLim].append(cornercode)
            else:
                db_minLim, db_maxLim, path = findPath(cube, cornercode,var,timepoint)
                if path not in Voxels_in_pool:
                    Voxels_in_pool[path] = {}
                    if db_minLim not in Voxels_in_pool[path]:
                        Voxels_in_pool[path][db_minLim] = {}
                        Voxels_in_pool[path][db_minLim] = ([cornercode])

    t3_1 = time.perf_counter()

    # further groupe adjacent cornercodes in the dictionary: {[path]:[db_minLim]:([cornercodes range], number of voxels)}
    c = get_constants()
    for path in Voxels_in_pool:
        grouped_voxels_pool = []
        for db_minLim in Voxels_in_pool[path]:
            Sorted_voxels = np.sort(Voxels_in_pool[path][db_minLim])
            #glue cornercodes if they are adjacent
            group_voxels = ranges(Sorted_voxels)
            for group_voxel in group_voxels:
                # the number of glued voxels
                number_vox = (group_voxel[1] - group_voxel[0])/512 + 1
                grouped_voxels_pool.append([group_voxel, number_vox])

            #update Voxels_in_pool dictionary with glued voxels
            Voxels_in_pool[path][db_minLim]=grouped_voxels_pool    

    ###################################################################################
    # step two: reading voxel sequentially and build voxels pool
    ###################################################################################

    t3_2 = time.perf_counter()
    # Reading data sequentially
    voxel_value_pool = []
    for db_file in Voxels_in_pool:
        for db_minLim in Voxels_in_pool[db_file]:
            morton_voxels_to_read = Voxels_in_pool[db_file][db_minLim]
            for morton_data in morton_voxels_to_read:
                    morton_index_range = morton_data[0]
                    num_voxels = morton_data[1]

                    seek_distance = num_values_per_datapoint * bytes_per_datapoint * (morton_index_range[0] - db_minLim)                    

                    read_length = num_values_per_datapoint * int(num_voxels) * 512
                    
                    
#                     print('read_value',np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance))
#                     print('morton_index_range',morton_index_range)
#                     print('num_voxels',num_voxels)
                    
                    # read voxels and reshape to (number of voxels, 8, 8, 8) because we read voxels in chunk
                    voxel_value = np.reshape(np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance), 
                                             (int(num_voxels),8,8,8,num_values_per_datapoint))

                    # build voxel pools
                    voxel_value_pool.extend(voxel_value)

    
    ###################################################################################
    # step three: Building bucket for each point and calculate interpolation
    ###################################################################################    
    t4 = time.perf_counter() 
    # Calculate interpolation for each point
    ui = []
    # Loop over all queried points
    for j in range(len(points)):
        voxel_point_value = []
        Bucket = []
        center_point = []
        # num_voxel[j] is always 8, because we need 8 voxels
        for i in range(num_voxel[j]):
            # pull out correspoding 8 voxels from the voxels pool
            index = np.where(points_cornercodes_all[j][i]==points_cornercodes_pool)[0][0]
            voxel_point_value.append(voxel_value_pool[index]) 
        # build bucket for each queried point    
        Bucket = formBucket(voxel_point_value,num_values_per_datapoint,Bucket_length = 16)
        # need the center coordinate of the bucket (e.g, (7.5, 7.5, 7.5) for the first bucket) to calculate interpolation.
        center_point = findCenter_gradient(B[j], dx)
    #    ui.append(interpLag4C(center_point,Bucket))
    #    ui.append(interpSplineM2Q8(center_point,Bucket))
    #    ui.append(LaplacianNone_Fd8(center_point,Bucket,dx))
    #    ui.append(HessianNone_Fd8(center_point,Bucket,dx))
    #    ui.append(InterpNone(center_point,Bucket))
    #    ui.append(LaplacianNone_Fd8(center_point,Bucket,dx))
    #    ui.append(HessianNone_Fd8(center_point,Bucket,dx))
        ui.append(GradientNone_Fd4(center_point,Bucket,dx))

    t5 = time.perf_counter()
    # TIC = (t2-t1)
#     print('Find all 8 corners coordinates',(t2-t1),' sec')
#     print('Find unique cornercode',(t3-t2),' sec')
#     print('Load voxel pool',(t4-t3),' sec')
#     print('Load voxel pool_creat_empty_dictionary',(t3_1-t3),' sec')
#     print('Load voxel pool_creat_voxel_dictionary',(t3_2-t3_1),' sec')
#     print('Load voxel pool_read_from dictionary',(t4-t3_2),' sec')
#     print('Interpolation, Build Bucket + Calc',(t5-t4),' sec')
#     print('In total',(t5-t1),' sec')
    
    return ui

def Lag_looKuptable_8(NB):
    frac = np.linspace(0,1-1/NB,NB)
    LW = []
    for fp in frac:
        LW.append(getLag8C(fp))
        
    return LW

def GetVelocity_Lag8int(cube, points, B, dx, num_values_per_datapoint, timepoint, bytes_per_datapoint, var, LW, NB):
    All_eight_corner_points=[]
    for point in points:
        x_range = [int(point[0]-3), int(point[0]+4)] ##here is changed 9 points are needed for no interpolation
        y_range = [int(point[1]-3), int(point[1]+4)]
        z_range = [int(point[2]-3), int(point[2]+4)]
        axes_ranges = assemble_axis_data([x_range, y_range, z_range])    
        box = [list(axis_range) for axis_range in axes_ranges]        

        # the order here is how we build bucket later in 'def formbucket'
        full_corner_points=((box[0][0],box[1][0],box[2][0]),(box[0][1],box[1][0],box[2][0]),(box[0][0],box[1][1],box[2][0]),
                       (box[0][1],box[1][1],box[2][0]),(box[0][0],box[1][0],box[2][1]),(box[0][1],box[1][0],box[2][1]),
                       (box[0][0],box[1][1],box[2][1]),(box[0][1],box[1][1],box[2][1]))

        All_eight_corner_points.append(full_corner_points)

    All_eight_corner_points=tuple(All_eight_corner_points) 

    t2 = time.perf_counter()

    # find corresponding cornercodes of the 8 corner points for each point
    points_cornercodes_pool=[]
    points_cornercodes_temp = []
    points_cornercodes_all = []

    num_voxel = []

    for point in All_eight_corner_points:
        points_cornercodes_temp = []
        for pp in point:    
            datapoint = [p % cube.N for p in pp]
            point_cornercode, offset = cube.get_offset(datapoint)
            points_cornercodes_temp.append(point_cornercode)
        # the number of voxel is always 8 for each queried point so that we can build bucket using 'def formBucket' for any case,
        # For point only need one voxel e.g, (3.5,3.5,3.5), points_cornercodes_temp = [0,0,0,0,0,0,0,0], 
        # For point only need two voxel e.g, (7.5,3.5,3.5), points_cornercodes_temp = [0,512,0,512,0,512,0,512]
        # For point only need four voxel e.g, (7.5,7.5,3.5), points_cornercodes_temp = [0,512,1024,1536,0,512,1024,1536]
        # ...for the most of cases, it needs 8 voxels
        # points_cornercodes_temp store the order that build bucket in 'def formBucket' 
        num_voxel.append(len(points_cornercodes_temp))
        points_cornercodes_all.append(points_cornercodes_temp)

    # find unique cornercodes for the voxels pool
    points_cornercodes_pool = np.unique(list(chain.from_iterable(points_cornercodes_all)))

    t3 = time.perf_counter()

    # groupe cornercodes in a dictionary: {[path]:[db_minLim]:[cornercodes]}
    Voxels_in_pool = {}
    for i in range(len(points_cornercodes_pool)):
        cornercode = points_cornercodes_pool[i]
        if i==0:
            db_minLim, db_maxLim, path = findPath(cube, cornercode,var,timepoint)
            if path not in Voxels_in_pool:
                Voxels_in_pool[path] = {}
                if db_minLim not in Voxels_in_pool[path]:
                    Voxels_in_pool[path][db_minLim] = {}
                    Voxels_in_pool[path][db_minLim] = ([cornercode])
        else:
            if cornercode <= db_maxLim:
                Voxels_in_pool[path][db_minLim].append(cornercode)
            else:
                db_minLim, db_maxLim, path = findPath(cube, cornercode,var,timepoint)
                if path not in Voxels_in_pool:
                    Voxels_in_pool[path] = {}
                    if db_minLim not in Voxels_in_pool[path]:
                        Voxels_in_pool[path][db_minLim] = {}
                        Voxels_in_pool[path][db_minLim] = ([cornercode])

    t3_1 = time.perf_counter()

    # further groupe adjacent cornercodes in the dictionary: {[path]:[db_minLim]:([cornercodes range], number of voxels)}
    c = get_constants()
    for path in Voxels_in_pool:
        grouped_voxels_pool = []
        for db_minLim in Voxels_in_pool[path]:
            Sorted_voxels = np.sort(Voxels_in_pool[path][db_minLim])
            #glue cornercodes if they are adjacent
            group_voxels = ranges(Sorted_voxels)
            for group_voxel in group_voxels:
                # the number of glued voxels
                number_vox = (group_voxel[1] - group_voxel[0])/512 + 1
                grouped_voxels_pool.append([group_voxel, number_vox])

            #update Voxels_in_pool dictionary with glued voxels
            Voxels_in_pool[path][db_minLim]=grouped_voxels_pool    

    ###################################################################################
    # step two: reading voxel sequentially and build voxels pool
    ###################################################################################

    t3_2 = time.perf_counter()
    # Reading data sequentially
    voxel_value_pool = []
    for db_file in Voxels_in_pool:
        for db_minLim in Voxels_in_pool[db_file]:
            morton_voxels_to_read = Voxels_in_pool[db_file][db_minLim]
            for morton_data in morton_voxels_to_read:
                    morton_index_range = morton_data[0]
                    num_voxels = morton_data[1]

                    seek_distance = num_values_per_datapoint * bytes_per_datapoint * (morton_index_range[0] - db_minLim)                    

                    read_length = num_values_per_datapoint * int(num_voxels) * 512

                    # read voxels and reshape to (number of voxels, 8, 8, 8) because we read voxels in chunk
                    voxel_value = np.reshape(np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance), (int(num_voxels),8,8,8,num_values_per_datapoint))

                    # build voxel pools
                    voxel_value_pool.extend(voxel_value)


    ###################################################################################
    # step three: Building bucket for each point and calculate interpolation
    ###################################################################################    
    t4 = time.perf_counter() 
    # Calculate interpolation for each point
    ui = []
    # Loop over all queried points
    for j in range(len(points)):
        voxel_point_value = []
        Bucket = []
        center_point = []
        # num_voxel[j] is always 8, because we need 8 voxels
        for i in range(num_voxel[j]):
            # pull out correspoding 8 voxels from the voxels pool
            index = np.where(points_cornercodes_all[j][i]==points_cornercodes_pool)[0][0]
            voxel_point_value.append(voxel_value_pool[index]) 
        # build bucket for each queried point    
        Bucket = formBucket(voxel_point_value,num_values_per_datapoint,Bucket_length = 16)
        # need the center coordinate of the bucket (e.g, (7.5, 7.5, 7.5) for the first bucket) to calculate interpolation.
    #    center_point = findCenter_gradient(B[j], dx)
        center_point = findCenter(B[j], dx)
    #    ui.append(interpLag8C(center_point,Bucket))
        ui.append(interpLag8L(center_point,Bucket,LW,NB))
    #     ui.append(interpSplineM2Q8(center_point,Bucket))
    #     ui.append(interpSplineM2Q8L(center_point,Bucket,LW,NB))

    
    return ui