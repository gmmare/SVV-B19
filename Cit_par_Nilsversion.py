from math import *
import numpy as np
from control.matlab import *
import matplotlib.pyplot as plt
import Reference_data_reader_num_model 
# Citation 550 - Linear simulation

# xcg = 0.25 * c

# Stationary flight condition

Start_hour_Phugoid, Start_min_Phugoid, Start_sec_Phugoid=
Start_hour_DR, Start_min_DR, Start_sec_DR=
Start_hour_SP, Start_min_SP Start_sec_SP=

test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, delta_a, delta_r, delta_e=Reference_data_reader_num_model.get_lists(tas,alt,pitch,AOA,d_a,d_r,d_e,t) #gets the list of all avriables irrespective of time

a='Phugoid' #Phugoid, DR, SP

if a='Phugoid':
    V0,hp0,th0,alpha0,PR,inputs_de=Reference_data_reader_num_model.get_Phugoid(test_list_tas, test_list_alt, theta_list, angle_of_attack_list, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec)
    hours,minu,sec=Start_hour_Phugoid, Start_min_Phugoid, Start_sec_Phugoid
elif a='DR':    
    V0,hp0,th0,alpha0,PR,inputs_da,inputs_dr=Reference_data_reader_num_model.get_DR(test_list_tas, test_list_alt, theta_list, angle_of_attack_list, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec)
    hours,minu,sec=Start_hour_DR, Start_min_DR, Start_sec_DR
elif a='SP':
    V0,hp0,th0,alpha0,PR,inputs_de=Reference_data_reader_num_model.get_SP(test_list_tas, test_list_alt, theta_list, angle_of_attack_list, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec)
    hours,minu,sec=Start_hour_SP, Start_min_SP, Start_sec_SP

#hp0: pressure altitude in the stationary flight condition [m]
#V0: true airspeed in the stationary flight condition [m/sec]
#alpha0: angle of attack in the stationary flight condition [rad]
#th0: pitch angle in the stationary flight condition [rad]

# Aircraft mass
m      =      Reference_data_reader_num_model.get_mass(hours,minu,sec)   # mass [kg] #mass at 30 min

# aerodynamic properties
e      =   0.8          # Oswald factor [ ]
CD0    =   0.04         # Zero lift drag coefficient [ ]
CLa    =   5.084          # Slope of CL-alpha curve [ ]

# Longitudinal stability
Cma    = -0.5626            # longitudinal stabilty [ ]
Cmde   =  -1.1642          # elevator effectiveness [ ]

# Aircraft geometry

S      = 30.00	          # wing area [m^2]
Sh     = 0.2 * S         # stabiliser area [m^2]
Sh_S   = Sh / S	          # [ ]
lh     = 0.71 * 5.968    # tail length [m]
c      = 2.0569	          # mean aerodynamic cord [m]
lh_c   = lh / c	          # [ ]
b      = 15.911	          # wing span [m]
bh     = 5.791	          # stabilser span [m]
A      = b ** 2 / S      # wing aspect ratio [ ]
Ah     = bh ** 2 / Sh    # stabilser aspect ratio [ ]
Vh_V   = 100.11566510739169 	          # [ ]
ih     = -2 * pi / 180   # stabiliser angle of incidence [rad]

# Constant values concerning atmosphere and gravity

rho0   = 1.2250          # air density at sea level [kg/m^3] 
lambd = -0.0065         # temperature gradient in ISA [K/m]
Temp0  = 288.15          # temperature at sea level in ISA [K]
R      = 287.05          # specific gas constant [m^2/sec^2K]
g      = 9.81            # [m/sec^2] (gravity constant)

# air density [kg/m^3]  
rho    = rho0 * pow( ((1+(lambd * hp0 / Temp0))), (-((g / (lambd*R)) + 1)))
W      = m * g            # [N]       (aircraft weight)

# Constant values concerning aircraft inertia

muc    = m / (rho * S * c)
mub    = m / (rho * S * b)
KX2    = 0.019
KZ2    = 0.042
KXZ    = 0.002
KY2    = 1.25 * 1.114

# Aerodynamic constants

Cmac   = 0                      # Moment coefficient about the aerodynamic centre [ ]
CNwa   = CLa                    # Wing normal force slope [ ]
CNha   = 2 * pi * Ah / (Ah + 2) # Stabiliser normal force slope [ ]
depsda = 4 / (A + 2)            # Downwash gradient [ ]

# Lift and drag coefficient

CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]

# Stabiblity derivatives

CX0    = W * sin(th0) / (0.5 * rho * V0 ** 2 * S)
CXu    = -0.02792
CXa    = +0.47966		# Positive! (has been erroneously negative since 1993) 
CXadot = +0.08330
CXq    = -0.28170
CXde   = -0.03728

CZ0    = -W * cos(th0) / (0.5 * rho * V0 ** 2 * S)
CZu    = -0.37616
CZa    = -5.74340
CZadot = -0.00350
CZq    = -5.66290
CZde   = -0.69612

Cmu    = +0.06990
Cmadot = +0.17800
Cmq    = -8.79415

CYb    = -0.7500
CYbdot =  0     
CYp    = -0.0304
CYr    = +0.8495
CYda   = -0.0400
CYdr   = +0.2300

Clb    = -0.10260
Clp    = -0.71085
Clr    = +0.23760
Clda   = -0.23088
Cldr   = +0.03440

Cnb    =  +0.1348
Cnbdot =   0     
Cnp    =  -0.0602
Cnr    =  -0.2061
Cnda   =  -0.0120
Cndr   =  -0.0939


#Symmetric

C1 = np.array([[-2*muc*c/V0**2, 0, 0, 0],
      [0, (CZa-2*muc)*c/V0, 0, 0],
      [0, 0, -c/V0, 0],
      [0, Cmadot*c/V0, 0, -2*muc*KY2*(c/V0)**2]])
C2 = np.array([[CXu/V0, CXa, CZ0, CXq*c/V0],
      [CZu/V0, CZa, -CX0, c/V0*(CZq+2*muc)],
      [0, 0, 0, c/V0],
      [Cmu/V0, Cma, 0, Cmq*c/V0]])
C3 = np.array([[CXde], [CZde], [0], [Cmde]])
C_extra=C2+C1

A = (-np.linalg.inv(C1)).dot(C2)
B = (-np.linalg.inv(C1)).dot(C3)
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
D = (np.linalg.inv(C_extra)).dot(C3)


sys=ss(A,B,C,D)


#A-Symmetric (_a)
C1_a = np.array([[(CYbdot-2*mub)*b/V0, 0, 0, 0],
               [0, -0.5*b/V0, 0, 0],
               [0, 0, -2*mub*KX2*(b/V0)**2, 2*mub*KXZ*(b/V0)**2],
               [Cnbdot*b/V0, 0, 2*mub*KXZ*(b/V0)**2, -2*mub*KZ2*(b/V0)**2]])
C2_a = np.array([[CYb, CL, CYp*b/(2*V0), (CYr-4*mub)*b/(2*V0)],
               [0, 0, (1-b/(2*V0)), 0],
               [Clb, 0, Clp*b/(2*V0), Clr*b/(2*V0)],
               [Cnb, 0, Cnp*b/(2*V0), Cnr*b/(2*V0)]])
C3_a = np.array([[CYda, CYdr],
               [0, 0],
               [Clda, Cldr],
               [Cnda, Cndr]])

C_extra_a=C2_a+C1_a


A_a = (-np.linalg.inv(C1_a)).dot(C2_a)
B_a = (-np.linalg.inv(C1_a)).dot(C3_a)
C_a = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
D_a =(np.linalg.inv(C_extra_a)).dot(C3)


if a='Phugoid' or a='SP':
    #Response for symmetric flight
    x0=np.matrix([[V0],[alpha0], [th0], [PR]]) 
    t=np.arange(0.0,10*len(inputs_da)-0.1,0.1) 
    sys=ss(A,B,C,D)
    
    #Defining inputs
    u1 = np.sin(t*w1) 
    y1=initial(sys,t,x0)  
        
    y1,tdum,xdum=lsim(sys,U=u1,T=t)
    plt.plot(t,speed)
    plt.show()

    print(np.linalg.eigvals(A))

elif a='DR':
    #Response for symmetric flight
    x0=np.matrix([[V0],[alpha0], [th0], [PR]]) 
    t=np.arange(0.0,10*len(inputs_da)-0.1,0.1) 
    sys=ss(A_a,B_a,C_a,D_a)
    
    #Defining inputs
    u1 = inputs_da
    y1=initial(sys,t,x0)  
        
    y1,tdum,xdum=lsim(sys,U=u1,T=t)
    plt.plot(t,speed)
    plt.show()

#eigen values


#phugoid
'''
(array([-1.28404348+1.89207559j, -1.28404348-1.89207559j,
        0.00547665+0.13477211j,  0.00547665-0.13477211j])

#Dutch Roll
(array([-4.25400348+0.j        , -0.33354818+2.04343478j,
       -0.33354818-2.04343478j, -0.01082982+0.j        ])


#short period
(array([-1.32306705+1.94243638j, -1.32306705-1.94243638j,
        0.00344641+0.13572562j,  0.00344641-0.13572562j])
'''

