from math import *
import numpy as np
from control import *
import matplotlib.pyplot as plt
import Reference_data_reader_num_model 
# Citation 550 - Linear simulation

# xcg = 0.25 * c

'''
 #for reference data
#Starting times of motions
Start_hour_Phugoid, Start_min_Phugoid, Start_sec_Phugoid=0,53,57 #0,53,57
Start_hour_AR, Start_min_AR, Start_sec_AR=0,59,10
Start_hour_SP, Start_min_SP, Start_sec_SP=1,0,34
Start_hour_DR, Start_min_DR, Start_sec_DR=1,1,57
Start_hour_DR_yaw, Start_min_DR_yaw, Start_sec_DR_yaw=1,2,47
Start_hour_spiral, Start_min_spiral, Start_sec_spiral=1,5,20

End_hour_Phugoid, End_min_Phugoid, End_sec_Phugoid=0,55,30
End_hour_AR, End_min_AR, End_sec_AR=0,59,22
End_hour_SP, End_min_SP, End_sec_SP=1,0,38
End_hour_DR, End_min_DR, End_sec_DR=1,2,20
End_hour_DR_yaw, End_min_DR_yaw, End_sec_DR_yaw=1,3,10
End_hour_spiral, End_min_spiral, End_sec_spiral=1,6,50
'''
# Stationary flight condition
tas,alt,pitch,AOA,PR,d_a,d_r,d_e,t='Dadc1_tas', 'Dadc1_alt', 'Ahrs1_Pitch', 'vane_AOA', 'Ahrs1_bPitchRate', 'delta_a', 'delta_r', 'delta_e', 'time' 
side_slip1, side_slip2, roll_angle, roll_rate, yaw_rate='Ahrs1_bLatAcc','Ahrs1_bLongAcc', 'Ahrs1_Roll','Ahrs1_bRollRate', 'Ahrs1_bYawRate'

#ACTUAL flight test
#Starting times of motion
Start_hour_Phugoid, Start_min_Phugoid, Start_sec_Phugoid=0,52,30
Start_hour_SP, Start_min_SP, Start_sec_SP=0,55,53
Start_hour_AR, Start_min_AR, Start_sec_AR=0,57,35
Start_hour_DR, Start_min_DR, Start_sec_DR=0,58,44
Start_hour_DR_yaw, Start_min_DR_yaw, Start_sec_DR_yaw=0,59,32
Start_hour_spiral, Start_min_spiral, Start_sec_spiral=1,2,59

End_hour_Phugoid, End_min_Phugoid, End_sec_Phugoid=0,55,20
End_hour_SP, End_min_SP, End_sec_SP=0,56,3
End_hour_AR, End_min_AR, End_sec_AR=0,58,0
End_hour_DR, End_min_DR, End_sec_DR=0,59,15
End_hour_DR_yaw, End_min_DR_yaw, End_sec_DR_yaw=1,0,6
End_hour_spiral, End_min_spiral, End_sec_spiral=1,4,25


#Getting the lists of variables

test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, delta_a, delta_r, delta_e,t=Reference_data_reader_num_model.get_lists(tas,alt,pitch,AOA,PR,d_a,d_r,d_e,t) #gets the list of all avriables irrespective of time
side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list=Reference_data_reader_num_model.get_lists_asymmetric(side_slip1, side_slip2, roll_angle, roll_rate, yaw_rate)



a='SP' #Phugoid, DR, SP, spiral, AR, DR_yaw       tas,alt,pitch,AOA,PR,d_a,d_r,d_e,t

if a=='Phugoid':
    start_hour,start_minu,start_sec=Start_hour_Phugoid, Start_min_Phugoid, Start_sec_Phugoid
    end_hour,end_minu,end_sec=End_hour_Phugoid, End_min_Phugoid, End_sec_Phugoid
    V1,hp1,th1,alpha1,PR1,inputs_de, time=Reference_data_reader_num_model.get_Phugoid(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec, delta_a, delta_e, delta_r)
    V0=V1*0.5144444444444
    hp0=hp1*0.3048 
    th0,alpha0,PR=th1*np.pi/180,alpha1*np.pi/180,PR1*np.pi/180
    
elif a=='SP':
    start_hour,start_minu,start_sec=Start_hour_SP, Start_min_SP, Start_sec_SP
    end_hour,end_minu,end_sec=End_hour_SP, End_min_SP, End_sec_SP
    V1,hp1,th1,alpha1,PR,inputs_de,time=Reference_data_reader_num_model.get_SP(test_list_tas, test_list_alt, theta_list, angle_of_attack_list, test_list_pitchrate, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec, delta_a, delta_e, delta_r)
    V0=V1*0.5144444444444
    hp0=hp1*0.3048 
    th0,alpha0,PR=th1*np.pi/180,alpha1*np.pi/180,PR*np.pi/180
    
elif a=='DR':
    start_hour,start_minu,start_sec=Start_hour_DR, Start_min_DR, Start_sec_DR
    end_hour,end_minu,end_sec=End_hour_DR, End_min_DR, End_sec_DR
    V1,hp1,th1,alpha1,PR1,inputs_da,inputs_dr,time=Reference_data_reader_num_model.get_DR(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec, delta_a, delta_e, delta_r)
    side_slip1,roll1,rollrate1,yawrate1=Reference_data_reader_num_model.get_DRasym(side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec)
    V0=V1*0.5144444444444
    hp0=hp1*0.3048 
    th0,alpha0,PR=th1*np.pi/180,alpha1*np.pi/180,PR1*np.pi/180
    side_slip0,roll0,rollrate0,yawrate0=side_slip1*np.pi/180,roll1*np.pi/180,rollrate1*np.pi/180,yawrate1*np.pi/180

elif a=='DR_yaw':
    start_hour,start_minu,start_sec=Start_hour_DR_yaw, Start_min_DR_yaw, Start_sec_DR_yaw
    end_hour,end_minu,end_sec=End_hour_DR_yaw, End_min_DR_yaw, End_sec_DR_yaw
    V1,hp1,th1,alpha1,PR1,inputs_da,inputs_dr,time=Reference_data_reader_num_model.get_DR(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec, delta_a, delta_e, delta_r)
    side_slip1,roll1,rollrate1,yawrate1=Reference_data_reader_num_model.get_DRasym(side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec)
        
    
    V0=V1*0.5144444444444
    hp0=hp1*0.3048 
    th0,alpha0,PR=th1*np.pi/180,alpha1*np.pi/180,PR1*np.pi/180
    side_slip0,roll0,rollrate0,yawrate0=side_slip1*np.pi/180,roll1*np.pi/180,rollrate1*np.pi/180,yawrate1*np.pi/180
    
elif a=='spiral':
    start_hour,start_minu,start_sec=Start_hour_spiral, Start_min_spiral, Start_sec_spiral
    end_hour,end_minu,end_sec=End_hour_spiral, End_min_spiral, End_sec_spiral
    V1,hp1,th1,alpha1,PR1,inputs_da,inputs_dr,time=Reference_data_reader_num_model.get_DR(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec, delta_a, delta_e, delta_r)
    side_slip1,roll1,rollrate1,yawrate1=Reference_data_reader_num_model.get_DRasym(side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec)
    
    V0=V1*0.5144444444444
    hp0=hp1*0.3048 
    th0,alpha0,PR=th1*np.pi/180,alpha1*np.pi/180,PR1*np.pi/180
    side_slip0,roll0,rollrate0,yawrate0=side_slip1*np.pi/180,roll1*np.pi/180,rollrate1*np.pi/180,yawrate1*np.pi/180
    
elif a=='AR':
    start_hour,start_minu,start_sec=Start_hour_AR, Start_min_AR, Start_sec_AR
    end_hour,end_minu,end_sec=End_hour_AR, End_min_AR, End_sec_AR
    V1,hp1,th1,alpha1,PR1,inputs_da,inputs_dr,time=Reference_data_reader_num_model.get_DR(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec, delta_a, delta_e, delta_r)
    side_slip1,roll1,rollrate1,yawrate1=Reference_data_reader_num_model.get_DRasym(side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, t, start_hour, start_minu, start_sec, end_hour, end_minu, end_sec)
    
    V0=V1*0.5144444444444
    hp0=hp1*0.3048 
    th0,alpha0,PR=th1*np.pi/180,alpha1*np.pi/180,PR1*np.pi/180
    side_slip0,roll0,rollrate0,yawrate0=side_slip1*np.pi/180,roll1*np.pi/180,rollrate1*np.pi/180,yawrate1*np.pi/180
    
side_slip0, ratios_u=Reference_data_reader_num_model.test(start_hour,start_minu,start_sec, end_hour,end_minu,end_sec,V0)
actual_TAS, actual_pitch, actual_AOA, actual_pitchrate,actual_sideslip, actual_roll, actual_rollrate, actual_yawrate=Reference_data_reader_num_model.get_graph_values(test_list_tas, theta_list, angle_of_attack_list,test_list_pitchrate, side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, start_hour,start_minu,start_sec, end_hour,end_minu,end_sec, V0)

#hp0: pressure altitude in the stationary flight condition [m]
#V0: true airspeed in the stationary flight condition [m/sec]
#alpha0: angle of attack in the stationary flight condition [rad]
#th0: pitch angle in the stationary flight condition [rad]

#redefining side_slip list for actual data

new_side_slip=[]
for i in range(len(inputs_dr)):
    new_side_slip.append(-inputs_dr[i]*np.pi/180)
actual_sideslip=list(new_side_slip)
side_slip0=actual_sideslip[0]


# Aircraft mass
m      =      Reference_data_reader_num_model.get_mass(start_hour,start_minu,start_sec,t,test_list_alt,test_list_tas)   # mass [kg] 


master_switch = True

# Stationary values or changed stationary values
switch = 1
switch_step=2 #which switch option will be compared against original
numerical_results=1 #will it be for validation or numerical model [0,1]
overall_lists_TAS, overall_lists_AOA, overall_lists_pitch, overall_lists_PR = [], [],[],[]

while master_switch==True and switch<=4:
    if switch==1: #previous unrefind data
        
        # aerodynamic properties
        e      =  0.8            # Oswald factor [ ]
        CD0    =  0.04           # Zero lift drag coefficient [ ]
        CLa    =  5.084          # Slope of CL-alpha curve [ ]
        
        # Longitudinal stability
        Cma    =  -0.5626          # longitudinal stabilty [ ]
        Cmde   =  -1.1642          # elevator effectiveness [ ]
    
    
    if switch==2: #slightly refined data
        e      =  0.984478051309284                 #fake value: 0.8          # Oswald factor [ ]
        CD0    =  0.03456048452067092               #fake value: 0.04         # Zero lift drag coefficient [ ]
        CLa    =  0.1110284431844329*180/np.pi      #fake value: 5.084          # Slope of CL-alpha curve [ ]
        
        # Longitudinal stability
        Cma    =-0.18697327092815078      # longitudinal stabilty [ ]
        Cmde   =-0.43413107112501953      # elevator effectiveness [ ]
    
    if switch==3:#manually refined data
        e      =  0.984478051309284                 #fake value: 0.8          # Oswald factor [ ]
        CD0    =  0.03456048452067092               #fake value: 0.04         # Zero lift drag coefficient [ ]
        CLa    =  0.1110284431844329*180/np.pi      #fake value: 5.084          # Slope of CL-alpha curve [ ]
        
        # Longitudinal stability
        Cma    = -0.35                            #-0.18697327092815078      # longitudinal stabilty [ ]
        Cmde   = -0.95                              #-0.43413107112501953     # elevator effectiveness [ ] 
        
    
    if switch ==4: #new slightly refined data
        e      =  0.984478051309284                 #fake value: 0.8          # Oswald factor [ ]
        CD0    =  0.03456048452067092               #fake value: 0.04         # Zero lift drag coefficient [ ]
        CLa    =  0.1110284431844329*180/np.pi      #fake value: 5.084          # Slope of CL-alpha curve [ ]
        
        # Longitudinal stability
        Cma    = -0.04856324424731579                              #-0.18697327092815078      # longitudinal stabilty [ ]
        Cmde   = -0.11275843406780188                              #-0.43413107112501953     # elevator effectiveness [ ] 
            
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
    Vh_V   = 1 	          # [ ]
    ih     = -2 * np.pi / 180   # stabiliser angle of incidence [rad]
    
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
    CXu    = -0.095  # CXu =-0.02792
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
    Clp    = -0.71085 #(-1 fits sometimes better)
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
          [0, (CZadot-2*muc)*c/V0, 0, 0],
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
    D = np.array([[0],[0],[0],[0]])
    
    print(A,B)
    
    
    #A-Symmetric (_a)
    C1_a = np.array([[(CYbdot-2*mub)*b/V0, 0, 0, 0],
                   [0, -0.5*b/V0, 0, 0],
                   [0, 0, -2*mub*KX2*(b/V0)**2, 2*mub*KXZ*(b/V0)**2],
                   [Cnbdot*b/V0, 0, 2*mub*KXZ*(b/V0)**2, -2*mub*KZ2*(b/V0)**2]])  
    C2_a = np.array([[CYb, CL, CYp*b/(2*V0), (CYr-4*mub)*b/(2*V0)], 
                   [0, 0, (b/(2*V0)), 0],
                   [Clb, 0, Clp*b/(2*V0), Clr*b/(2*V0)], 
                   [Cnb, 0, Cnp*b/(2*V0), Cnr*b/(2*V0)]]) 
    C3_a = np.array([[CYda, CYdr],
                   [0, 0],
                   [Clda, Cldr],
                   [Cnda, Cndr]])
    
    
    A_a = -(np.linalg.inv(C1_a)).dot(C2_a)
    B_a = -(np.linalg.inv(C1_a)).dot(C3_a)
    C_a = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    D_a = np.array([[0,0],[0,0],[0,0],[0,0]])
    
    print(np.linalg.eigvals(A_a))
    
    
    if a=='DR'or a=='DR_yaw':
        A_a[1]=[10e-20,10e-20,10e-20,10e-20]
        B_a[1]=[10e-20,10e-20]
        A_a[2]=[10e-20,10e-20,10e-20,10e-20]
        B_a[2]=[10e-20,10e-20]
    
    if a=='AR':
        A_a[0],A_a[3]= [10e-20,10e-20,10e-20,10e-20], [10e-20,10e-20,10e-20,10e-20]
        B_a[0],B_a[3]=[10e-20,10e-20],[10e-20,10e-20]
    
    if a=='Phugoid' or a=='SP':
        #Response for symmetric flight
        x0=np.matrix([[0],[0], [0], [PR]]) 
        
        #t=np.arange(0.0,len(inputs_de)/1000,0.001)
        t0=time[0]
        t1=time[-1]
        step=0.1
        t_plot=np.arange(0,time[-1]-time[0], (time[-1]-time[0])/len(time))
        t=time
         
        sys=ss(A,B,C,D)
        
        #Defining inputs
        new_u1=[]
        for i in range(len(inputs_de)):
            new_u1.append(inputs_de[i]*np.pi/180)
        
        u1 = new_u1
            
        tdum,y1,xdum=forced_response(sys,t,u1,x0)
        
        
        for i in range(len(tdum)):
            if switch==1:
                actual_AOA[i]=actual_AOA[i]-alpha0
                actual_TAS[i]=actual_TAS[i]+V0
                actual_pitch[i]=actual_pitch[i]-th0
            
            y1[0][i]=y1[0][i]+V0
            
           
        #PLOTTING
        if numerical_results==1:
                #TAS
            plt.title('Actual data versus Numerical model true air speed') 
            plt.xlabel('time [sec]')
            plt.ylabel('True airspeed [m/s]')
            plt.plot(t_plot,y1[0], label= 'Numerical model data')        
            plt.plot(t_plot, actual_TAS, label='Actual data')
            plt.show()
            
                #angle of attack
            plt.title('Actual data versus Numerical model angle of attack')
            plt.xlabel('time [sec]')
            plt.ylabel('angle of attack [rad]')
            plt.plot(t_plot,y1[1], label= 'Numerical model data')        
            plt.plot(t_plot, actual_AOA, label='Actual data')
            plt.show()
            
                #pitch   
            plt.title('Actual data versus Numerical model pitch')
            plt.xlabel('time [sec]')
            plt.ylabel('pitch [rad]')
            plt.plot(t_plot,y1[2], label= 'Numerical model data')        
            plt.plot(t_plot, actual_pitch, label='Actual data')
            plt.show()
            
                #pitch rate
            plt.title('Actual data versus Numerical model pitch rate')
            plt.xlabel('time [sec]')
            plt.ylabel('pitch rate [rad/sec]')
            plt.plot(t_plot,y1[3], label= 'Numerical model data')
            plt.plot(t_plot, actual_pitchrate, label='Actual data')
            plt.show()
            
            print('initial_TAS:', V0)
            print('initial_AOA:', alpha0)
            print('initial_pitch:', th0)
            print('initial_pitch rate:', PR)
        
        overall_lists_TAS.append(y1[0])
        overall_lists_AOA.append(y1[1])
        overall_lists_pitch.append(y1[2])
        overall_lists_PR.append(y1[3])
        
        if switch==1+switch_step:
                #TAS
            plt.title('Actual data versus Numerical model true air speed') 
            plt.xlabel('time [sec]')
            plt.ylabel('True airspeed [m/s]')
            plt.plot(t_plot,overall_lists_TAS[0], label= 'Numerical model data')     
            plt.plot(t_plot, actual_TAS, label='Actual data')
            plt.plot(t_plot,overall_lists_TAS[1])
            plt.show()
            
                #angle of attack
            plt.title('Actual data versus Numerical model angle of attack')
            plt.xlabel('time [sec]')
            plt.ylabel('angle of attack [rad]')
            plt.plot(t_plot,overall_lists_AOA[0], label= 'Numerical model data')
            plt.plot(t_plot, actual_AOA, label='Actual data')
            plt.plot(t_plot, overall_lists_AOA[1])
            plt.show()
            
                #pitch   
            plt.title('Actual data versus Numerical model pitch')
            plt.xlabel('time [sec]')
            plt.ylabel('pitch [rad]')
            plt.plot(t_plot,overall_lists_pitch[0], label= 'Numerical model data')
            plt.plot(t_plot, actual_pitch, label='Actual data')
            plt.plot(t_plot, overall_lists_pitch[1])
            plt.show()
            
                #pitch rate
            plt.title('Actual data versus Numerical model pitch rate')
            plt.xlabel('time [sec]')
            plt.ylabel('pitch rate [rad/sec]')
            plt.plot(t_plot,overall_lists_PR[0], label= 'Numerical model data') 
            plt.plot(t_plot, actual_pitchrate, label='Actual data')
            plt.plot(t_plot, overall_lists_PR[1])
            plt.show()
            
            master_switch=False
                
    if numerical_results==1:
        print('yes')    
        if a=='DR' or a=='spiral' or a=='DR_yaw' or a=='AR':
            #RESPONSE FOR ASYMMETRIC FLIGHT
            x0=np.matrix([[side_slip0],[roll0],[rollrate0],[yawrate0]]) 
            t0=time[0]
            t1=time[-1]
            step=(time[-1]-time[0])/len(time)
            t_plot=np.arange(0,time[-1]-time[0], step)
            t=time
            
                #formulating the state space
            sys=ss(A_a,B_a,C_a,D_a)
        
                #Defining inputs
            new_u1=[[],[]]
            for i in range(len(inputs_dr)):
                new_u1[0].append([-inputs_da[i]*np.pi/180])
                new_u1[1].append([-inputs_dr[i]*np.pi/180])
            
                #Reshaping the array
            u1=np.array(new_u1)
            u1=np.array(u1).reshape(2, len(new_u1[0]))
               
            #CREATING OUTPUTS
            tdum,y1,xdum=forced_response(sys,t,u1,x0)
        
            #PLOTTING
            
                #sideslip
            plt.title('Actual data versus Numerical model side slip')
            plt.xlabel('time [sec]')
            plt.ylabel('side slip angle [rad]')
            plt.plot(t_plot,y1[0], label= 'Numerical model data')        
            plt.plot(t_plot, actual_sideslip, label='Actual data')
            plt.show()
            
                #roll
            plt.title('Actual data versus Numerical model roll')
            plt.xlabel('time [sec]')
            plt.ylabel('roll angle [rad]')
            plt.plot(t_plot,y1[1], label= 'Numerical model data')        
            plt.plot(t_plot, actual_roll, label='Actual data')
            plt.show()
            
                #roll rate    
            plt.title('Actual data versus Numerical model roll rate')
            plt.xlabel('time [sec]')
            plt.ylabel('roll rate [rad/sec]')
            plt.plot(t_plot,y1[2], label= 'Numerical model data')        
            plt.plot(t_plot, actual_rollrate, label='Actual data')
            plt.show()
            
                #yaw rate
            plt.title('Actual data versus Numerical model yaw rate')
            plt.xlabel('time [sec]')
            plt.ylabel('yaw rate [rad/sec]')
            plt.plot(t_plot,y1[3], label= 'Numerical model data')
            plt.plot(t_plot, actual_yawrate, label='Actual data')
            plt.show()
        
            print('initial sideslip:', side_slip0)
            print('initial roll: ', roll0)
            print('initial rollrate:', rollrate0)
            print('initial yawrate:', yawrate0)
            master_switch=False
                
    switch+=switch_step
    print(switch)

