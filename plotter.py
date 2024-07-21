import struct

sample_size = 104
total_bytes = 42536
num_samples = total_bytes // sample_size
#print(num_samples)

samples = []

with open('step-178-171520.bin', 'rb') as file:
    for i in range(num_samples):
        sample_data = file.read(sample_size)
        sample = struct.unpack('>52h', sample_data)  
        samples.append(sample)


act1_command_li = []
act2_command_li = []
act3_command_li = []
act4_command_li = []

act1_inc_li = []
act2_inc_li = []
act3_inc_li = []
act4_inc_li = []

act1_abs_li = []
act2_abs_li = []
act3_abs_li = []
act4_abs_li = []

act1_current_li = []
act2_current_li = []
act3_current_li = []
act4_current_li = []

Actuator1_li = [act1_command_li, act1_inc_li, act1_abs_li, act1_current_li]
Actuator2_li = [act2_command_li, act2_inc_li, act2_abs_li, act2_current_li]
Actuator3_li = [act3_command_li, act3_inc_li, act3_abs_li, act3_current_li]
Actuator4_li = [act4_command_li, act4_inc_li, act4_abs_li, act4_current_li]



s_f_li = [0.00549,0.00549, 0.043945312, 0.001904762]




for i in range(1, 409):
    
    act1_command_li.append(samples[i][10]*0.00549)
    act2_command_li.append(samples[i][11]*0.00549)
    act3_command_li.append(samples[i][12]*0.00549)
    act4_command_li.append(samples[i][13]*0.00549)

    act1_inc_li.append(samples[i][14]*0.00549)
    act2_inc_li.append(samples[i][15]*0.00549)
    act3_inc_li.append(samples[i][16]*0.00549)
    act4_inc_li.append(samples[i][17]*0.00549)

    act1_abs_li.append(samples[i][18]*0.043945312)
    act2_abs_li.append(samples[i][19]*0.043945312)
    act3_abs_li.append(samples[i][20]*0.043945312)
    act4_abs_li.append(samples[i][21]*0.043945312)

    act1_current_li.append(samples[i][30]*0.001904762)
    act2_current_li.append(samples[i][31]*0.001904762)
    act3_current_li.append(samples[i][32]*0.001904762)
    act4_current_li.append(samples[i][33]*0.001904762)




# print("Actuator1:",Actuator1_li )
# print("Actuator2:",Actuator2_li)
# print("Actuator3:" ,Actuator3_li )
# print("Actuator4:" ,Actuator4_li )

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as signal
#from control import step_response

fig = plt.figure()


x=range(len(act1_command_li))
ax1=fig.add_subplot(2,2,1)
ax1.plot(x,act1_command_li,label='command')
ax1.plot(x,act1_inc_li,label='incremental')
ax1.plot(x,act1_abs_li,label='absolute')
ax1.plot(x,act1_current_li,label='current')
plt.xlabel("Time")
plt.ylabel("Actuator 1 in degrees")
plt.legend()


lower_threshold=0.1*np.max(act1_inc_li)
idx_lower=np.argmax(act1_inc_li>lower_threshold)

upper_threshold=0.9*np.max(act1_inc_li)
idx_upper=np.argmax(act1_inc_li>upper_threshold)

rise_time=x[idx_upper]-x[idx_lower]

idx_peak=np.argmax(act1_inc_li)

peak_time=x[idx_peak]

maximum_overshoot=(np.max(act1_inc_li)-np.min(act1_inc_li))/np.min(act1_inc_li)*100

tolerance=0.05*np.max(act1_inc_li)
#tolerance=0.02*np.max(act4_inc_li)

idx_enter_tolerance=np.argmax(np.abs(act1_inc_li))
idx_leave_tolerance=np.argmax(np.abs(act1_inc_li))
settling_time=x[idx_enter_tolerance+idx_leave_tolerance]-x[idx_enter_tolerance]
#desired_steady_state_error=0
steady_state_error=np.abs(act1_inc_li[-1])

print("time prameters for actuator1 (incremental)")
print("Rise Time:", rise_time)
print("Peak Overshoot:", maximum_overshoot)
print("Settling Time:", settling_time)
print("Steady-State Error:", steady_state_error)

     
ax2=fig.add_subplot(2,2,2)
ax2.plot(x,act2_command_li,label='command')
ax2.plot(x,act2_inc_li,label='incremental')
ax2.plot(x,act2_abs_li,label='absolute')
ax2.plot(x,act2_current_li,label='current')
plt.xlabel("Time")
plt.ylabel("Actuator 2 in degrees")
plt.legend()


lower_threshold=0.1*np.max(act2_inc_li)
idx_lower=np.argmax(act2_inc_li>lower_threshold)

upper_threshold=0.9*np.max(act2_inc_li)
idx_upper=np.argmax(act2_inc_li>upper_threshold)

rise_time=x[idx_upper]-x[idx_lower]

idx_peak=np.argmax(act2_inc_li)

peak_time=x[idx_peak]

maximum_overshoot=(np.max(act2_inc_li)-np.min(act2_inc_li))/np.min(act2_inc_li)*100

tolerance=0.05*np.max(act2_inc_li)
#tolerance=0.02*np.max(act4_inc_li)

idx_enter_tolerance=np.argmax(np.abs(act2_inc_li))
idx_leave_tolerance=np.argmax(np.abs(act2_inc_li))
settling_time=x[idx_enter_tolerance+idx_leave_tolerance]-x[idx_enter_tolerance]
#desired_steady_state_error=0
steady_state_error=np.abs(act2_inc_li[-1])


print("time prameters for actuator2 (incremental)")
print("Rise Time:", rise_time)
print("Peak Overshoot:", maximum_overshoot)
print("Settling Time:", settling_time)
print("Steady-State Error:", steady_state_error)


ax3 = fig.add_subplot(2,2,3)
ax3.plot(x, act3_command_li, label='command')
ax3.plot(x, act3_inc_li, label='incremental')
ax3.plot(x,act3_abs_li,label='absolute')
ax3.plot(x,act3_current_li,label='current')
plt.xlabel("Time")
plt.ylabel("Actuator 3 in degrees")
plt.legend()


lower_threshold=0.1*np.max(act3_inc_li)
idx_lower=np.argmax(act2_inc_li>lower_threshold)

upper_threshold=0.9*np.max(act3_inc_li)
idx_upper=np.argmax(act2_inc_li>upper_threshold)

rise_time=x[idx_upper]-x[idx_lower]

idx_peak=np.argmax(act3_inc_li)

peak_time=x[idx_peak]

maximum_overshoot=(np.max(act3_inc_li)-np.min(act3_inc_li))/np.min(act3_inc_li)*100

tolerance=0.05*np.max(act3_inc_li)
#tolerance=0.02*np.max(act4_inc_li)

idx_enter_tolerance=np.argmax(np.abs(act3_inc_li))
idx_leave_tolerance=np.argmax(np.abs(act3_inc_li))
settling_time=x[idx_enter_tolerance+idx_leave_tolerance]-x[idx_enter_tolerance]
#desired_steady_state_error=0
steady_state_error=np.abs(act3_inc_li[-1])


print("time prameters for actuator3 (incremental)")
print("Rise Time:", rise_time)
print("Peak Overshoot:", maximum_overshoot)
print("Settling Time:", settling_time)
print("Steady-State Error:", steady_state_error)


ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x,act4_command_li,label='command')
ax4.plot(x,act4_inc_li,label='incremental')
ax4.plot(x,act4_abs_li,label='absolute')
ax4.plot(x,act4_current_li,label='current')
plt.xlabel("Time")
plt.ylabel("Actuator 4 in degrees")
plt.legend()

lower_threshold=0.1*np.max(act2_inc_li)
idx_lower=np.argmax(act4_inc_li>lower_threshold)

upper_threshold=0.9*np.max(act4_inc_li)
idx_upper=np.argmax(act4_inc_li>upper_threshold)

rise_time=x[idx_upper]-x[idx_lower]

idx_peak=np.argmax(act4_inc_li)

peak_time=x[idx_peak]

maximum_overshoot=(np.max(act4_inc_li)-np.min(act2_inc_li))/np.min(act4_inc_li)*100

tolerance=0.05*np.max(act4_inc_li)
#tolerance=0.02*np.max(act4_inc_li)

idx_enter_tolerance=np.argmax(np.abs(act4_inc_li))
idx_leave_tolerance=np.argmax(np.abs(act4_inc_li))
settling_time=x[idx_enter_tolerance+idx_leave_tolerance]-x[idx_enter_tolerance]
#desired_steady_state_error=0
steady_state_error=np.abs(act4_inc_li[-1])



print("time prameters for actuator4 (incremental)")
print("Rise Time:", rise_time)
print("Peak Overshoot:", maximum_overshoot)
print("Settling Time:", settling_time)
print("Steady-State Error:", steady_state_error)



plt.tight_layout()
plt.show()






                  
                  
                  
                  
                  
                  
                  
    
    
                                 
    
    
    

