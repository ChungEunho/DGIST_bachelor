import numpy as np
import matplotlib.pyplot as plt

dt = 0.1  
T = 333  
steps = int(T / dt)

N = 4
v_I_mag = 2000.0
v_T_mag = 1500.0

r_I = np.array([0.0, 50_000.0])
r_T = np.array([500_000.0, 100_000.0])

target_direction = r_T - r_I
target_direction /= np.linalg.norm(target_direction)
v_I = v_I_mag * target_direction

v_T = np.array([-v_T_mag, 0.0])

r_I_list = [r_I.copy()]
r_T_list = [r_T.copy()]
distance_list = []
los_rate_list = []

intercept_time = None
intercept_index = None
intercept_threshold = 100.0  

for step in range(steps):
    lambda_vec = r_T - r_I
    lambda_norm = np.linalg.norm(lambda_vec)
    lambda_unit = lambda_vec / lambda_norm

    lambda_dot = v_T - v_I
    lambda_unit_dot = (lambda_dot / lambda_norm) - (np.dot(lambda_vec, lambda_dot) / lambda_norm**3) * lambda_vec
    los_rate = np.linalg.norm(lambda_unit_dot)
    los_rate_list.append(los_rate)

    v_c = -np.dot(lambda_vec, lambda_dot) / lambda_norm

    a_c = N * v_c * lambda_unit_dot

    v_I += a_c * dt
    r_I += v_I * dt

    r_T += v_T * dt

    r_I_list.append(r_I.copy())
    r_T_list.append(r_T.copy())
    distance_list.append(np.linalg.norm(r_T - r_I))

    if intercept_time is None and np.linalg.norm(lambda_vec) < intercept_threshold:
        intercept_time = step * dt
        intercept_index = step  
        break  


if intercept_index is not None:
    r_I_arr = np.array(r_I_list[:intercept_index+2]) 
    r_T_arr = np.array(r_T_list[:intercept_index+2])
    time_arr = np.arange(intercept_index+1) * dt
    distance_arr = np.array(distance_list[:intercept_index+1])
    los_rate_arr = np.array(los_rate_list[:intercept_index+1])
else:
    r_I_arr = np.array(r_I_list)
    r_T_arr = np.array(r_T_list)
    time_arr = np.arange(len(distance_list)) * dt
    distance_arr = np.array(distance_list)
    los_rate_arr = np.array(los_rate_list)

plt.figure(figsize=(10, 6))
plt.plot(r_I_arr[:, 0], r_I_arr[:, 1], label='Interceptor')
plt.plot(r_T_arr[:, 0], r_T_arr[:, 1], label='Target')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('PN Guidance Simulation (Improved Trajectories)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(time_arr, distance_arr)
plt.xlabel('Time (s)')
plt.ylabel('Distance to Target (m)')
plt.title('Range to Target Over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(time_arr, los_rate_arr)
plt.xlabel('Time (s)')
plt.ylabel('LOS Rate |d(lambda_hat)/dt| (1/s)')
plt.title('LOS Rate Over Time')
plt.grid(True)
plt.show()

if intercept_time:
    print(f"Intercept achieved at t = {intercept_time:.2f} s")
else:
    print("No intercept achieved")
