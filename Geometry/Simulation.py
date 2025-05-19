import numpy as np
import matplotlib.pyplot as plt

# 시뮬레이션 설정
dt = 0.1  # 시간 간격 (초)
T = 333   # 총 시뮬레이션 시간 (초)
steps = int(T / dt)

# PN 유도 상수 및 초기 속도
N = 4
v_I_mag = 2000.0
v_T_mag = 1500.0

# 초기 위치
r_I = np.array([0.0, 50_000.0])
r_T = np.array([500_000.0, 100_000.0])

# Interceptor 초기 속도를 target 방향으로 세팅
target_direction = r_T - r_I
target_direction /= np.linalg.norm(target_direction)
v_I = v_I_mag * target_direction

# Target은 왼쪽(-x축 방향)으로 이동 (수평)
v_T = np.array([-v_T_mag, 0.0])

# 기록용 리스트
r_I_list = [r_I.copy()]
r_T_list = [r_T.copy()]
distance_list = []
los_rate_list = []

intercept_time = None
intercept_index = None
intercept_threshold = 100.0  # 100m 이내면 요격 성공으로 간주

for step in range(steps):
    # 시선 벡터 및 단위벡터
    lambda_vec = r_T - r_I
    lambda_norm = np.linalg.norm(lambda_vec)
    lambda_unit = lambda_vec / lambda_norm

    # 시선 변화율 및 단위벡터 변화율 (정확 계산)
    lambda_dot = v_T - v_I
    lambda_unit_dot = (lambda_dot / lambda_norm) - (np.dot(lambda_vec, lambda_dot) / lambda_norm**3) * lambda_vec
    los_rate = np.linalg.norm(lambda_unit_dot)
    los_rate_list.append(los_rate)

    # 폐쇄 속도
    v_c = -np.dot(lambda_vec, lambda_dot) / lambda_norm

    # PN 가속도
    a_c = N * v_c * lambda_unit_dot

    # interceptor 업데이트
    v_I += a_c * dt
    r_I += v_I * dt

    # target 업데이트
    r_T += v_T * dt

    # 거리 기록
    r_I_list.append(r_I.copy())
    r_T_list.append(r_T.copy())
    distance_list.append(np.linalg.norm(r_T - r_I))

    # 요격 성공 여부 판단
    if intercept_time is None and np.linalg.norm(lambda_vec) < intercept_threshold:
        intercept_time = step * dt
        intercept_index = step  # 요격 시점 인덱스 저장
        break  # 요격 후 시뮬레이션 중단

# 배열 변환 (요격 시점까지만)
if intercept_index is not None:
    r_I_arr = np.array(r_I_list[:intercept_index+2])  # +2: step에서 break 직전까지 좌표 포함
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

# 시각화 1: 궤적
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

# 시각화 2: 거리 vs 시간
plt.figure(figsize=(10, 4))
plt.plot(time_arr, distance_arr)
plt.xlabel('Time (s)')
plt.ylabel('Distance to Target (m)')
plt.title('Range to Target Over Time')
plt.grid(True)
plt.show()

# 시각화 3: 시선(LOS) 단위벡터 변화율 vs 시간
plt.figure(figsize=(10, 4))
plt.plot(time_arr, los_rate_arr)
plt.xlabel('Time (s)')
plt.ylabel('LOS Rate |d(lambda_hat)/dt| (1/s)')
plt.title('LOS Rate Over Time')
plt.grid(True)
plt.show()

# 요격 성공 여부 출력
if intercept_time:
    print(f"Intercept achieved at t = {intercept_time:.2f} s")
else:
    print("No intercept achieved")
