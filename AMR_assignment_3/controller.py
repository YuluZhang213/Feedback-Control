import numpy as np
import math
import matplotlib.pyplot as plt

Ku_x_vel = 2.77638999
Tu_x_vel = 1/3

# PID parameters
Kp_pos = np.array([1.0, 1.5, 1.5, 1.0])
Ki_pos = np.array([0.0, 0.0, 0.0, 0.0])
Kd_pos = np.array([0.0, 0.0, 0.0, 0.0])

Kp_vel = np.array([1.5, 1.5, 0.75, 0.4])
Ki_vel = np.array([0.25, 0.25, 0.2, 0.2])
Kd_vel = np.array([0.09, 0.075, 0.0, 0.0])

MAX_VEL = 1.0
MAX_YAW_RATE = 0.5

# PID state
pos_integral = np.zeros(4)
pos_prev_error = np.zeros(4)
pos_prev = None

vel_integral = np.zeros(4)
vel_prev_error = np.zeros(4)

log_data = {
    'time': [],
    'vx': [], 'vy': [], 'vz': [], 'yaw_rate': [],
    'vx_actual': [], 'vy_actual': [], 'vz_actual': [], 'yaw_rate_actual': [],
    'error_x': [], 'error_y': [], 'error_z': [], 'error_yaw': []
}

def saturate(val, lim):
    return np.clip(val, -lim, lim)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def controller(state, target_pos, dt):
    global pos_prev, pos_integral, pos_prev_error
    global vel_integral, vel_prev_error

    pos = np.array(state[0:3])
    yaw = state[5]
    target_xyz = np.array(target_pos[0:3])
    target_yaw = target_pos[3]

    #test only
    #target_xyz = np.array([0.0, 0.0, 2.0])
    #target_yaw = 0.0

    # --- Outer loop position control ---
    pos_error = np.zeros(4)
    pos_error[0:3] = target_xyz - pos
    pos_error[3] = wrap_to_pi(target_yaw - yaw)

    pos_integral += pos_error * dt
    d_pos_error = (pos_error - pos_prev_error) / dt if dt > 1e-6 else np.zeros(4)
    pos_prev_error = pos_error

    desired_vel_world = Kp_pos * pos_error + Ki_pos * pos_integral + Kd_pos * d_pos_error

    R = np.array([
        [np.cos(-yaw), -np.sin(-yaw), 0],
        [np.sin(-yaw),  np.cos(-yaw), 0],
        [0,              0,           1]
    ])

    R = np.eye(3)
    desired_vel_body = R @ desired_vel_world[0:3]  # x, y, z
    desired_yaw_rate = desired_vel_world[3]        # yaw_rate 不变

    desired_vel = np.array([*desired_vel_body, desired_yaw_rate])
    #print(desired_vel)

    #desired_vel = np.array([0.5, 0.5, 0.5, 0.3])

    # --- Inner Loop velocity control ---
    if pos_prev is None:
        current_vel = np.zeros(3)  
        current_yaw_rate = 0.0
    else:
        current_vel = (pos - pos_prev[0]) / dt
        current_yaw_rate = wrap_to_pi(yaw - pos_prev[1]) / dt

    pos_prev = (pos, yaw)

    vel_error = np.zeros(4)
    vel_error[0:3] = desired_vel[0:3] - current_vel
    vel_error[3] = desired_vel[3] - current_yaw_rate

    vel_integral += vel_error * dt
    d_vel_error = (vel_error - vel_prev_error) / dt if dt > 1e-6 else np.zeros(4)
    vel_prev_error = vel_error

    control_output = Kp_vel * vel_error + Ki_vel * vel_integral + Kd_vel * d_vel_error

    vx = np.cos(yaw) * control_output[0] + np.sin(yaw) * control_output[1]
    vy = -np.sin(yaw) * control_output[0] + np.cos(yaw) * control_output[1]
    vz = np.tanh(control_output[2]) * MAX_VEL
    yaw_rate = np.tanh(control_output[3]) * MAX_YAW_RATE
    print("yaw:", yaw, "yaw_actual:", current_yaw_rate)

    #print(vx, vy, vz, yaw_rate)

    # --- log recorded ---
    log_data['time'].append(log_data['time'][-1] + dt if log_data['time'] else 0)
    log_data['error_x'].append(pos_error[0])
    log_data['error_y'].append(pos_error[1])
    log_data['error_z'].append(pos_error[2])
    log_data['error_yaw'].append(pos_error[3])
    log_data['vx'].append(vx)

    log_data['vx_actual'].append(current_vel[0])
    log_data['vy_actual'].append(current_vel[1])
    log_data['vz_actual'].append(current_vel[2])
    log_data['yaw_rate_actual'].append(current_yaw_rate)


    log_data['vy'].append(vy)
    log_data['vz'].append(vz)
    log_data['yaw_rate'].append(yaw_rate)

    return (vx, vy, vz, yaw_rate)

def plot_controller_logs():
    t = log_data['time']
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))

    # error plot
    plt.subplot(3, 1, 1)
    plt.plot(t, log_data['error_x'], label='error_x')
    plt.plot(t, log_data['error_y'], label='error_y')
    plt.plot(t, log_data['error_z'], label='error_z')
    plt.plot(t, log_data['error_yaw'], label='error_yaw')
    plt.title("Position and Yaw Errors")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    # actual velocities
    plt.subplot(3, 1, 2)
    plt.plot(t, log_data['vx_actual'], label='vx_actual')
    plt.plot(t, log_data['vy_actual'], label='vy_actual')
    plt.plot(t, log_data['vz_actual'], label='vz_actual')
    plt.plot(t, log_data['yaw_rate_actual'], label='yaw_rate_actual')
    plt.title("Actual Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # control outputs
    plt.subplot(3, 1, 3)
    plt.plot(t, log_data['vx'], label='vx (output)')
    plt.plot(t, log_data['vy'], label='vy (output)')
    plt.plot(t, log_data['vz'], label='vz (output)')
    plt.plot(t, log_data['yaw_rate'], label='yaw_rate (output)')
    plt.title("Control Outputs")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
