import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Constants
VOLUME = 20  # litres
FLOW_RATE = 10  # litres per minute
HEAT_CAPACITY = 3.8  # kJ/(kg*K)
DENSITY = 1.04  # kg/litre

# Non-linear ramp function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# PID-controlled temperature model with non-linear ramp-up and cool-down delay
def model(T, t, Q_edu, Kp, Ki, Kd, T_setpoint, integral, prev_error, ramp_up_time, ramp_down_time, prev_Q_cond):
    error = T_setpoint - T
    dt = t[1] - t[0]
    integral += error * dt
    derivative = (error - prev_error) / dt
    Q_pid = Kp * error + Ki * integral + Kd * derivative
    Q_cond = np.clip(Q_pid, -24, 24)

    # Apply non-linear ramp-up or cool-down if switching direction
    if np.sign(Q_cond) != np.sign(prev_Q_cond):
        ramp_time = ramp_down_time if Q_cond < 0 else ramp_up_time
        ramp_factor = sigmoid((t[1] / ramp_time) * 6 - 3)  # sigmoid centered at ramp_time/2
        Q_cond *= ramp_factor

    Q_total = Q_cond + Q_edu
    dTdt = Q_total / (FLOW_RATE * DENSITY * HEAT_CAPACITY)
    return dTdt, integral, error, Q_cond

# Streamlit UI
st.title("Fluid Conditioning Loop Model")

# Sidebar controls
st.sidebar.header("Adjust Parameters")
Q_edu = st.sidebar.slider("EDU Heat Output (kW)", 0.0, 10.0, 0.0)
Kp = st.sidebar.slider("PID Proportional Gain (Kp)", 0.0, 200.0, 100.0)
Ki = st.sidebar.slider("PID Integral Gain (Ki)", 0.0, 10.0, 0.0)
Kd = st.sidebar.slider("PID Derivative Gain (Kd)", 0.0, 10.0, 0.0)
T_setpoint = st.sidebar.slider("Setpoint Temperature (°C)", -20.0, 50.0, 10.0)
T_initial = st.sidebar.slider("Initial Temperature (°C)", 0.0, 100.0, 20.0)
duration = st.sidebar.slider("Simulation Duration (minutes)", 1, 500, 60)
ramp_up_time = st.sidebar.slider("Ramp-Up Time (minutes)", 0.1, 1000.0, 200.0)
ramp_down_time = st.sidebar.slider("Ramp-Down Time (minutes)", 0.1, 1000.0, 200.0)

# Time setup
time = np.linspace(0, duration, num=100)
T = T_initial
integral = 0
prev_error = 0
T_array = []
Q_cond_array = []
prev_Q_cond = 0

# Simulation loop
for t in range(len(time) - 1):
    dTdt, integral, prev_error, Q_cond = model(
        T, time[t:t+2], Q_edu, Kp, Ki, Kd, T_setpoint, integral, prev_error,
        ramp_up_time, ramp_down_time, prev_Q_cond
    )
    T += dTdt * (time[t+1] - time[t])
    T_array.append(T)
    Q_cond_array.append(Q_cond)
    prev_Q_cond = Q_cond

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Temperature plot
ax[0].plot(time[:-1], T_array, label="Output Temperature")
ax[0].axhline(y=T_setpoint, color='r', linestyle='--', label="Setpoint")
ax[0].set_xlabel("Time (minutes)")
ax[0].set_ylabel("Temperature (°C)")
ax[0].legend()

# Conditioning output plot with color-coded segments
for i in range(len(Q_cond_array) - 1):
    color = 'red' if Q_cond_array[i] > 0 else 'blue'
    ax[1].plot(time[i:i+2], Q_cond_array[i:i+2], color=color)

ax[1].set_xlabel("Time (minutes)")
ax[1].set_ylabel("Conditioning Output (kW)")
ax[1].legend(["Heating (red)", "Cooling (blue)"])

st.pyplot(fig)
