import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Constants
VOLUME = 20  # litres
FLOW_RATE = 10  # litres per minute
HEAT_CAPACITY = 3.8  # kJ/(kg*K) for 50% glycol-water
DENSITY = 1.04  # kg/litre

# PID-controlled temperature model
def model(T, t, Q_cond, Q_edu, Kp, Ki, Kd, T_setpoint, integral, prev_error):
    error = T_setpoint - T
    dt = t[1] - t[0]
    integral += error * dt
    derivative = (error - prev_error) / dt
    Q_pid = Kp * error + Ki * integral + Kd * derivative
    Q_total = Q_cond + Q_edu + Q_pid
    dTdt = Q_total / (FLOW_RATE * DENSITY * HEAT_CAPACITY)
    return dTdt, integral, error

# Streamlit UI
st.title("Fluid Conditioning Loop Model")

# Sidebar controls
st.sidebar.header("Adjust Parameters")
Q_cond = st.sidebar.slider("Conditioning Unit Capacity (kW)", -24.0, 24.0, 0.0)
Q_edu = st.sidebar.slider("EDU Heat Output (kW)", 0.0, 10.0, 0.0)
Kp = st.sidebar.slider("PID Proportional Gain (Kp)", 0.0, 10.0, 1.0)
Ki = st.sidebar.slider("PID Integral Gain (Ki)", 0.0, 10.0, 0.0)
Kd = st.sidebar.slider("PID Derivative Gain (Kd)", 0.0, 10.0, 0.0)
T_setpoint = st.sidebar.slider("Setpoint Temperature (°C)", 0.0, 100.0, 50.0)
T_initial = st.sidebar.slider("Initial Temperature (°C)", 0.0, 100.0, 20.0)
duration = st.sidebar.slider("Simulation Duration (minutes)", 1, 60, 10)

# Time setup
time = np.linspace(0, duration, num=100)
T = T_initial
integral = 0
prev_error = 0
T_array = []

# Simulation loop
for t in range(len(time) - 1):
    dTdt, integral, prev_error = model(
        T, time[t:t+2], Q_cond, Q_edu, Kp, Ki, Kd, T_setpoint, integral, prev_error
    )
    T += dTdt * (time[t+1] - time[t])
    T_array.append(T)

# Plotting
fig, ax = plt.subplots()
ax.plot(time[:-1], T_array, label="Output Temperature")
ax.axhline(y=T_setpoint, color='r', linestyle='--', label="Setpoint")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)
