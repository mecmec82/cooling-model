import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# PID Controller Class with Anti-Windup
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0
        self.output_limits = output_limits

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        # Anti-Windup: Clamp integral term
        if self.output_limits[0] is not None and self.integral < self.output_limits[0]:
            self.integral = self.output_limits[0]
        elif self.output_limits[1] is not None and self.integral > self.output_limits[1]:
            self.integral = self.output_limits[1]

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# System Model with warm-up and cool-down delays
def system_model(time_steps, target_temperatures, durations, Kp, Ki, Kd, heating_delay, cooling_delay):
    heating_power = 24  # kW
    cooling_power = {20: 35, 0: 25, -20: 14.5, -30: 6.5, -40: 2.6}  # kW
    fluid_capacity = 100  # Arbitrary unit
    dt = 1  # seconds

    # Build target temperature profile
    target_profile = []
    for temp, duration in zip(target_temperatures, durations):
        target_profile.extend([temp] * duration)

    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=target_profile[0], output_limits=(-10, 10))
    temperatures = [target_profile[0]]
    heating_transition = 0
    cooling_transition = 0
    heating_active = False
    cooling_active = False

    for t in range(1, len(target_profile)):
        current_temp = temperatures[-1]
        pid.setpoint = target_profile[t]
        control_signal = pid.update(current_temp, dt)

        if control_signal > 0:
            if not heating_active:
                heating_transition += dt
                if heating_transition >= heating_delay:
                    heating_active = True
                    heating_transition = heating_delay
            cooling_active = False
            cooling_transition = 0
            max_heating_rate = heating_power * (heating_transition / heating_delay if heating_delay > 0 else 1) / fluid_capacity
            temp_change = min(control_signal, max_heating_rate)
        else:
            if not cooling_active:
                cooling_transition += dt
                if cooling_transition >= cooling_delay:
                    cooling_active = True
                    cooling_transition = cooling_delay
            heating_active = False
            heating_transition = 0
            applicable_keys = list(filter(lambda x: x <= current_temp, cooling_power.keys()))
            cooling = cooling_power[max(applicable_keys)] if applicable_keys else cooling_power[min(cooling_power.keys())]
            max_cooling_rate = cooling * (cooling_transition / cooling_delay if cooling_delay > 0 else 1) / fluid_capacity
            temp_change = max(control_signal, -max_cooling_rate)

        new_temp = current_temp + temp_change * dt
        temperatures.append(new_temp)

    return temperatures, target_profile

# Streamlit UI
st.title("Temperature Control Simulation")

# Sidebar inputs
st.sidebar.header("PID Settings")
Kp = st.sidebar.slider("Kp", 0.0, 20.0, 1.0)
Ki = st.sidebar.slider("Ki", 0.0, 1.0, 0.001)
Kd = st.sidebar.slider("Kd", 0.0, 1.0, 0.005)

st.sidebar.header("Delay Settings")
heating_delay = st.sidebar.slider("Heating Delay (seconds)", 0, 5000, 300)
cooling_delay = st.sidebar.slider("Cooling Delay (seconds)", 0, 5000, 300)

st.sidebar.header("Target Temperatures and Durations")
target_temperatures = st.sidebar.text_input("Target Temperatures (comma-separated)", "20, 10, 30")
durations = st.sidebar.text_input("Durations (comma-separated, seconds)", "60, 120, 180")

# Convert input strings to lists
target_temperatures = [float(temp.strip()) for temp in target_temperatures.split(",")]
durations = [int(d.strip()) for d in durations.split(",")]

# Run simulation
time_steps = sum(durations)
temperatures, target_profile = system_model(time_steps, target_temperatures, durations, Kp, Ki, Kd, heating_delay, cooling_delay)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(temperatures, label='Actual Temperature')
ax.plot(target_profile, label='Target Temperature', linestyle='--')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature Evolution Over Time')
ax.legend()
ax.grid(True)

# Display plot
st.pyplot(fig)
