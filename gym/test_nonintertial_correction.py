import numpy as np
import matplotlib.pyplot as plt

# Assuming N time steps
N = 100
t = np.linspace(0, 10, N)  # Time steps

# Example positions of the two points over time
r1 = np.array([np.sin(t), np.cos(t)]).T  # Position for P1
r2 = np.array([np.sin(t + 0.5), np.cos(t + 0.5)]).T  # Position for P2

# Calculate the velocity of each point
v1 = np.gradient(r1, axis=0)
v2 = np.gradient(r2, axis=0)

# Calculate the linear velocity of the frame
v_frame = v1

# Calculate the orientation angle theta of the frame
d = r2 - r1
theta = np.arctan2(d[:, 1], d[:, 0])

# Unwrap the theta values to handle wrapping around
theta_unwrapped = np.unwrap(theta)

# Calculate the angular velocity omega of the frame
omega = np.gradient(theta_unwrapped, t)

# Calculate the acceleration of the frame
a_frame = np.gradient(v_frame, axis=0)

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot Linear Velocity
plt.subplot(2, 2, 1)
plt.plot(t, v_frame[:, 0], label='v_x (frame)')
plt.plot(t, v_frame[:, 1], label='v_y (frame)')
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity (m/s)')
plt.title('Linear Velocity of the Frame')
plt.legend()

# Plot Acceleration
plt.subplot(2, 2, 2)
plt.plot(t, a_frame[:, 0], label='a_x (frame)')
plt.plot(t, a_frame[:, 1], label='a_y (frame)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration of the Frame')
plt.legend()

# Plot Orientation Angle
plt.subplot(2, 2, 3)
plt.plot(t, theta_unwrapped)
plt.xlabel('Time (s)')
plt.ylabel('Orientation Angle (radians)')
plt.title('Orientation Angle of the Frame')

# Plot Angular Velocity
plt.subplot(2, 2, 4)
plt.plot(t, omega)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (radians/s)')
plt.title('Angular Velocity of the Frame')

plt.tight_layout()
plt.show()
