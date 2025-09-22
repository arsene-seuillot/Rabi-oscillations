import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# -------------------------------
# Paramètres physiques
# -------------------------------
Omega = 2.0
delta_ratio_init = 0.5
t_max = 10
n_frames = 200
dt = t_max / n_frames

# -------------------------------
# Fonction de rotation autour de B_eff
# -------------------------------
def rotate(state, B_eff, t):
    B_mag = np.linalg.norm(B_eff)
    B_unit = B_eff / B_mag
    theta = B_mag * t
    return (state*np.cos(theta) +
            np.cross(B_unit, state)*np.sin(theta) +
            B_unit*np.dot(B_unit, state)*(1-np.cos(theta)))

# -------------------------------
# Figure principale avec gridspec
# -------------------------------
plt.close('all')
fig = plt.figure(figsize=(7,9))
gs = GridSpec(2,1, height_ratios=[3,1], hspace=0.3)

# Axe sphère de Bloch
ax = fig.add_subplot(gs[0], projection='3d')

u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, color='lightblue', alpha=0.2)

theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta), 'k--', linewidth=0.8)
ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'k--', linewidth=0.8)

ax_lim = 1.1
ax.quiver(-ax_lim,0,0,2*ax_lim,0,0,color='k', arrow_length_ratio=0.05)
ax.quiver(0,-ax_lim,0,0,2*ax_lim,0,color='k', arrow_length_ratio=0.05)
ax.quiver(0,0,-ax_lim,0,0,2*ax_lim,color='k', arrow_length_ratio=0.05)
ax.text(ax_lim+0.05, 0, 0, 'X', color='k')
ax.text(0, ax_lim+0.05, 0, 'Y', color='k')
ax.text(0, 0, ax_lim+0.05, '|0⟩', color='k')
ax.text(0, 0, -ax_lim-0.05, '|1⟩', color='k')

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_box_aspect([1,1,1])

# -------------------------------
# Axe probabilité |1⟩
# -------------------------------
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, t_max)
ax2.set_ylim(0,1)
ax2.set_xlabel('Temps')
ax2.set_ylabel('Probabilité |1⟩')
ax2.set_title('Transition vers |1⟩')
line_prob, = ax2.plot([], [], 'b', lw=2)
t_vals = np.linspace(0, t_max, n_frames)
P1_vals = np.zeros_like(t_vals)

# -------------------------------
# Vecteurs et trajectoire
# -------------------------------
state_init = np.array([0,0,1])
line_state, = ax.plot([0,0],[0,0],[0,1], color='g', lw=2)
line_B, = ax.plot([0,0],[0,0],[0,1], color='r', lw=2)
traj_line, = ax.plot([], [], [], color='g', lw=1)

marker_state, = ax.plot([state_init[0]], [state_init[1]], [state_init[2]], 'go', markersize=6)
marker_B, = ax.plot([Omega], [0], [delta_ratio_init*Omega], 'ro', markersize=6)

x_traj, y_traj, z_traj = [], [], []

# -------------------------------
# Slider
# -------------------------------
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
slider = Slider(ax_slider, 'δ/Ω', 0.0, 10, valinit=delta_ratio_init, valstep=0.01)
fig.text(0.2, 0.06, r'$\delta$ = detuning', ha='center', fontsize=10)

def reset_all(val):
    global x_traj, y_traj, z_traj, P1_vals
    # Réinitialiser la trajectoire sur la sphère
    x_traj, y_traj, z_traj = [], [], []
    traj_line.set_data([], [])
    traj_line.set_3d_properties([])

    # Réinitialiser la probabilité |1⟩
    P1_vals[:] = 0
    line_prob.set_data([], [])


slider.on_changed(reset_all)

# -------------------------------
# Fonction de mise à jour
# -------------------------------
def update(frame):
    global x_traj, y_traj, z_traj, P1_vals
    delta = slider.val * Omega
    B_eff = np.array([Omega, 0, delta])
    t = frame * dt
    state_vec = rotate(state_init, B_eff, t)
    B_unit = B_eff / np.linalg.norm(B_eff)

    # Vecteurs sur la sphère
    line_state.set_data([0, state_vec[0]], [0, state_vec[1]])
    line_state.set_3d_properties([0, state_vec[2]])
    line_B.set_data([0, B_unit[0]], [0, B_unit[1]])
    line_B.set_3d_properties([0, B_unit[2]])

    marker_state.set_data([state_vec[0]], [state_vec[1]])
    marker_state.set_3d_properties([state_vec[2]])
    marker_B.set_data([B_unit[0]], [B_unit[1]])
    marker_B.set_3d_properties([B_unit[2]])

    # Trajectoire
    x_traj.append(state_vec[0])
    y_traj.append(state_vec[1])
    z_traj.append(state_vec[2])
    traj_line.set_data(x_traj, y_traj)
    traj_line.set_3d_properties(z_traj)

    # Probabilité |1⟩
    P1_vals[frame] = (1 - state_vec[2])/2
    line_prob.set_data(t_vals[:frame+1], P1_vals[:frame+1])

    ax.set_title(f'Oscillations de Rabi : δ/Ω = {slider.val:.2f}')
    return line_state, line_B, traj_line, marker_state, marker_B, line_prob

# -------------------------------
# Animation
# -------------------------------
anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
plt.show()
