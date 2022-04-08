import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import constants

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

P = 100
Z0 = 50

U_rms = np.sqrt(P*Z0)
I_rms = np.sqrt(P/Z0)

A = np.sqrt(2)*I_rms
f = 7.200E6
wl = constants.c/f

view_A = 2.5*A
wire_length = 1/4*wl

periods = 3
x = np.linspace(-wire_length, wire_length, 300)
t_frames = np.linspace(0, periods/f, 200*periods)

def wave_right(t):
    return A * np.cos(2*np.pi*(x/wl - f*t))

def wave_left(t):
    return A * np.cos(2*np.pi*(x/wl + f*t))

ln_wire, = ax.plot(x, 0*x, 'k')
ln_r, = ax.plot(x, wave_right(0), 'g')
ln_l, = ax.plot(x, wave_left(0), 'r')
ln_sum, = ax.plot(x, wave_right(0)+wave_left(0), 'tab:blue')

def init():
    ax.set_xlim(-1.5*wire_length, 1.5*wire_length)
    ax.set_ylim(-view_A, view_A)
    return ln_r, ln_l, ln_sum

def update(t_frame):
    w_r = wave_right(t_frame)
    w_l = wave_left(t_frame)
    ln_r.set_ydata(w_r)
    ln_l.set_ydata(w_l)
    ln_sum.set_ydata(w_r + w_l)
    return ln_r, ln_l, ln_sum

ani = FuncAnimation(fig, update, frames=t_frames,
                    init_func=init, blit=True, interval=40, repeat=False)
plt.show()