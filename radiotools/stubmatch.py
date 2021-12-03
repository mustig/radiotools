import numpy as np

import scipy.constants

import plotly.graph_objects as go

two_pi = 2*np.pi
HALF_PERIOD = np.linspace(0, np.pi, 157)
REFL_CIRCLE = np.exp(-2j*HALF_PERIOD)
N_PTS = 157

r2z = lambda r: (1 + r)/(1 - r)
z2r = lambda z: (z - 1)/(z + 1)
r2swr = lambda r: (1 + abs(r))/(1 - abs(r))

# Inputs

Z0 = 50

MHz = 14.1

RL = 200
XL = 0

vf = 0.83

def swr_circle(zL):
    refl_line = z2r(zL) * REFL_CIRCLE
    return r2z(refl_line)

def series_match(ZL, Z_tl):
    phase_range = np.linspace(0, np.pi, N_PTS)
    t = np.tan(phase_range)
    z_line = (Z_tl/Z0) * (ZL + Z_tl*1j*t) / (Z_tl + 1j*ZL*t)
    best_point = (abs(z_line)-1).argmin()
    refl_in = (z_line[best_point]-1) / (z_line[best_point]+1)
    swr_in = (1+abs(refl_in)) / (1-abs(refl_in))
    refl_line = refl_in * np.exp(-2j*phase_range)
    z_line_in = r2z(refl_line)
    match_length = phase_range[best_point]/(2*np.pi)
    return z_line, best_point, match_length, z_line_in, swr_in

def shunt_coil(ZL):
    phase_range = np.linspace(0, np.pi, N_PTS)
    YL = 1/ZL
    B_match = -YL.imag
    B_range = 1j*np.linspace(0, B_match, N_PTS)
    Z_match = 1/(YL + B_range)
    X_match = -1/B_match
    z_in = Z_match / Z0
    refl_line = z2r(z_in[-1]) * np.exp(-2j*phase_range)
    swr = r2swr(refl_line[0])
    return X_match, Z_match[-1], z_in, r2z(refl_line), swr

def stub_match(ZL):
    zl = ZL/Z0
    refl = z2r(zl)

    refl_mag = np.abs(refl)
    refl_ph = np.angle(refl)

    side = np.sqrt(1-(refl_mag**2)) / refl_mag

    _short_wl = np.arctan2(side, 2) / two_pi
    _open_wl =  np.arctan2(2, side) / two_pi
    _short_wl_alt = 0.5 - _short_wl
    _open_wl_alt = 0.5 - _open_wl

    short_stub_length_1 = _short_wl * wlm
    short_stub_length_2 = _short_wl_alt * wlm
    open_stub_length_1 = _open_wl * wlm
    open_stub_length_2 = _open_wl_alt * wlm

    shorts = []
    opens = []
    short_lengths = set()
    opens_lengths = set()
    for n in range(-1, 4):
        _d1_wl = (refl_ph + np.arcsin(refl_mag)) / (4*np.pi) + (n+(1/4))/2
        _d2_wl = (refl_ph - np.arcsin(refl_mag)) / (4*np.pi) + ((n+1)-(1/4))/2

        # if _d1_wl > 1:
        #     break

        d1 = _d1_wl * wlm
        d2 = _d2_wl * wlm

        if d1 > 0.1:
            if short_stub_length_1 not in short_lengths:
                short_lengths.add(short_stub_length_1)
                shorts.append((n, 'short 1', short_stub_length_1, d1))
            if open_stub_length_1 not in opens_lengths:
                opens_lengths.add(open_stub_length_1)
                opens.append((n, 'open 1', open_stub_length_1, d1))

        if d2 > 0.1:
            if short_stub_length_2 not in short_lengths:
                short_lengths.add(short_stub_length_2)
                shorts.append((n, 'short 2', short_stub_length_2, d2))
            if open_stub_length_2 not in opens_lengths:
                opens_lengths.add(open_stub_length_2)
                opens.append((n, 'open 2', open_stub_length_2, d2))



    return shorts, opens

def stub_match_trace(l, d, zL):
    visual_scaling = 1 + 0.01 * (-1 if d < 0.25 else 1)
    # First stub point
    _d1_wl_range = np.linspace(0, np.mod(d, 0.50))
    refl_d1 = z2r(zL) * np.exp(-4j*np.pi*_d1_wl_range)
    z_at_stub = r2z(refl_d1[-1])
    z_line = r2z(refl_d1*visual_scaling)

    z_stub = (1j*np.tan(two_pi*np.linspace(0.25, l, 157)))
    z_match = 1/(1/z_at_stub + 1/z_stub)
    z_trace = np.concatenate((z_line, z_match))
    return z_trace

# Calculations

f = MHz * 1e6

wl0 = scipy.constants.c / f
wlm = wl0 * vf

ZL = RL + 1j*XL
zL = ZL/Z0

SWR = r2swr(z2r(zL))
# return_loss_dB = -20*np.log10(abs(refl))

shorts, opens = stub_match(ZL)

print("Type           stub length (m)        distance (m)    wl")
for _, t, l, d in sorted(shorts+opens, key=lambda e: e[2]):
    print(f"{t:8s}       {l:.3f}                  {d:.3f}           {d/wlm:.3f}")


fig = go.Figure()
fig.update_layout(
    title=f"{RL:.1f} {XL:+.1f}j at {MHz:.1f} MHz"
)
plot_smith = lambda z, **kwargs: fig.add_trace(go.Scattersmith(imag=np.atleast_1d(z.imag), real=np.atleast_1d(z.real), **kwargs))

# Add zL
plot_smith(zL, marker_size=10, name=f"ZL {RL:.1f} {XL:+.1f}j  SWR {SWR:.1f}:1", marker_color="red")

# Circle of constant SWR
plot_smith(swr_circle(zL), line=dict(width=1, color="red", dash="dot"), name=f"SWR {SWR:.1f}:1", showlegend=False)

# Stub matches
_short1, _short2 = np.array(sorted(shorts, key=lambda e: e[2])[0:2])[:, 2:].astype(float) / wlm
plot_smith(stub_match_trace(*_short1, zL), line=dict(width=2, color="purple"), name=f"Stub match at {_short1[1]:.3f} wl")
plot_smith(stub_match_trace(*_short2, zL), line=dict(width=2, color="green"), name=f"Stub match at {_short2[1]:.3f} wl")

z_75, match_75, match_75_length, z_75_in, swr_75_in = series_match(ZL, 75)
plot_smith(z_75[:match_75+1], line=dict(width=2, color="orange"), mode="lines", name=f"75 ohm matching section ({match_75_length:.3f} wl)")
plot_smith(z_75[match_75+1:], line=dict(width=2, color="orange", dash="dash"), showlegend=False, mode="lines")
plot_smith(z_75_in, line=dict(width=1, color="orange", dash="dot"), name=f"SWR {swr_75_in:.1f}:1")

transformer_ratios = np.arange(2, 9)**2
z_transformed = (zL / transformer_ratios)
refl_line_transformed = z2r(z_transformed) * np.exp(-2j*HALF_PERIOD)[:, None]
swr_transformed = r2swr(refl_line_transformed[0, :])
for swr, r, n in sorted(zip(swr_transformed, refl_line_transformed.T, transformer_ratios.T)):
    if swr < SWR:
        print(f"{n:d}:1 transformer gives SWR {swr:.1f}:1")
        z_line_transformed = r2z(r)
        plot_smith(z_line_transformed, line=dict(width=1, color="red"), name=f"{n:d}:1 => SWR {swr:.1f}:1")
        break

if XL < -1:
    X_coil, Z_coil, z_shunt_coil, z_in_coil, swr_coil = shunt_coil(ZL)
    if X_coil > 0 and swr_coil < 3 and swr_coil < SWR-0.1:
        # Shunt inductance can possibly match this
        L_shunt = X_coil / (two_pi * f) * 1e6
        plot_smith(z_shunt_coil, line=dict(width=2, color="blue"), name=f"Shunt inductance ({L_shunt:2.2g} mH)")
        plot_smith(z_in_coil, line=dict(width=1, color="blue", dash="dot"), name=f"SWR {swr_coil:.1f}:1")

fig.show()