# %%theoretical calculations for bridge resistance
import numpy as np 


trace_width = [300,300,300, 500,500,500, 500]
neck_width = [50,100,150, 100, 100, 100, 20]
alignment_tol = [100, 100, 100, 50, 100, 200, 200]

trace_width = [w*1e-6 for w in trace_width]
neck_width = [w*1e-6 for w in neck_width]
SOI_h = 40e-6
Au1_h = 1e-6
Au2_h = 1e-6

rho_SOI = 20/100 # ohm-m
rho_Au = 0.022e-6 #ohm -m


def R(rho, l, width, h, width_2 = None):
    if not width_2 or (width==width_2):
        return rho*l/(width*h)
    else:
        return rho/(h *(width_2-width)/l) * np.log(width_2/width)

def par(*args):
    result = 1/sum([1/arg for arg in args])
    return result


def r_bridge(trace_width, neck_width, alignment_total):

    metal_gap = 5e-6
    inter_metal_gap =15e-6
    pad_size = 500e-6
    pad_gap = 30e-6
    trace_width_start = 500e-6

    ## section 1, pad
    r_soi = R(rho_SOI, 0.5*(pad_size-2*pad_gap), pad_size, SOI_h)
    r_met = R(rho_Au, 0.5*(pad_size-2*pad_gap),  (pad_size-2*pad_gap), Au1_h)

    r_1 = par(r_soi,r_met)
    # si only trace
    r_2 = R(rho_SOI, 780e-6, trace_width_start, SOI_h)
    r_3 = R(rho_SOI,505e-6, trace_width_start, trace_width, trace_width)

    # double_metalized

    r_4a_SOI = 2*(R(rho_SOI, 1480e-6 + 450e-6, trace_width, SOI_h) + R(rho_SOI, 1480e-6 + 450e-6, trace_width, SOI_h))
    r_4b_Au1 = 2*(R(rho_Au, 1480e-6 + 450e-6, trace_width-2*metal_gap, Au1_h) + R(rho_SOI, 1480e-6 + 450e-6, trace_width-2*metal_gap,Au1_h))
    r_4c_Au2 = 2*(R(rho_Au, 1480e-6 + 450e-6, trace_width-2*metal_gap-2*inter_metal_gap, Au2_h) + R(rho_SOI, 1480e-6 + 450e-6, trace_width-2*metal_gap-2*inter_metal_gap,Au2_h))

    r_4 = par(r_4a_SOI, r_4b_Au1, r_4c_Au2) 






    res_13 = 0
    res_24 = 0
    return res_13, res_24
# %%
