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

def main():
    SOI_only_res()
    
# function to calculate resistance with rho resistivity
# width and height and option second width
def R(rho, l, width, h, width_2 = None):
    if not width_2 or (width==width_2):
        return rho*l/(width*h)
    else:
        return rho/(h *(width_2-width)/l) * np.log(width_2/width)
	
# returns parallel resistance of input restistances
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

#trace measurements
    
SOI_rho = 0.2
leg_len = [783e-6, 780e-6, 780e-6, 1285e-6, 1285e-6,1285e-6]
leg_w = [500e-6, 500e-6, 500e-6, 500e-6,500e-6,500e-6]
    
hip_len = [500e-6, 504e-6, 505e-6, None, None, None]
hip_w1 = [500e-6, 500e-6, 500e-6, None, None, None]
hip_w2 = [300e-6, 300e-6, 300e-6, None, None, None]
    
ear_len = [30e-6, 55e-6, 80e-6, None, None, 55]
ear_w1 = [50e-6, 100e-6, 150e-6, None, None, 100]
ear_w2 = [125e-6, 205e-6, 265e-6, None, None, 205]
    
bridge_len = [200e-6, 200e-6, 200e-6, 210e-6, 310e-6, 400e-6]
bridge_w1 = [50e-6, 100e-6, 150e-6, 100e-6, 100e-6, 100e-6]

gold_bridge_w = [80e-6, 130e-6, 180e-6, 130e-6, 130e-6, 130e-6]
gold_bridge_len = [10e-6, 60e-6, 110e-6, 60e-6, 60e-6, 60e-6]

    
def SOI_only_res():
    for i in range(6):
        print('SOI_only_res for trace 1_3 of bridge ', i, '    :', SOI_only_res_1(i)/1000, ' kOhms')
        print('SOI_only_res for trace 1_3 of bridge ', i, '+Au :', SOI_with_gold_only_res_1(i)/1000, ' kOhms')
        print('                          difference        :', SOI_with_gold_only_res_1(i)/1000 - SOI_only_res_1(i)/1000, ' kOhms')
        print('SOI_only_res for trace 2_4 of bridge ', i, '    :', SOI_only_res_2(i)/1000, ' kOhms\n')

def SOI_only_res_1(i):
    return 2 * (leg_res(i) + hip_res(i))

def SOI_only_res_2(i):
    return SOI_bridge_res(i) + SOI_only_res_1(i)
    
def SOI_with_gold_only_res_1(i):
    return SOI_only_res_1(i) + gold_only_bridge(i)

def gold_only_bridge(i):
    return R(rho_Au, gold_bridge_len[i], gold_bridge_w[i], Au2_h)

def leg_res(i):
    return R(SOI_rho, leg_len[i], leg_w[i], SOI_h)
    
def hip_res(i):
    if hip_len[i] == None:
        return 0
    return R(SOI_rho, hip_len[i], hip_w1[i], SOI_h, hip_w2[i])
        
def SOI_bridge_res(i):
    if ear_len[i] == None:
        ear = 0
    else:
        ear = R(SOI_rho, ear_len[i], ear_w1[i], SOI_h, ear_w2[i])
    bridge = R(SOI_rho, bridge_len[i], bridge_w1[i], SOI_h)
    return bridge + (2 * ear)

    
if __name__ == "__main__":
    main()
