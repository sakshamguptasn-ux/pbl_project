import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

b2.start_scope()
b2.prefs.codegen.target = 'numpy'
duration = 100 * b2.ms
pause = 20 * b2.ms
n_input = 64
n_excitatory = 10
R_target = 10 * b2.Hz
tau_avg = 50 * b2.ms

digits = load_digits()
data = digits.data
scaler = MinMaxScaler(feature_range=(0, 60))
data_rates = scaler.fit_transform(data)

neuron_eqs = '''
dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
dR_avg/dt = -R_avg / tau_avg : Hz
v_rest : volt
tau_m : second
'''

input_group = b2.PoissonGroup(n_input, rates=np.zeros(n_input)*b2.Hz)

output_group = b2.NeuronGroup(n_excitatory, neuron_eqs, 
                              threshold='v > -50*mV', 
                              reset='v = -65*mV; R_avg += 1*Hz', 
                              refractory=5*b2.ms, 
                              method='euler')
output_group.v = -65 * b2.mV
output_group.v_rest = -65 * b2.mV
output_group.tau_m = 100 * b2.ms
output_group.R_avg = 0 * b2.Hz

stdp_eqs = '''
w : 1
dapre/dt = -apre / taupre : 1 (event-driven)
dapost/dt = -apost / taupost : 1 (event-driven)
taupre = 20*ms : second
taupost = 20*ms : second
Apre = 0.01 : 1
Apost = -0.012 : 1
'''

on_pre = '''
apre += Apre
M = (R_target / (R_avg_post + 1*Hz)) 
w = clip(w + M * apost, 0, 1)
v_post += w * 5*mV
'''
on_post = '''
apost += Apost
w = clip(w + apre, 0, 1)
'''

synapses = b2.Synapses(input_group, output_group,
                       model=stdp_eqs,
                       on_pre=on_pre,
                       on_post=on_post)
synapses.connect()
synapses.w = 'rand() * 0.5'

spikemon = b2.SpikeMonitor(output_group)
statemon = b2.StateMonitor(output_group, 'R_avg', record=True)

for i in range(1500):
    input_group.rates = data_rates[i] * b2.Hz
    b2.run(duration)
    input_group.rates = 0 * b2.Hz
    b2.run(pause)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for i in range(n_excitatory):
    plt.plot(statemon.t/b2.ms, statemon.R_avg[i], label=f'Neuron {i}')
plt.axhline(y=R_target/b2.Hz, color='r', linestyle='--')
plt.title("Homeostatic Regulation")
plt.xlabel("Time (ms)")
plt.ylabel("Avg Firing Rate (Hz)")

plt.subplot(1, 3, 2)
plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
plt.title("Spiking Activity")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron Index")

plt.subplot(1, 3, 3)
plt.hist(synapses.w, bins=20, color='purple', alpha=0.7)
plt.title("Weight Distribution")
plt.xlabel("Weight Strength")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
