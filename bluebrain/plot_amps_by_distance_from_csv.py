import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

amps = pd.read_csv('amplitudes', index_col=0)
amps = amps*1000
n_electrodes=30
distances = np.linspace(15, 100, n_electrodes)
fig3 = plt.figure(figsize=(10, 8))
fig3.suptitle('P2P amplitude measured from soma')
ax1 = fig3.add_subplot(3, 2, 1)
ax23 = fig3.add_subplot(3, 2, 2)
ax4 = fig3.add_subplot(3, 2, 3)
ax5 = fig3.add_subplot(3, 2, 4)
ax6 = fig3.add_subplot(3, 2, 5)
axes = [ax1, ax23, ax4, ax5, ax6]
for i, layer in enumerate(['L1', 'L23', 'L4', 'L5', 'L6']):
    L_df = amps[amps.index.str.startswith(layer)]
    models = [model[:-2] for model in L_df.index]
    unique_models = list(dict.fromkeys(models))
    for model in unique_models:
        model_df = L_df[L_df.index.str.startswith(model)]
        y = np.mean(model_df.values, axis=0)
        error = np.std(model_df.values, axis=0)
        axes[i].errorbar(distances, y, yerr=error, label=model)
    axes[i].plot(distances, [15]*n_electrodes, 'b--', label='15uV treshold')
    axes[i].set_title('Layer {}'.format(layer[1:]))
    if i ==3 or i == 4:
        axes[i].set_xlabel('Distance from soma (um)')
    if i == 0 or i == 2 or i == 4:
        axes[i].set_ylabel('Amplitude (uV)')
    #axes[i].set_xticks(np.arange(np.min(distances), np.max(distances), 5))
    axes[i].legend()

fig3.savefig('amps_by_distance2',dpi=400)