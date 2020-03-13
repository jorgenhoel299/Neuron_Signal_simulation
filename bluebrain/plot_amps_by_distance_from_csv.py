import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

amps = pd.read_csv('recordings/amplitudes_by_distance_multiway', index_col=0)
amps = amps*1000
n_electrodes=30
distances = np.linspace(8, 100, n_electrodes)
fig = plt.figure(figsize=(10, 8))
fig.suptitle('P2P amplitude measured from soma')
ax1 = fig.add_subplot(3, 2, 1)
ax23 = fig.add_subplot(3, 2, 2)
ax4 = fig.add_subplot(3, 2, 3)
ax5 = fig.add_subplot(3, 2, 4)
ax6 = fig.add_subplot(3, 2, 5)
axes = [ax1, ax23, ax4, ax5, ax6]
n_models = 3
for i, layer in enumerate(['L1', 'L23', 'L4', 'L5', 'L6']):
    L_df = amps[amps.index.str.startswith(layer)]
    L_df = L_df.reindex(L_df.mean(axis=1).sort_values().index, axis=0)
    models = [model[:-2] for model in L_df.index]
    unique_models = list(dict.fromkeys(models))
    for model in unique_models[:n_models]:
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
    axes[i].set_xticks(np.arange(np.min(distances), np.max(distances), 5))
    axes[i].legend()

fig.savefig('figures/amps_by_distance_multiway_onlysmall',dpi=400)

fig2 = plt.figure(figsize=(10, 8))
plt.title('P2P amplitude by recording distance')

colors = ['g', 'b', 'r', 'y', 'm']
for k, layer in enumerate(['L1', 'L23', 'L4', 'L5', 'L6']):
    ex_cells = np.zeros(n_electrodes)
    in_cells = np.zeros(n_electrodes)
    i, j = (0, 0)
    L_df = amps[amps.index.str.startswith(layer)]
    for row in L_df.iterrows():
        if 'PC' in row[0] or 'SS' in row[0]  or 'SP' in row[0]:
            i += 1
            ex_cells = np.add(ex_cells, row[1])
        else:
            j += 1
            in_cells = np.add(in_cells, row[1])
    plt.plot(distances, ex_cells/i, color=colors[k], label=layer + ' excitatory')
    plt.plot(distances, in_cells/j, color=colors[k], linestyle='dashed', label=layer +' inhibitory')
plt.plot(distances, [15]*n_electrodes, 'k--', label='15uV treshold')
plt.legend()
plt.xlabel('Distance from soma center [um]')
plt.ylabel('Amplitude [uV]')
fig2.savefig('figures/amps_by_distance_multiway_averages',dpi=400)