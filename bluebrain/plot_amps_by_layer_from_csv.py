import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
amps = pd.read_csv('recordings/amplitudes_by_distance_multiway', index_col=0)
amps = amps.iloc[:, 0]*1000
plt.figure(figsize=(10,8))
layers = ['L1', 'L23', 'L4', 'L5', 'L6']
for i, layer in enumerate(layers):
    j = [5, 4, 3, 2, 1][i]
    L_df = amps[amps.index.str.startswith(layer)]
    models = [model[:-2] for model in L_df.index]
    unique_models = list(dict.fromkeys(models))
    amps_in = []
    amps_ex = []
    for model in unique_models:
        model_df = L_df[L_df.index.str.startswith(model)]
        mean_amp = np.mean(model_df.values)
        if 'PC' in model or 'SS' in model or 'SP' in model:
            amps_ex.append(mean_amp)
            plt.plot(mean_amp, j*10, 'ro')
        else:
            amps_in.append(mean_amp)
            plt.plot(mean_amp, (j-0.5)*10, 'bo')
    plt.plot(np.mean(amps_ex), j*10, 'k*')
    plt.plot(np.mean(amps_in), (j-0.5)*10, 'k*')

plt.title('Amplitude measured 1 um outide of soma')
plt.vlines([15], ymin=5, ymax=55, colors='xkcd:sky blue', linestyles='dashed', label='15uV treshold')
plt.axvline(15, color='xkcd:sky blue', linestyle='dashed', label='15uV treshold')
plt.hlines([52.5, 42.5, 32.5, 22.5, 12.5], xmin=0, xmax=amps.max()+10)
plt.hlines([47.5, 37.5, 27.5, 17.5, 7.5], xmin=0, xmax=amps.max()+10, linestyles='dashed', colors='g')
plt.yticks([47.5, 37.5, 27.5, 17.5, 7.5], layers)
plt.xticks([0, 15, 100, 200, 300, 400, 500, 600, 700])
plt.axis(xlim=(0, 700), ylim=(0, 55))
plt.xlabel('Amplitude uV')
plt.savefig('AAmplitude by layer at 8um outside soma')

