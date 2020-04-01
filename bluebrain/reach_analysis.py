import matplotlib
matplotlib.use('AGG')
import os
import posixpath
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection, LineCollection
from warnings import warn
from glob import glob
import scipy.signal as ss
import neuron
import LFPy
from mpi4py import MPI
import operator


treshold = 0.03
syn_df = pd.read_csv('recordings/multiway_amplitude_from_1um_to_70_outside_soma').set_index('Unnamed: 0')
vclamp_df = pd.read_csv('vclamp/recordings/vclamp_amplitude_from_1um_to_70_outside_soma').set_index('Unnamed: 0')
neuron_counts_df = pd.read_csv('neuron_counts.csv', names=['neuron', 'ammount'], header=None)

count_dict = dict(zip(neuron_counts_df.neuron, neuron_counts_df.ammount))

layers = ['L1', 'L23', 'L4', 'L5', 'L6']
total_nrns_layer = dict(zip(layers, [338, 7524, 4656, 6114, 12651]))
layer_densities = dict(zip(layers, [14200, 164600, 177300, 83900, 131500]))

syn_count_dist = {}
vclamp_count_dist = {}
n_measured_syn = {}
n_measured_vclamp = {}
dens_vol_syn = {}
dens_vol_vclamp = {}

for NRN in count_dict.keys():
    layer = NRN[:2]
    layer = NRN[:3] if layer == 'L2' else layer
    syn_nrn_df = syn_df[syn_df.index.str.startswith(NRN)]
    vclamp_nrn_df = vclamp_df[vclamp_df.index.str.startswith(NRN)]
    all_rad_syn = []
    all_rad_vclamp  = []
    for row in syn_nrn_df.iterrows():
        spot = np.where(row[1].drop(['rec_spots']) <= treshold)[0][0]
        rec_spots = [float(i) for i in row[1].rec_spots.strip('[]').split(', ')]
        syn_rad = rec_spots[max(0, spot-1)]
        all_rad_syn.append(syn_rad)
    for row in vclamp_nrn_df.iterrows():
        spot = np.where(row[1].drop(['rec_spots']) <= treshold)[0][0]
        rec_spots = [float(i) for i in row[1].rec_spots.strip('[]').split(', ')]
        vclamp_rad = rec_spots[max(0, spot-1)]
        all_rad_vclamp.append(vclamp_rad)
    syn_count_dist[NRN] = np.mean(all_rad_syn)
    vclamp_count_dist[NRN] = np.mean(all_rad_vclamp)
    
    volume_syn = 4/3 * np.pi * (syn_rad/1000)**3
    volume_vclamp = 4/3 * np.pi * (vclamp_rad/1000)**3
    neuron_density = layer_densities[layer] * int(neuron_counts_df.loc[neuron_counts_df['neuron'] == 
                      NRN].ammount) / total_nrns_layer[layer]

    dens_vol_syn[NRN] = (volume_syn, neuron_density)
    dens_vol_vclamp[NRN] = (volume_vclamp, neuron_density)
    n_measured_syn[NRN] = neuron_density * volume_syn
    n_measured_vclamp[NRN] = neuron_density * volume_vclamp

L5_keys = [key for key in syn_count_dist.keys() if 'L5' in key]
L5_syn_dict = {key: syn_count_dist[key] for key in L5_keys}

# Number of cells measured in a layer
fig = plt.figure(figsize=(10, 8))
fig.suptitle('Measured neurons, synaptic input, treshold {} uV'.format(treshold*1000))
ax1 = fig.add_subplot(3, 2, 1)
ax23 = fig.add_subplot(3, 2, 2)
ax4 = fig.add_subplot(3, 2, 3)
ax5 = fig.add_subplot(3, 2, 4)
ax6 = fig.add_subplot(3, 2, 5)
axes = [ax1, ax23, ax4, ax5, ax6]
for i, ax in  enumerate(axes):
    layer = layers[i]
    nrns = [nrn for nrn in n_measured_syn.keys() if layer in nrn[:3]]
    vals = [val for i, val in enumerate(n_measured_syn.values()) if layer in list(n_measured_syn.keys())[i][:3]]
    vals, nrns = zip(*sorted(zip(vals, nrns), reverse=True))
    ax.bar(nrns, vals)
    ax.set_title('Layer {}'.format(layer[1:]))
    ax.set_xticklabels(nrns,rotation=45)
plt.savefig('figures/bar_every_neuron_syn_{}_uV_finer'.format(int(treshold*1000)))

fig2 = plt.figure(figsize=(10, 8))
fig2.suptitle('Measured neurons, voltage clamped, treshold {} uV'.format(treshold*1000))
ax1 = fig2.add_subplot(3, 2, 1)
ax23 = fig2.add_subplot(3, 2, 2)
ax4 = fig2.add_subplot(3, 2, 3)
ax5 = fig2.add_subplot(3, 2, 4)
ax6 = fig2.add_subplot(3, 2, 5)
axes = [ax1, ax23, ax4, ax5, ax6]
for i, ax in  enumerate(axes):
    layer = layers[i]
    nrns = [nrn for nrn in n_measured_syn.keys() if layer in nrn[:3]]
    vals = [val for i, val in enumerate(n_measured_syn.values()) if layer in list(n_measured_vclamp.keys())[i][:3]]
    vals, nrns = zip(*sorted(zip(vals, nrns), reverse=True))
    ax.bar(nrns, vals)
    ax.set_title('Layer {}'.format(layer[1:]))
    ax.set_xticklabels(nrns,rotation=45)
plt.savefig('figures/bar_every_neuron_vclamp_{}_uV_finer'.format(int(treshold*1000)))

# Plot for excitatory vs inhibitory cells
fig3 = plt.figure(figsize=(10, 15))
fig3.suptitle('Measured neurons, synaptic input, treshold {} uV'.format(treshold*1000))
ax1 = fig3.add_subplot(3, 2, 1)
ax23 = fig3.add_subplot(3, 2, 2)
ax4 = fig3.add_subplot(3, 2, 3)
ax5 = fig3.add_subplot(3, 2, 4)
ax6 = fig3.add_subplot(3, 2, 5)
axes = [ax1, ax23, ax4, ax5, ax6]
ylim = 5
for i, ax in  enumerate(axes):
    layer = layers[i]
    ex_nrns = []
    ex_vals = []
    in_nrns = []
    in_vals = []
    for i, nrn in enumerate(n_measured_syn.keys()):
        if layer in nrn[:3]:
            if 'PC' in nrn or 'SS' in nrn  or 'SP' in nrn:
                ex_nrns.append(nrn); ex_vals.append(n_measured_syn[nrn])
            else:
                in_nrns.append(nrn); in_vals.append(n_measured_syn[nrn])
    if layer == 'L1':
        ax.bar(1, [np.sum(in_vals)], color='b')
        ax.set_xticks([1])
        ax.set_xticklabels(['Inhibitory'],rotation=0)
    else:
        ax.bar([1, 2], [np.sum(ex_vals), np.sum(in_vals)], color=['r', 'b'])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Excitatory', 'Inhibitory'],rotation=0)
    ylim = np.sum(ex_vals) + 3 if ylim < np.sum(ex_vals) - 2 else ylim
    ax.set_xlim(0, 3)
    ax.set_title('Layer {}'.format(layer[1:]))
for ax in axes:
    ax.set_ylim(0, ylim)
plt.savefig('figures/bar_ex_in_syn_{}_uV_finer'.format(int(treshold*1000)))

ylim = 5
fig4 = plt.figure(figsize=(10, 15))
fig4.suptitle('Measured neurons, voltage clamped, treshold {} uV'.format(treshold*1000))
ax1 = fig4.add_subplot(3, 2, 1)
ax23 = fig4.add_subplot(3, 2, 2)
ax4 = fig4.add_subplot(3, 2, 3)
ax5 = fig4.add_subplot(3, 2, 4)
ax6 = fig4.add_subplot(3, 2, 5)
axes = [ax1, ax23, ax4, ax5, ax6]
for i, ax in  enumerate(axes):
    layer = layers[i]
    ex_nrns = []
    ex_vals = []
    in_nrns = []
    in_vals = []
    for i, nrn in enumerate(n_measured_vclamp.keys()):
        if layer in nrn[:3]:
            if 'PC' in nrn or 'SS' in nrn  or 'SP' in nrn:
                ex_nrns.append(nrn); ex_vals.append(n_measured_vclamp[nrn])
            else:
                in_nrns.append(nrn); in_vals.append(n_measured_vclamp[nrn])
    if layer == 'L1':
        ax.bar(1, [np.sum(in_vals)], color='b')
        ax.set_xticks([1])
        ax.set_xticklabels(['Inhibitory'],rotation=0)
    else:
        ax.bar([1, 2], [np.sum(ex_vals), np.sum(in_vals)], color=['r', 'b'])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Excitatory', 'Inhibitory'],rotation=0)
    ylim = np.sum(ex_vals) + 3 if ylim < np.sum(ex_vals) - 2 else ylim
    ax.set_xlim(0, 3)
    ax.set_title('Layer {}'.format(layer[1:]))
for ax in axes:
    ax.set_ylim(0, ylim)
plt.savefig('figures/bar_ex_in_vclamp_{}_uV_finer'.format(int(treshold*1000)))

# Illustration of different volumes and morphologies(surfaces)
n_exc = 2
n_inh = 2
n_tot = n_exc + n_inh
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
#working dir
CWD = os.getcwd()
NMODL = 'hoc_combos_syn.1_0_10.allmods'

#load some required neuron-interface files
neurons_starts = [nrn[0] for nrn in sorted(L5_syn_dict.items(), key = operator.itemgetter(1), reverse = True)[:n_exc]]
for nrn in sorted(L5_syn_dict.items(), key = operator.itemgetter(1))[:n_inh][::-1]:
    neurons_starts.append(nrn[0])
neurons = [glob(os.path.join('hoc_combos_syn.1_0_10.allzips', neurons_starts[0]+'*'))[0]]
for i in range(len(neurons_starts)-1):
    neurons.append(glob(os.path.join('hoc_combos_syn.1_0_10.allzips', neurons_starts[i+1]+'*'))[0])
print(neurons)
neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")
if RANK == 0:
    if not os.path.isdir(NMODL):
        os.mkdir(NMODL)
    for NRN in neurons:
        for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
            while not os.path.isfile(os.path.join(NMODL,
                                                  os.path.split(nmodl)[-1])):
                if "win32" in sys.platform:
                    os.system("copy {} {}".format(nmodl, NMODL))
                else:
                    os.system('cp {} {}'.format(nmodl,
                                                os.path.join(NMODL,
                                                         '.')))
    os.chdir(NMODL)
    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows. " 
         + "Run mknrndll from NEURON bash in the folder %s and rerun example script" % NMODL)
    else:
        os.system('nrnivmodl')        
    os.chdir(CWD)
COMM.Barrier()
if "win32" in sys.platform:
    if not NMODL in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(NMODL+"/nrnmech.dll")
    neuron.nrn_dll_loaded.append(NMODL)
else:
    neuron.load_mechanisms(NMODL)

os.chdir(CWD)
#load the LFPy SinSyn mechanism for stimulus
if "win32" in sys.platform:
    pth = os.path.join(LFPy.__path__[0], "test").replace(os.sep, posixpath.sep)
    if not pth in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth + "/nrnmech.dll")
    neuron.nrn_dll_loaded.append(pth)
else:
    neuron.load_mechanisms(os.path.join(LFPy.__path__[0], "test"))

def posixpth(pth):
    """
    Replace Windows path separators with posix style separators
    """
    return pth.replace(os.sep, posixpath.sep)

def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within
    
    Arguments
    ---------
    f : file, mode 'r'
    
    Returns
    -------
    templatename : str
    
    '''    
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print('template {} found!'.format(templatename))
            continue
    
    return templatename


fig5 = plt.figure(figsize=(15, 15))
layer = layers[-2]
fig5.suptitle('Layer {0}, synaptic input, treshold {1} uV'.format(layer[1:], treshold*1000))
i=0

for NRN, rad in sorted(L5_syn_dict.items(), key = operator.itemgetter(1), reverse = True)[:n_exc]:
    ax_n = fig5.add_subplot(n_tot, 4, i*4+1, frameon=False)
    ax_m = fig5.add_subplot(n_tot, 4, i*4+2, frameon=False, aspect=1, xlim=[-200, 200])
    ax_c = fig5.add_subplot(n_tot, 4, i*4+3, frameon=False)
    ax_d = fig5.add_subplot(n_tot, 4, i*4+4, frameon=False)
    ax_n.xaxis.set_visible(False)
    if 'PC' in NRN or 'SS' in NRN  or 'SP' in NRN:
        color='r'
    else:
        color='b'
    circle = plt.Circle((0, 0), rad, color=color)
    ax_c.text(0, 0, 'rad: {:.1f} um'.format(rad), size=12)
    ax_c.set_xlim(-70, 70)
    ax_c.set_ylim(-70, 70)
    ax_c.add_artist(circle)
    ax_d.text(.5, .5, '{0:.2f} nrns/mm\nfraction {1:.2f}\nE(measured): {2:.2f}'.format(dens_vol_syn[NRN][1], count_dict[NRN]/total_nrns_layer[layer], n_measured_syn[NRN]), size=15)
    ax_n.text(.5, .5, NRN, size=15)
    for ax in [ax_m, ax_c, ax_d, ax_n]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    os.chdir(neurons[i])

    #get the template name
    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()
    
    #get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()
    
    #get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()
    
    #get synapses template name
    f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = get_templatename(f)
    f.close()
    

    print('Loading constants')
    neuron.h.load_file('constants.hoc')
    
        
    if not hasattr(neuron.h, morphology): 
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics): 
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, posixpth(os.path.join('synapses', 'synapses.hoc')))
    if not hasattr(neuron.h, templatename): 
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    COUNTER = 0 
    for morphologyfile in glob(os.path.join('morphology', '*')):
        if COUNTER % SIZE == RANK:
            # Instantiate the cell(s) using LFPy
            cell = LFPy.TemplateCell(morphology=morphologyfile,
                                templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                                templatename=templatename,
                                templateargs=0,
                                v_init=-75,
                                tstart=-200,
                                tstop=100,
                                dt=2**-5,
                                nsegs_method=None)
            cell.set_rotation(x=np.pi/2)
            [ax_m.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c='k', lw=cell.diam[i]) for i in range(cell.totnsegs)]
        COUNTER += 1
        os.chdir(CWD)
    i+=1

i=n_exc
for NRN, rad in sorted(L5_syn_dict.items(), key = operator.itemgetter(1))[:n_inh][::-1]:
    ax_n = fig5.add_subplot(n_tot, 4, i*4+1, frameon=False)
    ax_m = fig5.add_subplot(n_tot, 4, i*4+2, frameon=False, aspect=1, xlim=[-200, 200])
    ax_c = fig5.add_subplot(n_tot, 4, i*4+3, frameon=False)
    ax_d = fig5.add_subplot(n_tot, 4, i*4+4, frameon=False)
    ax_n.xaxis.set_visible(False)
    if 'PC' in NRN or 'SS' in NRN  or 'SP' in NRN:
        color='r'
    else:
        color='b'
    circle = plt.Circle((0, 0), rad, color=color)
    ax_c.text(0, 0, 'rad: {:.1f} um'.format(rad), size=12)
    ax_c.xaxis.set_visible(True)
    ax_c.set_xlim(-70, 70)
    ax_c.set_ylim(-70, 70)
    ax_c.add_artist(circle)
    ax_d.text(.5, .5, '{0:.2f} nrns/mm\nfraction {1:.2f}\nE(measured): {2:.2e}'.format(dens_vol_syn[NRN][1], count_dict[NRN]/total_nrns_layer[layer], n_measured_syn[NRN]), size=15)
    ax_n.text(.5, .5, NRN, size=15)
    for ax in [ax_m, ax_c, ax_d, ax_n]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    os.chdir(neurons[i])

    #get the template name
    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()
    
    #get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()
    
    #get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()
    
    #get synapses template name
    f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = get_templatename(f)
    f.close()
    

    print('Loading constants')
    neuron.h.load_file('constants.hoc')
    
        
    if not hasattr(neuron.h, morphology): 
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics): 
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, posixpth(os.path.join('synapses', 'synapses.hoc')))
    if not hasattr(neuron.h, templatename): 
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    COUNTER = 0 
    for morphologyfile in glob(os.path.join('morphology', '*')):
        if COUNTER % SIZE == RANK:
            # Instantiate the cell(s) using LFPy
            cell = LFPy.TemplateCell(morphology=morphologyfile,
                                templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                                templatename=templatename,
                                templateargs=0,
                                v_init=-75,
                                tstart=-200,
                                tstop=100,
                                dt=2**-5,
                                nsegs_method=None)
            cell.set_rotation(x=np.pi/2)
            [ax_m.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c='k', lw=cell.diam[i]) for i in range(cell.totnsegs)]
        COUNTER += 1
        os.chdir(CWD)
    i+=1

fig5.savefig('figures/circles_syn_{}_uV_finer'.format(int(treshold*1000)))