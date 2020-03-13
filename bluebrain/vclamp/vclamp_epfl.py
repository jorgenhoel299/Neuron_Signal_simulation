#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that the complete set of cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads is unzipped in this folder. 

Execution:

    python example_EPFL_neurons.py
    
Copyright (C) 2017 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
import matplotlib
matplotlib.use('AGG')
import os
import posixpath
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection, LineCollection
from glob import glob
import numpy as np
from warnings import warn
import scipy.signal as ss
import neuron
import LFPy
from mpi4py import MPI
import pandas as pd
from neuron import h

plt.rcParams.update({
    'axes.labelsize' : 8,
    'axes.titlesize' : 8,
    #'figure.titlesize' : 8,
    'font.size' : 8,
    'ytick.labelsize' : 8,
    'xtick.labelsize' : 8,
})
control_simulation = True
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

amps = pd.DataFrame(columns=['amplitude', 'r_soma'])

#working dir
CWD = os.getcwd()
NMODL = 'hoc_combos_syn.1_0_10.allmods'

#load some required neuron-interface files
neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

#load only some layer 5 pyramidal cell types
neurons = glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5*'))
# neurons = glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_TTPC*'))[:1]
# neurons += glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_MC*'))[:1]
# neurons += glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_LBC*'))[:1]
# neurons += glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_NBC*'))[:1]

#flag for cell template file to switch on (inactive) synapses
add_synapses = False

#attempt to set up a folder with all unique mechanism mod files, compile, and
#load them all
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

FIGS = 'hoc_combos_syn.1_0_10.allfigures'
if not os.path.isdir(FIGS):
    os.mkdir(FIGS)


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

# Make cells passive
def remove_mechanisms(remove_list=None):
    if remove_list is None:
        remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1", "SK_E2",
                       "K_Tst", "K_Pst", "Im", "Ih",
                       "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA", "KdShu2007", "StochKv"]
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove()
# PARAMETERS

#sim duration
tstop = 10.
dt = 2**-5

PointProcParams = {
    'idx' : 0,
    'pptype' : 'SinSyn',
    'delay' : 200.,
    'dur' : tstop - 300.,
    'pkamp' : 0.5,
    'freq' : 0.,
    'phase' : np.pi/2,
    'bias' : 0.,
    'record_current' : False
}

#spike sampling
threshold = -20 #spike threshold (mV)
samplelength = int(2. / dt)
            
#filter settings for extracellular traces
b, a = ss.butter(N=3, Wn=(300*dt*2/1000, 5000*dt*2/1000), btype='bandpass')
apply_filter = True

#communication buffer where all simulation output will be gathered on RANK 0
COMM_DICT = {}

largest_soma_diam = 15.6597
COUNTER = 0
for i, NRN in enumerate(neurons):
    os.chdir(NRN)

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


    for morphologyfile in glob(os.path.join('morphology', '*')):
        if COUNTER % SIZE == RANK:
            two_up =  os.path.abspath(os.path.join(__file__ ,"../../../"))
            # Instantiate the cell(s) using LFPy
            cell = LFPy.TemplateCell(morphology=morphologyfile,
                             templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             v_init=-75,
                             tstart=-200,
                             tstop=tstop,
                             dt=dt,
                             nsegs_method=None)

            if control_simulation:
                #h("soma[0] {insert hh}")
                synapse_parameters = {
                    'idx' : 0,
                    'e' : 0.,                   # reversal potential
                    'syntype' : 'ExpSyn',       # synapse type
                    'tau' : 5.,                 # synaptic time constant
                    'weight' : .05,            # synaptic weight
                    'record_current' : False,    # record synapse current
                }

                # Create synapse and set time of synaptic input
                synapse = LFPy.Synapse(cell, **synapse_parameters)
                synapse.set_spike_times(np.array([2.]))
            else:
                i=0
                remove_mechanisms()
                # for seg in cell.allseclist:
                #     if i == 0:
                #         vclamp = h.SEClamp_i(seg(0.5))
                #     i+=1
                for sec in h.allsec():

                    vclamp = h.SEClamp_i(sec(0.5))
                    break
                
                vclamp.dur1 = 1e9
                # vclamp.amp1 = -75
                # vclamp.dur2 = 2
                # vclamp.amp2 = -80
                vclamp.rs = .01
                vmem_to_insert = h.Vector(np.load(two_up+'/'+NRN[30:]+"_soma_spike.npy"))
                # vmem_to_insert = h.Vector(np.array([-65]*1000))
                vmem_to_insert.play(vclamp._ref_amp1, h.dt)
            
            cell.set_rotation(x=np.pi/2)
            i=0
            for seg in cell.allseclist:
                if i == 0:
                    soma_diam = seg.diam
                i+=1
            rec_spot = np.ceil(largest_soma_diam/2)
            x = np.linspace(25, 55, 4)
            y = np.zeros(x.shape)
            z = np.zeros(x.shape)
            # Define electrode parameters
            elec_params = {
                'sigma' : 0.3,      # extracellular conductivity
                'x' : x,  # electrode requires 1d vector of positions
                'y' : y,
                'z' : z,
                'method': 'soma_as_point'
            }
            elec_clr = lambda idx: plt.cm.viridis(idx / (len(x)))
            electrode = LFPy.RecExtElectrode(**elec_params)
            cell.simulate(electrode=electrode)

            plt.close("all")
            
            fig = plt.figure(figsize=[9, 9])

            ax_m = fig.add_subplot(121, frameon=False, aspect=1, xlim=[-200, 200])

            ax1 = fig.add_subplot(222, xlabel="time (ms)", ylabel="mV", title="somatic Vm")
            ax3 = fig.add_subplot(224, xlabel="time (ms)", ylabel="$\mu$V", title="extracellular potential")

            [ax_m.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c='k', lw=cell.diam[i]) for i in range(cell.totnsegs)]

            peak2peaks = np.zeros(len(x))
            for e_idx in range(len(x)):
                ax_m.plot(electrode.x[e_idx], electrode.z[e_idx], 'o', c=elec_clr(e_idx))
                ax3.plot(cell.tvec, 1000 * electrode.LFP[e_idx], c=elec_clr(e_idx))

                peak2peaks[e_idx] = 1000 * (np.max(electrode.LFP[e_idx]) - np.min(electrode.LFP[e_idx]))
                ax3.plot([cell.tvec[-1] - 1, cell.tvec[-1] - 1], [1000 * np.min(electrode.LFP[e_idx]),
                                                                1000 * np.max(electrode.LFP[e_idx])], c=elec_clr(e_idx))

            ax1.plot(cell.tvec, cell.somav)

            if control_simulation:
                print(NRN[30:])
                np.save(two_up+'/'+NRN[30:]+"_soma_spike.npy", cell.somav)

                plt.savefig(two_up+'/'+"vclamp_control_on_{}".format(NRN[30:]), format='png')
            else:
                plt.savefig(two_up+'/'+"vclamp_test on {}".format(NRN[30:]))
            #electrode.calc_lfp()
            LFP = electrode.LFP
            if apply_filter:
                LFP = ss.filtfilt(b, a, LFP, axis=-1)
                    
        COUNTER += 1
        os.chdir(CWD)
        

COMM.Barrier()
