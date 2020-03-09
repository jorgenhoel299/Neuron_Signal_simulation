import pandas as pd
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

csv_ready = True
if csv_ready:
    amps = pd.read_csv('opa')
else:
    # Load data
    amps = pd.read_csv('amplitudes_by_distance_multiway', index_col=0)
    amps['r/d'] = ""

    # Load cells and determine morpholgy parameters
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()

    #working dir
    CWD = os.getcwd()
    NMODL = 'hoc_combos_syn.1_0_10.allmods'

    #load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    #load only some layer 5 pyramidal cell types
    neurons = glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L*'))

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
    add_synapses = False

    tstop = 50.
    dt = 2**-6
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
    cell_to_morph = {}
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
                # Instantiate the cell(s) using LFPy
                cell = LFPy.TemplateCell(morphology=morphologyfile,
                                templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                                templatename=templatename,
                                templateargs=1 if add_synapses else 0,
                                tstop=tstop,
                                dt=dt,
                                nsegs_method=None)
                dend_diams = []

                for sec in cell.allseclist:
                    if 'soma' in sec.name():
                        r_soma = 0.5*sec.diam
                    try:
                        parent_name = sec.parentseg().sec.name()
                    except:
                        continue  
                    if "soma" in parent_name:
                        for seg in sec:  
                            dend_diams.append(seg.diam)
                            break
                
                value = np.sum(np.array(dend_diams)**1.5)/r_soma
                amps.at[NRN[30:], 'r/d'] = value
            amps.to_csv('../opa')
            COUNTER += 1
            os.chdir(CWD)
    amps.to_csv('opa')

amps = amps.set_index('Unnamed: 0')
fig = plt.figure(figsize=(10, 8))
layers = ['L1', 'L23', 'L4', 'L5', 'L6']
styles = ['ro', 'bo', 'co', 'go', 'yo']
for i, layer in enumerate(layers):
    L_df = amps[amps.index.str.startswith(layer)]
    plt.plot(L_df['r/d'], L_df['8.0']*1000, styles[i], label=layer)

plt.title('Amplitudes at 8um')
plt.xlabel('∑')
plt.ylabel('Amplitude uV')
plt.legend()
plt.savefig('Amps_at_8um')