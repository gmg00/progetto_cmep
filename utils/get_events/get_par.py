"""This module is used in the CMS virtual machine to get parameters."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Written in python 2.6
import sys
import gc
import numpy as np
from DataFormats.FWLite import Events, Handle
import ROOT

events = Events(['root://eospublic.cern.ch//eos/opendata/cms/Run2012A/ElectronHad/AOD/22Jan2013-v1/20000/FEE9E03A-F581-E211-8758-002618943901.root'])
handle = Handle('std::vector<reco::Track>')
label = ('generalTracks')

folder = '/home/cms-opendata/PAR2/' #folder path.

ROOT.gROOT.SetBatch()

par_all = [] #Inizialize matrix

#Because of a memory problem we have to make smaller files with N events.
N = 2000 #Number of events

i = 1 
k = 0
for event in events:
	event.getByLabel(label,handle)
	tracks = handle.product()
	for tr in tracks:
		#Save 15 significative values of the covariance matrix.
		arr = [tr.chi2(), tr.ndof(), tr.pt(), tr.eta(), tr.phi()]
		par_all.append(arr)
	if (i % 1000 == 0): print('Events completed:' + str(i))
	if (i % N == 0): #Close a file and open a new one.
		print('Creating par_' + str(k) + '.npy and freeing memory')
		np.save(folder+'par_' + str(k) + '.npy', par_all)
		k = k + 1
		print('Saved par_all in file')
		#Delete some arrays to collect memory.
		del par_all
		print('Deleated par_all')
		del tracks
		print('Deleated tracks')
		del tr
		print('Deleated tr')
		gc.collect()
		print('Collected')
		par_all = []
		print('Restart from ' + str(i + 1))
	i = i + 1
	
#Save last file
np.save(folder + 'par_' + str(k) + '.npy', par_all)

del par_all
del events
del event
del tracks
del tr
gc.collect()

	
