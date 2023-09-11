"""This module is used in the CMS virtual machine to get covariance matrix."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Written in python 2.6
import gc
import numpy as np
from DataFormats.FWLite import Events, Handle
import ROOT

events = Events(['root://eospublic.cern.ch//eos/opendata/cms/Run2012A/ElectronHad/AOD/22Jan2013-v1/20000/FEE9E03A-F581-E211-8758-002618943901.root'])
handle = Handle('std::vector<reco::Track>')
label = 'generalTracks'

folder = '/home/cms-opendata/COV2/' #folder path.

ROOT.gROOT.SetBatch()

cov_all = [] #Inizialize matrix

#Because of a memory problem we have to make smaller files with N events.
N = 2000 #Number of events

i = 1
k = 0
for event in events:
	event.getByLabel(label,handle)
	tracks = handle.product()
	for tr in tracks:
		#Save 15 significative values of the covariance matrix.
		arr = [tr.covariance(0,0), tr.covariance(0,1), tr.covariance(0,2), tr.covariance(0,3),
		tr.covariance(0,4), tr.covariance(1,1), tr.covariance(1,2), tr.covariance(1,3),
		tr.covariance(1,4), tr.covariance(2,2), tr.covariance(2,3), tr.covariance(2,4),
		tr.covariance(3,3), tr.covariance(3,4), tr.covariance(4,4)]
		cov_all.append(arr)
	if i % 1000 == 0:
		print('Events completed:' + str(i))
	if i % N == 0: #Close a file and open a new one.
		print('Creating cov_' + str(k) + '.npy and freeing memory')
		np.save(folder+'cov_' + str(k) + '.npy', cov_all)
		k = k + 1
		print('Saved cov_all in file')
		#Delete some arrays to collect memory.
		del cov_all
		print('Deleated cov_all')
		del tracks
		print('Deleated tracks')
		del tr
		print('Deleated tr')
		gc.collect()
		print('Collected')
		cov_all = []
		print('Restart from ' + str(i + 1))
	i = i + 1

#Save last file
np.save(folder + 'cov_' + str(k) + '.npy', cov_all)

del cov_all
del events
del event
del tracks
del tr
gc.collect()

