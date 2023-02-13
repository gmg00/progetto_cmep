import sys
import gc
import numpy as np
from DataFormats.FWLite import Events, Handle
import ROOT
np.random.seed(42)

# Convert (x,y) in 0,..,14 
dict = {(0,0):0, (0,1):1, (1,0):1, (2,0):2, (0,2):2, (3,0):3, (0,3):3, (4,0):4, (0,4):4,
		(1,1):5, (2,1):6, (1,2):6, (3,1):7, (1,3):7, (4,1):8, (1,4):8,
		(2,2):9, (3,2):10, (2,3):10, (4,2):11, (2,4):11,
		(3,3):12, (4,3):13, (3,4):13,
		(4,4):14}

N_of_files = 8

#Load first file
cov = np.load('/home/cms-opendata/COV2/cov_0.npy')
par = np.load('/home/cms-opendata/PAR2/par_0.npy')
#Keep inventory of event already passed.
size_before = 0

events = Events(['root://eospublic.cern.ch//eos/opendata/cms/Run2012A/ElectronHad/AOD/22Jan2013-v1/20000/FEE9E03A-F581-E211-8758-002618943901.root'])
handle = Handle('std::vector<reco::Track>')
label = ('generalTracks')

ROOT.gROOT.SetBatch()

k = 0 #File index.
N = 5 #Number of random tests.
index = np.random.randint(0, cov.shape[0], N) #Index array.  

i = 0 #Tracks
for event in events:
	event.getByLabel(label,handle)
	tracks = handle.product()
	for tr in tracks:
		if (i % 50000 == 0): print('i = %d' %i)
		#If i in index test if tr.covariance and the registered one are equal.
		if i in index:
			tmp = cov[i - size_before, :]
			tmp2 = par[i - size_before, :]
			x = int(np.random.rand()*4) 
			y = int(np.random.rand()*4)
			condition_1 = (tr.covariance(x,y) == tmp[dict[(x,y)]])
			condition_2 = np.array_equal(tmp2, [tr.chi2(), tr.ndof(), tr.pt(), tr.eta(), tr.phi()])
			#Evaluate if a random element of matrix is equal to the one registered.
			if condition_1 & condition_2: print('OK')
			else: 
				raise ValueError
		i = i + 1
		#Load another file when one end.
		if i - size_before == cov.shape[0]:
			if k == N_of_files: break
			k = k + 1 
			size_before = cov.shape[0] + size_before #Update size_before.
			del cov
			del par
			gc.collect() #Collect memory.
			print('Loading next .npy')
			cov = np.load('/home/cms-opendata/COV2/cov_' + str(k) + '.npy')
			par = np.load('/home/cms-opendata/PAR2/par_' + str(k) + '.npy')
			index = np.random.randint(0, cov.shape[0], N) + size_before # index array  
			print('Restarting from %d' %i)
			

