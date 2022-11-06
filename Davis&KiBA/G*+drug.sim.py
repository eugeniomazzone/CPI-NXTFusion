import NXTfusion.NXTfusion as NX
import NXTfusion.DataMatrix as DM 
import NXTfusion.NXLosses as L
from NXTfusion.NXmodels import NXmodelProto
from NXTfusion.NXmultiRelSide import NNwrapper
from NXTfusion.NXFeaturesConstruction import buildPytorchFeats
import numpy as np
from scipy.io import mmread
import torch as t
from scipy.sparse import coo_matrix as sp
import random
from sklearn.model_selection import KFold 
from lifelines.utils import concordance_index as ci
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

DEVICE = "cpu:0" #change here to cuda:X if you want GPU acceleration
IGNORE_INDEX = -9999999
n_Fold=5
n_Epoch=70

datapath='data/davis/' #'mat_drug_protein_remove_homo.txt' 

def main(args):

	########### indicing all protein and compound list
	proteins=[]
	drugs=[]
	cpi=[]

	with open(datapath + 'drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt',"r") as data:
		x= data.readlines()
	for drug, lines  in enumerate(x):
		line=lines.strip().split()
		for prot, intera in enumerate(line):
			react=float(intera)
			react=-np.log10(react/1e9)
			proteins.append(prot)
			drugs.append(drug)
			cpi.append(react-5)
                
	all_prot=range(len(set(proteins)))
	all_drug=range(len(set(drugs)))
	shape=(len(all_prot),len(all_drug))
	#########################################
	lister=[]
	shape1=(len(all_drug),len(all_drug))
	with open(datapath + 'drug-drug_similarities_2D.txt','r') as data:
		x=data.readlines()
		for drug1, lines  in enumerate(x):
			line=lines.strip().split()
			for drug2, sim in enumerate(line):
				lister.append([drug1,drug2, float(sim)])
				
	dd_train=sp(([i[2] for i in lister],([i[0] for i in lister], [i[1] for i in lister])), shape=shape1)
	#########################################
	lister=[]
	shape2=(len(all_prot),len(all_prot))
	with open(datapath + 'target-target_similarities_WS.txt','r') as data:
		x=data.readlines()
		for drug1, lines  in enumerate(x):
			line=lines.strip().split()
			for drug2, sim in enumerate(line):
				lister.append([drug1,drug2, float(sim)/100])
				
	pp_train=sp(([i[2] for i in lister],([i[0] for i in lister], [i[1] for i in lister])), shape=shape2)
	
	y1=[]
	y2=[]
	############# train ########################
	kf = KFold(n_splits = n_Fold, shuffle=True)

	protEnt = NX.Entity("proteins", all_prot, np.int16)
	drugEnt = NX.Entity("compounds", all_drug, np.int16)
	protDrugLoss = L.LossWrapper(t.nn.MSELoss(), type="regression", ignore_index = IGNORE_INDEX)
	drugDrugLoss = L.LossWrapper(t.nn.L1Loss(), type="regression", ignore_index = IGNORE_INDEX)
	protProtLoss = L.LossWrapper(t.nn.L1Loss(), type="regression", ignore_index = IGNORE_INDEX)

	results=[]
	for train_index, test_index in kf.split(proteins):

		###########################
		cpi_train= sp(([cpi[i] for i in train_index],([proteins[i] for i in train_index], [drugs[i] for i in train_index])), shape=shape)

		protDrugMat = DM.DataMatrix("protDrugMatrix", protEnt, drugEnt, cpi_train)
		protDrugRel = NX.MetaRelation("prot-drug", protEnt, drugEnt, None, None)
		protDrugRel.append(NX.Relation("drugInteraction", protEnt, drugEnt, protDrugMat, "regression", protDrugLoss, relationWeight=1))

		drugDrugMat = DM.DataMatrix("drugDrugMatrix", drugEnt, drugEnt, dd_train)
		drugDrugRel = NX.MetaRelation("drug-drug", drugEnt, drugEnt, None, None)
		drugDrugRel.append(NX.Relation("ddrugInteraction", drugEnt, drugEnt, drugDrugMat, "regression", drugDrugLoss, relationWeight=1))
				
		protProtMat = DM.DataMatrix("protProtMatrix", protEnt, protEnt, pp_train)
		protProtRel = NX.MetaRelation("prot-prot", protEnt, protEnt, None, None)
		protProtRel.append(NX.Relation("protInteraction", protEnt, protEnt, protProtMat, "regression", protProtLoss, relationWeight=1))
				
		ERgraph = NX.ERgraph([protDrugRel, drugDrugRel, protProtRel])

		model = Model(ERgraph, "mod")
		wrapper = NNwrapper(model, dev = DEVICE, ignore_index = IGNORE_INDEX)
		wrapper.fit(ERgraph, epochs=n_Epoch)

		Yt=[cpi[i] for i in test_index]
		Xt=[ [proteins[i], drugs[i]] for i in test_index]

		Yp = wrapper.predict(ERgraph, Xt, "prot-drug", "drugInteraction", None, None)

		results.append([ci(Yt,Yp), mse(Yt,Yp)])
		break
	print(np.mean([i[0] for i in results]),np.mean([i[1] for i in results]))
	
class Model(NXmodelProto):
	def __init__(self, ERG, name):
		super(Model, self).__init__()
		self.name = name
		##########DEFINE NN HERE##############
		protEmbLen = ERG["prot-drug"]["lenDomain1"]
		drugEmbLen = ERG["prot-drug"]["lenDomain2"]
		
		PROT_LATENT_SIZE = 300
		DRUG_LATENT_SIZE = 200
		
		SIZE1=30
		SIZE2=20
		
		ACTIVATION = t.nn.Tanh
		self.protEmb = t.nn.Embedding(protEmbLen, PROT_LATENT_SIZE)
		self.protHid = t.nn.Sequential(t.nn.Linear(PROT_LATENT_SIZE, SIZE1), t.nn.LayerNorm(SIZE1), ACTIVATION())
		
		self.drugEmb = t.nn.Embedding(drugEmbLen, DRUG_LATENT_SIZE)
		self.drugHid = t.nn.Sequential(t.nn.Linear(DRUG_LATENT_SIZE, SIZE2), t.nn.LayerNorm(SIZE2), ACTIVATION())
		
		self.biProtDrug = t.nn.Bilinear(SIZE1, SIZE2, 40)
		self.outProtDrug = t.nn.Sequential( t.nn.LayerNorm(40), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(40,1))
		
		self.biDrugDrug = t.nn.Bilinear(SIZE2, SIZE2, 10)
		self.outDrugDrug = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,1))
		
		self.biProtProt = t.nn.Bilinear(SIZE1, SIZE1, 10)
		self.outProtProt = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,1))

		self.apply(self.init_weights)

	def forward(self, relName, i1, i2, s1=None, s2=None):
		if relName == "prot-drug":
			u = self.protEmb(i1)
			v = self.drugEmb(i2)
			u = self.protHid(u).squeeze()
			v = self.drugHid(v).squeeze()
			o = self.biProtDrug(u, v)
			o = self.outProtDrug(o)
			
		if relName == "drug-drug":
			u = self.drugEmb(i1)
			v = self.drugEmb(i2)
			u = self.drugHid(u).squeeze()
			v = self.drugHid(v).squeeze()
			o = self.biDrugDrug(u, v)
			o = self.outDrugDrug(o)

		if relName == "prot-prot":
			u = self.protEmb(i1)
			v = self.protEmb(i2)
			u = self.protHid(u).squeeze()
			v = self.protHid(v).squeeze()
			o = self.biProtProt(u, v)
			o = self.outProtProt(o)	
		return o

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
