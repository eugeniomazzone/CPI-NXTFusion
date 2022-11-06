import NXTfusion.NXTfusion as NX
import NXTfusion.DataMatrix as DM 
import NXTfusion.NXLosses as L
from NXTfusion.NXmodels import NXmodelProto
from NXTfusion.NXmultiRelSide import NNwrapper
from NXTfusion.NXFeaturesConstruction import buildPytorchFeats

import numpy as np
import torch as t
from scipy.sparse import coo_matrix as sp

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import precision_score as pc
from sklearn.metrics import recall_score as re
from sklearn.model_selection import KFold 

from rdkit import DataStructs as ds
from rdkit import Chem 
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem

DEVICE = "cpu:0" #change here to cuda:X if you want GPU acceleration
IGNORE_INDEX = -9999
	
n_Fold=5
n_Epoch=30
gamma = 4.
alpha= 0.5

def main(args):

	############ reading full protein and compound list
	
	with open("data/protein_list.fasta","r") as data:
		all_prot= data.readlines()
		all_prot=[all_prot[i].strip() for i in range(1,len(all_prot),2)]
	with open("data/compound_list.txt","r") as data:
		all_drug= data.readlines()
		all_drug=[drug.strip() for drug in all_drug]

	shape=(len(all_prot), len(all_drug))
	print(shape)
	
	########### indicing all protein and compound list Human.dat
	with open("data/dataH.txt","r") as data:
		x= data.readlines()
	
	proteins=[]
	drugs=[]
	cpi=[]
		
	for j, lines in enumerate(x):
		compound, protein,react =lines.strip().split()
		react=float(react)
		for i, y in enumerate(all_prot):
		  	if y==protein: 
		  		proteins.append(i)
		  		break
		for i, y in enumerate(all_drug):
		   	if y==compound: 
		   		drugs.append(i)
		   		break
		cpi.append(react)
	
	shape=(len(all_prot), len(all_drug))
	
	################# drug sim #######################
	print("Start")
	shape1=(len(all_drug), len(all_drug))
	drug_mat=np.zeros(shape1)
	all_drug2=[Chem.MolFromSmiles(d) for d in all_drug]	
	all_drug3 = [AllChem.GetMorganFingerprintAsBitVect(d,2,nBits=1024) for d in all_drug2]

	for i, d1 in enumerate(all_drug3):
		for j, d2 in enumerate(all_drug3):
			drug_mat[i][j]=ds.DiceSimilarity(d1,d2)
	print("Done")
	############# train ########################
	kf = KFold(n_splits = n_Fold, shuffle=True)
	
	protEnt = NX.Entity("proteins", all_prot, np.int16)
	drugEnt = NX.Entity("compounds", all_drug, np.int16)
	protDrugLoss = L.LossWrapper(FocalLoss(gamma,alpha), type="regression", ignore_index = IGNORE_INDEX)
	drugDrugLoss = L.LossWrapper(t.nn.MSELoss(), type="regression", ignore_index = IGNORE_INDEX) 
	
	results=[]
	for train_index, test_index in kf.split(drugs):
		
		############################
		Yt=[cpi[i] for i in test_index]
		Xt=[ [proteins[i], drugs[i]] for i in test_index]	
			
		cpi_train= sp(([cpi[i] for i in train_index],([proteins[i] for i in train_index], [drugs[i] for i in train_index])), shape=shape)
		
		protDrugMat = DM.DataMatrix("protDrugMatrix", protEnt, drugEnt, cpi_train)
		protDrugRel = NX.MetaRelation("prot-drug", protEnt, drugEnt, None, None)
		protDrugRel.append(NX.Relation("drugInteraction", protEnt, drugEnt, protDrugMat, "regression", protDrugLoss, relationWeight=1))
			
			
		drugDrugMat = DM.DataMatrix("drugDrugMatrix", drugEnt, drugEnt, drug_mat)
		drugDrugRel = NX.MetaRelation("drug-drug", drugEnt, drugEnt, None, None)
		drugDrugRel.append(NX.Relation("ddrugInteraction", drugEnt, drugEnt, drugDrugMat, "regression", drugDrugLoss, relationWeight=1))
			
		ERgraph = NX.ERgraph([protDrugRel,drugDrugRel])
			
		model = Model(ERgraph, "mod")
		wrapper = NNwrapper(model, dev = DEVICE, ignore_index = IGNORE_INDEX)
		
		Yp = wrapper.predict(ERgraph, Xt, "prot-drug", "drugInteraction", None, None)
		
		wrapper.fit(ERgraph, epochs=n_Epoch)
		
		Yp = wrapper.predict(ERgraph, Xt, "prot-drug", "drugInteraction", None, None)
		
		results.append([auc(Yt, Yp),auprc(Yt, Yp)])
		break
	print("AUC mean : " , np.mean([i for i,_ in results]), "AUPR mean : " ,np.mean([i for _,i in results]) )




class FocalLoss(t.nn.Module):
	def __init__(self, gamma =0., alpha=0.5):
		super(FocalLoss, self).__init__()
		self.gamma = t.tensor(gamma, dtype = t.float32)
		self.alpha = t.tensor(alpha, dtype = t.float32)
		self.eps = 1e-16

	def forward(self, input, target):
		probs = input
		#print(self.gamma)
		focal_loss = self.alpha*t.pow(1-probs + self.eps, self.gamma).mul(-t.log(probs)).mul(target) + (1. - self.alpha)*t.pow(probs + self.eps, self.gamma).mul(-t.log(1 - probs)).mul(1 - target)
		return focal_loss.mean()
		
		
class Model(NXmodelProto):
	def __init__(self, ERG, name):
		super(Model, self).__init__()
		self.name = name
		##########DEFINE NN HERE##############
		protEmbLen = ERG["prot-drug"]["lenDomain1"]
		drugEmbLen = ERG["prot-drug"]["lenDomain2"]
		PROT_LATENT_SIZE = 30
		DRUG_LATENT_SIZE = 20
		ACTIVATION = t.nn.Sigmoid
		self.protEmb = t.nn.Embedding(protEmbLen, PROT_LATENT_SIZE)
		self.protHid = t.nn.Sequential(t.nn.Linear(PROT_LATENT_SIZE, 20), t.nn.LayerNorm(20), ACTIVATION())
		
		self.drugEmb = t.nn.Embedding(drugEmbLen, DRUG_LATENT_SIZE)
		self.drugHid = t.nn.Sequential(t.nn.Linear(DRUG_LATENT_SIZE, 10), t.nn.LayerNorm(10), ACTIVATION())
		
		self.biProtDrug = t.nn.Bilinear(20, 10, 10)
		self.outProtDrug = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,1), t.nn.Sigmoid())
		
		self.biDrugDrug = t.nn.Bilinear(10, 10, 10)
		self.outDrugDrug = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,1),t.nn.Sigmoid())
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
		
		return o
		
if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
