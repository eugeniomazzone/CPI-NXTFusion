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
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import precision_score as pc
from sklearn.metrics import recall_score as re
from sklearn.metrics import precision_recall_curve as prc
from sklearn.model_selection import KFold 

#/home/euge/anaconda3/lib/python3.8/site-packages/NXTfusion/MultirRelSide
# change y[_,_] to y[_]



DEVICE = "cpu:0" #change here to cuda:X if you want GPU acceleration
IGNORE_INDEX = -9999

n_Fold=10
n_Epoch=40

gamma =2.#[0.,10.]
alpha=0.8#[3.,1.]

datamat='mat_drug_protein.txt' #'mat_drug_protein_remove_homo.txt' 

prop=1

def main(args):
	
	############ reading full protein and compound list
	all_prot=[]
	prot_dic={}
	l=0
	with open("data/protein.txt","r") as data:
		all_p= data.readlines()
		for i,prot in enumerate(all_p):
			if prot not in all_prot:
				all_prot.append(prot)
				prot_dic[i]=l
				l+=1
					
			elif prot in all_prot:
				for j, p2 in enumerate(all_prot):
					if prot==p2: prot_dic[i]=l
				
	with open("data/drug.txt","r") as data:
		all_drug= data.readlines()
		all_drug=[drug.strip() for drug in all_drug]

	shape=(len(all_prot), len(all_drug))	
	print(shape)
	########### indicing all protein and compound list
	proteins=[]
	drugs=[]
	cpi=[]
	neg_idx=[]
	pos_idx=[]
	
	with open("data/" + datamat,"r") as data:
		x= data.readlines()
		for drug, lines  in enumerate(x):
			line=lines.strip().split()
			for prot, intera in enumerate(line):
				react=float(intera)
				 
				proteins.append(prot_dic[prot])
				drugs.append(drug)
				cpi.append(react)

				if react==1: pos_idx.append(len(cpi)-1)
				else: neg_idx.append(len(cpi)-1)
	print(len(set(proteins)),len(set(drugs)))
	print(len(set([drugs[i] for i in pos_idx])))
	print(len(set([drugs[i] for i in neg_idx])))
	############# train ########################
	
	kf = KFold(n_splits = n_Fold, shuffle=True)
	
	
	protEnt = NX.Entity("proteins", all_prot, np.int16)
	drugEnt = NX.Entity("compounds", all_drug, np.int16)
	protDrugLoss = L.LossWrapper(FocalLoss(gamma,alpha), type="regression", ignore_index = IGNORE_INDEX)
	s_protProtLoss = L.LossWrapper(t.nn.MSELoss(), type="regression", ignore_index = IGNORE_INDEX)
	s_drugDrugLoss = L.LossWrapper(t.nn.MSELoss(), type="regression", ignore_index = IGNORE_INDEX)
	protProtLoss = L.LossWrapper(t.nn.MSELoss(), type="regression", ignore_index = IGNORE_INDEX)
	drugDrugLoss = L.LossWrapper(t.nn.MSELoss(), type="regression", ignore_index = IGNORE_INDEX)
	
	cpi1=[]
	proteins1=[]
	drugs1=[]
	
	neg_idx1=[random.randrange(len(neg_idx)) for i in range(len(pos_idx)*prop)]
			
	for i in neg_idx1:
		cpi1.append(cpi[neg_idx[i]])
		proteins1.append(proteins[neg_idx[i]])
		drugs1.append(drugs[neg_idx[i]])

	print(len(proteins1), len(pos_idx))
		
	for i in pos_idx:
		cpi1.append(cpi[i])
		proteins1.append(proteins[i])
		drugs1.append(drugs[i])
	
	############## prot similarity ##############
	shape2=(len(all_prot),len(all_prot))
	pp_sim=np.zeros(shape2)
	
	with open("data/Similarity_Matrix_Proteins.txt","r") as data:
		x=data.readlines()
		for i, lines in enumerate(x):
			line=lines.strip().split()
			for j, value in enumerate(line):
				pp_sim[prot_dic[i],prot_dic[j]]=float(value)/100

	############## drug similarity ##############
	shape1=(len(all_drug),len(all_drug))
	dd_sim=np.zeros(shape1) 
	
	with open("data/Similarity_Matrix_Drugs.txt","r") as data:
		x=data.readlines()
		for i, lines in enumerate(x):
			line=lines.strip().split()
			for j, value in enumerate(line):
				dd_sim[i,j]=float(value)
	
	############## prot-prot ##############
	shape2=(len(all_prot),len(all_prot))
	pp=[]
	ppos_idx=[]
	pneg_idx=[]
	
	with open("data/mat_protein_protein.txt","r") as data:
		x=data.readlines()
		for i, lines in enumerate(x):
			line=lines.strip().split()
			for j, value in enumerate(line):
				pp.append([prot_dic[i],prot_dic[j],float(value)])
				if float(value)==1.: ppos_idx.append(len(pp)-1)
				else: pneg_idx.append(len(pp)-1)
	
	pneg_idx1=[random.randrange(len(pneg_idx)) for i in range(len(ppos_idx)*1)]	
	pp_1=[pp[pneg_idx[i]] for i in pneg_idx1]
	
	print(len(pneg_idx1), len(ppos_idx))
		
	for i in ppos_idx:
		pp_1.append(pp[i])
	
	pp= sp(([pp_1[i][2] for i in range(len(pp_1))],([pp_1[i][0] for i in range(len(pp_1))], [pp_1[i][1] for i in range(len(pp_1))])), shape=shape2)
	
	############## drug-drug ##############
	shape1=(len(all_drug),len(all_drug))
	dd=[]
	dpos_idx=[]
	dneg_idx=[]
	
	with open("data/mat_drug_drug.txt","r") as data:
		x=data.readlines()
		for i, lines in enumerate(x):
			line=lines.strip().split()
			for j, value in enumerate(line):
				dd.append([i,j,float(value)])
				if float(value)==1.: dpos_idx.append(len(dd)-1)
				else: dneg_idx.append(len(dd)-1)
				
	dneg_idx1=[random.randrange(len(dneg_idx)) for i in range(len(dpos_idx)*1)]	
	dd_1=[dd[dneg_idx[i]] for i in dneg_idx1]
	
	print(len(dneg_idx1), len(dpos_idx))
		
	for i in dpos_idx:
		dd_1.append(dd[i])
	
	dd= sp(([dd_1[i][2] for i in range(len(dd_1))],([dd_1[i][0] for i in range(len(dd_1))], [dd_1[i][1] for i in range(len(dd_1))])), shape=shape1)

	############## drug disease ##############
	with open("data/mat_drug_disease.txt","r") as data:
		x=data.readlines()
		val_vec=np.zeros((len(x[0].strip().split()),))
		for i, lines in enumerate(x):
			line=lines.strip().split()
			for j, value in enumerate(line):
				val_vec[j]+=int(value)
	dis_list={}
	count=0
	for i, val in enumerate(val_vec):
		if val>250: 
			dis_list[i]=count 
			count+=1

	print(count)

	ddis_sim=np.zeros((len(all_drug),count))
	for i, lines in enumerate(x):
		line=lines.strip().split()
		for j, value in enumerate(line):
			try: ddis_sim[i,dis_list[j]]+=int(value)
			except: 1
			
	############## prot disease ##############
	with open("data/mat_protein_disease.txt","r") as data:
		x=data.readlines()
	val_vec=np.zeros((len(x[0].strip().split()),))
	for i, lines in enumerate(x):
		line=lines.strip().split()
		for j, value in enumerate(line):
			val_vec[j]+=int(value)
	dis_list={}
	count=0
	for i, val in enumerate(val_vec):
		if val>1300: 
			dis_list[i]=count 
			count+=1

	print(count)

	pdis_sim=np.zeros((len(all_prot),count))
	for i, lines in enumerate(x):
		line=lines.strip().split()
		for j, value in enumerate(line):
			try: pdis_sim[i,dis_list[j]]+=int(value)
			except: 1
	################################
		
	count=[]
	for i in range(len(all_drug)):
		vec=[0,0]
		for j in range(len(drugs1)): 
			if i==drugs1[j]: 
				if cpi1[j]==0: vec[0]+=1
				elif cpi1[j]==1: vec[1]+=1
		count.append(vec)
		#print(vec)
	
	cou=0
	cou1=0
	cou2=0
	for i in count:
		if i[0] != 0 and i[1] == 0: cou+=1
		if i[0] == 0 and i[1] != 0: cou1+=1
		if i[0] != 0 and i[1] != 0: cou2+=1
	print(len(all_drug),len(count), cou, cou1, cou2)

	#################################

	results=[]
	drugSideMat = DM.SideInfo("drugSideMatrix", drugEnt, ddis_sim)	
	protSideMat = DM.SideInfo("protSideMatrix", protEnt, pdis_sim)	
	for train_index, test_index in kf.split(range(len(drugs1))):
		
		###########################
		cpi_train= sp(([cpi1[i] for i in train_index],([proteins1[i] for i in train_index], [drugs1[i] for i in train_index])), shape=shape)
		
		protDrugMat = DM.DataMatrix("protDrugMatrix", protEnt, drugEnt, cpi_train)
		protDrugRel = NX.MetaRelation("prot-drug", protEnt, drugEnt, protSideMat, drugSideMat)
		protDrugRel.append(NX.Relation("drugInteraction", protEnt, drugEnt, protDrugMat, "regression", protDrugLoss, relationWeight=1))
		
		protProtMat = DM.DataMatrix("protProtMatrix", protEnt, protEnt, pp)
		s_protProtMat = DM.DataMatrix("s_protProtMatrix", protEnt, protEnt, pp_sim)
		protProtRel = NX.MetaRelation("prot-prot", protEnt, protEnt, protSideMat, protSideMat)
		protProtRel.append(NX.Relation("PprotInteraction", protEnt, protEnt, protProtMat, "regression", protProtLoss, relationWeight=1))
		protProtRel.append(NX.Relation("s_PprotInteraction", protEnt, protEnt, s_protProtMat, "regression", s_protProtLoss, relationWeight=1))	

		drugDrugMat = DM.DataMatrix("drugDrugMatrix", drugEnt, drugEnt, dd)
		s_drugDrugMat = DM.DataMatrix("s_drugDrugMatrix", drugEnt, drugEnt, dd_sim)
		drugDrugRel = NX.MetaRelation("drug-drug", drugEnt, drugEnt, drugSideMat, drugSideMat)
		drugDrugRel.append(NX.Relation("DdrugInteraction", drugEnt, drugEnt, drugDrugMat, "regression", drugDrugLoss, relationWeight=1))
		drugDrugRel.append(NX.Relation("s_DdrugInteraction", drugEnt, drugEnt, s_drugDrugMat, "regression", s_drugDrugLoss, relationWeight=1))

		ERgraph = NX.ERgraph([protDrugRel, protProtRel, drugDrugRel])
			
		model = Model(ERgraph, "mod")
		wrapper = NNwrapper(model, dev = DEVICE, ignore_index = IGNORE_INDEX)
		wrapper.fit(ERgraph, epochs=n_Epoch)
			
		Yt=[cpi1[i] for i in test_index]
		Xt=[ [proteins1[i], drugs1[i]] for i in test_index]
			
		Yp = wrapper.predict(ERgraph, Xt, "prot-drug", "drugInteraction", protSideMat, drugSideMat)
		
		results.append([auc(Yt, Yp),auprc(Yt, Yp)])

		break
		
	print(np.mean([i for i,_ in results]), np.mean([i for _,i in results]))

	
class Model(NXmodelProto):
	def __init__(self, ERG, name):
		super(Model, self).__init__()
		self.name = name
		##########DEFINE NN HERE##############
		protEmbLen = ERG["prot-drug"]["lenDomain1"]
		drugEmbLen = ERG["prot-drug"]["lenDomain2"]
		PROT_LATENT_SIZE = 30
		DRUG_LATENT_SIZE = 20
		D_SIDE_LATENT_SIZE = 20
		P_SIDE_LATENT_SIZE = 30
		ACTIVATION = t.nn.Sigmoid
		self.protEmb = t.nn.Embedding(protEmbLen, PROT_LATENT_SIZE)
		self.sideHid1 = t.nn.Sequential(t.nn.Linear(332, P_SIDE_LATENT_SIZE), t.nn.LayerNorm(P_SIDE_LATENT_SIZE), ACTIVATION())
		self.sideMix1 = t.nn.Bilinear(PROT_LATENT_SIZE, P_SIDE_LATENT_SIZE, 30)			
		self.protHid = t.nn.Sequential(t.nn.Linear(30, 20), t.nn.LayerNorm(20), ACTIVATION())
		
		self.drugEmb = t.nn.Embedding(drugEmbLen, DRUG_LATENT_SIZE)
		self.sideHid2 = t.nn.Sequential(t.nn.Linear(216, D_SIDE_LATENT_SIZE), t.nn.LayerNorm(D_SIDE_LATENT_SIZE), ACTIVATION())
		self.sideMix2 = t.nn.Bilinear(DRUG_LATENT_SIZE, D_SIDE_LATENT_SIZE, 20)			
		self.drugHid = t.nn.Sequential(t.nn.Linear(20, 10), t.nn.LayerNorm(10), ACTIVATION())
		
		self.biProtDrug = t.nn.Bilinear(20, 10, 10)
		self.outProtDrug = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,1), t.nn.Sigmoid())
		
		self.biprotProt = t.nn.Bilinear(20, 20, 10)
		self.outprotProt = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,2))
		
		self.biDrugDrug = t.nn.Bilinear(10, 10, 10)
		self.outDrugDrug = t.nn.Sequential( t.nn.LayerNorm(10), ACTIVATION(), t.nn.Dropout(0.3), t.nn.Linear(10,2))
		
		self.apply(self.init_weights)

	def forward(self, relName, i1, i2, s1=None, s2=None):
		if relName == "prot-drug":
			u = self.protEmb(i1)
			v = self.drugEmb(i2)
			v = self.sideMix2(v,self.sideHid2(s2))
			u = self.sideMix1(u,self.sideHid1(s1))
			v = self.drugHid(v).squeeze()
			u = self.protHid(u).squeeze()
			o = self.biProtDrug(u, v)
			o = self.outProtDrug(o)
		if relName == "prot-prot":
			u = self.protEmb(i1)
			v = self.protEmb(i2)
			u = self.sideMix1(u,self.sideHid1(s1))
			v = self.sideMix1(v,self.sideHid1(s2))
			u = self.protHid(u).squeeze()
			v = self.protHid(v).squeeze()
			o = self.biprotProt(u, v)
			o = self.outprotProt(o)
		if relName == "drug-drug":
			u = self.drugEmb(i1)
			v = self.drugEmb(i2)
			u = self.sideMix2(v,self.sideHid2(s1))
			v = self.sideMix2(v,self.sideHid2(s2))
			u = self.drugHid(u).squeeze()
			v = self.drugHid(v).squeeze()
			o = self.biDrugDrug(u, v)
			o = self.outDrugDrug(o)
			
		return o

class FocalLoss(t.nn.Module):

	def __init__(self, gamma,alpha):
		super(FocalLoss, self).__init__()
		self.gamma = t.tensor(gamma, dtype = t.float32)
		self.alpha = t.tensor(alpha, dtype = t.float32)
		self.eps = 1e-6

	def forward(self, input, target):
		#print(input)
		probs = t.sigmoid(input)
		focal_loss = self.alpha*t.pow(1-probs + self.eps, self.gamma).mul(-t.log(probs)).mul(target) + (1 - self.alpha)*t.pow(probs + self.eps, self.gamma).mul(-t.log(1 - probs)).mul(1 - target)
		return focal_loss.mean() 


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))

