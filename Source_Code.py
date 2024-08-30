# ---------------------------------------------------------
# Does Air Quality Really Impact COVID-19 Clinical Severity
# ---------------------------------------------------------
# Based on PyTorch Geometric Temporal 
# ---------------------------------------------------------
# Paper Submission to KDD 2021
# ---------------------------------------------------------
# V1

#%% Libraries  
# Torch-related libraries 
import torch  
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
# Other libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from tqdm import tqdm
import time   

start_time = time.time() # To measure time

#%% Functions
def getDaily_Dataframe(dfData, initialDay, finalDay):
    # Starting index 
    indStart = dfData.index.get_loc(initialDay)
    # Ending index 
    indFinal = dfData.index.get_loc(finalDay)
    # Create new daily cases/deaths dataframe
    dfData_NewD = dfData.iloc[0:5].copy()
    for iD in range(indStart, indFinal+1):  
        cases_T = [float(a) for a in list(dfData.loc[dfData.index[iD]])]
        cases_T_DBack = [float(a) for a in list(dfData.loc[dfData.index[iD-1]])]
        arrayNewCases = np.subtract(cases_T, cases_T_DBack)
        arrayNewCases[arrayNewCases<0] = 0.0  # In case error in monitoring-data 
        df_Row_aux = pd.DataFrame([arrayNewCases.tolist()], columns=dfData.columns)
        # To append in the dataframe 
        dfData_NewD = dfData_NewD.append(df_Row_aux)
    # Change Index name 
    lisIDX_OUT = list(dfData.index[indStart:(indFinal+1)])
    lisIDX_OUT.insert(0, dfData.index[4])
    lisIDX_OUT.insert(0, dfData.index[3])
    lisIDX_OUT.insert(0, dfData.index[2])
    lisIDX_OUT.insert(0, dfData.index[1])
    lisIDX_OUT.insert(0, dfData.index[0])
    dfData_NewD.index = lisIDX_OUT
    dfData_NewD.index.name = dfData.index.name
    return dfData_NewD.copy()

def replace_greater_Dataframe(dfData, maxVal):
    dfData_NewD = dfData.copy() 
    for strFIPS in dfData.columns:  
        lisStr = dfData_NewD[strFIPS].iloc[5:].values.tolist()
        arrayAux = np.array( [float(a) for a in lisStr] ) 
        if(np.sum(arrayAux<=maxVal)>0):
            valReplace = np.quantile(arrayAux[arrayAux<=maxVal], 0.2)
        else:
            valReplace = 0.05
        lisFloat = np.where(arrayAux>maxVal, valReplace, arrayAux).tolist()
        lisNewStr = [str(a) for a in lisFloat]
        dfData_NewD[strFIPS].iloc[5:] = lisNewStr
    # Return
    return dfData_NewD.copy()

#%% Parameters
# Features's Selection
#
# Training Parameters
percenTrain = 0.8   # Percentage for training: value in (0,1] 
NEpochs = 600 # Number of Epochs for training
# Historical and Forecasting parameters
LBack = 4 # Historical: LBack>=0 , where 0 means only takes current X observation
LForward = 15 # Forecasting: LForward>=0 , where 0 means only takes current Y value
# Period of time (Range) 
initialDay = '02/01/2020' 
finalDay = '12/31/2020'   
# Dataset Selection 
#letter2State = 'CA'  
#letter2State = 'PA'
letter2State = 'TX' 

folderIndexes = 'Sel_0_1_2_4_5_6_7_8'
pathInputALL = letter2State+'/'+folderIndexes+'/'  
# Adjacent Matrix
nameFileCounties = 'IG_List_County.csv' 
nameFileADJ = 'IG_County_ADJ.csv' 
# Input Files
nameIN_RISK = 'NX_RISK_Covid19.csv'
nameIN_CASES = 'NX_CASES_Covid19.csv'
nameIN_DEATH = 'NX_DEATH_Covid19.csv'
nameIN_RT = 'NX_RT_Covid19.csv'
nameIN_AOD = 'NX_AOD_Satellite.csv'
nameIN_TEMP = 'NX_TEMP_Satellite.csv'
nameIN_RH = 'NX_RH_Satellite.csv'
nameIN_HOSPI = 'NX_HOSPI_Covid19.csv'
nameIN_ICU = 'NX_ICU_Covid19.csv'
nameIN_SEVERITY = 'NX_SEVERITY_Covid19.csv'
# Output files
pathOutput = 'OUTPUT/'
# Selected Features

#****** SECOND PART OF EXPERIMENTS
#sel_FEATURES = [nameIN_HOSPI] 
#sel_FEATURES = [nameIN_HOSPI, nameIN_AOD]
#sel_FEATURES = [nameIN_HOSPI, nameIN_DEATH]
#sel_FEATURES = [nameIN_HOSPI, nameIN_TEMP]  
#sel_FEATURES = [nameIN_HOSPI, nameIN_RH]
sel_FEATURES = [nameIN_HOSPI, nameIN_DEATH, nameIN_AOD]
#sel_FEATURES = [nameIN_HOSPI, nameIN_DEATH, nameIN_TEMP]
#sel_FEATURES = [nameIN_HOSPI, nameIN_DEATH, nameIN_RH]

#**** Selected Target 
sel_TARGET = nameIN_HOSPI


#%% Open Datasets
# Open Adjacency Matrix and County
df_ADJ = pd.read_csv(pathInputALL+nameFileADJ) # Adjacency matrix
df_County = pd.read_csv(pathInputALL+nameFileCounties) # List of Counties
# The target Serie
df_Target_Aux = pd.read_csv(pathInputALL+sel_TARGET, index_col=0) # Selected target
#if((sel_TARGET==nameIN_CASES) or (sel_TARGET==nameIN_DEATH) or (sel_TARGET==nameIN_HOSPI)):  
if((sel_TARGET==nameIN_CASES) or (sel_TARGET==nameIN_DEATH)):  
    df_Target = getDaily_Dataframe(df_Target_Aux, initialDay, finalDay)
elif(sel_TARGET==nameIN_AOD):
    df_Target = replace_greater_Dataframe(df_Target_Aux, 5)
else:
    df_Target = df_Target_Aux.copy() 

# Open Features
lis_DF_FEATURES = []
for k in range(len(sel_FEATURES)):
    dfData = pd.read_csv(pathInputALL+sel_FEATURES[k], index_col=0)
    if((sel_FEATURES[k]==nameIN_CASES) or (sel_FEATURES[k]==nameIN_DEATH)):
        dfData_NewD = getDaily_Dataframe(dfData, initialDay, finalDay)
    elif(sel_FEATURES[k]==nameIN_AOD):
        dfData_NewD = replace_greater_Dataframe(dfData, 5)
    else:
        dfData_NewD = dfData.copy()
    # Append New Dataframe/Dataset  
    lis_DF_FEATURES.append(dfData_NewD.copy())


#%% Build Data for each Day
DATABASE_lis = []
# Range of dates
datesRange = pd.date_range(start=initialDay, end=finalDay, freq='D') 
datesRangeTXT = list(datesRange.strftime('%m/%d/%Y'))
# Dynamic Graph Parameters
DGP_num_nodes = df_County.shape[0]
DGP_num_node_features = len(lis_DF_FEATURES)
DGP_num_edges = df_ADJ.shape[0]
DGP_NGraphs = len(datesRange)
# Graph connectivity in COO format with shape [2, num_edges] and type torch.long 
array_ADJ = np.concatenate((df_ADJ[['From','To']].values, df_ADJ[['To','From']].values))
tensor_edge_index = torch.from_numpy(array_ADJ).t().contiguous()
# Edge feature matrix 
tensor_edge_attr = torch.tensor([1.0]*tensor_edge_index.shape[1], dtype=torch.float)
# To save information for normalization and standarization
array_min_Y = 100000000*np.ones(DGP_num_nodes)
array_max_Y = -100000000*np.ones(DGP_num_nodes)
array_min_X = 100000000*np.ones((DGP_num_nodes, DGP_num_node_features))
array_max_X = -100000000*np.ones((DGP_num_nodes, DGP_num_node_features))
array_mean_Y = np.zeros(DGP_num_nodes) 
array_mean_X = np.zeros((DGP_num_nodes, DGP_num_node_features)) 
# Build DAILY data: feature matrix 'x' and target 'y'  
for txtDay in tqdm(datesRangeTXT):  
    # Target 'y' using value  
    lisY = [1]*df_County.shape[0]  # List same size of number of counties
    for iC in range(df_County.shape[0]):  
        val_Y = float(df_Target[str(df_County.iloc[iC].FIPS)].loc[txtDay])
        lisY[iC] = val_Y
        if(val_Y<array_min_Y[iC]): # Minimum
            array_min_Y[iC] = val_Y
        if(val_Y>array_max_Y[iC]): # Maximum
            array_max_Y[iC] = val_Y
        array_mean_Y[iC] = array_mean_Y[iC] + val_Y
    tensor_Y = torch.tensor(lisY, dtype=torch.float) 
    # Node feature matrix 'x' 
    lisX = []
    for iC in range(df_County.shape[0]):  
        aux_X = [0]*len(sel_FEATURES)
        for kF in range(len(sel_FEATURES)): 
            valAux = float(lis_DF_FEATURES[kF][str(df_County.iloc[iC].FIPS)].loc[txtDay]) 
            aux_X[kF] = valAux
            if(valAux<array_min_X[iC,kF]): # Minimum
                array_min_X[iC,kF] = valAux
            if(valAux>array_max_X[iC,kF]): # Maximum
                array_max_X[iC,kF] = valAux
            array_mean_X[iC,kF] = array_mean_X[iC,kF] + valAux
        lisX.append(aux_X)
    tensor_X = torch.tensor(lisX, dtype=torch.float)
    # Save Graph in List
    DATABASE_lis.append( Data(edge_index=tensor_edge_index, edge_attr=tensor_edge_attr, x=tensor_X,  y=tensor_Y) )

#%% Build new DATABASE including HISTORICAL and FORECASTING
DATABASE_HISFORE = []
for kB in range(LBack, DGP_NGraphs-LForward):
    # Forecasting Y
    tensor_YFore = DATABASE_lis[kB+LForward].y 
    # Rebuild X features
    torch_XCat = torch.empty(DGP_num_nodes, 0)
    for iH in range(kB-LBack,kB+1):
        torch_XCat = torch.cat((torch_XCat, DATABASE_lis[iH].x), 1)
        #print(f'  kB:  {kB}    iH:  {iH}')
        #print(DATABASE_lis[iH].x.shape)
        #print(DATABASE_lis[iH].x)
        #print(torch_XCat.shape)
        #print(torch_XCat)
    # To add Merged Features and Forecasting Y
    DATABASE_HISFORE.append( Data(edge_index=tensor_edge_index, edge_attr=tensor_edge_attr, x=torch_XCat,  y=tensor_YFore) )
# Some general variables will change
HIFO_num_node_features = len(lis_DF_FEATURES)*(LBack+1) 
HIFO_NGraphs = len(DATABASE_HISFORE)  
HIFO_num_nodes = df_County.shape[0]
HIFO_num_edges = df_ADJ.shape[0]

#%% Separate in TRAIN and TEST  
indTop_Train = int(percenTrain*HIFO_NGraphs)
train_dataset = DATABASE_HISFORE[0:indTop_Train]
test_dataset = DATABASE_HISFORE[indTop_Train:HIFO_NGraphs]

#%% To define a Recurrent Graph Neural Network architecture
# *******  The used at the moment for experiments: For paper  *******
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 256, 1)
        self.linear = torch.nn.Linear(256, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(h)
        h = self.linear(h)
        return h

#%% To train and test several times
NTT = 10 # Number of times
arrayMSE = np.zeros((NTT, 2)) # To save MSE
arrayMSE_County = np.zeros((NTT, HIFO_num_nodes)) # To save MSE 
for iTT in range(NTT):
    print(f'  ----------- Run [{iTT}] ----------- ')
    #%% Training 
    model = RecurrentGCN(node_features = HIFO_num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()  
    for epoch in tqdm(range( NEpochs )):   # make your loops show a smart progress meter 
        cost = 0
        for timeCount, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            #cost = cost + torch.mean((y_hat.t().contiguous()-snapshot.y)**2)
            cost = cost + torch.sqrt( torch.mean((y_hat.t().contiguous()-snapshot.y)**2) )
        cost = cost / (timeCount+1)
        #print("Training >>> Final MSE: {:.10f}".format(cost.item()))
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()  

    print("Training >>> Final MSE: {:.4f}".format(cost.item()))
    arrayMSE[iTT, 0] = cost.item()
    #%% Evaluation 
    arrayTorch_Counties_MSE = torch.tensor([0]*HIFO_num_nodes, dtype=torch.float) # To save individual counties errors 
    model.eval()  
    cost = 0      
    for timeCount, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        #cost = cost + torch.mean((y_hat.t().contiguous()-snapshot.y)**2)
        cost = cost + torch.sqrt( torch.mean((y_hat.t().contiguous()-snapshot.y)**2) )
        arrayTorch_Counties_MSE = arrayTorch_Counties_MSE + (y_hat.t().contiguous()-snapshot.y)**2     
    cost = cost / (timeCount+1)
    cost = cost.item()
    # To save county-individuals MSE
    #arrayTorch_Counties_MSE = arrayTorch_Counties_MSE / (timeCount+1)
    arrayTorch_Counties_MSE = torch.sqrt( arrayTorch_Counties_MSE / (timeCount+1) )
    arrayMSE_County[iTT] = arrayTorch_Counties_MSE.detach().numpy()[0]

    print("Test >>> MSE: {:.4f}".format(cost))
    arrayMSE[iTT, 1] = cost

# Add the mean by row 
arrayMSE = np.concatenate((arrayMSE, np.mean(arrayMSE, axis=0).reshape(1,2)))

# Convert to dataframe and save in CSV 
txtIndex = [str(a) for a in range(NTT)]
txtIndex.append('average')
dfResul_MSE = pd.DataFrame(index=txtIndex, data=arrayMSE, columns=['Train_MSE', 'Test_MSE'])  
dfResul_MSE.index.name = 'Run'  
# Save to CSV 
nameFile_CSV = ''
for kN in range(len(sel_FEATURES)):
    nameFile_CSV = nameFile_CSV + sel_FEATURES[kN].split('_')[1]+'_'
nameFile_CSV = nameFile_CSV + '_' + sel_TARGET.split('_')[1].lower() + '.csv'
dfResul_MSE.to_csv(pathOutput+'IMGS/'+nameFile_CSV, header=True, index=True)

# MSE at COUNTY level
txt_FIPS = [str(a) for a in df_County.FIPS]
arrayMSE_County = np.concatenate((arrayMSE_County, np.mean(arrayMSE_County, axis=0).reshape(1, DGP_num_nodes)))
dfResul_MSE_County = pd.DataFrame(index=txtIndex, data=arrayMSE_County, columns=txt_FIPS)
dfResul_MSE_County.index.name = 'Run'
dfResul_MSE_County.to_csv(pathOutput+'IMGS/county_'+nameFile_CSV, header=True, index=True)  

#%% Final Time
print("RUNNING TIME: "+str((time.time() - start_time))+" Seg ---  "+str((time.time() - start_time)/60)+" Min ---  "+str((time.time() - start_time)/(60*60))+" Hr ") 
print('DONE !!!')

#%%
print('Plot Training and Testing >> Time series')
allMSE_train = [] 
allMSE_test = [] 
for selNode in tqdm(range(HIFO_num_nodes)):
    #selNode = 8 
    # --- For training 
    train_vecY_hat = [0]*len(train_dataset)
    train_vecY_GT = [0]*len(train_dataset)
    for time, snapshot in enumerate(train_dataset):
        train_y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        train_vecY_hat[time] = float(train_y_hat[selNode])
        train_vecY_GT[time] = float(snapshot.y[selNode])
    # --- For testing 
    test_vecY_hat = [0]*len(test_dataset)
    test_vecY_GT = [0]*len(test_dataset)
    for time, snapshot in enumerate(test_dataset):
        test_y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        test_vecY_hat[time] = float(test_y_hat[selNode])
        test_vecY_GT[time] = float(snapshot.y[selNode])
    # --- Save County level MSE
    #allMSE_train.append(np.mean((np.array(train_vecY_hat)-np.array(train_vecY_GT))**2))  
    #allMSE_test.append(np.mean((np.array(test_vecY_hat)-np.array(test_vecY_GT))**2))  
    allMSE_train.append(  np.sqrt(np.mean((np.array(train_vecY_hat)-np.array(train_vecY_GT))**2))  ) 
    allMSE_test.append(  np.sqrt(np.mean((np.array(test_vecY_hat)-np.array(test_vecY_GT))**2))  )
    # --- Plot 
    #plt.figure(num=None, figsize=(7, 4), dpi=80, facecolor='w', edgecolor='k')  
    vecX = range(HIFO_NGraphs) 
    #plt.plot(vecX[:indTop_Train], train_vecY_GT, 'k')  
    #plt.plot(vecX[:indTop_Train], train_vecY_hat, 'g') 
    #plt.plot(vecX[indTop_Train:HIFO_NGraphs], test_vecY_GT, 'b')  
    #plt.plot(vecX[indTop_Train:HIFO_NGraphs], test_vecY_hat, 'r') 
    #plt.title(f' {selNode}: FIPS[ {txt_FIPS[selNode]} ]   MSE: [{np.round(allMSE_train[-1], 3)}] and [{np.round(allMSE_test[-1], 3)}]')  
    #plt.savefig(pathOutput+'IMGS/Plot_'+txt_FIPS[selNode]+'.pdf')
    #plt.show()



