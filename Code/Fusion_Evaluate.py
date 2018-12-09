from Ensemble_Fusion import *
from DataLoader import BatchDataset


#Build the Dataloader for our dataset
fileDir='../../../../Data/Ensemble_Datasets/'
Dataset='Opp79'
BatchDataset = BatchDataset(fileDir=fileDir, Dataset=Dataset)
#BatchDataset = BatchDataset()

#Evaluate N times
for trial in range(6):
    #Save each ensemble models for N times
    Probs_ensemble = []

    #Read each time
    exp_id = 'T_' + str(trial) + '_CE_79'
    #Rank the model in each time
    #shown 30 is because of use from 10th epoch and want top 20 epoches
    exp = choose(exp_id, bestM=20, shown_TopN=30)

    #Get the score fusion for each time(Top 20 models / Top 30 models / 100epoch)
    prob_M1 = score_fusion(exp, BatchDataset, setnum=1, win=5000, trial=trial, M=20)

    #Save the model performance in some part
    prob_avg_1 = np.mean(prob_M1[:1, :, :], axis=0)
    prob_avg_5 = np.mean(prob_M1[:5, :, :], axis=0)
    prob_avg_10 = np.mean(prob_M1[:10, :, :], axis=0)
    prob_avg_20 = np.mean(prob_M1[:20, :, :], axis=0)

    Probs_ensemble.append(prob_avg_1)
    Probs_ensemble.append(prob_avg_5)
    Probs_ensemble.append(prob_avg_10)
    Probs_ensemble.append(prob_avg_20)

    print(prob_avg_20)
    np.save('./Trial/Opp/' + exp_id + '.npy', Probs_ensemble)