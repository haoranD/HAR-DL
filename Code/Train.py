from DataLoader import BatchDataset
from Generate_SingleModel_LSTM import RNN, train

#Get the necessary parameters
#
#
#Number of epoches for each time
nm_epochs = 100
#Input rnn size for lstm layer
rnn_size = 256
#Number of lstm layers
number_of_layers = 2
#Drop and Keep in rnn
keep_rate = 0.5
#For test window
win = 5000


#Build the Dataloader for our dataset
fileDir='../../../../Data/Ensemble_Datasets/'
Dataset='Opp79'
BatchDataset = BatchDataset(fileDir=fileDir, Dataset=Dataset)
#Get the Dimesion for the dataset
dim = BatchDataset.X_train.shape[-1]
#Get the numer of class for the dataset
n_classes = BatchDataset.y_train.shape[-1]
#Send to the model
opts = [n_classes, nm_epochs, rnn_size, number_of_layers, keep_rate, dim, win]


#Start Training and save the results
#Set 30 times
for trial in range(30):
    #Set the saved model
    model_name = 'T_' + str(trial) + '_CE'
    #Build the model
    model = RNN(dim, keep_rate, rnn_size, number_of_layers, n_classes)
    #each trainning
    train(BatchDataset, model, opts, range_mb=[128, 256], range_win=[16, 32], model_name=model_name)

    #Plot the acc and loss changing trend
    #plot_history(history, 'results/{0}.png'.format(model_name))

print('Traing completed')