import numpy as np
import scipy.io
import pandas as pd
import os

class BatchDataset:

    def __init__(self, fileDir='../../../../Data/Ensemble_Datasets/', Dataset='Opp79'):
        
        if Dataset=='Opp79': 
            matfile = fileDir + str(Dataset) + '.mat'
            print(matfile)
            data = scipy.io.loadmat(matfile)

            X_train = np.transpose(data['trainingData'])
            X_valid = np.transpose(data['valData'])
            X_test = np.transpose(data['testingData'])
            print('normalising... zero mean, unit variance')
            mn_trn = np.mean(X_train, axis=0)
            std_trn = np.std(X_train, axis=0)
            X_train = (X_train - mn_trn)/std_trn
            X_valid = (X_valid - mn_trn)/std_trn
            X_test = (X_test - mn_trn)/std_trn
            print('normalising...X_train, X_valid, X_test... done')
            
            y_train = data['trainingLabels'].reshape(-1)-1
            y_valid = data['valLabels'].reshape(-1)-1
            y_test = data['testingLabels'].reshape(-1)-1
            
            y_train = pd.get_dummies( y_train , prefix='labels')
            y_valid = pd.get_dummies( y_valid , prefix='labels' )
            y_test = pd.get_dummies( y_test , prefix='labels' )
            
            y_valid.insert(17, 'labels_17', 0, allow_duplicates=False)#as validation only has 17 lables
            print('loading the 79-dim matData successfully . . .')
        
        if Dataset=='SKODA':
            matfile = fileDir + str(Dataset) + '.mat'
            data = scipy.io.loadmat(matfile)
            
            X_train = data['X_train']
            X_valid = data['X_valid']
            X_test = data['X_test']
            y_train = data['y_train'].reshape(-1)
            y_valid = data['y_valid'].reshape(-1)
            y_test = data['y_test'].reshape(-1)
            y_train = pd.get_dummies( y_train , prefix='labels' )
            y_valid = pd.get_dummies( y_valid , prefix='labels' )
            y_test = pd.get_dummies( y_test , prefix='labels' )

            print('the Skoda dataset was normalized to zero-mean, unit variance')
            print('loading the 33HZ 60d matData successfully . . .')
            
        if Dataset=='PAMAP2':
            matfile = fileDir + str(Dataset) + '.mat'
            data = scipy.io.loadmat(matfile)
            
            X_train = data['X_train']
            X_valid = data['X_valid']
            X_test = data['X_test']
            y_train = data['y_train'].reshape(-1)
            y_valid = data['y_valid'].reshape(-1)
            y_test = data['y_test'].reshape(-1)
            
            y_train = pd.get_dummies( y_train , prefix='labels' )
            y_valid = pd.get_dummies( y_valid , prefix='labels' )
            y_test = pd.get_dummies( y_test , prefix='labels' )
            
            print('the PAMAP2 dataset was normalized to zero-mean, unit variance')
            print('loading the 33HZ PAMAP2 52d matData successfully . . .')

        self.X_train = X_train.astype(np.float32)
        self.X_valid = np.expand_dims(X_valid.astype(np.float32), 0)
        self.X_test  = np.expand_dims(X_test.astype(np.float32), 0)
        self.y_train = y_train.astype(np.int64)
        self.y_valid = np.expand_dims(y_valid.astype(np.int64), 0)
        self.y_test  = np.expand_dims(y_test.astype(np.int64), 0)

        if not os.path.exists('./model'):
            os.mkdir('./model')
        if not os.path.exists('./results'):
            os.mkdir('./results')

    def next_train_batch(self, batch_size):
        dim = len(self.X_train[0])
        n_classes = self.y_train.shape[-1]
        seqence_len = len(self.X_train) // batch_size

        # the default one
        indices_start = np.random.randint(low=0, high=len(self.X_train) - seqence_len, size=(batch_size,))

        ########################## coverage #########################################################
        indices_all_2d = np.zeros((batch_size, seqence_len))
        for i in range(batch_size):
            indices_all_2d[i, :] = np.arange(indices_start[i], indices_start[i] + seqence_len)
        indices_all = np.reshape(indices_all_2d, (-1))

        coverage = 100 * len(np.unique(indices_all)) / len(indices_all)
        #####################################################################################################


        X_train3D = np.zeros((batch_size, seqence_len, dim), dtype=np.float32)
        y_train3D = np.zeros((batch_size, seqence_len, n_classes), dtype=np.uint8)
        for i in range(batch_size):
            idx_start = indices_start[i]
            idx_end = idx_start + seqence_len
            X_train3D[i, :, :] = self.X_train[idx_start:idx_end, :]
            y_train3D[i, :, :] = self.y_train.iloc[idx_start:idx_end, :]

        return X_train3D, y_train3D, coverage

    def next_valid_or_test_batch(self, iter, win, mode='valid'):
        #iter means the next start
        start = iter * win
        if mode == 'valid':
            end = np.min((self.X_valid.shape[1], start + win))
            x = self.X_valid[:,start:end,:]
            y = self.y_valid[:, start:end, :]
        elif mode == 'test':
            end = np.min((self.X_test.shape[1], start + win))
            x = self.X_test[:, start:end, :]
            y = self.y_test[:, start:end, :]
        return x, y

#BatchDataset(fileDir='../../../Data/Ensemble_Datasets/', Dataset='Opp79')