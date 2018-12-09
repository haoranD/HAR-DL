import torch
from torch.autograd import Variable
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

class RNN(torch.nn.Module):
    def __init__(self, dim, dropout_keep_prob, rnn_size, number_of_layers, n_classes):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.rnn_size = rnn_size
        self.rnn = torch.nn.LSTM(input_size=dim, hidden_size=rnn_size, num_layers=number_of_layers,
                                 dropout=dropout_keep_prob, batch_first=True)
        #self.linear1 = torch.nn.Linear(in_features=rnn_size, out_features=256)
        self.linear2 = torch.nn.Linear(in_features=rnn_size, out_features=n_classes)
        #self.relu = torch.nn.ReLU()

    def forward(self, x, nn_tuple_state):
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size]
        x, (h_n, c_n) = self.rnn(x, nn_tuple_state)
        #x = self.linear1(x)
        #x = self.relu(x)
        x = self.linear2(x)
        #x = self.relu(x)
        return x, (h_n, c_n)

def compute_acc_loss(pred, truth, loss_F, return_pred_truth=False):

    # Get the batch size and current data size for each
    # pred,truth: [batch_size, win, n_classes]
    batch_size = pred.size()[0]
    win = pred.size()[1]

    #Use the contiguous() in order to use view
    #Use view to reshape the tensor and flattern it (-1)means automatically change the row
    pred = pred.contiguous().view(batch_size * win, -1)#[batch_size*win, n_classes]
    truth = truth.contiguous().view(batch_size * win, -1)#[batch_size*win, n_classes]

    #Make the max probability to be the class
    truth = torch.argmax(truth, 1)#[batch_size*win]

    #Calculate loss  using loss function
    #.squeeze() delete the aittional dimension
    loss = loss_F(pred, truth.squeeze())
    pred_cls = torch.argmax(pred, 1)#[batch_size*win]
    pred1d = pred_cls.cpu()
    truth1d = truth.cpu()

    #Calculate the accuracy
    acc = (pred1d == truth1d).sum().numpy() / pred_cls.size()[0]
    if return_pred_truth:
        return acc, loss, pred1d, truth1d
    else:
        return acc, loss

#Initial the necessary h0, c0 as rnn components
def state_ini(number_of_layers, batch_size, rnn_size):
    h0 = Variable(torch.zeros(number_of_layers, batch_size, rnn_size))
    c0 = Variable(torch.zeros(number_of_layers, batch_size, rnn_size))
    if torch.cuda.is_available():
        h0, c0 = h0.cuda(), c0.cuda()
    state = (h0, c0)
    return state

def validation(BatchDataset, model, win, loss_F):

    #The different of Validation from train is:
    #batch size equal to 1
    #Bigger fixed windows
    batch_size = 1
    with torch.no_grad():
        sample_sz, accuracy, loss, sz, pred, truth = 0, 0, 0, 0, [], []
        state = state_ini(model.number_of_layers, batch_size, model.rnn_size)
        num_val_process = BatchDataset.X_valid.shape[1] // win + 1
        for j in range(num_val_process):
            x, y = BatchDataset.next_valid_or_test_batch(j, win, 'valid')
            x = Variable(torch.Tensor(x))
            y = Variable(torch.LongTensor(y))
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred_y, _ = model(x, state)
            acc, loss, pred1d, truth1d = compute_acc_loss(pred_y, y, loss_F, True)
            sample_sz = batch_size * win
            accuracy += acc * sample_sz
            loss += loss * sample_sz
            sz += sample_sz
            pred.extend(pred1d)
            truth.extend(truth1d)
        valid_accuracy = accuracy / sz
        valid_loss = loss / sz
        valid_f1 = f1_score(truth, pred, average='macro')

        sample_sz, accuracy, loss, sz, pred, truth = 0, 0, 0, 0, [], []
        state = state_ini(model.number_of_layers, batch_size, model.rnn_size)
        num_val_process = BatchDataset.X_test.shape[1] // win + 1
        for j in range(num_val_process):
            x, y = BatchDataset.next_valid_or_test_batch(j, win, 'test')
            x = Variable(torch.Tensor(x))
            y = Variable(torch.LongTensor(y))
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred_y, _ = model(x, state)
            acc, loss, pred1d, truth1d = compute_acc_loss(pred_y, y, loss_F, True)
            sample_sz = batch_size * win
            accuracy += acc * sample_sz
            loss += loss * sample_sz
            sz += sample_sz
            pred.extend(pred1d)
            truth.extend(truth1d)
        test_accuracy = accuracy / sz
        test_loss = loss / sz
        test_f1 = f1_score(truth, pred, average='macro')

    return valid_accuracy, valid_loss, valid_f1, test_accuracy, test_loss, test_f1

def train(BatchDataset, model, opts, range_mb=(128, 256), range_win=(16, 32), model_name = ''):

    #Use the GPU
    if torch.cuda.is_available():
        print('speed up by cuda and cudnn')
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    else:
        print('Can not find the GPU')

    # Initial the necessary parameters
    # classes    100         256          2               0.5     79   5000
    n_classes, nm_epochs, rnn_size, number_of_layers, keep_rate, dim, win = opts
    history = np.empty([0, 6], dtype=np.float32)
    results = np.empty([0, 5], dtype=np.float32)

    #Initial the function of optimizer and losser
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_F = torch.nn.CrossEntropyLoss()

    #100 epoches
    for epoch in range(nm_epochs):

        print('Start Training Epoch : {}'.format(epoch))

        #eg : batch_size = 18
        #X_train = [18,36165,79]
        #18 * 36165 = whole data one time
        train_loss, train_accuracy, train_sz, i, pos_end = 0, 0, 0, 0, 0
        batch_size = int(np.random.randint(low=range_mb[0], high=range_mb[1], size=1)[0])

        # train_x3D shape: [batch_size,train_len,dim]
        train_x3D, train_y3D, coverage = BatchDataset.next_train_batch(batch_size)
        train_len = BatchDataset.X_train.shape[0] // batch_size

        #initial the c,h
        #For each epoch, the state will transplante
        state = state_ini(number_of_layers, batch_size, rnn_size)

        #sub process : for each epoch : sub process has own model
        #states in sub process are sending between sub process
        while pos_end < train_len:

            pos_start = pos_end

            #randomly choose the current window length
            #Sum(different windows in all the times of the loop) = 36165(final pos_end)
            curr_win_len = int(np.random.randint(low=range_win[0], high=range_win[1], size=1)[0])
            pos_end += curr_win_len

            #[batch_size, curr_win_len, dim]
            #eg:[18, 20, 79]
            x = torch.Tensor(train_x3D[:, pos_start:pos_end, :])
            y = torch.LongTensor(train_y3D[:, pos_start:pos_end, :])

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            #Flattern and feed the data to model
            #[18*20,79]
            pred, state = model(x, state)
            acc, loss = compute_acc_loss(pred, y, loss_F)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            #not use each point but use each sample
            sample_sz = batch_size * curr_win_len
            train_accuracy += float(acc) * sample_sz
            train_loss += float(loss) * sample_sz
            train_sz += sample_sz

            print("Traing %06i, Train Accuracy = %.2f, Train Loss = %.3f" % (pos_end * batch_size, acc, loss))

        #use sample wise 2
        train_accuracy /= train_sz
        train_loss /= train_sz

        valid_accuracy, valid_loss, valid_f1, test_accuracy, test_loss, test_f1 = validation(BatchDataset, model, win, loss_F)
        print("Valid Accuracy = %.3f, Valid Loss = %.3f \nTest Accuracy = %.3f, Test Loss = %.3f"
              % (valid_accuracy, valid_loss, test_accuracy, test_loss))

        print('£££££££££££££££££££££££££££££££££££££££££££££££')
        print(valid_f1)

        epoch_history = np.array([train_accuracy, train_loss, valid_accuracy, valid_loss, test_accuracy, test_loss])
        history = np.float32(np.vstack((history, epoch_history)))
        epoch_wise_results = np.array([train_loss, valid_loss, test_loss, valid_f1, test_f1])
        results = np.float32(np.vstack((results, epoch_wise_results)))
        print('saving results ...')
        np.save('results/' + model_name + '_' + str(dim) + '.npy', results)

        #save model after 10 epoch
        if epoch >= 10:
            path = './model/{0}_{1}_{2}'.format(model_name, dim, epoch)
            torch.save(model, path)
            print("Model saved to %s" % path)

    return history