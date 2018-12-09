import torch
from torch.autograd import Variable
import torch.nn.functional as F
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
    batch_size = 1
    with torch.no_grad():
        accuracy, loss, sz, pred, truth = 0, 0, 0, [], []
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

        accuracy, loss, sz, pred, truth = 0, 0, 0, [], []
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



def pre_score_Rank(results, shown_TopN=30, valid_col=3):

    #Use valid f1 score or other standard
    print('valid_col', valid_col)
    test_col = valid_col + 1
    idx_set = np.argsort(results[:, valid_col])[::-1]#sort and reverse
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

    valid_set = np.zeros(shown_TopN)
    test_set = np.zeros(shown_TopN)
    for i in range(shown_TopN):

        valid_f1 = results[idx_set[i], valid_col]  # epoch_wise_results[3] = f1_valid
        test_f1 = results[idx_set[i], test_col]  # epoch_wise_results[4] = f1_test
        idx = idx_set[i]

        valid_set[i] = valid_f1
        test_set[i] = test_f1

        if idx >= 10:
            print('idx {}, f1_valid: {:.3f}, f1_test: {:.3f}'.format(idx, valid_f1, test_f1))

    return idx_set, valid_set, test_set


def choose(exp_id, bestM=20, shown_TopN=30):
    # shown_TopN=30 when doesn't deliver the parameter
    # Get the 100 results for each epoch
    # Find the model by the index from results
    a = np.load('results/' + exp_id + '.npy')

    #Rank by specific standard in trainnig phase, such as rank by valid f1 score
    #Return the model index
    idx_set, valid_set_a, test_set_a = pre_score_Rank(a, shown_TopN)

    # select the location of best model, idx_set>=10 as only saved models with epoch>=10
    Best_idx = idx_set[idx_set >= 10]

    #Get the index of top 20 model
    choice = []
    choice.append(Best_idx[:bestM])
    choice.append(exp_id)

    return choice

#make the mean of top cumulative model f1score as the final output
def score_fusion(exp, BatchDataset, setnum, win, trial, M=20):

    #Set the necessary parameters
    batch_size = 1
    test_x, test_y = BatchDataset.X_test, BatchDataset.y_test
    # each len
    lable_len, n_classes = test_x.shape[1], test_y.shape[-1]
    prob_M = np.zeros((M, lable_len, n_classes))
    f1_list = []
    loss_F = torch.nn.CrossEntropyLoss()


    #Load one model calculate the cumulative model score
    for i in range(M):

        #After ranking and choosing, use index to get the model
        idx = exp[0][i]
        modelpath = './model/' + exp[1] + '_' + str(idx)
        print(modelpath)
        model = torch.load(modelpath)

        prob_2d = np.zeros((lable_len, n_classes))

        #Initial the parameters
        sample_sz,accuracy, loss, sz, start, end, pred, truth = 0,0, 0, 0, 0, 0, [], []
        state = state_ini(model.number_of_layers, batch_size, model.rnn_size)
        # Use the fixed window
        num_val_process = BatchDataset.X_test.shape[1] // win + 1

        #Test the model and get the performance for each

        #Iterate the total/(batchsize * window) times
        for j in range(num_val_process):
            x, y = BatchDataset.next_valid_or_test_batch(j, win, 'test')
            x = Variable(torch.Tensor(x))
            y = Variable(torch.LongTensor(y))
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred_y, _ = model(x, state)
            acc, loss, pred1d, truth1d = compute_acc_loss(pred_y, y, loss_F, True)

            #Sample wise
            sample_sz = batch_size * win
            accuracy += acc * sample_sz
            loss += loss * sample_sz
            sz += sample_sz
            pred.extend(pred1d)
            truth.extend(truth1d)
            end += y.shape[1]
            pred_y_ = pred_y.contiguous().view(end - start, n_classes)
            pred_y_ = F.softmax(pred_y_)
            prob_2d[start:end, :] = np.reshape(np.array(pred_y_.detach().cpu()), ((end - start), n_classes))
            start = end

        test_accuracy = accuracy / sz
        test_loss = loss / sz
        test_f1 = f1_score(truth, pred, average='macro')

        print("  Test F1: %.3f" % test_f1)

        print("  Average Test Accuracy: %.3f" % test_accuracy)
        print("  Average Test Loss:     %.3f" % test_loss)

        #Initial the parameter
        prob_M[i, :, :] = prob_2d

        #top n mean f1
        #Calculate the mean of cumulative models score
        curr_prob_avg = np.mean(prob_M[:i + 1, :, :], axis=0)  # extract from 0-1, 0-2 ... 0-M, so, the first f1_fused will same with f1_value.
        fused_pred = np.argmax(curr_prob_avg, axis=1)

        f1_fused = f1_score(truth, fused_pred, average='macro')  # actualvalue means test_y with shape (lable_len, n_classes)

        print('curr_f1 {:.3f} and sz {} fused_f1 {:.3f}'.format(test_f1, i + 1, f1_fused))

        #Save the results
        if i == 0 or i == 4 or i == 9 or i == 19:
            f1_list.append(f1_fused)
            if setnum == 1:
                with open('f1_fused_Opp.txt', 'a') as the_file:
                    context = "In top " + str(i + 1) + " models, "
                    context += "the f1_fused of Trial "
                    context += str(trial) + " is: " + str(f1_fused)
                    context += "\n"
                    the_file.write(context)
                name = "Opptunity_" + str(trial)

            if setnum == 2:
                with open('f1_fused_Pamap2.txt', 'a') as the_file:
                    context = "In top " + str(i + 1) + ", "
                    context += "the f1_fused of Trial "
                    context += str(trial) + " is: " + str(f1_fused)
                    context += "\n"
                    the_file.write(context)
                name = "Pamap2_" + str(trial)

            if setnum == 3:
                with open('f1_fused_skoda.txt', 'a') as the_file:
                    context = "In top " + str(i + 1) + ", "
                    context += "the f1_fused of Trial "
                    context += str(trial) + " is: " + str(f1_fused)
                    context += "\n"
                    the_file.write(context)
                name = "Skoda_" + str(trial)
    csv = open('79.csv', 'a')
    np.savetxt(csv, f1_list, fmt='%.3f', delimiter=' ', header=name)
    return prob_M

def plot_history(history, save_name='acc_loss.png'):

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history[:, 0])
    plt.plot(history[:, 2])
    plt.plot(history[:, 4])
    plt.legend(['train_acc', 'val_acc', 'test_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(history[:, 1])
    plt.plot(history[:, 3])
    plt.plot(history[:, 5])
    plt.legend(['train_loss', 'val_loss', 'test_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.savefig(save_name)

    plt.show()