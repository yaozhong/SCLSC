# models part
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# The supervised contrastive loss
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1

        # I CHANGE TORCH.SUM AXIS
        dist_sq = torch.sum(torch.pow(diff, 2), -1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]

        return loss


# The MLP encoder structure, Yusri's current implementation
class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.batchnorm1(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        return out

class MLP_nobatchnorm(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP_nobatchnorm, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        return out

class MLP_nodo(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP_nodo, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.batchnorm1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.batchnorm2(out)

        out = self.linear3(out)
        return out

# use the CNN module previously used
class CNN(nn.Module):
        def __init__(self, input_dim, kernel_size=7, dr=0.1):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv1d(1,16,kernel_size=kernel_size, stride=2)
                self.bn1 = nn.BatchNorm1d(16)
                self.conv2 = nn.Conv1d(16,32, kernel_size=kernel_size, stride=2)
                self.bn2 = nn.BatchNorm1d(32)
                self.relu = nn.ReLU()

                self.maxpool = nn.MaxPool1d(2)

                self.fc1 = nn.Linear(7936, 32)
                #self.fc1 = nn.Linear(4608, 1474)


        def forward(self, x):
                x = self.bn1(self.relu(self.conv1(x)))
                x = self.bn2(self.relu(self.conv2(x)))
                x = self.maxpool(x)

                x = self.fc1(torch.flatten(x, 1))

                return x



# Function to transform raw representation to embedded representation
def project(encoder, X, device, encoder_model):

    if encoder_model == "CNN":
        X = np.expand_dims(X, 1)

    with torch.no_grad():
        encoder.eval()
        if torch.is_tensor(X):
            X = X.to(device=device, dtype=torch.float32)
        else:
            X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        emb_X = encoder(X).detach().cpu().numpy()
    return emb_X


# early stopping criterion
class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):

            self.counter += 1
            print("Counter = %d, Val loss = %f, threshold= %f " %
                  (self.counter, validation_loss,
                   self.min_validation_loss + self.min_delta))
            if self.counter >= self.patience:
                return True
        return False

