import torch
import torch.nn as nn
import torch.optim as optim
from helpers import index_splitter
from helpers import make_balanced_sampler
from data_generation import get_reddit_labeled_starting_hours
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from RedditHoursModel import RedditHoursModel
from trainer import StepByStep
from sklearn.preprocessing import StandardScaler
import numpy as np

x, y = get_reddit_labeled_starting_hours(10000, 24)
train_idx, val_idx = index_splitter(len(x), [80,20])

x_tensor = torch.reshape(torch.as_tensor(x), (torch.as_tensor(x).shape[0],torch.as_tensor(x).shape[1],1))
y_tensor = torch.as_tensor(y)

x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]

#scale x
reshaped_x_train_tensor = x_train_tensor.reshape(-1,1)

scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(reshaped_x_train_tensor)

scaled_x_train_tensor = torch.as_tensor(scaler.transform(reshaped_x_train_tensor).reshape(x_train_tensor.shape))

x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

reshaped_x_val_tensor = x_val_tensor.reshape(-1,1)
scaled_x_val_tensor = torch.as_tensor(scaler.transform(reshaped_x_val_tensor).reshape(x_val_tensor.shape))


#making the samples
#sampler = make_balanced_sampler(y_train_tensor)

train_dataset = TensorDataset(scaled_x_train_tensor.float(), y_train_tensor)
test_dataset = TensorDataset(scaled_x_val_tensor.float(), y_val_tensor)

#train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
#The examples should be uniformly distributed among classes
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16)

num_layers = 2
rnn_layer = nn.LSTM
bidirectional = False
n_outputs = torch.unique(y_train_tensor, return_counts=True)[0].shape[0]
hidden_dim = int(n_outputs/2)

torch.manual_seed(21)
model = RedditHoursModel(n_features=1, hidden_dim=hidden_dim, n_outputs=n_outputs, num_layers=num_layers, bidirectional=bidirectional, rnn_layer=rnn_layer)
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(model.parameters())

sbs_rnn = StepByStep(model, loss, optimizer)
sbs_rnn.set_loaders(train_loader, test_loader)
sbs_rnn.train(500)

fig = sbs_rnn.plot_losses()
fig.show()
accuracy_matrix = (StepByStep.loader_apply(test_loader, sbs_rnn.correct))
print(accuracy_matrix)
accuracy = [row[0]/row[1]for row in accuracy_matrix]
for hour, accuracy_hour in enumerate(accuracy_matrix):
    print(f'Accuracy at hour {hour}: {accuracy_hour[0]/accuracy_hour[1] * 100} %')
print(f'Total accuracy: {np.mean(accuracy)*100} %')


