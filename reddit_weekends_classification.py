import torch
import torch.nn as nn
import torch.optim as optim
from helpers import index_splitter
from helpers import make_balanced_sampler
from data_generation import get_reddit_labeled_weekends
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from RedditWeekendModel import RedditWeekendModel
from trainer import StepByStep
from sklearn.preprocessing import StandardScaler

x, y = get_reddit_labeled_weekends()
train_idx, val_idx = index_splitter(len(x), [80,20])

x_tensor = torch.reshape(torch.as_tensor(x), (torch.as_tensor(x).shape[0],torch.as_tensor(x).shape[1],1))
#(365,24,1)
y_tensor = torch.reshape(torch.as_tensor(y), (torch.as_tensor(y).shape[0],torch.as_tensor(y).shape[1],1))
#(365,24,1) -> #(365,1)

x_train_tensor = x_tensor[train_idx]
y_train_tensor = torch.as_tensor([y_value[0] for y_value in y_tensor[train_idx]])

#scale x
reshaped_x_train_tensor = x_train_tensor.reshape(-1,1)
#(8760*0.8,1)
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(reshaped_x_train_tensor)

#parte 1 = tensor(8760*0.8,1)->array(365*0.8,24,1)->tensor(365*0.8,24,1)
scaled_x_train_tensor = torch.as_tensor(scaler.transform(reshaped_x_train_tensor).reshape(x_train_tensor.shape))
#tensor(365*0.8,24,1)

x_val_tensor = x_tensor[val_idx]
y_val_tensor = torch.as_tensor([y_value[0] for y_value in y_tensor[val_idx]])

reshaped_x_val_tensor = x_val_tensor.reshape(-1,1)
#tensor(8760*0.2,1)
scaled_x_val_tensor = torch.as_tensor(scaler.transform(reshaped_x_val_tensor).reshape(x_val_tensor.shape))
#tensor(365*0.2,24,1)


#making the samples
sampler = make_balanced_sampler(y_train_tensor)

#x: tensor(0.8*365,24,1), y: tensor(0.8*365,1)
train_dataset = TensorDataset(scaled_x_train_tensor.float(), y_train_tensor.view(-1, 1).float())
test_dataset = TensorDataset(scaled_x_val_tensor.float(), y_val_tensor.view(-1, 1).float())
#x: tensor(0.2*365,24,1), y: tensor(0.2*365,1)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(dataset=test_dataset, batch_size=16)


num_layers = 2
rnn_layer = nn.LSTM
hidden_dim = 2
bidirectional = False

torch.manual_seed(21)
model = RedditWeekendModel(n_features=1, hidden_dim=hidden_dim, n_outputs=1, num_layers=num_layers, bidirectional=bidirectional, rnn_layer=rnn_layer)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

sbs_rnn = StepByStep(model, loss, optimizer)
sbs_rnn.set_loaders(train_loader, test_loader)
sbs_rnn.train(100)

fig = sbs_rnn.plot_losses()
fig.show()
print(StepByStep.loader_apply(test_loader, sbs_rnn.correct))

