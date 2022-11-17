The goal of this project is to use a data driven method to greatly simplify the simulation of a beam of charged particles going through a series of magnetic lenses. This method brings computation times from hours of complicated and overdetailed simulation (which were also difficult to correlate to reality) down to real time simulation. The main limitation is currently the small sample size. However, the model can easily be scaled once more data becomes available.

This is the notebook used to train and test various models. It's not the cleanest implementation as this was written in prototyping phase and many different approaches were tried. 

First of all, we import libraries and datasets:


```python
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.optim import SGD
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from os.path import join
import datetime
from matplotlib import pyplot as plt
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"

data = pd.read_excel("./CATANA_clean.xls", sheet_name=0, header=0).fillna(0)
data1 = pd.read_excel("./MAGNEX.xls", sheet_name=0, header=0).fillna(0)
data2 = pd.read_excel("./TEBE.xls", sheet_name=0, header=0).fillna(0)
data3 = pd.read_excel("./ZERO GRADI.xls", sheet_name=0, header=0).fillna(0)

data = data.drop(labels=["EXFC1", "EXFC2"], axis=1)
data1 = data1.drop(labels=["Data"], axis=1)
data2 = data2.drop(labels=["Data"], axis=1)
data3 = data3.drop(labels=["Data"], axis=1)

# Merging data
data = data.append(data1, ignore_index=True)
data = data.append(data2, ignore_index=True)
data = data.append(data3, ignore_index=True)

data = data.fillna(0)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CMx-EXQU1</th>
      <th>CMy-EXQU1</th>
      <th>STDx-EXQU1</th>
      <th>STDy-EXQU1</th>
      <th>EXST2x</th>
      <th>EXQP1</th>
      <th>EXST3x</th>
      <th>EXQP2</th>
      <th>EXST4y</th>
      <th>EXQP3</th>
      <th>EXST5y</th>
      <th>EXST5x</th>
      <th>CMx-EXQU2</th>
      <th>CMy-EXQU2</th>
      <th>STDx-EXQU2</th>
      <th>STDy-EXQU2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>340</td>
      <td>370</td>
      <td>44</td>
      <td>9</td>
      <td>3.0</td>
      <td>67.0</td>
      <td>0.0</td>
      <td>83.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-2.0</td>
      <td>368</td>
      <td>355</td>
      <td>49</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>340</td>
      <td>366</td>
      <td>34</td>
      <td>8</td>
      <td>2.0</td>
      <td>70.5</td>
      <td>-1.7</td>
      <td>84.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.8</td>
      <td>0.0</td>
      <td>349</td>
      <td>364</td>
      <td>27</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>334</td>
      <td>366</td>
      <td>32</td>
      <td>8</td>
      <td>2.0</td>
      <td>70.5</td>
      <td>-1.7</td>
      <td>84.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.8</td>
      <td>0.0</td>
      <td>351</td>
      <td>362</td>
      <td>29</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>339</td>
      <td>367</td>
      <td>34</td>
      <td>10</td>
      <td>3.5</td>
      <td>67.5</td>
      <td>0.9</td>
      <td>85.5</td>
      <td>3.1</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>359</td>
      <td>360</td>
      <td>35</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>377</td>
      <td>346</td>
      <td>42</td>
      <td>17</td>
      <td>4.1</td>
      <td>72.0</td>
      <td>0.6</td>
      <td>85.0</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>411</td>
      <td>339</td>
      <td>40</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



This data contains info about the beam of particles and the configuration of the magnetic lenses. 
The first 4 columns contain the position an dimension of the beam before encountering a series of magnetic lenses. These magnetic lenses are fixed in position so that the only parameter needed to study them is the electric current flowing through each lens which determines all its deflecting properties. These currents are recorded in the 'middle' columns. 
The last 4 columns contain the position and dimensions of the beam after the series of lenses. 
The goal is to predict these last 4 columns using all the rest of the data as input of a neural network. 

Data is normalized and shuffled:


```python
data_norm = ((data - data.min()) / (data.max() - data.min())).fillna(0)

random_seed = 1234
data_shuffle = data_norm.sample(frac=1, random_state=random_seed)
test_len = int(len(data)*0.25)
datarray = np.array(data_shuffle[:-test_len])
datarray_test = np.array(data_shuffle[-test_len:])
datarray.shape, datarray_test.shape
```




    ((75, 16), (24, 16))




```python
datarray_in = torch.Tensor(datarray[:, :-4]).to(device)
datarray_out = torch.Tensor(datarray[:, -4:]).to(device)
datatest_in = torch.Tensor(datarray_test[:, :-4]).to(device)
datatest_out = torch.Tensor(datarray_test[:, -4:]).to(device)

```

Next we have the model and a training pipeline


```python
class BeamNetV4(nn.Module):
    def __init__(self):
        super(BeamNetV4, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fc1 = nn.Sequential(
            nn.Linear(12, 100), nn.Linear(100, 100), nn.Linear(100, 150)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(150, 100), nn.Linear(100, 75), nn.Linear(75, 4)
        )

    def forward(self, x):

        out = self.fc1(x.float().view(-1, 12))
        out = self.fc2(out).view(-1, 4)

        return out
    
def train(
    model,
    epochs,
    datarray_in,
    datarray_out,
    datatest_in,
    datatest_out,
    device=device,
    lr=0.001,
    momentum=0.9,
    logdir="logs",
    exp_name="secondo",
    log_interval=100,
):
    #     criterion = nn.MSELoss()
    losses_train = []
    losses_test = []
    datarray_in = datarray_in.to(device)
    datarray_out = datarray_out.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # writer = SummaryWriter(join(logdir, exp_name, str(run_date)))
    for e in tqdm(range(epochs)):
        #         print(model(datarray_in))

        # Training
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            outs = model(datarray_in)
            loss = criterion(outs.view(-1, 4), datarray_out.view(-1, 4))
            #             writer.add_scalar('Loss', loss, global_step=e)
            loss.backward()
            losses_train.append(loss.item())
            optimizer.step()

        # Testing
        with torch.set_grad_enabled(False):
            test_out = model(datatest_in)
            loss_test = criterion(test_out.view(-1, 4), datatest_out.view(-1, 4))
            losses_test.append(loss_test.item())
        #             writer.add_scalar('Loss_test', loss_test, global_step=e)

        if e % log_interval == 0 and e != 0:
            # writer.add_scalar("Loss", loss, global_step=e)
            # writer.add_scalar("Loss_test", loss_test, global_step=e)
            plt.plot(losses_train, label="train")
            plt.plot(losses_test, label="test")
            plt.legend()
            plt.show()
```

We can now start training!


```python
model = BeamNetV4().to(device)

# %% ##############################################################################################
################################## Train ##########################################################
train(
    model,
    epochs = 300,
    datarray_in=datarray_in,
    datarray_out=datarray_out,
    datatest_in=datatest_in,
    datatest_out=datatest_out,
    lr=0.0001,
    log_interval=290,
    exp_name="Bigger_dataset_BeamV4_prova",
)
```

     96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████     | 288/300 [00:03<00:00, 82.52it/s]


    
![png](train_files/train_9_1.png)
    


    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 76.03it/s]
    

We can now compare the model when trained on different sized datasets


```python
model1 = BeamNetV4().to(device)
model1 = torch.load("15k_Normalized_CATANA_data_BeamNetV4.py")

model2 = BeamNetV4().to(device)
model2 = torch.load("15k_Normalized_Bigger_data_BeamNetV4.py")

# Prepare data for plotting
x_train = datarray_out[:, 0].detach().cpu()
y_train = datarray_out[:, 1].detach().cpu()
sx_train = datarray_out[:, 2].detach().cpu()
sy_train = datarray_out[:, 3].detach().cpu()
x_test = datatest_out[:, 0].detach().cpu()
y_test = datatest_out[:, 1].detach().cpu()
sx_test = datatest_out[:, 2].detach().cpu()
sy_test = datatest_out[:, 3].detach().cpu()

# Prepare data for plotting
x_out_train = model1(datarray_in)[:, 0].detach().cpu()
y_out_train = model1(datarray_in)[:, 1].detach().cpu()
sx_out_train = model1(datarray_in)[:, 2].detach().cpu()
sy_out_train = model1(datarray_in)[:, 3].detach().cpu()

x_out_train_new = model2(datarray_in)[:, 0].detach().cpu()
y_out_train_new = model2(datarray_in)[:, 1].detach().cpu()
sx_out_train_new = model2(datarray_in)[:, 2].detach().cpu()
sy_out_train_new = model2(datarray_in)[:, 3].detach().cpu()


# Prepare data for plotting
x_out = model1(datatest_in)[:, 0].detach().cpu()
y_out = model1(datatest_in)[:, 1].detach().cpu()
sx_out = model1(datatest_in)[:, 2].detach().cpu()
sy_out = model1(datatest_in)[:, 3].detach().cpu()

x_out_new = model2(datatest_in)[:, 0].detach().cpu()
y_out_new = model2(datatest_in)[:, 1].detach().cpu()
sx_out_new = model2(datatest_in)[:, 2].detach().cpu()
sy_out_new = model2(datatest_in)[:, 3].detach().cpu()

##################################################################################################
# Plot Train set

plt.figure(figsize=(20, 40))
for i in range(len(x_train)):
    ax = plt.subplot(15, 5, i + 1)
    ax.errorbar(
        x_train[i],
        y_train[i],
        sx_train[i],
        sy_train[i],
        linestyle="None",
        label="Ground Truth",
    )
    ax.errorbar(
        x_out_train[i],
        y_out_train[i],
        sx_out_train[i],
        sy_out_train[i],
        linestyle="None",
        label="Model 1 Predict",
    )
    ax.errorbar(
        x_out_train_new[i],
        y_out_train_new[i],
        sx_out_train_new[i],
        sy_out_train_new[i],
        linestyle="None",
        label="Model 2 Predict",
    )
    #     ax.set_xlim([min(x_train)*0.8, max(x_train)*1.3])
    #     ax.set_ylim([min(y_train)*0.8, max(y_train)*1.3])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
plt.suptitle("Train Dataset")
plt.figlegend(["Ground Truth", "Trained on partial Dataset", "Trained on all Dataset"])
plt.show()


################################################################################
# Plot Test
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
for i in range(len(x_test)):
    ax = plt.subplot(5, 5, i + 1)
    ax.errorbar(
        x_test[i],
        y_test[i],
        sx_test[i],
        sy_test[i],
        linestyle="None",
        label="Ground Truth",
    )
    ax.errorbar(
        x_out[i],
        y_out[i],
        sx_out[i],
        sy_out[i],
        linestyle="None",
        label="Model 1 Predict",
    )
    ax.errorbar(
        x_out_new[i],
        y_out_new[i],
        sx_out_new[i],
        sy_out_new[i],
        linestyle="None",
        label="Model 2 Predict",
    )

    # Stessa normalizzazione di visualizzazione del train set
    #     ax.set_xlim([min(x_train)*0.8, max(x_train)*1.3])
    #     ax.set_ylim([min(y_train)*0.8, max(y_train)*1.3])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
plt.suptitle("Test Dataset")
plt.figlegend(["Ground Truth", "Trained on partial Dataset", "Trained on all Dataset"])
plt.show()
```


    
![png](train_files/train_11_0.png)
    



    
![png](train_files/train_11_1.png)
    


One can clearly see that the performance is generally much better increasing the dataset size. 
