# Selezionare una casella di input e usare le frecce su/giù  per vedere l'effetto di ogni parametro
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

data_norm = ((data - data.min()) / (data.max() - data.min())).fillna(0)

random_seed = 1234
data_shuffle = data_norm.sample(frac=1, random_state=random_seed)
test_len = int(len(data)*0.25)
datarray = np.array(data_shuffle[:-test_len])
datarray_test = np.array(data_shuffle[-test_len:])
datarray.shape, datarray_test.shape

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
    
# Model selection
model = BeamNetV4().to(device)
model = torch.load("15k_Normalized_Bigger_data_BeamNetV4.py")
model.eval()

matplotlib.use("TkAgg")

root = tk.Tk()
root.wm_title("Demo Beam Project")


def graph_output():
    xs = [float(x.get()) for x in entries.values()]
    xs = torch.Tensor(xs).to(device)
    #     print(xs)
    out = model(xs)[0].detach().cpu()
    xs = xs.detach().cpu()
    #     print(out)
    ax.cla()
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.errorbar(xs[0], xs[1], xs[3], xs[2], linestyle="None")
    ax.errorbar(out[0], out[1], out[3], out[2], linestyle="None")
    fig.canvas.draw_idle()


def change_input():
    #         print(idx.get())
    for i, column in enumerate(data_norm.columns[:-4]): 
        entries[column].delete(0, 999)
        entries[column].insert(
            tk.END, data_norm[column][int(idx.get())]
        )  # L'ultimo indice determina il dato che si prende come default
    graph_output()


def up_number(e):
    old_value = float(e.widget.get())
    e.widget.delete(0, 999)
    e.widget.insert(tk.END, str(round(old_value / 0.05) * 0.05 + 0.05))
    graph_output()


def down_number(e):
    old_value = float(e.widget.get())
    e.widget.delete(0, 999)
    e.widget.insert(tk.END, str(round(old_value / 0.05) * 0.05 - 0.05))
    graph_output()


canvas1 = tk.Canvas(root, width=400, height=300)
canvas1.pack()
fig = plt.Figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

button1 = tk.Button(root, text="Get the Graph", command=graph_output)
canvas1.create_window(200, 180, window=button1)

entries = dict()

for i, column in enumerate(data_norm.columns[:-4]):
    entries[column] = tk.Entry(root, width=10)
    entries[column].bind("<Up>", up_number)
    entries[column].bind("<Down>", down_number)
    entries[column].insert(
        tk.END, data_norm[column][0]
    )  # L'ultimo indice determina il dato che si prende come default
    label1 = tk.Label(root, text=column, anchor="w")
    canvas1.create_window(20, 20 + i * 20, window=entries[column])
    canvas1.create_window(100, 20 + i * 20, window=label1)


# Takes idx of data to import
idx = tk.Entry(root, width=10)
idx.insert(tk.END, 0)
canvas1.create_window(200, 20, window=idx)
idx_button = tk.Button(root, text="Import from Dataset", command=change_input)
canvas1.create_window(250, 20, window=idx_button)
change_input()

root.mainloop()