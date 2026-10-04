import numpy as np
import glob
import os
from scipy.signal import butter, iirnotch, filtfilt
import pywt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

FS = 1000

def load(f):
    try: x = np.loadtxt(f, delimiter=',')
    except: x = np.loadtxt(f, delimiter=',', skiprows=1)
    if x.shape[0] < x.shape[1]: x = x.T
    return x

def preprocess(x_data, fs=FS):
    bn, an = iirnotch(60, 30, fs)
    xf = filtfilt(bn, an, x_data, axis=0)
    b, a = butter(4, [20/(fs/2), 499/(fs/2)], btype='band')
    return filtfilt(b, a, xf, axis=0)

def make_windows(x_sig, win=300, hop=150):
    n = (len(x_sig) - win) // hop + 1
    return np.stack([x_sig[i*hop : i*hop+win] for i in range(n)])

def minmax(w_data, eps=1e-8):
    mn = w_data.min(axis=(1, 2), keepdims=True)
    mx = w_data.max(axis=(1, 2), keepdims=True)
    denom = np.maximum((mx - mn), eps)
    return (w_data - mn) / denom

SCALES = np.arange(1, 33)
def to_cwt(one_window, wavelet='morl'):
    maps = []
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        maps.append(np.abs(coef))
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)

files = sorted(glob.glob('data/**/*.csv', recursive=True))
unique_users = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in files)))
label_map = {u: i for i, u in enumerate(unique_users)}

def label_of(f): return label_map[os.path.basename(os.path.dirname(f))]
labels = [label_of(f) for f in files]

tr_files, te_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)

def build(flist):
    X, y = [], []
    for f in flist:
        sig = preprocess(load(f))
        for w in minmax(make_windows(sig)):
            X.append(to_cwt(w))
            y.append(label_of(f))
    return np.stack(X), np.array(y)

Xtr, ytr = build(tr_files)
Xte, yte = build(te_files)

print("\n[PyTorch 모델 학습 준비]")
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 장치(Device): {dev}")

train_dataset = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = densenet161(weights=None)
model.classifier = nn.Linear(2208, len(unique_users)) # 실제 피험자 수에 맞게 자동 설정
model = model.to(dev)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

epochs = 45
print(f"총 {epochs} Epoch 학습을 시작합니다.")

for epoch in range(epochs):
    model.train() 
    tot = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(dev), yb.to(dev)
        loss = crit(model(xb), yb)
        
        opt.zero_grad() 
        loss.backward() 
        opt.step()
        
        tot += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {tot/len(train_loader):.4f}")

print("모델 학습이 완료되었습니다.")