import os, glob, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt
import pywt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, resnet18

FS = 1000

def load(f):
    try: x = np.loadtxt(f, delimiter=',')
    except: x = np.loadtxt(f, delimiter=',', skiprows=1)
    return x if x.shape[0] >= x.shape[1] else x.T

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
    return (w_data - mn) / np.maximum((mx - mn), eps)

SCALES = np.arange(1, 33)
def to_cwt(one_window, wavelet='morl'):
    maps = []
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        maps.append(np.abs(coef))
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)

print("데이터 처리 중... ")
files = sorted(glob.glob('data/**/*.csv', recursive=True))
unique_users = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in files)))
label_map = {u: i for i, u in enumerate(unique_users)}

def label_of(f): return label_map[os.path.basename(os.path.dirname(f))]
labels = [label_of(f) for f in files]

def build(flist):
    X, y = [], []
    for f in flist:
        sig = preprocess(load(f))
        for w in minmax(make_windows(sig)):
            X.append(to_cwt(w))
            y.append(label_of(f))
    return np.stack(X), np.array(y)

tr_files, te_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)
Xtr, ytr = build(tr_files)
Xte, yte = build(te_files)

BATCH_SIZE = 4 
EPOCHS = 3

train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_2dcnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 5)
    )

def get_resnet():
    m = resnet18(weights=None); m.fc = nn.Linear(512, 5)
    return m

def get_densenet():
    m = densenet161(weights=None); m.classifier = nn.Linear(2208, 5)
    return m

def train_and_evaluate(model, name, tr_loader, te_loader):
    print(f"\n[{name}] 모델 학습 및 평가 시작...")
    model = model.to(dev)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    t0 = time.time()
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0
    
    model.eval()
    xb = next(iter(te_loader))[0][:1].to(dev) 
    with torch.no_grad():
        for _ in range(10): model(xb)
        if dev == 'cuda': torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(100): model(xb)
        if dev == 'cuda': torch.cuda.synchronize()
        infer_time = (time.time() - t1) * 10
    
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            out = model(xb.to(dev))
            preds += out.argmax(1).cpu().tolist()
            trues += yb.tolist()
            
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    
    print(f"-> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print(f"-> 파라미터 수: {params:,}개 | 총 학습: {train_time:.1f}초 | 1건당 추론: {infer_time:.2f}ms")
    
    return model, acc, f1, trues, preds


results = {}
models_dict = {'2D CNN': get_2dcnn, 'ResNet18': get_resnet, 'DenseNet161': get_densenet}

for name, get_model_fn in models_dict.items():
    trained_model, acc, f1, tr, pr = train_and_evaluate(get_model_fn(), name, train_loader, test_loader)
    results[name] = {'trues': tr, 'preds': pr}


print("\n[각 모델별 혼동행렬(Confusion Matrix) 이미지 생성 및 분석]")
L = ['A', 'B', 'C', 'D', 'E']

for model_name, res in results.items():
    tr, pr = res['trues'], res['preds']
    
    print(f"\n[{model_name} Classification Report]")
    print(classification_report(tr, pr, digits=3))
    
    cm = confusion_matrix(tr, pr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(L); ax.set_yticklabels(L)
    
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center')
            
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.colorbar(im); plt.tight_layout()
    

    filename = f"confusion_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=120)
    plt.close(fig) 
    print(f"-> '{filename}' 이미지 저장 완료")
    

    np.fill_diagonal(cm, 0)
    max_err_idx = np.unravel_index(cm.argmax(), cm.shape)
    print(f"-> {model_name} 최대 오류 쌍: 실제 '{L[max_err_idx[0]]}' 클래스를 '{L[max_err_idx[1]]}'(으)로 가장 많이 오분류함.")


print("\n[5-Fold 교차검증 진행]")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(files, labels)):
    Xtr_cv, ytr_cv = build([files[i] for i in tr_idx])
    Xva_cv, yva_cv = build([files[i] for i in va_idx])
    
    cv_tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr_cv), torch.tensor(ytr_cv, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    cv_va_loader = DataLoader(TensorDataset(torch.tensor(Xva_cv), torch.tensor(yva_cv, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
    
    m_cv = get_2dcnn().to(dev) 
    opt_cv = torch.optim.Adam(m_cv.parameters(), lr=1e-3)
    crit_cv = nn.CrossEntropyLoss()
    
    m_cv.train()
    for epoch in range(EPOCHS):
        for xb, yb in cv_tr_loader:
            opt_cv.zero_grad(); crit_cv(m_cv(xb.to(dev)), yb.to(dev)).backward(); opt_cv.step()
            
    m_cv.eval()
    pr_cv, tr_cv = [], []
    with torch.no_grad():
        for xb, yb in cv_va_loader:
            pr_cv += m_cv(xb.to(dev)).argmax(1).cpu().tolist()
            tr_cv += yb.tolist()
            
    acc_cv = accuracy_score(tr_cv, pr_cv)
    cv_scores.append(acc_cv)
    print(f"-> Fold {fold+1} 정확도: {acc_cv:.4f}")

print(f"\n최종 5-Fold 평균 정확도: {np.mean(cv_scores)*100:.2f} ± {np.std(cv_scores)*100:.2f}%")