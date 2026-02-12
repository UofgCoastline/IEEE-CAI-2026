import os
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

import torch
import torch.nn as nn

# ================= 配置 =================
DATA_DIR = r"E:\ground"
TRAIN_YEARS = [2017, 2018, 2019, 2020]
TEST_YEARS  = [2021, 2022, 2023, 2024, 2025]
NUM_TRANSECTS = 50
EPOCHS = 300

# ================= 读 PDF =================
def read_pdf(path):
    doc = fitz.open(path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    doc.close()
    return img

# ================= 白线提取 =================
def white_mask(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return (r > 220) & (g > 220) & (b > 220)

def extract_transects(mask, n):
    h, w = mask.shape
    xs = np.linspace(0, w-1, n, dtype=int)
    ys = []
    for x in xs:
        col = np.where(mask[:, x])[0]
        ys.append(np.median(col) if len(col) else np.nan)
    return xs, np.array(ys)

# ================= LSTM =================
class SimpleLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# ================= 训练 =================
def train_lstm(transect_series):
    X, Y = [], []
    for ys in transect_series.values():
        if len(ys) == 4:
            X.append(ys[:3])
            Y.append(ys[3])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    model = SimpleLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model

# ================= 预测 =================
def predict_future(model, init_seq, years):
    seq = init_seq.copy()
    preds = []
    for _ in range(years):
        x = torch.tensor(seq[-3:], dtype=torch.float32).view(1,3,1)
        y = model(x).item()
        preds.append(y)
        seq.append(y)
    return preds

# ================= 主流程 =================
def main():
    yearly_transects = {}
    images = {}

    # ---- 读所有年份 ----
    for y in TRAIN_YEARS + TEST_YEARS:
        img = read_pdf(os.path.join(DATA_DIR, f"ground_{y}.pdf"))
        mask = white_mask(img)
        xs, ys = extract_transects(mask, NUM_TRANSECTS)
        yearly_transects[y] = ys
        images[y] = img

    # ---- 构建训练数据（按 transect）----
    transect_series = {}
    for i in range(NUM_TRANSECTS):
        series = []
        for y in TRAIN_YEARS:
            series.append(yearly_transects[y][i])
        if not any(np.isnan(series)):
            transect_series[i] = series

    print(f"Training on {len(transect_series)} transects")

    # ---- 训练 LSTM ----
    model = train_lstm(transect_series)

    # ---- 预测并可视化 ----
    for idx, year in enumerate(TEST_YEARS):
        img = images[year].copy()
        gt_y = yearly_transects[year]

        pred_y = []
        for i in range(NUM_TRANSECTS):
            if i not in transect_series:
                pred_y.append(np.nan)
                continue
            init = transect_series[i][-3:]
            future = predict_future(model, init, idx+1)
            pred_y.append(future[-1])

        # ---- 画图 ----
        vis = img.copy()

        # GT（绿）
        for x, y in zip(xs, gt_y):
            if not np.isnan(y):
                vis[int(y)-1:int(y)+1, x-1:x+1] = [0,255,0]

        # Pred（红）
        for x, y in zip(xs, pred_y):
            if not np.isnan(y):
                vis[int(y)-1:int(y)+1, x-1:x+1] = [255,0,0]

        plt.figure(figsize=(10,6))
        plt.imshow(vis)
        plt.title(f"LSTM Prediction vs GT – {year}\nTrain: 2017–2020")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# ================= 入口 =================
if __name__ == "__main__":
    main()
