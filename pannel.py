"""
Coastline Prediction v7 - 基于v5，只修复局部上跳问题
"""
import argparse, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler

try:
    import fitz
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf' and PDF_SUPPORT:
        doc = fitz.open(path)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4: img = img[:,:,:3]
        doc.close()
        return img
    img = plt.imread(path)
    if img.dtype in (np.float32, np.float64): img = (img*255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4: img = img[:,:,:3]
    return img.astype(np.uint8)

def extract_coastline(img):
    r,g,b = img[:,:,0].astype(np.int16), img[:,:,1].astype(np.int16), img[:,:,2].astype(np.int16)
    water = (b > 90) & (b > g + 15) & (b > r + 15)
    water = ndimage.binary_fill_holes(ndimage.binary_closing(water, np.ones((5,5))))
    lbl, n = ndimage.label(water)
    if n > 1: water = lbl == (np.argmax(ndimage.sum(water, lbl, range(1,n+1))) + 1)
    boundary = ndimage.binary_dilation(water, np.ones((3,3))) ^ ndimage.binary_erosion(water, np.ones((3,3)))
    white = (r >= 220) & (g >= 220) & (b >= 220)
    coast = white & ndimage.binary_dilation(boundary, np.ones((7,7)))
    coast = ndimage.binary_closing(coast, np.ones((3,3)))
    lbl, n = ndimage.label(coast)
    if n > 1: coast = lbl == (np.argmax(ndimage.sum(coast, lbl, range(1,n+1))) + 1)
    return coast

def extract_transects_bottom(mask, num=200):
    """取下边缘(max y)，只修复局部上跳"""
    h, w = mask.shape
    x_positions = np.linspace(0, w-1, num, dtype=int)

    # 第一轮：取max y
    raw_y = []
    for x in x_positions:
        ys = np.where(mask[:, x])[0]
        if len(ys) > 0:
            raw_y.append(int(np.max(ys)))
        else:
            raw_y.append(-1)

    raw_y = np.array(raw_y, dtype=float)

    # 第二轮：只修复局部上跳（某点比左右邻居都小30+像素）
    fixed_y = raw_y.copy()
    for i in range(len(raw_y)):
        if raw_y[i] <= 0:
            continue

        # 取左右各3个有效邻居
        left_vals = [raw_y[j] for j in range(max(0,i-3), i) if raw_y[j] > 0]
        right_vals = [raw_y[j] for j in range(i+1, min(len(raw_y),i+4)) if raw_y[j] > 0]

        if left_vals and right_vals:
            left_med = np.median(left_vals)
            right_med = np.median(right_vals)
            # 如果当前点比左右邻居都小30像素以上（往上跳），用插值替代
            if raw_y[i] < left_med - 30 and raw_y[i] < right_med - 30:
                fixed_y[i] = (left_med + right_med) / 2

    transects = [(x, int(y)) for x, y in zip(x_positions, fixed_y) if y > 0]
    return transects

if TORCH_AVAILABLE:
    class SimpleLSTM(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, 2, batch_first=True, dropout=0.1)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

class LSTMPredictor:
    def __init__(self, seq_len=4, hidden=32, epochs=200):
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers = {}

    def fit(self, yearly_transects):
        self.years = sorted(yearly_transects.keys())
        all_x = set()
        for y in self.years:
            for x, _ in yearly_transects[y]: all_x.add(x)
        self.all_x = sorted(all_x)

        self.series = {}
        for x_pos in self.all_x:
            y_vals = []
            for year in self.years:
                for x, y in yearly_transects[year]:
                    if x == x_pos:
                        y_vals.append(y)
                        break
            if len(y_vals) == len(self.years):
                self.series[x_pos] = np.array(y_vals, dtype=np.float32)

        print(f"  {len(self.series)} transects with complete data")

        if not TORCH_AVAILABLE or len(self.series) < 10:
            self._fit_fallback()
            return

        print(f"  Training LSTM...")
        X_all, Y_all = [], []
        for x_pos, vals in self.series.items():
            scaler = MinMaxScaler()
            vals_norm = scaler.fit_transform(vals.reshape(-1,1)).flatten()
            self.scalers[x_pos] = scaler
            for i in range(self.seq_len, len(vals_norm)):
                X_all.append(vals_norm[i-self.seq_len:i])
                Y_all.append(vals_norm[i])

        X = torch.FloatTensor(np.array(X_all)).unsqueeze(-1)
        Y = torch.FloatTensor(np.array(Y_all)).unsqueeze(-1)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=32, shuffle=True)

        self.model = SimpleLSTM(self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.005)

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                loss = nn.MSELoss()(self.model(bx), by)
                loss.backward()
                opt.step()
        self.model.eval()
        self.use_lstm = True

    def _fit_fallback(self):
        self.use_lstm = False
        self.base, self.trend = {}, {}
        for x_pos, vals in self.series.items():
            self.base[x_pos] = np.mean(vals[-3:])
            self.trend[x_pos] = np.clip((vals[-1] - vals[-3]) / 2, -2, 2)

    def predict(self, num_years=5):
        predictions = {}
        for offset in range(1, num_years + 1):
            year_preds = []
            for x_pos in self.all_x:
                if x_pos not in self.series: continue
                if hasattr(self, 'use_lstm') and self.use_lstm:
                    vals = list(self.series[x_pos])
                    for prev in range(1, offset):
                        for px, py in predictions.get(prev, []):
                            if px == x_pos: vals.append(py); break
                    scaler = self.scalers[x_pos]
                    recent = np.array(vals[-self.seq_len:], dtype=np.float32)
                    recent_norm = scaler.transform(recent.reshape(-1,1)).flatten()
                    inp = torch.FloatTensor(recent_norm).unsqueeze(0).unsqueeze(-1).to(self.device)
                    with torch.no_grad():
                        pred_y = scaler.inverse_transform([[self.model(inp).cpu().numpy()[0,0]]])[0,0]
                else:
                    pred_y = self.base[x_pos] + self.trend[x_pos] * offset
                year_preds.append((x_pos, int(round(pred_y))))
            predictions[offset] = year_preds
        return predictions

def transects_to_mask(transects, h, w, thick=2):
    mask = np.zeros((h, w), dtype=bool)
    if len(transects) < 2: return mask
    pts = sorted(transects, key=lambda t: t[0])
    for i in range(len(pts)-1):
        x0,y0 = pts[i]; x1,y1 = pts[i+1]
        n = max(abs(x1-x0), abs(y1-y0)) + 1
        for x, y in zip(np.linspace(x0,x1,n).astype(int), np.linspace(y0,y1,n).astype(int)):
            if 0<=x<w and 0<=y<h:
                mask[max(0,y-thick):min(h,y+thick+1), max(0,x-thick):min(w,x+thick+1)] = True
    return mask

def metrics(gt, pred):
    gd, pd = {x:y for x,y in gt}, {x:y for x,y in pred}
    errs = [abs(gd[x]-pd[x]) for x in set(gd)&set(pd)]
    return (np.mean(errs), np.sqrt(np.mean(np.array(errs)**2)), len(errs)) if errs else (np.nan,np.nan,0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=r"E:\ground")
    parser.add_argument("--output_path", default=r"E:\prediction_results")
    parser.add_argument("--num_transects", type=int, default=200)
    args = parser.parse_args()

    print("="*50)
    print("Coastline Prediction v7 (Bottom + Local Fix + LSTM)")
    print("="*50)

    train_data, gt_data, yearly_tr = {}, {}, {}
    for fn in sorted(os.listdir(args.input_path)):
        if os.path.splitext(fn)[1].lower() not in ['.png','.jpg','.pdf']: continue
        try: year = int(''.join(filter(str.isdigit, fn.split('.')[0])))
        except: continue
        if not 2000 <= year <= 2030: continue

        img = read_image(os.path.join(args.input_path, fn))
        mask = extract_coastline(img)
        tr = extract_transects_bottom(mask, args.num_transects)

        if 2015 <= year <= 2020:
            train_data[year] = {'image':img, 'mask':mask, 'transects':tr}
            yearly_tr[year] = tr
            print(f"  Train: {year} ({len(tr)} pts)")
        elif 2021 <= year <= 2025:
            gt_data[year] = {'image':img, 'mask':mask, 'transects':tr}
            print(f"  GT:    {year} ({len(tr)} pts)")

    predictor = LSTMPredictor(seq_len=4, hidden=32, epochs=200)
    predictor.fit(yearly_tr)
    preds = predictor.predict(5)

    os.makedirs(args.output_path, exist_ok=True)

    print("\n" + "="*50)
    print(f"{'Year':<6} {'MAE':<10} {'RMSE':<10} {'N':<6}")
    print("-"*50)
    for i, year in enumerate(range(2021, 2026)):
        gt_tr = gt_data.get(year, {}).get('transects', [])
        pred_tr = preds.get(i+1, [])
        mae, rmse, n = metrics(gt_tr, pred_tr)
        print(f"{year:<6} {mae:<10.2f} {rmse:<10.2f} {n:<6}")

        if year in gt_data:
            d = gt_data[year]
            h, w = d['image'].shape[:2]
            overlay = d['image'].copy()
            gt_m = transects_to_mask(gt_tr, h, w)
            pred_m = transects_to_mask(pred_tr, h, w)
            overlay[gt_m & ~pred_m] = [0,255,0]
            overlay[pred_m & ~gt_m] = [255,0,0]
            overlay[gt_m & pred_m] = [255,255,0]

            fig, ax = plt.subplots(1,2,figsize=(16,6))
            ax[0].imshow(d['image']); ax[0].set_title(f'{year} Original'); ax[0].axis('off')
            ax[1].imshow(overlay); ax[1].set_title(f'{year} MAE={mae:.1f}px', fontweight='bold'); ax[1].axis('off')
            ax[1].legend(handles=[Patch(facecolor='g',label='GT'),Patch(facecolor='r',label='Pred'),Patch(facecolor='y',label='Overlap')], loc='upper right')
            plt.savefig(os.path.join(args.output_path, f'overlay_{year}.png'), dpi=150)
            plt.close()
    print("="*50)

if __name__ == "__main__":
    main()