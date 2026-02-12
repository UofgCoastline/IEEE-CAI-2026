"""
TERRA Project - Coastline Prediction System
Multi-page Streamlit Dashboard with UK Regional Navigation

Run: streamlit run terra_app.py
"""
import streamlit as st
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

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

# ==================== Page Config ====================
st.set_page_config(
    page_title="TERRA - Coastline Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Prediction Code ====================
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
    h, w = mask.shape
    x_positions = np.linspace(0, w-1, num, dtype=int)
    raw_y = []
    for x in x_positions:
        ys = np.where(mask[:, x])[0]
        raw_y.append(int(np.max(ys)) if len(ys) > 0 else -1)
    raw_y = np.array(raw_y, dtype=float)
    fixed_y = raw_y.copy()
    for i in range(len(raw_y)):
        if raw_y[i] <= 0: continue
        left_vals = [raw_y[j] for j in range(max(0,i-3), i) if raw_y[j] > 0]
        right_vals = [raw_y[j] for j in range(i+1, min(len(raw_y),i+4)) if raw_y[j] > 0]
        if left_vals and right_vals:
            left_med, right_med = np.median(left_vals), np.median(right_vals)
            if raw_y[i] < left_med - 30 and raw_y[i] < right_med - 30:
                fixed_y[i] = (left_med + right_med) / 2
    return [(x, int(y)) for x, y in zip(x_positions, fixed_y) if y > 0]

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
        self.seq_len, self.hidden, self.epochs = seq_len, hidden, epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers = {}

    def fit(self, yearly_transects, progress_callback=None):
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
                    if x == x_pos: y_vals.append(y); break
            if len(y_vals) == len(self.years):
                self.series[x_pos] = np.array(y_vals, dtype=np.float32)

        if not TORCH_AVAILABLE or len(self.series) < 10:
            self._fit_fallback()
            return

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
                nn.MSELoss()(self.model(bx), by).backward()
                opt.step()
            if progress_callback: progress_callback((epoch+1)/self.epochs)
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

def create_overlay(img, gt_tr, pred_tr):
    h, w = img.shape[:2]
    overlay = img.copy()
    gt_m = transects_to_mask(gt_tr, h, w)
    pred_m = transects_to_mask(pred_tr, h, w)
    overlay[gt_m & ~pred_m] = [0,255,0]
    overlay[pred_m & ~gt_m] = [255,0,0]
    overlay[gt_m & pred_m] = [255,255,0]
    return overlay

# ==================== UK Regions Data ====================
UK_REGIONS = {
    "Scotland": {
        "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
        "cities": ["St Andrews", "Edinburgh", "Aberdeen", "Dundee", "Inverness"]
    },
    "England": {
        "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "cities": ["London", "Liverpool", "Brighton", "Bristol", "Newcastle"]
    },
    "Wales": {
        "flag": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
        "cities": ["Cardiff", "Swansea", "Newport", "Bangor"]
    },
    "Northern Ireland": {
        "flag": "🇬🇧",
        "cities": ["Belfast", "Derry", "Lisburn", "Newry"]
    }
}

# ==================== Session State Init ====================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.metrics_data = {}
    st.session_state.overlays = {}

# ==================== Page Functions ====================

def show_home_page():
    """Main Home Page with UK Map and Regional Navigation"""

    # Data path (adjust as needed)
    data_path = r"E:\ground"

    # Header with 3 Logos: TERRA | EU (center) | Glasgow
    # Add padding columns on sides to bring logos closer together
    pad1, col1, col2, col3, pad2 = st.columns([1, 1, 1.2, 1, 1])

    # Uniform logo size
    LOGO_SIZE = 280

    # Left - TERRA Logo
    with col1:
        terra_path = os.path.join(data_path, "terra.png")
        if not os.path.exists(terra_path):
            for name in ["terra.PNG", "Terra.png", "TERRA.png"]:
                alt = os.path.join(data_path, name)
                if os.path.exists(alt): terra_path = alt; break
        if os.path.exists(terra_path):
            st.image(terra_path, width=LOGO_SIZE)
        else:
            st.markdown("### 🌊 TERRA")

    # Center - EU Logo
    with col2:
        eu_path = os.path.join(data_path, "EU.png")
        if not os.path.exists(eu_path):
            for name in ["eu.png", "Eu.png", "EU.PNG"]:
                alt = os.path.join(data_path, name)
                if os.path.exists(alt): eu_path = alt; break
        if os.path.exists(eu_path):
            st.image(eu_path, width=LOGO_SIZE)
        else:
            st.markdown("### 🇪🇺 EU Horizon")

    # Right - University of Glasgow Logo
    with col3:
        uofg_path = os.path.join(data_path, "Uofg.png")
        if not os.path.exists(uofg_path):
            for name in ["uofg.png", "UofG.png", "UOFG.png", "glasgow.png"]:
                alt = os.path.join(data_path, name)
                if os.path.exists(alt): uofg_path = alt; break
        if os.path.exists(uofg_path):
            st.image(uofg_path, width=LOGO_SIZE)
        else:
            st.markdown("### 🎓 UofG")

    # Title
    st.markdown("<h1 style='text-align: center; color: #1e3a5a;'>Coastal Monitoring System</h1>",
               unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Coastline Prediction & Analysis | EU Horizon TERRA Project</p>",
               unsafe_allow_html=True)

    st.divider()

    # UK Map in Center
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        map_path = os.path.join(data_path, "ukmap.jpg")
        if os.path.exists(map_path):
            st.image(map_path, caption="United Kingdom - Select a Region Below", use_container_width=True)
        else:
            # Try alternative names
            for name in ["ukmap.JPG", "UKmap.jpg", "uk_map.jpg"]:
                alt_path = os.path.join(data_path, name)
                if os.path.exists(alt_path):
                    st.image(alt_path, caption="United Kingdom - Select a Region Below", use_container_width=True)
                    break
            else:
                # Placeholder
                st.info("🗺️ UK Map\n\nPlace 'ukmap.jpg' in your data folder")

    st.divider()

    # Regional Selection - 4 Columns
    st.markdown("### 📍 Select a Region")

    region_cols = st.columns(4)

    for idx, (region, data) in enumerate(UK_REGIONS.items()):
        with region_cols[idx]:
            st.markdown(f"#### {data['flag']} {region}")

            # City buttons
            for city in data['cities']:
                # Only St Andrews is active
                if city == "St Andrews":
                    if st.button(f"📍 {city}", key=f"btn_{city}", use_container_width=True, type="primary"):
                        st.session_state.page = 'st_andrews'
                        st.rerun()
                else:
                    if st.button(f"🔒 {city}", key=f"btn_{city}", use_container_width=True, disabled=False):
                        st.toast(f"⚠️ {city} - Under Development", icon="🚧")

    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.caption("TERRA Project | EU Horizon Programme | © 2024-2026")


def show_st_andrews_page():
    """St Andrews Coastline Prediction Page"""

    # Back button
    if st.button("← Back to Home", type="secondary"):
        st.session_state.page = 'home'
        st.rerun()

    st.title("🏴󠁧󠁢󠁳󠁣󠁴󠁿 St Andrews Coastline Prediction")
    st.caption("LSTM-based Coastline Prediction System | Training Data: 2015-2020")

    # Sidebar config
    with st.sidebar:
        st.header("⚙️ Configuration")
        input_path = st.text_input("Data Path", r"E:\ground")
        num_transects = st.slider("Number of Transects", 100, 300, 200)

        st.divider()
        run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

        st.divider()
        st.info("**Model Parameters**\n- LSTM hidden: 32\n- Seq length: 4\n- Epochs: 200")

    # Run prediction
    if run_btn:
        if not os.path.exists(input_path):
            st.error(f"Path not found: {input_path}")
            return

        with st.spinner("Processing..."):
            progress = st.progress(0, "Loading data...")

            train_data, gt_data, yearly_tr = {}, {}, {}
            files = [f for f in sorted(os.listdir(input_path))
                    if os.path.splitext(f)[1].lower() in ['.png','.jpg','.pdf']]

            for idx, fn in enumerate(files):
                try: year = int(''.join(filter(str.isdigit, fn.split('.')[0])))
                except: continue
                if not 2000 <= year <= 2030: continue

                img = read_image(os.path.join(input_path, fn))
                mask = extract_coastline(img)
                tr = extract_transects_bottom(mask, num_transects)

                if 2015 <= year <= 2020:
                    train_data[year] = {'image':img, 'mask':mask, 'transects':tr}
                    yearly_tr[year] = tr
                elif 2021 <= year <= 2025:
                    gt_data[year] = {'image':img, 'mask':mask, 'transects':tr}

                progress.progress((idx+1)/len(files), f"Processing {fn}")

            # Train model
            progress.progress(0, "Training LSTM model...")
            predictor = LSTMPredictor(seq_len=4, hidden=32, epochs=200)
            predictor.fit(yearly_tr, lambda p: progress.progress(p, f"Training {int(p*100)}%"))
            preds = predictor.predict(5)

            # Calculate metrics and generate overlays
            metrics_data, overlays = {}, {}
            for i, year in enumerate(range(2021, 2026)):
                gt_tr = gt_data.get(year, {}).get('transects', [])
                pred_tr = preds.get(i+1, [])
                mae, rmse, n = metrics(gt_tr, pred_tr)
                metrics_data[year] = {"MAE": mae, "RMSE": rmse, "N": n}

                if year in gt_data:
                    overlays[year] = {
                        'image': gt_data[year]['image'],
                        'overlay': create_overlay(gt_data[year]['image'], gt_tr, pred_tr),
                        'gt_tr': gt_tr,
                        'pred_tr': pred_tr
                    }

            st.session_state.metrics_data = metrics_data
            st.session_state.overlays = overlays
            st.session_state.results = True
            progress.empty()

        st.success("✅ Prediction Complete!")

    # Display results
    if st.session_state.results:
        st.divider()

        # Year selection
        years = list(st.session_state.overlays.keys())
        if years:
            st.subheader("📅 Select Prediction Year")
            cols = st.columns(len(years))
            for i, year in enumerate(years):
                with cols[i]:
                    if st.button(f"🗓️ {year}", use_container_width=True,
                                type="primary" if st.session_state.get('selected_year') == year else "secondary"):
                        st.session_state.selected_year = year

            selected = st.session_state.get('selected_year', years[0])

            st.divider()

            # Display image and metrics
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(f"📍 Coastline Prediction - {selected}")

                tab1, tab2 = st.tabs(["Overlay Comparison", "Original Image"])
                with tab1:
                    st.image(st.session_state.overlays[selected]['overlay'],
                            caption="🟢 Ground Truth | 🔴 Prediction | 🟡 Overlap")
                with tab2:
                    st.image(st.session_state.overlays[selected]['image'])

            with col2:
                st.subheader("📊 Error Metrics")
                m = st.session_state.metrics_data[selected]
                st.metric("MAE", f"{m['MAE']:.2f} px", help="Mean Absolute Error")
                st.metric("RMSE", f"{m['RMSE']:.2f} px", help="Root Mean Square Error")
                st.metric("Sample Points", f"{m['N']}", help="Number of valid transects")

                st.divider()
                st.caption("**MAE Trend**")
                chart = {str(y): st.session_state.metrics_data[y]["MAE"]
                        for y in st.session_state.metrics_data}
                st.bar_chart(chart, height=150)

        # Summary table
        st.divider()
        st.subheader("📋 Summary Results")
        import pandas as pd
        df = pd.DataFrame(st.session_state.metrics_data).T
        df.index.name = "Year"
        st.dataframe(df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "N": "{:.0f}"}),
                    use_container_width=True)


def show_under_development():
    """Placeholder page for cities under development"""
    st.warning("🚧 This location is currently under development. Please check back later!")
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()


# ==================== Main Router ====================
def main():
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'st_andrews':
        show_st_andrews_page()
    else:
        show_under_development()


if __name__ == "__main__":
    main()