import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Land Cover Classifier",
    page_icon="🛰️",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 60%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; color: #e2e8f0; margin: 0 0 0.4rem 0; }
.hero p  { color: #64748b; font-size: 0.9rem; margin: 0; }

.badge {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 0.25rem 1rem;
    font-size: 0.78rem;
    color: #475569;
    margin-bottom: 1.5rem;
}

.result-card {
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
    border: 1px solid #22c55e;
    background: #052e16;
}
.result-label { font-size: 1.7rem; font-weight: 700; color: #22c55e; }
.result-conf  { font-size: 0.95rem; color: #94a3b8; margin-top: 0.2rem; }

.bar-row   { display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }
.bar-label { width: 160px; font-size: 0.8rem; color: #94a3b8; text-align: right; flex-shrink: 0; }
.bar-bg    { flex: 1; background: #1e293b; border-radius: 6px; height: 9px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 6px; background: #3b82f6; }
.bar-pct   { width: 45px; font-size: 0.8rem; color: #cbd5e1; flex-shrink: 0; }

.xai-label {
    text-align: center;
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-style: italic;
}

.footer {
    text-align: center;
    color: #334155;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

CLASS_EMOJIS = {
    'AnnualCrop': '🌾', 'Forest': '🌲', 'HerbaceousVegetation': '🌿',
    'Highway': '🛣️', 'Industrial': '🏭', 'Pasture': '🐄',
    'PermanentCrop': '🍇', 'Residential': '🏘️', 'River': '🌊', 'SeaLake': '🏞️'
}

CLASS_DESCRIPTIONS = {
    'AnnualCrop':            'Seasonal farmland — planted and harvested within one year',
    'Forest':                'Dense tree cover — natural or managed woodland',
    'HerbaceousVegetation':  'Low-lying plants and grasses without woody stems',
    'Highway':               'Major road infrastructure visible from satellite',
    'Industrial':            'Factories, warehouses, and manufacturing zones',
    'Pasture':               'Open grassland used for livestock grazing',
    'PermanentCrop':         'Long-term crops like orchards and vineyards',
    'Residential':           'Housing areas — urban or suburban settlements',
    'River':                 'Flowing water body — rivers and streams',
    'SeaLake':               'Standing or open water — sea, lake, or reservoir'
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=10
        )
        model.load_state_dict(
            torch.load('swin_transformer_eurosat.pth', map_location=device)
        )
        model.eval()
        model.to(device)
        return model, device, None
    except Exception as e:
        return None, None, str(e)

# ── Transform ─────────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

# ── Predict + GradCAM ─────────────────────────────────────────────────────────
def predict_with_gradcam(image_pil, model, device):
    transform = get_transform()
    tensor = transform(image_pil).unsqueeze(0).to(device)

    cam_extractor = GradCAM(model, target_layer='norm')
    model.eval()

    with torch.enable_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        pred   = int(np.argmax(probs))
        activation = cam_extractor(pred, output)

    # Build CAM heatmap
    cam_map = activation[0].squeeze(0).cpu()
    cam_map = F.interpolate(
        cam_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze()

    # Denorm original for display
    mean_t = torch.tensor(MEAN)[:, None, None]
    std_t  = torch.tensor(STD)[:, None, None]
    img_show = (tensor.squeeze().cpu() * std_t + mean_t).clamp(0, 1)
    img_pil_224 = to_pil_image(img_show)

    # Overlay
    overlay = overlay_mask(img_pil_224, to_pil_image(cam_map, mode='F'), alpha=0.5)

    cam_extractor.remove_hooks()

    return pred, probs, img_pil_224, cam_map.numpy(), overlay

# ── Heatmap figure ────────────────────────────────────────────────────────────
def make_heatmap_figure(cam_map):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cam_map, cmap='jet')
    ax.axis('off')
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0f172a', dpi=120)
    buf.seek(0)
    plt.close()
    return buf

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛰️ Satellite Land Cover Classifier</h1>
    <p>Upload a satellite image — get predicted land use class + Grad-CAM explanation</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="badge">Swin Transformer · EuroSAT · 98.89% Accuracy · 10 Classes · XAI: Grad-CAM</div>',
    unsafe_allow_html=True
)

# Load model
model, device, load_error = load_model()

if load_error:
    st.error(f"Could not load model: {load_error}")
    st.info("""
**To run this app:**
1. Make sure `swin_transformer_eurosat.pth` is in the same folder as this file
2. Install dependencies: `pip install streamlit timm torchcam torch torchvision`
3. Run: `streamlit run satellite_app.py`
    """)
    st.stop()

# Upload
uploaded = st.file_uploader(
    "Upload a satellite image (JPG or PNG):",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

st.markdown("**Or try an example class to understand what each land type looks like:**")
st.caption("AnnualCrop: flat cultivated fields · Forest: dense dark green · Highway: thin linear road · SeaLake: uniform blue · Residential: grid of rooftops")

if uploaded:
    image = Image.open(uploaded).convert('RGB')

    with st.spinner("Analyzing satellite image..."):
        pred_idx, probs, img_224, cam_map, overlay = predict_with_gradcam(
            image, model, device
        )

    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    emoji = CLASS_EMOJIS[pred_name]
    desc  = CLASS_DESCRIPTIONS[pred_name]

    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">{emoji} {pred_name}</div>
        <div class="result-conf">{confidence:.1%} confidence &nbsp;·&nbsp; {desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Three images side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_224, use_container_width=True)
        st.markdown('<div class="xai-label">Original Image (224×224)</div>',
                    unsafe_allow_html=True)
    with col2:
        heatmap_buf = make_heatmap_figure(cam_map)
        st.image(heatmap_buf, use_container_width=True)
        st.markdown('<div class="xai-label">Grad-CAM Heatmap (WHERE)</div>',
                    unsafe_allow_html=True)
    with col3:
        st.image(overlay, use_container_width=True)
        st.markdown('<div class="xai-label">Overlay (What the model focused on)</div>',
                    unsafe_allow_html=True)

    # Probability bars
    st.markdown("**Confidence across all 10 classes:**")
    sorted_classes = sorted(
        zip(CLASS_NAMES, probs), key=lambda x: -x[1]
    )
    bars_html = ""
    for cls, prob in sorted_classes:
        width = prob * 100
        bold  = "font-weight:600; color:#e2e8f0;" if cls == pred_name else ""
        color = "#22c55e" if cls == pred_name else "#3b82f6"
        em    = CLASS_EMOJIS[cls]
        bars_html += f"""
        <div class="bar-row">
            <div class="bar-label" style="{bold}">{em} {cls}</div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>
            </div>
            <div class="bar-pct">{prob:.1%}</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    # XAI explanation
    with st.expander("What does Grad-CAM show?"):
        st.markdown(f"""
**Grad-CAM (Gradient-weighted Class Activation Mapping)** highlights *which regions* of the image the Swin Transformer focused on when predicting **{pred_name}**.

- 🔴 **Red/hot areas** = regions with the highest influence on the prediction
- 🔵 **Blue/cold areas** = regions the model largely ignored

For **{pred_name}**, the model correctly focuses on the characteristic visual patterns:
{CLASS_DESCRIPTIONS[pred_name]}.

This is your XAI component — it proves the model is making decisions based on *real* land cover features, not random noise or artifacts.
        """)

else:
    st.info("👆 Upload a satellite image above to classify it.")
    st.markdown("**Expected input:** 64×64 to 224×224 pixel satellite image from EuroSAT-style Sentinel-2 data. The app automatically resizes to 224×224.")

# Footer
st.markdown("""
<div class="footer">
    BSE-634 Final Year Project · Swin Transformer · EuroSAT · 
    Grad-CAM XAI · PyTorch + timm · 98.89% Test Accuracy
</div>
""", unsafe_allow_html=True)
