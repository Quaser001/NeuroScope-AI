import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import google.generativeai as genai
from fpdf import FPDF
from monai.bundle import download, ConfigParser
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    Resized, ToTensord, ConcatItemsd
)

# --- 1. APP CONFIGURATION & MEDICAL CSS ---
st.set_page_config(
    page_title="NeuroScope AI | Clinical Tumor Board",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {font-family: 'Helvetica', sans-serif; color: #E0E0E0;}
    .report-header {color: #FF4B4B; font-weight: bold;}
    .warning-box {
        border-left: 5px solid #ffcc00; 
        background-color: #262730; 
        padding: 15px; 
        border-radius: 5px; 
        color: #e0e0e0;
        font-size: 0.9em;
    }
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold; 
        height: 3em;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST VISION ENGINE (CPU-OPTIMIZED) ---
@st.cache_resource
def load_vision_model():
    # Force CPU for stability on Hugging Face Free Tier
    device = torch.device("cpu")
    bundle_name = "brats_mri_segmentation"
    bundle_dir = "./model_zoo"
    
    # Auto-Download Weights
    if not os.path.exists(bundle_dir):
        try:
            download(name=bundle_name, bundle_dir=bundle_dir, version="0.3.5")
        except:
            st.warning("⚠️ Automatic model download failed. Please check network connectivity.")
            
    model_path = os.path.join(bundle_dir, "brats_mri_segmentation", "models", "model.pt")
    config_path = os.path.join(bundle_dir, "brats_mri_segmentation", "configs", "inference.json")
    
    try:
        parser = ConfigParser()
        parser.read_config(config_path)
        model = parser.get_parsed_content("network").to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Critical Model Load Error: {e}")
        return None, None

def process_multimodal(files_dict, model, device):
    """
    Processes 4 MRI sequences (FLAIR, T1, T1ce, T2) into a 3D segmentation map.
    Returns: input_tensor, mask, core_vol, whole_vol, location_str
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = {}
        for key, file_obj in files_dict.items():
            path = os.path.join(temp_dir, f"{key}.nii")
            with open(path, "wb") as f:
                f.write(file_obj.read())
            paths[key] = path
            
        transforms = Compose([
            LoadImaged(keys=["flair", "t1ce", "t1", "t2"]),
            EnsureChannelFirstd(keys=["flair", "t1ce", "t1", "t2"]),
            ScaleIntensityd(keys=["flair", "t1ce", "t1", "t2"]),
            Resized(keys=["flair", "t1ce", "t1", "t2"], spatial_size=(240, 240, 160)),
            ConcatItemsd(keys=["flair", "t1ce", "t1", "t2"], name="image", dim=0),
            ToTensord(keys=["image"])
        ])
        
        try:
            data = transforms(paths)
            input_tensor = data["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                mask = (torch.sigmoid(output) > 0.5).float()
            
            # --- VOLUMETRICS (FIXED CHANNEL MAPPING) ---
            # Channel 0: Tumor Core (TC) - Active Tumor + Necrosis
            # Channel 1: Whole Tumor (WT) - Core + Edema
            vol_core = torch.sum(mask[0, 0]).item() / 1000.0
            vol_whole = torch.sum(mask[0, 1]).item() / 1000.0
            
            # --- LOCATION HEURISTIC ---
            # Uses Core (Channel 0) center of mass
            coords = torch.nonzero(mask[0, 0, :, :, :])
            if coords.shape[0] > 0:
                mean_x = torch.mean(coords[:,0].float()).item()
                loc = "Left Hemisphere" if mean_x > 120 else "Right Hemisphere"
            else:
                loc = "Undetected / Small Lesion"
            
            return input_tensor, mask, vol_core, vol_whole, loc
            
        except Exception as e:
            st.error(f"Processing Pipeline Error: {e}")
            return None, None, 0, 0, "Error"

# --- 3. MEDICAL AGENT ENGINE (SMART-SELECT + PROTOCOLS) ---
class MedicalBoardAgents:
    def __init__(self, api_key, vol, loc, name, age, gender, history):
        genai.configure(api_key=api_key)
        
        # --- SMART MODEL SELECTOR (PREVENTS 404 CRASHES) ---
        self.model = None
        working_model = None
        try:
            # 1. Try to find Flash (Fast/New)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name:
                        working_model = m.name
                        break
                    elif 'pro' in m.name and not working_model:
                        working_model = m.name
            
            # 2. Fallback
            if not working_model: 
                working_model = "models/gemini-1.5-flash"
            
            self.model = genai.GenerativeModel(working_model)
        except:
            # 3. Hard Fallback
            self.model = genai.GenerativeModel("gemini-1.5-flash")

        # Clinical Context Vector
        self.context = {
            "mri_metrics": f"Active Core: {vol:.2f} cm3, Location: {loc}",
            "patient_demographics": f"{name}, {age}yo {gender}",
            "clinical_history": history,
            "setting": "Tertiary Care Tumor Board (India)"
        }

    def _consult(self, role, guidelines, task):
        full_prompt = f"""
        **SYSTEM ROLE**: {role}
        **STRICT PROTOCOL**: Follow {guidelines}.
        **PATIENT DATA**: {self.context}
        **TASK**: {task}
        **COMPLIANCE**: Use professional medical terminology. Cite specific guidelines (NCCN/WHO/Stupp) where applicable. Include standard disclaimer.
        """
        try:
            return self.model.generate_content(full_prompt).text
        except Exception as e:
            return f"⚠️ Agent Consultation Unavailable: {e}"

    def run_board(self):
        # 1. Neurosurgeon (NCCN)
        surg_rep = self._consult(
            role="Senior Consultant Neurosurgeon (M.Ch)",
            guidelines="NCCN Guidelines v2.2024 for CNS Cancers",
            task="""
            1. **Urgency Score**: Assess herniation risk. (Vol > 30cm3 is critical).
            2. **Surgical Strategy**: Flowchart for Biopsy vs. GTR (Gross Total Resection).
            3. **Intraoperative**: Recommend tools (5-ALA Fluorescence, Neuronavigation, CUSA).
            """
        )
        
        # 2. Oncologist (Stupp)
        res_rep = self._consult(
            role="Molecular Oncologist",
            guidelines="Stupp Protocol (2005) & WHO CNS Classification 2021",
            task="""
            1. **Standard of Care**: Detail Concurrent Chemo-Radiation (TMZ 75mg/m2 + RT 60Gy).
            2. **Molecular Prognosis**: Explain significance of IDH-Mutation & MGMT Promoter Methylation.
            3. **Adjuvant Phase**: Detail Maintenance TMZ cycles (150-200mg/m2).
            """
        )
        
        # 3. Metabolic Specialist (Warburg)
        diet_rep = self._consult(
            role="Metabolic Psychiatry Specialist",
            guidelines="Ketogenic Metabolic Therapy (Warburg Effect)",
            task="""
            1. **Mechanism**: Explain glucose dependency of GBM.
            2. **Prescription**: Indian Ketogenic Diet (Ghee, Paneer, No Rice/Roti).
            3. **Monitoring**: Mandatory warning for Liver Function Tests (LFT) & Lipid Panel.
            """
        )
        
        # 4. Patient Advocate (Empathy)
        patient_rep = self._consult(
            role="Compassionate Medical Guide",
            guidelines="Patient-Centered Care & Health Literacy",
            task="""
            Translate the medical findings into a reassuring guide for the family.
            1. **The Diagnosis**: Explain tumor size/location simply.
            2. **The Plan**: Simplify the Surgery -> Chemo -> Diet roadmap.
            3. **Actionable Hope**: 3 things to focus on today (Sleep, Stress, Nutrition).
            """
        )
        
        return surg_rep, res_rep, diet_rep, patient_rep

# --- 4. PDF REPORT ENGINE ---
def create_pdf(name, age, gender, history, surg, onc, diet, patient):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "NeuroScope AI - Clinical Tumor Board Report", ln=True, align='C')
    pdf.ln(5)
    
    # Patient Info Box
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient: {name} | {age}yo {gender}", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, f"Clinical History: {history}")
    pdf.ln(5)
    
    # Sections
    def add_section(title, body):
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, title, ln=True, fill=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=10)
        # Clean special chars for PDF
        clean_text = body.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 5, clean_text)
        pdf.ln(5)
        
    add_section("1. Surgical Triage (NCCN Guidelines)", surg[:2000])
    add_section("2. Oncology Protocol (Stupp)", onc[:2000])
    add_section("3. Metabolic Intervention", diet[:2000])
    add_section("4. Patient Guide", patient[:2000])
    
    return pdf.output(dest='S').encode('latin-1')

# --- 5. MAIN UI LAYOUT ---

# Session State for API Key Security
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = None

st.title("🧠 NeuroScope AI")
st.markdown("### Clinical Grade Tumor Board & Multi-Modal Segmentation")

# SIDEBAR: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Secure API Key Handling
    if not st.session_state.gemini_key:
        user_key = st.text_input("Gemini API Key", type="password", placeholder="Paste Key & Press Enter")
        if user_key:
            st.session_state.gemini_key = user_key
            st.rerun()
    else:
        st.success("✅ API Key Active")
        if st.button("🔄 Reset API Key"):
            st.session_state.gemini_key = None
            st.rerun()
            
    st.markdown("---")
    st.write("📂 **Upload Multi-Modal Scans:**")
    st.caption("Select all 4 NIfTI files (FLAIR, T1, T1ce, T2).")
    
    # Smart File Sorter
    uploaded_files = st.file_uploader("Drag & Drop Files Here", type=['nii', 'nii.gz'], accept_multiple_files=True)
    
    f_flair, f_t1ce, f_t1, f_t2 = None, None, None, None
    if uploaded_files:
        for f in uploaded_files:
            n = f.name.lower()
            if "flair" in n: f_flair = f
            elif "t1ce" in n: f_t1ce = f
            elif "t1" in n: f_t1 = f
            elif "t2" in n: f_t2 = f
        
        # Visual Validation
        if f_flair: st.success("✅ FLAIR (Edema) Loaded")
        if f_t1ce: st.success("✅ T1-Ce (Active Core) Loaded") 
        if f_t1: st.success("✅ T1 (Anatomy) Loaded")
        if f_t2: st.success("✅ T2 (Texture) Loaded")

# MAIN PAGE: Patient Demographics
st.subheader("📋 Patient Demographics")
c1, c2, c3 = st.columns(3)
p_name = c1.text_input("Full Name", "Patient-001")
p_age = c2.number_input("Age", min_value=1, max_value=120, value=45)
p_gender = c3.selectbox("Gender", ["Male", "Female", "Other"])
p_hist = st.text_area("Clinical History / Symptoms", "Presentation: Focal seizures (right upper limb), persistent headaches (3 months). No prior history of craniotomy.")

# EXECUTION LOGIC
if st.button("🚀 Initiate Clinical Analysis", type="primary"):
    # Validation
    if not st.session_state.gemini_key:
        st.error("⚠️ Protocol Halted: Please provide a Gemini API Key.")
    elif not (f_flair and f_t1ce and f_t1 and f_t2):
        st.error("⚠️ Protocol Halted: Incomplete MRI Sequence. Please upload FLAIR, T1, T1ce, and T2.")
    else:
        # A. Vision Phase
        with st.spinner("🔄 Processing 4-Channel Volumetrics..."):
            model, device = load_vision_model()
            img, mask, v_core, v_whole, loc = process_multimodal(
                {"flair": f_flair, "t1ce": f_t1ce, "t1": f_t1, "t2": f_t2}, 
                model, device
            )
        
        if img is not None:
            # B. Metrics Dashboard
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tumor Location", loc)
            m2.metric("Active Core (Surgery Target)", f"{v_core:.2f} cm³")
            m3.metric("Total Spread (Radiation Target)", f"{v_whole:.2f} cm³")
            
            # C. Dual-View Visualization
            st.subheader("👁️ Radiological Assessment")
            col1, col2 = st.columns(2)
            slice_idx = 100
            
            # Extract Slices
            t1ce_bg = img[0, 1, :, :, slice_idx].cpu().numpy()
            flair_bg = img[0, 0, :, :, slice_idx].cpu().numpy()
            mask_core = mask[0, 0, :, :, slice_idx].cpu().numpy()
            mask_whole = mask[0, 1, :, :, slice_idx].cpu().numpy()
            
            with col1:
                st.caption("🟥 SURGEON'S VIEW (T1-Ce + Active Core)")
                fig, ax = plt.subplots()
                ax.imshow(t1ce_bg, cmap="gray")
                ax.imshow(mask_core, cmap="Reds", alpha=0.6)
                ax.axis("off")
                st.pyplot(fig)
            with col2:
                st.caption("🟩 ONCOLOGIST'S VIEW (FLAIR + Edema)")
                fig, ax = plt.subplots()
                ax.imshow(flair_bg, cmap="gray")
                ax.imshow(mask_whole, cmap="Greens", alpha=0.5)
                ax.axis("off")
                st.pyplot(fig)

            # D. Expert Agent Phase
            st.markdown("---")
            st.subheader("🏥 Multi-Disciplinary Tumor Board Consensus")
            
            with st.spinner("⚡ Convening Specialists (Gemini 1.5)..."):
                board = MedicalBoardAgents(st.session_state.gemini_key, v_core, loc, p_name, p_age, p_gender, p_hist)
                surg, onc, diet, patient_guide = board.run_board()
                
            # E. Report Display
            tab1, tab2, tab3, tab4 = st.tabs(["🧠 Surgical Triage", "💊 Stupp Protocol", "🥑 Metabolic Rehab", "💙 Patient Guide"])
            with tab1: st.markdown(surg)
            with tab2: st.markdown(onc)
            with tab3: st.markdown(diet)
            with tab4: 
                st.info("ℹ️ This section is simplified for the patient & family.")
                st.markdown(patient_guide)
            
            # F. PDF Generation
            st.markdown("---")
            pdf_bytes = create_pdf(p_name, p_age, p_gender, p_hist, surg, onc, diet, patient_guide)
            st.download_button(
                label="📄 Download Official Tumor Board Report (PDF)",
                data=pdf_bytes,
                file_name=f"NeuroScope_Report_{p_name.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

            # G. Legal Guardrail
            st.markdown("""
            <div class='warning-box'>
            <b>LEGAL DISCLAIMER: INVESTIGATIONAL DEVICE.</b><br>
            This software utilizes Artificial Intelligence (Gemini 1.5 + Monai Swin-UNETR) for clinical decision support. 
            It is NOT a replacement for professional medical judgment. All outputs must be verified by a board-certified Neuro-Oncologist.
            </div>
            """, unsafe_allow_html=True)