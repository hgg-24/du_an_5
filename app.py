import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from stmol import showmol
import py3Dmol
from PIL import Image
import io

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="ED-ODYSSEY | Chem-Lab", layout="wide")

def inject_custom_css():
    st.markdown(r"""
    <style>
        :root {
            --primary-color: #007AFF;
        }
        .stApp {
            background-color: #F4F7FA;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid #E5E5EA !important;
            padding: 1.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# 2. HÀM XỬ LÝ HÓA TIN HỌC (DATA ENGINES)
# ==========================================
@st.cache_data
def get_smiles_from_formula(formula):
    """Sử dụng PubChem API để lấy SMILES từ công thức phân tử."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{formula}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Bổ sung dòng kiểm tra an toàn: Chỉ lấy dữ liệu nếu tồn tại key 'PropertyTable'
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                
        # Nếu không có data hoặc lỗi, trả về None để hệ thống tự hiện cảnh báo vàng
        return None
        
    except Exception:
        return None

def draw_bohr_model(atomic_number):
    """Vẽ mô hình Bohr 2D bằng Plotly."""
    # Quy tắc phân bố electron cơ bản (Klechkowski giản lược)
    shells = [2, 8, 18, 32]
    electrons_left = atomic_number
    config = []
    
    for capacity in shells:
        if electrons_left <= 0: break
        if electrons_left >= capacity:
            config.append(capacity)
            electrons_left -= capacity
        else:
            config.append(electrons_left)
            electrons_left = 0

    fig = go.Figure()
    
    # Vẽ hạt nhân
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', 
                             marker=dict(size=40, color='#FF3B30'),
                             text=[f"+{atomic_number}"], textfont=dict(color='white', size=14),
                             name="Hạt nhân"))
    
    # Vẽ các lớp vỏ và electron
    for i, num_e in enumerate(config):
        radius = i + 1
        # Vẽ vòng tròn quỹ đạo
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(x=radius*np.cos(theta), y=radius*np.sin(theta), 
                                 mode='lines', line=dict(color='gray', dash='dash'), hoverinfo='skip', showlegend=False))
        
        # Phân bố electron trên quỹ đạo
        angles = np.linspace(0, 2*np.pi, num_e, endpoint=False)
        e_x = radius * np.cos(angles)
        e_y = radius * np.sin(angles)
        fig.add_trace(go.Scatter(x=e_x, y=e_y, mode='markers', 
                                 marker=dict(size=12, color='#007AFF'), name=f"Lớp {i+1} ({num_e}e)"))

    fig.update_layout(width=400, height=400, template="plotly_white", 
                      xaxis=dict(visible=False, range=[-5, 5]), yaxis=dict(visible=False, range=[-5, 5]),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig, config

# ==========================================
# 3. GIAO DIỆN CHÍNH (MAIN APP)
# ==========================================
st.title("🧪 Chem-Lab: Molecular & Thermo Engine")
st.markdown("Hệ thống trực quan hóa cấu trúc nguyên tử và phân tử đa chiều của ED-ODYSSEY.")

# Nút chọn chế độ thông minh
search_mode = st.radio(
    "Lựa chọn chế độ phân tích:",
    ["⚛️ Nguyên tố (Bohr & Cấu hình e)", "🔗 Phân tử (Cấu trúc 2D & 3D)"],
    horizontal=True
)

st.markdown("---")

# Tạo 3 Tabs để tối ưu không gian hiển thị
tab1, tab2, tab3 = st.tabs(["⚛️ Cấu tạo nguyên tử", "🔗 Cấu trúc 2D (Lewis)", "🌐 Mô hình 3D (WebGL)"])

# Khu vực giữ chỗ (Placeholders)
bohr_placeholder = tab1.empty()
lewis_placeholder = tab2.empty()
model3d_placeholder = tab3.empty()
thermo_placeholder = st.empty()

# ---------------------------------------------------------
# CHẾ ĐỘ 1: XỬ LÝ NGUYÊN TỐ (DÙNG SELECTBOX)
# ---------------------------------------------------------
if "Nguyên tố" in search_mode:
    # Từ điển 30 nguyên tố đầu tiên (cậu có thể tự bổ sung thêm đến 118)
    # Từ điển toàn bộ 118 nguyên tố hóa học trong Bảng tuần hoàn
    elements_db = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
        "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
        "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
        "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
        "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
        "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
        "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
        "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
        "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
        "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98,
        "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
        "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
        "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
    }
    
    selected_element = st.selectbox(
        "🔍 Chọn Nguyên tố hóa học:", 
        list(elements_db.keys())
    )
    
    atomic_num = elements_db[selected_element]
    
    with bohr_placeholder.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            # Hiển thị số hiệu nguyên tử (Z) ở góc dưới (Subscript) bằng LaTeX
            st.markdown("##### Ký hiệu:")
            st.latex(rf"_{{{atomic_num}}}\text{{{selected_element}}}")
        
        with col2:
            st.subheader(f"Mô hình Bohr: {selected_element}")
            fig_bohr, e_config = draw_bohr_model(atomic_num)
            st.plotly_chart(fig_bohr, use_container_width=False, key="bohr_chart")
            st.latex(rf"\text{{Cấu hình e: }} " + " \\ ".join([f"{i+1}s^{{...}}" for i in range(len(e_config))]))
            st.caption(f"Tổng số electron: {sum(e_config)}")
            
    lewis_placeholder.info("Cấu trúc Lewis 2D thường áp dụng cho phân tử. Vui lòng chuyển sang tab Phân tử.")
    model3d_placeholder.info("Mô hình 3D áp dụng cho phân tử hoặc mạng tinh thể. Vui lòng chuyển sang chế độ Phân tử.")

# ---------------------------------------------------------
# CHẾ ĐỘ 2: XỬ LÝ PHÂN TỬ (DÙNG TEXT INPUT + API)
# ---------------------------------------------------------
else:
    search_query = st.text_input(
        "🔍 Nhập Công thức phân tử (VD: H2O, CO2, CH4):", 
        value="H2O"
    ).strip()
    
    if search_query:
        bohr_placeholder.info("Mô hình Bohr áp dụng cho đơn nguyên tử. Vui lòng chuyển sang chế độ Nguyên tố.")
        
        try:
            smiles = get_smiles_from_formula(search_query)
            
            if smiles:
                # --- PANEL 2: RDKit 2D Lewis Structure ---
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 400), fitImage=True)
                    with lewis_placeholder.container():
                        st.image(img, caption=f"Cấu trúc 2D của {search_query}", use_container_width=False)
                
                # --- PANEL 3: py3Dmol Interactive 3D ---
                with model3d_placeholder.container():
                    st.subheader(f"Trình diễn WebGL 3D: {search_query}")
                    st.markdown("*Lưu ý: Dùng chuột xoay 360° và cuộn chuột để phóng to/thu nhỏ.*")
                    
                    view = py3Dmol.view(query=f'smiles:{smiles}', width=600, height=400)
                    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
                    view.zoomTo()
                    showmol(view, height=400, width=600)
                    
                # --- TÍNH NĂNG NHIỆT HÓA HỌC ---
                # --- TÍNH NĂNG NHIỆT HÓA HỌC (MVP DEMO) ---
                with thermo_placeholder.container(border=True):
                    st.subheader("🔥 Động cơ Nhiệt hóa học (Enthalpy Engine)")
                    st.latex(rf"\Delta_r H_{{298}}^0 = \sum \Delta_f H_{{298}}^0 (\text{{Sản phẩm}}) - \sum \Delta_f H_{{298}}^0 (\text{{Chất tham gia}})")
                    st.markdown("---")
                    
                    # Cơ sở dữ liệu nhiệt tạo thành chuẩn Mini (kJ/mol) để test Demo
                    # Cậu có thể tự bổ sung thêm các chất thường gặp trong SGK vào đây
                    enthalpy_db = {
                        "H2O (l)": -285.8, "H2O (g)": -241.8, "CO2 (g)": -393.5, 
                        "CH4 (g)": -74.8, "NH3 (g)": -45.9, "O2 (g)": 0.0, 
                        "H2 (g)": 0.0, "N2 (g)": 0.0, "HCl (g)": -92.3, "NaCl (s)": -411.2
                    }
                    
                    st.write("**Công cụ tính nhanh Enthalpy Phản ứng:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("Chất Tham Gia")
                        react_sub = st.selectbox("Chọn chất tham gia:", list(enthalpy_db.keys()), key="react_sub")
                        react_coef = st.number_input("Hệ số cân bằng:", min_value=1, value=1, key="react_coef")
                        react_total = react_coef * enthalpy_db[react_sub]
                        st.write(f"Tổng nhiệt: **{react_total:.1f} kJ/mol**")
                        
                    with col2:
                        st.success("Sản Phẩm")
                        prod_sub = st.selectbox("Chọn sản phẩm:", list(enthalpy_db.keys()), index=2, key="prod_sub")
                        prod_coef = st.number_input("Hệ số cân bằng:", min_value=1, value=1, key="prod_coef")
                        prod_total = prod_coef * enthalpy_db[prod_sub]
                        st.write(f"Tổng nhiệt: **{prod_total:.1f} kJ/mol**")
                        
                    # Tính toán và hiển thị kết quả
                    delta_H = prod_total - react_total
                    st.markdown("---")
                    
                    if delta_H < 0:
                        st.error(f"**Kết quả: $\\Delta_r H_{{298}}^0$ = {delta_H:.1f} kJ/mol**")
                        st.caption("🔥 Phản ứng **TỎA NHIỆT** (Môi trường xung quanh nóng lên).")
                    elif delta_H > 0:
                        st.info(f"**Kết quả: $\\Delta_r H_{{298}}^0$ = +{delta_H:.1f} kJ/mol**")
                        st.caption("❄️ Phản ứng **THU NHIỆT** (Môi trường xung quanh lạnh đi).")
                    else:
                        st.warning(f"**Kết quả: $\\Delta_r H_{{298}}^0$ = 0 kJ/mol**")
            else:
                st.warning(f"Đang gặp sự cố mạng hoặc không tìm thấy '{search_query}'. Vui lòng nhấn 'Clear Cache' ở góc phải Streamlit hoặc thử chất khác.")
                
        except Exception as e:
            st.error("Có lỗi xảy ra trong quá trình kết xuất đồ họa Hóa học.")
            st.caption(f"Mã lỗi hệ thống: {e}")

# ==========================================
# 4. FOOTER
# ==========================================
st.markdown("---")
st.caption("© 2026 ED-ODYSSEY Hub | Khởi chạy bằng RDKit & py3Dmol Engine")
