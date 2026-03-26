import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import signal

# =============================================================================
# 1. LATEX FORMATTER & NUCLIDE DATABASE
# =============================================================================
def format_latex(nuclide_string):
    """Converts a standard string like '13c' or 'alpha' into LaTeX scientific format."""
    clean = str(nuclide_string).strip()
    low = clean.lower()
    
    # Handle specific light particles
    if low in ['a', 'alpha']: return r"\alpha"
    if low == 'n': return r"\text{n}"
    if low == 'p': return r"\text{p}"
    if low == 'd': return r"\text{d}"
    if low == 't': return r"\text{t}"
    if low in ['he3', '3he']: return r"^3\text{He}"
    
    # Handle standard isotopes (e.g., '13C', '10b')
    match = re.match(r'^(\d+)([A-Za-z]+)$', clean)
    if match:
        mass_num = match.group(1)
        element = match.group(2).capitalize() # Forces 'c' to 'C'
        return f"^{{{mass_num}}}\\text{{{element}}}"
        
    # Fallback if it doesn't match standard formats
    return f"\\text{{{clean}}}"

MASS_DB = {
    # Light Particles & Hydrogen Isotopes
    'n': 1.008665, 'p': 1.007276, 'd': 2.014102, 't': 3.016049,
    'a': 4.002603, 'alpha': 4.002603, 'he3': 3.016029,
    '1h': 1.007825, '2h': 2.014102, '3h': 3.016049,
    '3he': 3.016029, '4he': 4.002603,
    
    # Lithium to Oxygen (Original + Additions)
    '6li': 6.015122, '7li': 7.016003,
    '7be': 7.016928, '8be': 8.005305, '9be': 9.012183, '10be': 10.013534,
    '10b': 10.012937, '11b': 11.009305,
    '11c': 11.011433, '12c': 12.000000, '13c': 13.003355, '14c': 14.003242,
    '13n': 13.005739, '14n': 14.003074, '15n': 15.000109,
    '15o': 15.003065, '16o': 15.994915, '17o': 16.999131, '18o': 17.999160,
    
    # Fluorine to Silicon
    '18f': 18.000938, '19f': 18.998403,
    '20ne': 19.992440, '21ne': 20.993846, '22ne': 21.991385,
    '23na': 22.989769,
    '24mg': 23.985041, '25mg': 24.985836, '26mg': 25.982592,
    '27al': 26.981538,
    '28si': 27.976926, '29si': 28.976494, '30si': 29.973770,
    
    # Phosphorus to Calcium
    '31p': 30.973762, '32p': 31.973907,
    '32s': 31.972071, '33s': 32.971458, '34s': 33.967867, '36s': 35.967081,
    '35cl': 34.968852, '36cl': 35.968306, '37cl': 36.965903,
    '36ar': 35.967545, '38ar': 37.962732, '40ar': 39.962383,
    '39k': 38.963706, '40k': 39.963998, '41k': 40.961825,
    '40ca': 39.962591, '42ca': 41.958618, '44ca': 43.955481, '48ca': 47.952534,
    
    # Scandium to Zinc (The Iron Peak)
    '45sc': 44.955910,
    '46ti': 45.952628, '48ti': 47.947946, '50ti': 49.944791,
    '51v': 50.943960,
    '50cr': 49.946044, '52cr': 51.940508, '53cr': 52.940649, '54cr': 53.938880,
    '55mn': 54.938045,
    '54fe': 53.939610, '56fe': 55.934937, '57fe': 56.935394, '58fe': 57.933276,
    '59co': 58.933195,
    '58ni': 57.935343, '60ni': 59.930786, '61ni': 60.931056, '62ni': 61.928345, '64ni': 63.927966,
    '63cu': 62.929597, '65cu': 64.927789,
    '64zn': 63.929142, '66zn': 65.926033, '67zn': 66.927127, '68zn': 67.924844,
    
    # Gallium to Zirconium
    '69ga': 68.925574, '71ga': 70.924701,
    '70ge': 69.924247, '72ge': 71.922076, '74ge': 73.921178,
    '75as': 74.921596,
    '78se': 77.917309, '80se': 79.916521, '82se': 81.916699,
    '79br': 78.918337, '81br': 80.916290,
    '80kr': 79.916379, '84kr': 83.911507, '86kr': 85.910610,
    '85rb': 84.911789, '87rb': 86.909180,
    '88sr': 87.905612, '90sr': 89.907738,
    '89y': 88.905848,
    '90zr': 89.904704, '91zr': 90.905645, '92zr': 91.905040, '94zr': 93.906315,
    
    # Niobium to Ruthenium (Approaching Mass 100)
    '93nb': 92.906378,
    '92mo': 91.906811, '95mo': 94.905842, '96mo': 95.904679, '98mo': 97.905408, '100mo': 99.907477,
    '98ru': 97.905287, '99ru': 98.905939, '100ru': 99.904220, '101ru': 100.905582, '102ru': 101.904350,

    # Silver to Tin (100-120)
    '107ag': 106.905093, '109ag': 108.904752,
    '110cd': 109.903002, '112cd': 111.902757, '114cd': 113.903358,
    '115in': 114.903878,
    '112sn': 111.904818, '118sn': 117.901603, '120sn': 119.902194, '124sn': 123.905273,
    
    # Antimony to Barium (120-140)
    '121sb': 120.903815, '123sb': 122.904214,
    '127i': 126.904473,
    '124xe': 123.905893, '131xe': 130.905082, '132xe': 131.904153, '136xe': 135.907219,
    '133cs': 132.905451,
    '136ba': 135.904575, '138ba': 137.905247,
    
    # Lanthanides (140-180)
    '140ce': 139.905438, '142ce': 141.909244,
    '141pr': 140.907652,
    '142nd': 141.907723, '144nd': 143.910087,
    '152sm': 151.919732, '154sm': 153.922209,
    '153eu': 152.921230,
    '158gd': 157.924103, '160gd': 159.927054,
    '164dy': 163.929174,
    '165ho': 164.930322,
    '166er': 165.930293,
    '174yb': 173.938862,
    '175lu': 174.940771,
    
    # Hafnium to Lead (180-210)
    '180hf': 179.946550,
    '181ta': 180.947995,
    '184w': 183.950931, '186w': 185.954364,
    '187re': 186.955753,
    '192os': 191.961481,
    '193ir': 192.962926,
    '195pt': 194.964791,
    '197au': 196.966568,
    '202hg': 201.970643,
    '205tl': 204.974427,
    '204pb': 203.973043, '206pb': 205.974465, '207pb': 206.975896, '208pb': 207.976652,
    '209bi': 208.980398,
    
    # Radium to Uranium (210-240)
    '222rn': 222.017577,
    '226ra': 226.025409,
    '232th': 232.038055,
    '231pa': 231.035884,
    '234u': 234.040952, '235u': 235.043929, '238u': 238.050788,
    
    # Transuranics (240-270)
    '237np': 237.048173,
    '239pu': 239.052163, '244pu': 244.064204,
    '243am': 243.061381,
    '247cm': 247.070354,
    '247bk': 247.070307,
    '251cf': 251.079587,
    '252es': 252.082980,
    '257fm': 257.095105,
    '258md': 258.098431,
    '259no': 259.10103,
    '262lr': 262.10963,
    
    # Superheavy Elements (270-300)
    '267rf': 267.1281,
    '268db': 268.1287,
    '271sg': 271.1334,
    '270bh': 270.133,
    '277hs': 277.149,
    '278mt': 278.156,
    '281ds': 281.164,
    '282rg': 282.169,
    '285cn': 285.177,
    '286nh': 286.182,
    '289fl': 289.190,
    '290mc': 290.196,
    '293lv': 293.204,
    '294ts': 294.211,
    '294og': 294.214 # The heaviest known element
}

def get_mass(nuclide_string):
    clean_string = str(nuclide_string).strip().lower()
    if clean_string not in MASS_DB:
        return None
    return MASS_DB[clean_string]

def solve_kinematics(m_p, m_t, m_out, m_other, Ep, Q, angles_deg):
    theta = np.radians(angles_deg)
    a = m_out + m_other
    b = -2 * np.sqrt(m_p * m_out * Ep) * np.cos(theta)
    c = -(m_other * Q + (m_other - m_p) * Ep)
    
    discriminant = b**2 - 4*a*c
    valid = discriminant >= 0 
    
    E_out = np.full_like(theta, np.nan, dtype=float)
    if np.any(valid):
        sqrt_E = (-b[valid] + np.sqrt(discriminant[valid])) / (2 * a)
        positive = sqrt_E > 0
        valid_idx = np.where(valid)[0]
        final_idx = valid_idx[positive]
        E_out[final_idx] = sqrt_E[positive]**2
    return E_out

# =============================================================================
# 2. UI LAYOUT & SIDEBAR INPUTS
# =============================================================================
st.set_page_config(page_title="Kinematics Calculator", layout="wide")
st.title("☢️ Nuclear Kinematics Calculator")

st.sidebar.header("Reaction Parameters")
col1, col2 = st.sidebar.columns(2)
projectile = col1.text_input("Projectile", value="alpha")
target     = col2.text_input("Target", value="10B")
ejectile   = col1.text_input("Ejectile", value="n")
recoil     = col2.text_input("Recoil", value="13N")

# --- NEW: Synchronized Incident Energy Input ---
if 'energy_num' not in st.session_state:
    st.session_state.energy_num = 5.0
if 'energy_slide' not in st.session_state:
    st.session_state.energy_slide = 5.0

def update_energy_slider():
    st.session_state.energy_slide = st.session_state.energy_num

def update_energy_num():
    st.session_state.energy_num = st.session_state.energy_slide

st.sidebar.markdown("**Incident Energy (MeV)**")
st.sidebar.number_input("Type Energy:", min_value=0.1, max_value=100.0, step=0.1, 
                        key='energy_num', on_change=update_energy_slider, label_visibility="collapsed")
st.sidebar.slider("Slide Energy:", min_value=0.1, max_value=100.0, step=0.1, 
                  key='energy_slide', on_change=update_energy_num, label_visibility="collapsed")

incident_energy_MeV = st.session_state.energy_num

# --- NEW: Synchronized Angle Input ---
if 'angle_num' not in st.session_state:
    st.session_state.angle_num = 45.0
if 'angle_slide' not in st.session_state:
    st.session_state.angle_slide = 45.0

def update_slider():
    st.session_state.angle_slide = st.session_state.angle_num

def update_num():
    st.session_state.angle_num = st.session_state.angle_slide

st.sidebar.markdown("**Detector Angle (Degrees)**")
st.sidebar.number_input("Type Angle:", min_value=0.0, max_value=180.0, step=0.1, 
                        key='angle_num', on_change=update_slider, label_visibility="collapsed")
st.sidebar.slider("Slide Angle:", min_value=0.0, max_value=180.0, step=0.1, 
                  key='angle_slide', on_change=update_num, label_visibility="collapsed")

detector_angle_deg = st.session_state.angle_num


# --- UPDATED: Total Process Tree Shutdown ---
st.sidebar.markdown("---")
if st.sidebar.button("🛑 Shutdown App", use_container_width=True):
    # 1. UI Feedback for the user
    st.components.v1.html(
        """
        <script>
            window.parent.document.body.innerHTML = 
            "<div style='font-family:sans-serif; text-align:center; margin-top:50px;'>"
            + "<h1>☢️ Session Terminated</h1>"
            + "<p>The Python engine and shell script have been killed.</p>"
            + "<p>You can now safely close this tab.</p></div>";
        </script>
        """,
        height=0
    )
    
    # 2. Brief pause to ensure the browser receives the HTML above
    import time
    time.sleep(0.5)
    
    # 3. KILL THE ENTIRE PROCESS GROUP
    # This targets the Python script AND the shell script that started it.
    import os
    import signal
    
    # os.getpgrp() gets the Process Group ID. 
    # Killing the group ensures the Automator 'robot' stops immediately.
    os.killpg(os.getpgrp(), signal.SIGTERM)

st.sidebar.markdown("---")
angle_step_deg = st.sidebar.selectbox("CSV Angle Resolution", [1.0, 0.5, 0.1, 0.05], index=2)

# Verify masses
m_projectile = get_mass(projectile)
m_target     = get_mass(target)
m_ejectile   = get_mass(ejectile)
m_recoil     = get_mass(recoil)

if None in (m_projectile, m_target, m_ejectile, m_recoil):
    st.error("⚠️ One or more particles not found in the database. Please check your spelling.")
    st.stop()

# =============================================================================
# 3. KINEMATICS CALCULATIONS
# =============================================================================
Q_value = (m_projectile + m_target - m_ejectile - m_recoil) * 931.494

num_steps = int(180.0 / angle_step_deg) + 1
angles = np.linspace(0, 180, num_steps)

E_ejectile_curve = solve_kinematics(m_projectile, m_target, m_ejectile, m_recoil, incident_energy_MeV, Q_value, angles)
E_recoil_curve   = solve_kinematics(m_projectile, m_target, m_recoil, m_ejectile, incident_energy_MeV, Q_value, angles)

E_ej_target = solve_kinematics(m_projectile, m_target, m_ejectile, m_recoil, incident_energy_MeV, Q_value, np.array([detector_angle_deg]))
E_rec_target = solve_kinematics(m_projectile, m_target, m_recoil, m_ejectile, incident_energy_MeV, Q_value, np.array([detector_angle_deg]))

val_ejectile = E_ej_target[0] if len(E_ej_target) > 0 and not np.isnan(E_ej_target[0]) else None
val_recoil   = E_rec_target[0] if len(E_rec_target) > 0 and not np.isnan(E_rec_target[0]) else None

# =============================================================================
# 4. MAIN DASHBOARD & PLOT
# =============================================================================
# LaTeX Formatted Header
tex_target   = format_latex(target)
tex_proj     = format_latex(projectile)
tex_ejec     = format_latex(ejectile)
tex_recoil   = format_latex(recoil)

reaction_latex = f"{tex_target}({tex_proj}, {tex_ejec}){tex_recoil}"
st.markdown(f"### Reaction: ${reaction_latex}$")

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Q-Value", f"{Q_value:.3f} MeV")
m2.metric(f"Angle Selected", f"{detector_angle_deg}°")
m3.metric(f"{ejectile} Energy", f"{val_ejectile:.3f} MeV" if val_ejectile else "Forbidden")
m4.metric(f"{recoil} Energy", f"{val_recoil:.3f} MeV" if val_recoil else "Forbidden")

st.markdown("---")

# Expandable Math Formula Section
with st.expander("📐 View Governing Kinematics Equations"):
    st.markdown("The energy of the outgoing particle is determined by solving the conservation of energy and momentum equations, which form a quadratic equation in terms of $\sqrt{E_{out}}$:")
    
    st.latex(r''' A = m_{out} + m_{other} ''')
    st.latex(r''' B = -2 \sqrt{m_p m_{out} E_p} \cos(\theta) ''')
    st.latex(r''' C = -\left[ m_{other} Q + (m_{other} - m_p) E_p \right] ''')
    
    st.markdown("Taking the positive physical root, the energy is:")
    st.latex(r''' E_{out} = \left( \frac{-B + \sqrt{B^2 - 4AC}}{2A} \right)^2 ''')

st.markdown("---")

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(angles, E_ejectile_curve, color='mediumseagreen', linewidth=2.5, label=f'Ejectile ({ejectile})')
ax.plot(angles, E_recoil_curve, color='royalblue', linewidth=2.5, label=f'Recoil Nucleus ({recoil})')
ax.axvline(detector_angle_deg, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

if val_ejectile is not None:
    ax.scatter([detector_angle_deg], [val_ejectile], color='red', zorder=5, s=80)
if val_recoil is not None:
    ax.scatter([detector_angle_deg], [val_recoil], color='red', zorder=5, s=80)

ax.set_title(f"Kinematic Curves", fontweight='bold')
ax.set_xlabel("Laboratory Observation Angle [Degrees]")
ax.set_ylabel("Particle Energy [MeV]")
ax.set_xlim(0, 180)
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# =============================================================================
# 5. CSV EXPORT BUTTON
# =============================================================================
df = pd.DataFrame({
    "Angle_Laboratory_Deg": angles,
    f"Ejectile_Energy_{ejectile}_MeV": E_ejectile_curve,
    f"Recoil_Energy_{recoil}_MeV": E_recoil_curve
})

csv_data = df.to_csv(index=False).encode('utf-8')

st.sidebar.markdown("---")
st.sidebar.download_button(
    label="⬇️ Download Kinematics CSV",
    data=csv_data,
    file_name=f"Kinematics_{target}_{projectile}_{ejectile}_{recoil}.csv",
    mime="text/csv"
)