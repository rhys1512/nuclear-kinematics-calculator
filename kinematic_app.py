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
    'n': 1.008665, 'p': 1.007276, 'd': 2.014102, 't': 3.016049,
    'a': 4.002603, 'alpha': 4.002603, 'he3': 3.016029,
    '1h': 1.007825, '2h': 2.014102, '3h': 3.016049,
    '3he': 3.016029, '4he': 4.002603,
    '6li': 6.015122, '7li': 7.016003,
    '7be': 7.016928, '8be': 8.005305, '9be': 9.012183, '10be': 10.013534,
    '10b': 10.012937, '11b': 11.009305,
    '11c': 11.011433, '12c': 12.000000, '13c': 13.003355, '14c': 14.003242,
    '13n': 13.005739, '14n': 14.003074, '15n': 15.000109,
    '15o': 15.003065, '16o': 15.994915, '17o': 16.999131, '18o': 17.999160,
    '18f': 18.000938, '19f': 18.998403,
    '20ne': 19.992440, '21ne': 20.993846, '22ne': 21.991385,
    '23na': 22.989769,
    '24mg': 23.985041, '25mg': 24.985836, '26mg': 25.982592,
    '27al': 26.981538,
    '28si': 27.976926, '29si': 28.976494, '30si': 29.973770
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
st.title("⚛️ Nuclear Kinematics Calculator")

st.sidebar.header("Reaction Parameters")
col1, col2 = st.sidebar.columns(2)
projectile = col1.text_input("Projectile", value="alpha")
target     = col2.text_input("Target", value="13C")
ejectile   = col1.text_input("Ejectile", value="n")
recoil     = col2.text_input("Recoil", value="16O")

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
            + "<h1>⚛️ Session Terminated</h1>"
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

ax.set_title(f"Kinematic Drop-off over Angle", fontweight='bold')
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