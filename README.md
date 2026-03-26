# ⚛️ Nuclear Kinematics Calculator

A streamlined, interactive web application built with Python and Streamlit for calculating two-body non-relativistic nuclear kinematics. Designed for experimental nuclear physicists, this tool provides real-time kinematic drop-off curves, specific angle interrogation, and raw data export.

## ✨ Features
* **Interactive UI:** Real-time updates via Streamlit sliders and synchronized number inputs.
* **Built-in Nuclide Database:** Automatically parses strings like `13C` or `alpha` and fetches exact atomic masses.
* **Dynamic LaTeX Rendering:** Formats your input strings into textbook-quality reaction equations.
* **CSV Export:** One-click download of the raw kinematic data at customizable angular resolutions.

## 🚀 Installation & Setup
1. Clone the repository: `git clone https://github.com/rhys1512/nuclear-kinematics-calculator.git`
2. Install the dependencies: `pip install streamlit numpy matplotlib pandas`
3. Run the application: `streamlit run kinematic_app.py`
