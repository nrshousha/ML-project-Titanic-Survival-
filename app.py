import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import time
from userPredict2 import TitanicPassengerPredictor

# Page Configuration
st.set_page_config(
    page_title="Titanic 1912: The Sink-O-Matic™",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sarcastic luxury / icy doom aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

    :root {
        --ice-blue: #00e5ff;
        --deep-ocean: #070d18;
        --gold-lux: #e6b758;
        --blood-rust: #ff4757;
        --emerald-safe: #2ed573;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #0c203b 0%, #060b13 70%, #03060a 100%);
        color: #e0e8f5;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Headlines */
    h1, h2, h3, h4 {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 1px;
    }

    .hero-title {
        font-family: 'Cinzel', serif;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 20%, #7ee8fa 60%, #e6b758 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-shadow: 0 0 30px rgba(0, 229, 255, 0.3);
    }

    .hero-subtitle {
        text-align: center;
        color: #9bbcdb;
        font-size: 1.15rem;
        font-style: italic;
        margin-bottom: 25px;
    }

    .ticker-tape {
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 12px;
        padding: 10px 18px;
        text-align: center;
        font-size: 0.95rem;
        color: #7ee8fa;
        margin-bottom: 25px;
        backdrop-filter: blur(8px);
    }

    /* Result Banners */
    .verdict-box-dead {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.18) 0%, rgba(20, 10, 15, 0.85) 100%);
        border: 2px solid #ff4757;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 35px rgba(255, 71, 87, 0.35);
        animation: pulseDoom 3s infinite alternate;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .verdict-box-alive {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.18) 0%, rgba(10, 25, 20, 0.85) 100%);
        border: 2px solid #2ed573;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 35px rgba(46, 213, 115, 0.35);
        animation: pulseLife 3s infinite alternate;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    @keyframes pulseDoom {
        0% { box-shadow: 0 0 20px rgba(255, 71, 87, 0.2); }
        100% { box-shadow: 0 0 40px rgba(255, 71, 87, 0.5); }
    }
    @keyframes pulseLife {
        0% { box-shadow: 0 0 20px rgba(46, 213, 115, 0.2); }
        100% { box-shadow: 0 0 40px rgba(46, 213, 115, 0.5); }
    }

    /* Sarcastic Certificate */
    .certificate-card {
        background: #fbf7ee;
        color: #1a1611;
        border: 8px double #8c6d31;
        border-radius: 12px;
        padding: 30px;
        font-family: 'Playfair Display', serif;
        position: relative;
        box-shadow: 0 15px 35px rgba(0,0,0,0.7);
        margin-top: 20px;
    }
    .cert-title {
        font-family: 'Cinzel', serif;
        font-size: 1.6rem;
        font-weight: 900;
        color: #8c6d31;
        text-align: center;
        text-transform: uppercase;
        border-bottom: 2px solid #8c6d31;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }

    /* Model comparison card */
    .model-card-custom {
        background: rgba(13, 24, 43, 0.85);
        border: 1px solid rgba(230, 183, 88, 0.3);
        border-left: 5px solid #e6b758;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .model-card-sklearn {
        background: rgba(13, 24, 43, 0.85);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-left: 5px solid #00e5ff;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .model-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-custom {
        background: rgba(230, 183, 88, 0.2);
        color: #e6b758;
        border: 1px solid #e6b758;
    }
    .badge-sklearn {
        background: rgba(0, 229, 255, 0.2);
        color: #00e5ff;
        border: 1px solid #00e5ff;
    }

    /* Native containers border styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(13, 24, 43, 0.75);
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px);
    }

    /* Input text & number fields */
    .stTextInput input, .stNumberInput input {
        background-color: rgba(6, 15, 29, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #00e5ff !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.6) !important;
    }

    /* Brighten all Widget Labels (Passenger Name, Class, Age, Fare, etc.) */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"],
    .stTextInput label,
    .stTextInput label p,
    .stSelectbox label,
    .stSelectbox label p,
    .stNumberInput label,
    .stNumberInput label p,
    .stRadio label,
    .stRadio label p,
    .stSlider label,
    .stSlider label p,
    div[data-testid="stMarkdownContainer"] p strong,
    label p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.3px !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.8), 0 0 10px rgba(0, 229, 255, 0.25) !important;
    }

    /* Subheadings inside cards */
    h3, h4, .stSubheader {
        color: #f0f7ff !important;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important;
    }

    /* Radio button options text */
    div[data-testid="stRadio"] label div p {
        color: #e8f4fc !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Help tooltip icons */
    [data-testid="stWidgetLabel"] svg {
        fill: #00e5ff !important;
        color: #00e5ff !important;
    }

    /* Custom sidebar style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091322 0%, #050a12 100%);
        border-right: 1px solid rgba(0, 229, 255, 0.2);
    }

    /* Custom button */
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-family: 'Cinzel', serif;
        letter-spacing: 1px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 198, 255, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(0, 198, 255, 0.7);
    }

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(13, 24, 43, 0.6);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 10px;
        color: #9bbcdb;
        padding: 10px 20px;
        font-family: 'Cinzel', serif;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 229, 255, 0.15) !important;
        border-color: #00e5ff !important;
        color: #00e5ff !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⚓ Commissioning 1912 AI Engines from Scratch...")
def load_predictor():
    return TitanicPassengerPredictor()

predictor = load_predictor()

# Header & Sarcastic Intro (Removed religious quote)
st.markdown('<div class="hero-title">TITANIC 1912: THE SINK-O-MATIC™</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">"Billed as the absolute pinnacle of unsinkable engineering... until Machine Learning audited the passenger manifest."</div>', unsafe_allow_html=True)

# Sarcastic Live News Ticker
roast_tickers = [
    "🚨 ICEBERG INTEL: Visibility is 0.02 miles. Lookouts report binocular shortage.",
    "🎟️ LIFEBOAT POLICY: Reserved strictly for first-class aristocrats, their luggage, and 3 prize-winning Pomeranians.",
    "🎻 ORCHESTRA STATUS: Currently tuning violins for the final hypothermia concert.",
    "🚪 DOOR CAPACITY UPDATE: The door clearly fits two people, but Rose disagrees.",
    "🌊 WATER TEMP: -2°C (28°F) — Optimal for a brisk 3-minute survival sprint."
]
st.markdown(f'<div class="ticker-tape">{random.choice(roast_tickers)}</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_survival, tab_fare, tab_door, tab_insights = st.tabs([
    "💀 Sarcastic Survival Oracle", 
    "💸 Extortion Fare Appraiser", 
    "🚪 The Rose's Door Physics Lab", 
    "📊 1912 Ship Demographics"
])

# ==========================================
# TAB 1: SURVIVAL ORACLE (CLASSIFICATION)
# ==========================================
with tab_survival:
    st.markdown("### 🎩 Configure Your 1912 Passenger Profile")
    st.caption("Enter your passenger details and let our custom gradient descent algorithms evaluate your frozen fate.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("📋 Passenger Identity")

            passenger_name = st.text_input(
                "👤 Passenger Name", 
                value="Jack Dawson",
                help="Type any passenger name to generate a personalized manifest certificate."
            )

            pclass = st.selectbox(
                "🎟️ Cabin Class Privilege",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1st Class 🎩 (Butler, Caviar & Lifeboat Priority Access)",
                    2: "2nd Class 🧐 (Polite middle class, 50/50 toss up)",
                    3: "3rd Class / Steerage 🚪 (Locked below deck with the luggage)"
                }[x],
                index=2
            )

            gender_str = st.radio(
                "⚧️ 1912 Maritime Gender Protocol",
                options=["Female", "Male"],
                format_func=lambda x: "Lady 👗 (Automatic VIP Lifeboat Boarding Pass)" if x == "Female" else "Gentleman 👔 ('Women and children first, sorry old chap')",
                horizontal=True
            )
            sex = 1 if gender_str == "Male" else 0

            age = st.number_input(
                "🎂 Age in Years",
                min_value=0.5,
                max_value=100.0,
                value=28.0,
                step=1.0,
                help="Children get pity points. Elderly men are politely asked to take up smoking on the stern."
            )

            # Dynamic sarcastic age commentary
            if age < 12:
                st.info("👶 Baby Shield: Crew members will literally throw you into a lifeboat.")
            elif age > 60 and sex == 1:
                st.warning("👴 Vintage Gentleman: You're going down with a glass of port in hand.")
            elif 18 <= age <= 35 and sex == 1:
                st.caption("🏊 Prime Age: Good luck swimming 500 miles to Newfoundland.")

            fare = st.number_input(
                "💰 Ticket Fare Paid (£ 1912 Sterling)",
                min_value=0.0,
                max_value=600.0,
                value=8.05 if pclass == 3 else (26.0 if pclass == 2 else 85.0),
                step=1.0,
                help="Higher fare = more mahogany lifeboats. Free ticket = boiler room coal shovel."
            )

    with col2:
        with st.container(border=True):
            st.subheader("🚢 Family & Group Travel")
            
            c_a, c_b = st.columns(2)
            with c_a:
                sibsp = st.number_input("👫 Siblings / Spouse aboard", min_value=0, max_value=10, value=0, help="More family = more people holding you back at the stairs")
            with c_b:
                parch = st.number_input("👶 Parents / Children aboard", min_value=0, max_value=10, value=0, help="Parental panic factor")

            st.markdown("---")
            st.markdown("#### ⚙️ ML Model Architecture")
            st.markdown("""
            - **Custom Scratch Model**: Ridge Logistic Regression (`lr=0.01`, `epochs=1000`, `lmbd=0.1`)
            - **Benchmark Model**: Scikit-Learn Logistic Regression (`saga` solver, `C=0.5`)
            """)

            predict_button = st.button("🔮 PREDICT MY FROZEN DESTINY", use_container_width=True)

    # Perform Prediction
    # Scale inputs for classification [pclass, sex, age, fare]
    passenger_scaled = predictor.scaler_class.transform([[pclass, sex, age, fare]])

    # Custom Logistic Regression prediction & probability
    custom_pred = predictor.custom_classi_model.predict(passenger_scaled.tolist())[0]
    custom_linear = sum([predictor.custom_classi_model.weights[j] * passenger_scaled[0][j] for j in range(len(passenger_scaled[0]))]) + predictor.custom_classi_model.bias
    custom_prob = predictor.sigmoid(custom_linear)

    # Sklearn Logistic Regression prediction & probability
    sklearn_pred = predictor.sklearn_classi_model.predict(passenger_scaled)[0]
    sklearn_prob = predictor.sklearn_classi_model.predict_proba(passenger_scaled)[0][1]

    avg_prob = (custom_prob + sklearn_prob) / 2.0
    survived = avg_prob >= 0.50

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sarcastic Result Banner
    safe_name = passenger_name.strip() if passenger_name.strip() else "Anonymous Passenger"
    if survived:
        verdict_html = f"""
        <div class="verdict-box-alive">
            <h2 style="color: #2ed573; margin:0;">🎉 VERDICT: {safe_name.upper()} SURVIVED (Barely)</h2>
            <h4 style="color: #a8f5c8; margin-top: 8px;">Survival Probability: {avg_prob*100:.1f}%</h4>
            <p style="font-size: 1.1rem; color: #e0e8f5; margin-top: 15px;">
                <strong>Historical Roast:</strong> You claimed the entire wooden door for yourself, demanded extra fur coats, 
                and filed a formal complaint to the Carpathia crew regarding room temperature soup.
            </p>
        </div>
        """
    else:
        verdict_html = f"""
        <div class="verdict-box-dead">
            <h2 style="color: #ff4757; margin:0;">💀 VERDICT: {safe_name.upper()} BECAME FISH FOOD</h2>
            <h4 style="color: #ffb8b8; margin-top: 8px;">Survival Probability: {avg_prob*100:.1f}% (Ice Cold)</h4>
            <p style="font-size: 1.1rem; color: #e0e8f5; margin-top: 15px;">
                <strong>Historical Roast:</strong> The 3rd class steerage gate was securely locked, your ticket was cheap, and the deck orchestra played their final dramatic symphony exclusively for you. 
                You will be honored in a blockbuster film 85 years later.
            </p>
        </div>
        """
    st.markdown(verdict_html, unsafe_allow_html=True)

    # Side by side model comparison
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="model-card-custom">
            <span class="model-badge badge-custom">Custom Scratch Model</span>
            <h3 style="color: #e6b758; margin-top: 8px;">{'🟢 Survived' if custom_pred == 1 else '🔴 Perished'}</h3>
            <p>Calculated Odds: <strong>{custom_prob*100:.2f}%</strong></p>
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 10px; width: 100%;">
                <div style="background: {'#2ed573' if custom_pred == 1 else '#ff4757'}; height: 10px; width: {max(min(custom_prob*100, 100), 0):.1f}%; border-radius: 8px;"></div>
            </div>
            <p style="font-size: 0.8rem; color: #8fa3be; margin-top: 10px;">Gradient descent with manual L2 Ridge penalty.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="model-card-sklearn">
            <span class="model-badge badge-sklearn">Scikit-Learn Logistic Model</span>
            <h3 style="color: #00e5ff; margin-top: 8px;">{'🟢 Survived' if sklearn_pred == 1 else '🔴 Perished'}</h3>
            <p>Calculated Odds: <strong>{sklearn_prob*100:.2f}%</strong></p>
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 10px; width: 100%;">
                <div style="background: {'#2ed573' if sklearn_pred == 1 else '#ff4757'}; height: 10px; width: {max(min(sklearn_prob*100, 100), 0):.1f}%; border-radius: 8px;"></div>
            </div>
            <p style="font-size: 0.8rem; color: #8fa3be; margin-top: 10px;">Standard Scikit-Learn SAGA optimizer.</p>
        </div>
        """, unsafe_allow_html=True)

    # Sarcastic Certificate Generator
    st.markdown("### 📜 Official 1912 Board of Inquiry Certificate")
    cert_status = "CERTIFIED ICEBERG EVADER & DOOR OCCUPANT" if survived else "OFFICIAL NORTH ATLANTIC CORAL & CRAB REEF"
    cert_color = "#1b6d39" if survived else "#8c2020"
    
    st.markdown(f"""
    <div class="certificate-card">
        <div class="cert-title">WHITE STAR LINE • CASUALTY & LIFEBOAT MANIFEST</div>
        <p style="text-align:center; font-size: 1.1rem; color: #443c33;">Port of Southampton to New York • April 15, 1912</p>
        <hr style="border: 0; border-top: 1px solid #8c6d31; margin: 15px 0;">
        <p><strong>Passenger Name:</strong> {safe_name}</p>
        <p><strong>Passenger Profile:</strong> {gender_str}, Age {age:.1f}, Traveling in Class {pclass}</p>
        <p><strong>Recorded Fare:</strong> £{fare:.2f} (Delivered outcome: {'Warm lifeboat seat' if survived else 'Freezing saline bath'})</p>
        <p><strong>Official Inquest Verdict:</strong> <span style="color: {cert_color}; font-weight: bold; font-size: 1.2rem;">{cert_status}</span></p>
        <p style="font-style: italic; color: #554e44; margin-top: 15px;">
            "Recorded by the High Maritime Court of Southampton. No refunds will be issued for damp luggage, floating violins, or hypothermia."
        </p>
        <div style="text-align: right; font-size: 0.85rem; color: #8c6d31; margin-top: 10px;">
            <strong>SEAL OF APPROVAL:</strong> 🔏 White Star Line AI Registry
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: FARE EXTORTION (REGRESSION)
# ==========================================
with tab_fare:
    st.markdown("### 💸 Sarcastic 1912 Fare Appraiser (Regression Model)")
    st.caption("How much did the White Star Line charge passengers to embark on history's most famous disaster?")

    col_r1, col_r2 = st.columns([1, 1], gap="large")
    with col_r1:
        with st.container(border=True):
            r_pclass = st.selectbox("Cabin Class Desired", options=[1, 2, 3], index=0, key="reg_pclass")
            r_gender = st.radio("Gender Identity", options=["Male", "Female"], key="reg_gender")
            r_sex = 1 if r_gender == "Male" else 0
            r_age = st.number_input("Age of Passenger", min_value=1, max_value=100, value=32, key="reg_age")
            r_sibsp = st.number_input("Accompanying Siblings / Spouse", min_value=0, max_value=10, value=1, key="reg_sibsp")
            r_parch = st.number_input("Accompanying Parents / Children", min_value=0, max_value=10, value=2, key="reg_parch")

    with col_r2:
        # Scale inputs for regression: [pclass, sex, age, sibsp, parch]
        reg_input_scaled = predictor.scaler_reg.transform([[r_pclass, r_sex, r_age, r_sibsp, r_parch]])

        custom_fare_pred = predictor.custom_reg_model.predict(reg_input_scaled.tolist())[0]
        sklearn_fare_pred = predictor.sklearn_reg_model.predict(reg_input_scaled)[0]
        avg_fare = max((custom_fare_pred + sklearn_fare_pred) / 2.0, 4.0)

        # 1912 £ to Modern USD inflation (~ 1 £ in 1912 = ~ $145 today)
        modern_usd = avg_fare * 145.0

        with st.container(border=True):
            st.markdown(f"""
            <h3 style="color: #e6b758; margin-top:0;">Estimated 1912 Extortion Price</h3>
            <div style="font-size: 2.8rem; font-family: 'Cinzel', serif; color: #7ee8fa; font-weight: 800;">
                £{avg_fare:.2f}
            </div>
            <p style="color: #9bbcdb;">Equivalent to approximately <strong>${modern_usd:,.2f} USD</strong> in modern money.</p>
            <hr style="border: 0; border-top: 1px solid rgba(0, 229, 255, 0.2); margin: 15px 0;">
            <p><strong>Custom Scratch Ridge:</strong> £{custom_fare_pred:.2f}</p>
            <p><strong>Scikit-Learn Ridge:</strong> £{sklearn_fare_pred:.2f}</p>
            <div style="background: rgba(230, 183, 88, 0.1); border: 1px dashed #e6b758; border-radius: 8px; padding: 12px; margin-top: 15px;">
                💡 <strong>White Star Value Assessment:</strong> 
                {
                    "You paid a fortune to eat caviar and sink in luxury." if r_pclass == 1 else
                    ("Moderate pricing with zero guarantee of life jacket quality." if r_pclass == 2 else
                    "Bargain basement fare! Includes 1 slice of dry bread and a bunk 3 floors below the waterline.")
                }
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 3: THE ROSE'S DOOR PHYSICS LAB
# ==========================================
with tab_door:
    st.markdown("### 🚪 The Infamous Floating Door Buoyancy Simulator")
    st.caption("Addressing the greatest controversy in cinematic & maritime history: Could Jack have fit?")

    col_d1, col_d2 = st.columns([1, 1], gap="large")
    with col_d1:
        with st.container(border=True):
            door_area = st.slider("Door Surface Area (sq meters)", 1.0, 3.5, 2.2, step=0.1, help="Standard 1912 oak door panel")
            rose_selfishness = st.slider("Rose's Selfishness Quotient (%)", 0, 100, 85, help="Propensity to hoard space while promising 'I will never let go'")
            jack_weight = st.slider("Jack Dawson's Weight (kg)", 50, 100, 72)

    with col_d2:
        # Satirical buoyancy calculation
        wood_buoyancy_kg = door_area * 55.0  # Approx weight capacity in freezing salt water
        total_people_weight = 54 + jack_weight  # Rose ~ 54kg
        
        can_float_both = (wood_buoyancy_kg >= total_people_weight) and (rose_selfishness < 60)

        with st.container(border=True):
            st.markdown("#### 📐 Buoyancy & Ego Equation Results")
            st.write(f"- **Door Maximum Buoyancy Capacity:** {wood_buoyancy_kg:.1f} kg")
            st.write(f"- **Combined Passenger Weight (Jack + Rose):** {total_people_weight:.1f} kg")
            st.write(f"- **Rose's Effective Door Monopoly:** {rose_selfishness}%")

            if can_float_both:
                st.success("✅ **SURPRISE: Jack Survives!** Both fit comfortably. You have rewritten history and spared Leonardo DiCaprio hypothermia.")
            elif rose_selfishness >= 60 and wood_buoyancy_kg >= total_people_weight:
                st.error("❌ **Physics allowed it, but Rose's ego did not.** Jack sinks anyway because Rose wanted room to stretch her arms.")
            else:
                st.error("❌ **Both sink together!** The oak door capsizes in freezing water. Titanic tragedy confirmed.")

# ==========================================
# TAB 4: HISTORICAL DATA & MODEL WEIGHTS
# ==========================================
with tab_insights:
    st.markdown("### 📊 Historical Realities & Machine Learning Weights")
    st.caption("How your custom-trained machine learning model learned human prejudice from 1912 data.")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        with st.container(border=True):
            st.subheader("⚖️ Custom Logistic Regression Weights")
            feature_names = ["Passenger Class", "Sex (Male=1)", "Age", "Fare (£)"]
            weights = predictor.custom_classi_model.weights
            
            df_weights = pd.DataFrame({
                "Feature": feature_names,
                "Model Weight": [f"{w:.4f}" for w in weights],
                "Impact on Survival": [
                    "🔻 Strong Negative (Higher class number = 3rd class = doom)",
                    "🔻 Massive Negative (Being male dropped odds by over 70%)",
                    "🔻 Moderate Negative (Older = less prioritized)",
                    "🔺 Positive (Wealth directly bought survival)"
                ]
            })
            st.table(df_weights)
            st.caption(f"**Model Bias Term (Intercept):** {predictor.custom_classi_model.bias:.4f}")

    with col_i2:
        with st.container(border=True):
            st.subheader("💡 The 1912 Survival Cheat Codes")
            st.markdown("""
            1. **Be Female (Highest Weight)**: The 'Women & Children First' Birkenhead drill was strictly enforced.
            2. **Be in 1st Class (High Weight)**: 1st class passengers were closest to the top boat deck.
            3. **Pay Higher Fare**: Lifeboats were lowered before 3rd class steerage gates were opened.
            4. **Be Young**: Children had high priority access to early lifeboats.
            """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #5a738e; font-size: 0.85rem;">'
    'Built with Streamlit & Custom ML Models from Scratch • Academic ML Project • Sarcastic 1912 Titanic Survival Predictor'
    '</p>',
    unsafe_allow_html=True
)
