import os
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Spondylitis Risk Predictor",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# CUSTOM CSS INJECTION
# ------------------------------------------------------------------------------
# This attempts to recreate the modern, professional UI requested.
custom_css = """
<style>
/* Base overrides */
div.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
}
p, div, span, label {
    font-family: 'Inter', sans-serif !important;
}

/* Card-like containers for content */
.custom-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    margin-bottom: 24px;
}

/* Hero Section */
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7e22ce;
}
.hero-eyebrow::before {
    content: "";
    width: 28px;
    height: 1px;
    background: #7e22ce;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #7e22ce;
    margin-bottom: 8px;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 28px;
    font-weight: 300;
    color: #0f172a;
    margin-bottom: 20px;
}
.hero-divider {
    width: 56px;
    height: 3px;
    background: #d8b4fe;
    border-radius: 999px;
    margin-bottom: 22px;
}
.hero-desc {
    color: #475569;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 14px;
}

/* Buttons */
.stButton>button {
    background-color: #d8b4fe !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 16px rgba(166, 122, 232, 0.15) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    box-shadow: 0 12px 24px rgba(166, 122, 232, 0.25) !important;
    transform: translateY(-2px);
    color: #0f172a !important;
}

/* Questionnaire Headers */
.q-header {
    color: #7e22ce;
    font-weight: 700;
    margin-bottom: 8px;
    font-size: 24px;
}

/* Result Section */
.result-banner {
    background-color: #fef9c3;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
}
.metric-box {
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 10px;
}
.metric-title {
    font-weight: 600;
    color: #7e22ce;
    font-size: 14px;
}
.metric-title.warn {
    color: #ef4444;
}
.metric-desc {
    font-size: 13px;
    color: #475569;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load models safely and cache them using st.cache_resource."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        ss_seated = joblib.load(os.path.join(base_dir, 'scaler_seated_activities.pkl'))
        ss_ohe = joblib.load(os.path.join(base_dir, 'scaler_ohe_gender.pkl'))
        ss_oe = joblib.load(os.path.join(base_dir, 'scaler_ordinal_encoding.pkl'))
        pca_model = joblib.load(os.path.join(base_dir, 'pca_model.pkl'))
        kmeans_model = joblib.load(os.path.join(base_dir, 'kmeans_model (2).pkl'))
        return ss_seated, ss_ohe, ss_oe, pca_model, kmeans_model, None
    except Exception as e:
        return None, None, None, None, None, str(e)

ss_seated, ss_ohe, ss_oe, pca_model, kmeans_model, error_msg = load_models()

# ------------------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# ------------------------------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def change_page(new_page):
    st.session_state.page = new_page

# ------------------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------------------

def show_home():
    st.markdown("""
        <div class="custom-card">
            <div class="hero-eyebrow">Spinal Health Assessment</div>
            <div class="hero-title">Spondylitis</div>
            <div class="hero-subtitle">Student Project Risk Form</div>
            <div class="hero-divider"></div>
            <p class="hero-desc">
                Spondylitis is an inflammatory condition that affects the spine and nearby joints. It can lead to pain,
                stiffness, and reduced flexibility if symptoms are ignored.
            </p>
            <p class="hero-desc">
                This web application is a project by students of MNIT Jaipur. It gives a brief educational form based on posture, sleep,
                lifestyle, and pain-related questions.
            </p>
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 18px; padding: 12px 18px; background: #fef9c3; border: 1px solid #e2e8f0; border-radius: 999px; font-size: 14px; color: #0f172a;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #eab308;"></div>
                <span>Prepared by <strong>MNIT Jaipur students</strong> for educational use</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="custom-card" style="padding: 20px;">
            <h3 style="color:#7e22ce; font-size:18px; margin-bottom:8px;">Before You Start</h3>
            <ul style="color:#475569; padding-left:20px; font-size:14px;">
                <li><strong>Short form:</strong> The next page asks a brief set of questions.</li>
                <li><strong>Data mapping:</strong> Selected options are saved and mapped to numerical data.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="custom-card" style="padding: 20px;">
            <h3 style="color:#7e22ce; font-size:18px; margin-bottom:8px;">Important Notice</h3>
            <ul style="color:#475569; padding-left:20px; font-size:14px;">
                <li><strong>Educational project:</strong> This is not a medical diagnosis tool.</li>
                <li><strong>Student work:</strong> Created as a project by students of MNIT Jaipur.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; padding: 20px 0;">', unsafe_allow_html=True)
    st.button("Enter Details", on_click=change_page, args=('form',))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #475569; font-size: 13px; margin-top: 20px;">
        Developed by students of <strong>Malaviya National Institute of Technology, Jaipur</strong><br>
        This tool is for educational purposes only and does not constitute medical advice.
    </div>
    """, unsafe_allow_html=True)

def show_form():
    st.markdown('<div class="q-header" style="text-align:center;">Your Lifestyle Profile</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#475569; margin-bottom:30px;">Answer all questions honestly. Option questions are displayed to build your profile.</p>', unsafe_allow_html=True)

    with st.form(key='risk_form'):
        # Q1: Sitting hours
        sitting_hours = st.slider(
            "How many hours do you typically sit in a day? (work, study, screen time, travel, etc.)",
            min_value=0, max_value=16, value=6, step=1
        )
        st.markdown("<hr style='border:1px solid #e2e8f0;'>", unsafe_allow_html=True)

        # Q2: Gender
        gender_options = ["Female", "Male"]
        gender_label = st.radio("What is your gender?", gender_options)
        gender_val = gender_options.index(gender_label)
        st.markdown("<hr style='border:1px solid #e2e8f0;'>", unsafe_allow_html=True)

        # Q3: Morning Stiffness
        ms_options = [
            "No, I feel completely fine", 
            "Mild stiffness that fades quickly", 
            "Moderate stiffness that takes time to ease", 
            "Severe stiffness lasting a long time"
        ]
        ms_label = st.selectbox("Do you feel stiffness in your body when you wake up in the morning?", ms_options)
        ms_val = ms_options.index(ms_label)
        
        # Q4: Persistent Pain
        pp_options = [
            "No pain at all", 
            "Yes - started less than 3 months ago", 
            "Yes - ongoing for about 3 months", 
            "Yes - persisting for more than 3 months"
        ]
        pp_label = st.selectbox("Do you have persistent pain in your back, neck, or heel?", pp_options)
        pp_val = pp_options.index(pp_label)
        
        # Q5: Sitting Posture
        sp_options = [
            "Straight and upright", 
            "Slightly slouched", 
            "Noticeably hunched or bent forward"
        ]
        sp_label = st.radio("How would you describe your usual sitting posture?", sp_options)
        sp_val = sp_options.index(sp_label)
        st.markdown("<hr style='border:1px solid #e2e8f0;'>", unsafe_allow_html=True)

        # Q6: Sleeping Posture
        slp_options = [
            "On my back", 
            "On my side", 
            "I change positions frequently", 
            "On my stomach"
        ]
        slp_label = st.selectbox("What is your usual sleeping position?", slp_options)
        slp_val = slp_options.index(slp_label)
        
        # Q7: Sleep Turning
        st_options = [
            "No difficulty at all", 
            "Mild difficulty sometimes", 
            "Moderate difficulty", 
            "Significant pain or difficulty turning"
        ]
        st_label = st.selectbox("Do you have difficulty turning sides while sleeping?", st_options)
        st_val = st_options.index(st_label)
        
        # Q8: Sleep Quality
        sq_options = [
            "Excellent - I wake up feeling refreshed", 
            "Good - mostly restful", 
            "Average - could be better", 
            "Poor - I rarely feel well-rested"
        ]
        sq_label = st.selectbox("How would you rate the overall quality of your sleep?", sq_options)
        sq_val = sq_options.index(sq_label)
        st.markdown("<hr style='border:1px solid #e2e8f0;'>", unsafe_allow_html=True)
        
        # Q9: Physical Activity
        pa_options = [
            "Every day", 
            "A few times a week", 
            "Rarely - once in a while", 
            "Never"
        ]
        pa_label = st.radio("How often do you engage in physical activity - exercise, sports, or regular walking?", pa_options)
        pa_val = pa_options.index(pa_label)
        
        # Q10: OTT Usage
        ott_options = [
            "Very rarely - once in a few months", 
            "About once a month", 
            "Weekly", 
            "Regularly - almost every day"
        ]
        ott_label = st.selectbox("How often do you watch OTT platforms or stream content (Netflix, YouTube, etc.)?", ott_options)
        ott_val = ott_options.index(ott_label)
        st.markdown("<hr style='border:1px solid #e2e8f0;'>", unsafe_allow_html=True)

        # Q11: Family Backpain
        fb_options = [
            "No, not at all", 
            "Possibly, but never formally diagnosed", 
            "Yes - diagnosed by a doctor"
        ]
        fb_label = st.radio("Do your parents or close family members have a history of back pain?", fb_options)
        fb_val = fb_options.index(fb_label)
        
        # Q12: Gut Infection
        gi_options = [
            "No, never", 
            "Yes - a moderate, diagnosed case", 
            "Yes - a severe, diagnosed case"
        ]
        gi_label = st.radio("Have you ever been diagnosed with a gut or intestinal infection?", gi_options)
        gi_val = gi_options.index(gi_label)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            submit_button = st.form_submit_button(label='Analyse My Risk ➔', use_container_width=True)

    if submit_button:
        # Check if models loaded successfully
        if error_msg:
            st.error(f"Cannot process request. Model loading failed: {error_msg}")
            return
            
        st.session_state.answers = {
            'sitting_hours': sitting_hours,
            'gender': gender_val,
            'morning_stiffness': ms_val,
            'persistent_pain': pp_val,
            'sitting_posture': sp_val,
            'sleeping_posture': slp_val,
            'sleep_turning': st_val,
            'sleep_quality': sq_val,
            'physical_activity': pa_val,
            'ott_usage': ott_val,
            'family_backpain': fb_val,
            'gut_infection': gi_val
        }
        change_page('result')
        st.rerun()

    st.button("Back to Home", on_click=change_page, args=('home',))

def show_result():
    if 'answers' not in st.session_state:
        change_page('form')
        st.rerun()

    ans = st.session_state.answers
    
    try:
        # Preprocess logic
        seated = np.array([[ans['sitting_hours']]])
        gender = np.array([[ans['gender']]])
        ordinals = np.array([[
            ans['morning_stiffness'],
            ans['persistent_pain'],
            ans['sitting_posture'],
            ans['sleeping_posture'],
            ans['sleep_turning'],
            ans['sleep_quality'],
            ans['physical_activity'],
            ans['ott_usage'],
            ans['family_backpain'],
            ans['gut_infection']
        ]])
        
        seated_scaled = ss_seated.transform(seated)
        gender_scaled = ss_ohe.transform(gender)
        ordinals_scaled = ss_oe.transform(ordinals)
        
        features = np.hstack([seated_scaled, gender_scaled, ordinals_scaled])
        features_pca = pca_model.transform(features)
        cluster_id = int(kmeans_model.predict(features_pca)[0])
        
        cluster_risk_mapping = {
            0: "High",
            1: "Very High",
            2: "Low",
            3: "Moderate"
        }
        risk = cluster_risk_mapping[cluster_id]
        
        suggestions = {
            "Low": "Maintain your healthy lifestyle. Keep up the good posture, regular physical activity, and adequate sleep.",
            "Moderate": "Your responses indicate moderate risk. Focus on ergonomic improvements, consistent targeted exercise, and consider consulting a physiotherapist.",
            "High": "Your pattern aligns with significant risk factors for spondylitis. We recommend consulting a healthcare professional for an evaluation.",
            "Very High": "Your pattern aligns with severe risk factors for spondylitis. We strongly recommend consulting a rheumatologist immediately."
        }
        
        if risk == "Low":
            desc = "Your responses map to a lower-risk cluster. Keep up healthy posture and activity!"
        elif risk == "Mild":
            desc = "Your answers place you in a mild-risk cluster. Minor lifestyle changes can help you maintain health."
        elif risk == "Moderate":
            desc = "Your answers place you in a moderate-risk cluster. Consider incorporating more movement and ergonomic improvements."
        else:
            desc = "Your pattern aligns with a higher-risk cluster. We recommend consulting a healthcare professional for an evaluation."
            
        # Display Banner
        st.markdown(f"""
        <div class="result-banner">
            <div class="hero-eyebrow">Risk Analysis Result</div>
            <h1 style="color: #0f172a; margin-top:0; font-size: 38px;">{risk} Risk Zone</h1>
            <p style="color: #475569; font-size: 16px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Suggestion
        st.markdown(f"""
        <div class="custom-card" style="background-color: #f8fafc;">
            <h3 style="color: #7e22ce; margin-top:0; font-size: 20px;">Health Suggestion</h3>
            <p style="color: #0f172a;">{suggestions[risk]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Matplotlib Chart
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #7e22ce; margin-top:0; font-size: 20px;">Cluster Analysis Graph</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        centers = kmeans_model.cluster_centers_
        
        colors = {
            0: "orange",
            1: "red",
            2: "green",
            3: "blue"
        }
        
        n_points_per_cluster = 150
        synthetic_X = []
        synthetic_y = []
        np.random.seed(42)
        
        for cid in range(kmeans_model.n_clusters):
            c_color = colors[cid]
            c_risk = cluster_risk_mapping[cid]
            pts = np.random.normal(loc=centers[cid], scale=0.6, size=(n_points_per_cluster, centers.shape[1]))
            synthetic_X.append(pts)
            synthetic_y.extend([cid] * n_points_per_cluster)
            ax.scatter(pts[:, 0], pts[:, 1], c=c_color, s=40, alpha=1.0, edgecolors='none', label=f'Cluster {cid}: {c_risk} Risk')
            
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=80, marker='o', edgecolors='none', label='Centroids')
        
        # User Point
        ax.scatter(features_pca[0, 0], features_pca[0, 1], c='yellow', s=120, marker='o', edgecolors='black', label='YOUR POSITION')
        
        ax.set_title("Clusters of people", fontsize=14)
        ax.set_xlabel("PC1: Pain (Pain, Turning, Family, Stiffness)")
        ax.set_ylabel("PC2: Lifestyle (Sitting, OTT, Gut, Activity)")
        
        handles, lbls = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Plot in Streamlit
        st.pyplot(fig)
        st.markdown(f'<p style="color: #475569; font-size: 13px; text-align: center; margin-top: 10px;">Assigned to Risk Cluster {cluster_id} based on your 12 behavioral inputs.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metrics
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #7e22ce; margin-top:0; font-size: 20px;">Model Evaluation Metrics</h3>', unsafe_allow_html=True)
        
        sil_score = 0.336 
        db_score = 0.960
        ch_score = 180.257
        
        st.markdown(f"""
        <div class="metric-box">
            <span class="metric-title">Silhouette Score:</span> <span style="font-weight: 700;">{sil_score}</span><br>
            <span class="metric-desc">Measures cluster density. Closer to 1 is better.</span>
        </div>
        <div class="metric-box">
            <span class="metric-title warn">Davies-Bouldin Index:</span> <span style="font-weight: 700;">{db_score}</span><br>
            <span class="metric-desc">Measures cluster separation. Lower score is better.</span>
        </div>
        <div class="metric-box">
            <span class="metric-title warn">Calinski-Harabasz Index:</span> <span style="font-weight: 700;">{ch_score}</span><br>
            <span class="metric-desc">Measures cluster variance ratio. Higher score is better.</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.button("Retake Assessment", on_click=change_page, args=('home',))

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.button("Back to Form", on_click=change_page, args=('form',))

# ------------------------------------------------------------------------------
# APP ROUTING
# ------------------------------------------------------------------------------
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'form':
    show_form()
elif st.session_state.page == 'result':
    show_result()
