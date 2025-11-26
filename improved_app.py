import streamlit as st
import pandas as pd
import numpy as np
from fraud_detection_system import FraudDetectionSystem
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .fraud-alert {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        animation: pulse 2s infinite;
    }
    .legitimate-alert {
        background: linear-gradient(90deg, #00b894, #00cec9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource
def load_fraud_system():
    system = FraudDetectionSystem()
    system.setup_database()
    return system

fraud_system = load_fraud_system()

# Currency conversion
USD_TO_INR = 83

def usd_to_inr(amount):
    return amount * USD_TO_INR

def inr_to_usd(amount):
    return amount / USD_TO_INR

def seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"

def time_to_seconds(hours, minutes):
    return hours * 3600 + minutes * 60

# Sound effects simulation
def play_sound_effect(is_fraud):
    if is_fraud:
        st.error("🔊 ALERT SOUND: BEEP BEEP BEEP!")
        st.balloons()  # Visual effect for fraud
    else:
        st.success("🔊 SUCCESS SOUND: DING!")

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏦 Credit Card Fraud Detection System</h1>
    <p>AI-Powered Security for Your Financial Transactions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.markdown("### 🧭 Navigation Panel")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Choose a feature:", [
    "🔍 Predict Transaction", 
    "📊 Train Model", 
    "📈 Transaction History", 
    "🔎 Search Transactions",
    "📋 Analytics",
    "🚨 Fraud Alerts",
    "📱 Quick Test"
])

# Main content
if page == "🔍 Predict Transaction":
    st.header("🔍 Transaction Analysis")
    
    if not fraud_system.load_model():
        st.error("❌ No trained model found. Please train a model first.")
    else:
        st.success("✅ AI Model loaded and ready!")
        
        # Enhanced input form
        with st.form("prediction_form"):
            st.markdown("### 💳 Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💰 Amount**")
                amount_inr = st.number_input("Amount (₹)", min_value=1.0, value=5000.0, step=100.0)
                amount_usd = inr_to_usd(amount_inr)
                st.info(f"💱 Equivalent: ${amount_usd:.2f} USD")
            
            with col2:
                st.markdown("**⏰ Transaction Time**")
                col2a, col2b = st.columns(2)
                with col2a:
                    hours = st.selectbox("Hours", range(0, 24), index=12)
                with col2b:
                    minutes = st.selectbox("Minutes", range(0, 60, 15), index=0)
                
                time_seconds = time_to_seconds(hours, minutes)
                st.info(f"🕐 Time: {hours:02d}:{minutes:02d}")
            
            # Transaction type
            st.markdown("**🏪 Transaction Type**")
            transaction_type = st.selectbox("Select type:", 
                                          ["💳 Online Purchase", "🏧 ATM Withdrawal", "🛒 POS Payment", "📱 UPI Transfer"])
            
            # Submit button with enhanced styling
            submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)
            
            if submitted:
                try:
                    with st.spinner("🤖 AI is analyzing your transaction..."):
                        time.sleep(1)  # Add slight delay for effect
                        result = fraud_system.predict_single_transaction(amount_usd, time_seconds)
                    
                    if result:
                        # Results with enhanced visuals
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        
                        # Metrics in attractive cards
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("💰 Amount", f"₹{amount_inr:,.0f}", f"${result['amount']:.2f}")
                        
                        with col2:
                            risk_pct = result['probability'] * 100
                            st.metric("⚠️ Risk Score", f"{risk_pct:.1f}%")
                        
                        with col3:
                            st.metric("🕐 Time", f"{hours:02d}:{minutes:02d}")
                        
                        with col4:
                            st.metric("🏪 Type", transaction_type.split()[1])
                        
                        # Risk visualization
                        st.markdown("### 🎯 Risk Assessment")
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_pct,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Final verdict with sound effects
                        st.markdown("### 🎭 Final Verdict")
                        
                        if result['prediction'] == 1:
                            st.markdown("""
                            <div class="fraud-alert">
                                <h2>🚨 FRAUD DETECTED!</h2>
                                <p><strong>This transaction appears to be fraudulent!</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            play_sound_effect(True)
                            
                            st.error("""
                            **🚨 IMMEDIATE ACTIONS REQUIRED:**
                            
                            1. 🔒 **Block your card immediately**
                            2. 📞 **Call your bank**: 1800-XXX-XXXX
                            3. 📧 **Report online**: cybercrime.gov.in
                            4. 💬 **SMS 'BLOCK'** to your bank
                            
                            **⚠️ DO NOT:**
                            - Share OTP/PIN with anyone
                            - Click suspicious links
                            - Provide card details over phone
                            """)
                        
                        else:
                            st.markdown("""
                            <div class="legitimate-alert">
                                <h2>✅ TRANSACTION APPROVED!</h2>
                                <p><strong>This transaction appears to be legitimate.</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            play_sound_effect(False)
                            
                            st.success("""
                            **✅ Transaction Security Tips:**
                            
                            - 🔐 Always verify merchant details
                            - 📱 Check SMS alerts regularly  
                            - 🌐 Use secure networks only
                            - 🔔 Enable transaction notifications
                            """)
                        
                        # Additional insights
                        st.markdown("### 💡 Transaction Insights")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if amount_inr > 50000:
                                st.warning("💰 High-value transaction detected")
                            if hours < 6 or hours > 22:
                                st.info("🌙 Unusual timing - Off-hours transaction")
                        
                        with col2:
                            if "ATM" in transaction_type and amount_inr > 25000:
                                st.warning("🏧 Large ATM withdrawal")
                            if "Online" in transaction_type and risk_pct > 30:
                                st.info("🛒 Online purchase - Extra verification recommended")
                    
                except Exception as e:
                    st.error(f"❌ Analysis Error: {e}")

elif page == "📊 Train Model":
    st.header("📊 AI Model Training")
    
    st.info("🤖 Train the AI model with transaction data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            try:
                with st.spinner("🧠 Training AI model..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📂 Loading data...")
                    progress_bar.progress(25)
                    df = fraud_system.ingest_data()
                    
                    status_text.text("🔧 Preprocessing...")
                    progress_bar.progress(50)
                    df = fraud_system.preprocess_data(df)
                    
                    status_text.text("🤖 Training models...")
                    progress_bar.progress(75)
                    results = fraud_system.train_robust_models(df)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training complete!")
                    
                    st.success("🎉 Model trained successfully!")
                    
                    # Results
                    st.subheader("📈 Training Results")
                    for model, metrics in results.items():
                        st.write(f"**{model}**: {metrics['test_auc']:.1%} accuracy")
                        
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    
    with col2:
        st.markdown("""
        **📋 Training Info:**
        - Uses real transaction data
        - Multiple AI algorithms
        - Cross-validation testing
        - Automatic model selection
        """)

elif page == "📱 Quick Test":
    st.header("📱 Quick Transaction Tests")
    
    st.markdown("### 🎮 Test Common Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Normal Transactions")
        
        if st.button("🛒 Shopping ₹2,500 (2 PM)", use_container_width=True):
            result = fraud_system.predict_single_transaction(inr_to_usd(2500), time_to_seconds(14, 0))
            if result:
                if result['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED!")
                    play_sound_effect(True)
                else:
                    st.success("✅ APPROVED")
                    play_sound_effect(False)
        
        if st.button("☕ Coffee ₹350 (10 AM)", use_container_width=True):
            result = fraud_system.predict_single_transaction(inr_to_usd(350), time_to_seconds(10, 0))
            if result:
                if result['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED!")
                    play_sound_effect(True)
                else:
                    st.success("✅ APPROVED")
                    play_sound_effect(False)
    
    with col2:
        st.subheader("⚠️ Suspicious Scenarios")
        
        if st.button("🌙 Late Night ₹50,000 (2 AM)", use_container_width=True):
            result = fraud_system.predict_single_transaction(inr_to_usd(50000), time_to_seconds(2, 0))
            if result:
                if result['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED!")
                    play_sound_effect(True)
                else:
                    st.success("✅ APPROVED")
                    play_sound_effect(False)
        
        if st.button("💰 Large Amount ₹2,00,000 (3 AM)", use_container_width=True):
            result = fraud_system.predict_single_transaction(inr_to_usd(200000), time_to_seconds(3, 0))
            if result:
                if result['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED!")
                    play_sound_effect(True)
                else:
                    st.success("✅ APPROVED")
                    play_sound_effect(False)

# Other pages with INR conversion
elif page == "📈 Transaction History":
    st.header("📈 Transaction History")
    
    try:
        history = fraud_system.get_transaction_history(limit=50)
        
        if history is not None and not history.empty:
            history['amount_inr'] = history['amount'].apply(usd_to_inr)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total", len(history))
            with col2:
                st.metric("💰 Total Amount", f"₹{history['amount_inr'].sum():,.0f}")
            with col3:
                st.metric("🚨 Fraud Count", history['prediction'].sum())
            with col4:
                st.metric("📈 Fraud Rate", f"{history['prediction'].mean():.1%}")
            
            # Enhanced table
            display_history = history.copy()
            display_history['Amount (₹)'] = display_history['amount_inr'].apply(lambda x: f"₹{x:,.0f}")
            display_history['Status'] = display_history['prediction'].apply(lambda x: "🚨 Fraud" if x == 1 else "✅ Safe")
            display_history['Risk'] = display_history['fraud_probability'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_history[['timestamp', 'Amount (₹)', 'Status', 'Risk']], use_container_width=True)
        else:
            st.info("📝 No transactions yet. Make some predictions first!")
            
    except Exception as e:
        st.error("💾 Database error. Run: python fix_database.py")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("🔒 **Secure & Safe**")
with col2:
    st.markdown("🤖 **AI-Powered**")
with col3:
    st.markdown("📞 **24/7 Support**")