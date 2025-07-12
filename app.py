import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4f46e5;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4f46e5;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .offer-card {
        background: #f8fafc;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .high-risk {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    .medium-risk {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    .low-risk {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing objects."""
    try:
        with open('models/churn_prediction_model.pkl', 'rb') as f:
            model = cloudpickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = cloudpickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = cloudpickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = cloudpickle.load(f)
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the training notebook first. Error: {e}")
        return None, None, None, None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = {}

def generate_offers(churn_probability, customer_features=None):
    """Generate personalized offers based on churn probability and customer features."""
    offers = []
    
    if churn_probability >= 0.8:
        # High risk customers - aggressive retention
        offers = [
            "🎯 URGENT: 50% discount on next 3 months + Free premium services",
            "🎁 Exclusive loyalty bonus: $100 credit + Priority customer support",
            "📞 Personal account manager assigned for better service",
            "🔄 Flexible contract terms with no early termination fees",
            "🎉 Free upgrade to premium plan for 6 months"
        ]
        risk_level = "Very High"
        risk_color = "high-risk"
    elif churn_probability >= 0.6:
        # Medium-high risk - strong incentives
        offers = [
            "💰 30% discount on monthly charges for next 6 months",
            "🎁 Free additional services (Streaming, Security, etc.)",
            "📱 Latest device upgrade at 50% off",
            "🔄 Switch to annual plan with 20% savings",
            "🎯 Personalized service package based on your usage"
        ]
        risk_level = "High"
        risk_color = "high-risk"
    elif churn_probability >= 0.4:
        # Medium risk - moderate incentives
        offers = [
            "💡 20% discount on next month's bill",
            "🎁 Free premium feature trial for 3 months",
            "📞 Enhanced customer support priority",
            "🔄 Flexible billing options available",
            "🎯 Customized service recommendations"
        ]
        risk_level = "Medium"
        risk_color = "medium-risk"
    elif churn_probability >= 0.2:
        # Low-medium risk - engagement offers
        offers = [
            "⭐ 10% loyalty discount available",
            "🎁 Try our new premium features free for 1 month",
            "📱 Exclusive early access to new services",
            "🎯 Refer a friend and get $25 credit",
            "💡 Service optimization consultation"
        ]
        risk_level = "Low-Medium"
        risk_color = "medium-risk"
    else:
        # Low risk - engagement and upselling
        offers = [
            "🌟 Thank you for being a loyal customer!",
            "🎁 Explore our premium add-on services",
            "🔄 Consider upgrading for better value",
            "🎯 Refer friends and earn rewards",
            "💡 Service enhancement opportunities available"
        ]
        risk_level = "Low"
        risk_color = "low-risk"
    
    return offers, risk_level, risk_color

def main():
    # App title
    st.markdown('<h1 class="main-header">📞 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model artifacts
    model, scaler, label_encoders, feature_names = load_model_artifacts()
    
    if model is None:
        st.error("Please run the training notebook first to generate the required model files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Customer Survey", "Prediction Results", "Business Insights"]
    )
    
    if page == "Customer Survey":
        show_customer_survey(label_encoders, feature_names)
    elif page == "Prediction Results":
        show_prediction_results(model, scaler, label_encoders, feature_names)
    else:
        show_business_insights()

def show_customer_survey(label_encoders, feature_names):
    st.markdown('<h2 class="sub-header">📋 Customer Information Survey</h2>', unsafe_allow_html=True)
    st.write("Please fill out the following information to assess churn risk and receive personalized offers.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Demographics")
        
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            help="Customer's gender"
        )
        
        senior_citizen = st.selectbox(
            "Senior Citizen",
            options=["No", "Yes"],
            help="Is the customer a senior citizen (65+)?"
        )
        
        partner = st.selectbox(
            "Partner",
            options=["No", "Yes"],
            help="Does the customer have a partner?"
        )
        
        dependents = st.selectbox(
            "Dependents",
            options=["No", "Yes"],
            help="Does the customer have dependents?"
        )
        
        st.subheader("📞 Phone Services")
        
        phone_service = st.selectbox(
            "Phone Service",
            options=["No", "Yes"],
            help="Does the customer have phone service?"
        )
        
        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=["No", "Yes", "No phone service"],
            help="Does the customer have multiple phone lines?"
        )
        
        st.subheader("🌐 Internet Services")
        
        internet_service = st.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"],
            help="Type of internet service"
        )
        
        online_security = st.selectbox(
            "Online Security",
            options=["No", "Yes", "No internet service"],
            help="Does the customer have online security service?"
        )
        
        online_backup = st.selectbox(
            "Online Backup",
            options=["No", "Yes", "No internet service"],
            help="Does the customer have online backup service?"
        )
    
    with col2:
        st.subheader("🛡️ Additional Services")
        
        device_protection = st.selectbox(
            "Device Protection",
            options=["No", "Yes", "No internet service"],
            help="Does the customer have device protection?"
        )
        
        tech_support = st.selectbox(
            "Tech Support",
            options=["No", "Yes", "No internet service"],
            help="Does the customer have tech support?"
        )
        
        streaming_tv = st.selectbox(
            "Streaming TV",
            options=["No", "Yes", "No internet service"],
            help="Does the customer have streaming TV service?"
        )
        
        streaming_movies = st.selectbox(
            "Streaming Movies",
            options=["No", "Yes", "No internet service"],
            help="Does the customer have streaming movies service?"
        )
        
        st.subheader("💳 Account Information")
        
        contract = st.selectbox(
            "Contract",
            options=["Month-to-month", "One year", "Two year"],
            help="Type of contract"
        )
        
        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=["No", "Yes"],
            help="Does the customer use paperless billing?"
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            help="Customer's payment method"
        )
        
        st.subheader("💰 Financial Information")
        
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            help="Number of months the customer has been with the company"
        )
        
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=18.0,
            max_value=120.0,
            value=65.0,
            step=0.5,
            help="Monthly charges in dollars"
        )
        
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=9000.0,
            value=float(monthly_charges * tenure),
            step=1.0,
            help="Total charges in dollars"
        )
    
    # Store customer data in session state
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    st.session_state.customer_data = customer_data
    
    # Predict button
    if st.button("🔮 Predict Churn Risk & Get Offers", type="primary"):
        st.session_state.prediction_made = True
        st.success("✅ Data collected! Please go to 'Prediction Results' to see your churn risk assessment and personalized offers.")

def show_prediction_results(model, scaler, label_encoders, feature_names):
    st.markdown('<h2 class="sub-header">🔮 Churn Prediction Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_made or not st.session_state.customer_data:
        st.warning("⚠️ Please complete the customer survey first!")
        return
    
    # Prepare data for prediction
    customer_data = st.session_state.customer_data.copy()
    
    # Create DataFrame
    df_input = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    for col in df_input.columns:
        if col in label_encoders and col != 'Churn':
            if col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
                continue  # Skip numerical columns
            df_input[col] = label_encoders[col].transform(df_input[col])
    
    # Ensure columns are in the correct order
    df_input = df_input[feature_names]
    
    # Scale the features (Naive Bayes doesn't need scaling but we'll prepare it anyway)
    df_scaled = scaler.transform(df_input)
    
    # Make prediction
    churn_probability = model.predict_proba(df_input)[0][1]
    churn_prediction = model.predict(df_input)[0]
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <h2>Churn Risk Assessment</h2>
            <h1>{churn_probability:.1%}</h1>
            <h3>Likelihood to Churn</h3>
            <p>Status: {'❌ HIGH RISK' if churn_prediction == 1 else '✅ LOW RISK'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate and display offers
        offers, risk_level, risk_color = generate_offers(churn_probability, customer_data)
        
        st.markdown(f'<h3 class="sub-header">🎁 Personalized Retention Offers</h3>', unsafe_allow_html=True)
        st.markdown(f"**Risk Level: {risk_level}**")
        
        for i, offer in enumerate(offers, 1):
            st.markdown(f"""
            <div class="offer-card {risk_color}">
                <strong>Offer {i}:</strong> {offer}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk %"},
            delta = {'reference': 26.5},  # Average churn rate
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 60], 'color': "orange"},
                    {'range': [60, 80], 'color': "lightcoral"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer summary
        st.subheader("📊 Customer Profile")
        st.write(f"**Tenure:** {customer_data['tenure']} months")
        st.write(f"**Monthly Charges:** ${customer_data['MonthlyCharges']:.2f}")
        st.write(f"**Contract:** {customer_data['Contract']}")
        st.write(f"**Internet Service:** {customer_data['InternetService']}")
        st.write(f"**Payment Method:** {customer_data['PaymentMethod']}")
    
    # Feature importance visualization (if available)
    if hasattr(model, 'feature_importances_'):
        st.markdown('<h3 class="sub-header">📈 Key Factors Influencing Prediction</h3>', unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance.tail(10), 
            x='Importance', 
            y='Feature',
            title="Top 10 Most Important Features",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Action recommendations
    st.markdown('<h3 class="sub-header">💡 Recommended Actions</h3>', unsafe_allow_html=True)
    
    if churn_probability >= 0.6:
        st.error("""
        **Immediate Action Required:**
        - Contact customer within 24 hours
        - Assign dedicated account manager
        - Offer immediate incentives
        - Schedule retention call
        """)
    elif churn_probability >= 0.4:
        st.warning("""
        **Monitor Closely:**
        - Schedule check-in call within 1 week
        - Send targeted offers via email
        - Monitor usage patterns
        - Provide proactive support
        """)
    else:
        st.success("""
        **Engagement Opportunities:**
        - Send satisfaction survey
        - Offer service upgrades
        - Include in loyalty program
        - Request referrals
        """)

def show_business_insights():
    st.markdown('<h2 class="sub-header">📊 Business Insights & Analytics</h2>', unsafe_allow_html=True)
    
    # Model performance summary
    st.subheader("🤖 Model Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best Model",
            "Naive Bayes",
            delta="Highest F1-Score"
        )
    
    with col2:
        st.metric(
            "F1-Score",
            "60.42%",
            delta="2.29% vs Gradient Boosting"
        )
    
    with col3:
        st.metric(
            "ROC AUC",
            "82.04%",
            delta="Good discrimination"
        )
    
    with col4:
        st.metric(
            "Training Time",
            "0.008s",
            delta="Very fast"
        )
    
    # Key insights
    st.subheader("🔍 Key Business Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **High-Risk Customer Characteristics:**
        - Month-to-month contract customers
        - Fiber optic internet service users
        - Electronic check payment method
        - Short tenure (< 12 months)
        - High monthly charges without contract commitment
        """)
        
        st.markdown("""
        **Protective Factors:**
        - Long-term contracts (1-2 years)
        - Automatic payment methods
        - Multiple service subscriptions
        - Longer tenure (> 12 months)
        """)
    
    with insights_col2:
        st.markdown("""
        **Recommended Retention Strategies:**
        - Target month-to-month customers with contract incentives
        - Improve fiber optic service quality/pricing
        - Promote automatic payment adoption
        - Bundle services for better value proposition
        - Implement early intervention for new customers
        """)
        
        st.markdown("""
        **Expected Business Impact:**
        - Potential 15-20% reduction in churn rate
        - Improved customer lifetime value
        - More targeted marketing spend
        - Proactive customer service
        """)
    
    # Cost-benefit analysis
    st.subheader("💰 Cost-Benefit Analysis")
    
    st.markdown("""
    **Retention Offer Costs vs. Customer Value:**
    - Average customer lifetime value: $2,000-$3,000
    - Cost of acquisition: $200-$400 per customer
    - Retention offer costs: $50-$500 (depending on risk level)
    - **ROI of retention efforts: 300-500%**
    """)
    
    # Implementation roadmap
    st.subheader("🗺️ Implementation Roadmap")
    
    st.markdown("""
    **Phase 1 (Month 1-2):** Deploy prediction system
    - Integrate with customer database
    - Train customer service team
    - Set up automated alerts
    
    **Phase 2 (Month 3-4):** Optimize offers
    - A/B test different offer types
    - Refine risk thresholds
    - Measure retention impact
    
    **Phase 3 (Month 5-6):** Scale and improve
    - Expand to all customer segments
    - Implement real-time scoring
    - Develop predictive maintenance alerts
    """)

if __name__ == "__main__":
    main()
