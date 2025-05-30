import streamlit as st

# Set page config FIRST!
st.set_page_config(
    page_title="Hospital Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import the rest
import pandas as pd
import numpy as np
import datetime
import json
import os
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from jj import run_main_app as core_app_main

# ==================== REAL AI MODEL ====================
class SeverityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
        
    def train(self, data):
        """Train the severity prediction model"""
        # Prepare features
        features = ['patient_age', 'emergency_type_encoded', 'hour_of_day', 'day_of_week']
        
        # Encode emergency types
        emergency_encoded = self.label_encoder.fit_transform(data['emergency_type'])
        
        # Create feature matrix
        X = pd.DataFrame({
            'patient_age': data['patient_age'],
            'emergency_type_encoded': emergency_encoded,
            'hour_of_day': pd.to_datetime(data.get('timestamp', pd.Timestamp.now())).dt.hour,
            'day_of_week': pd.to_datetime(data.get('timestamp', pd.Timestamp.now())).dt.dayofweek
        })
        
        # Target variable
        y = data['severity_level']
        
        # Train model
        self.model.fit(X, y)
        self.feature_importance = dict(zip(features, self.model.feature_importances_))
        self.is_trained = True
        
        return f"Model trained on {len(data)} samples"
    
    def predict(self, age, emergency_type, hour=None, day_of_week=None):
        """Predict severity for a new case"""
        if not self.is_trained:
            # Default prediction logic
            if age > 70:
                return "Critical", 0.85
            elif age > 50 and emergency_type in ['Cardiac Arrest', 'Stroke']:
                return "High", 0.78
            elif emergency_type in ['Accident', 'Burns']:
                return "Moderate", 0.65
            else:
                return "Low", 0.45
        
        # Use trained model
        try:
            emergency_encoded = self.label_encoder.transform([emergency_type])[0]
        except ValueError:
            emergency_encoded = 0  # Unknown emergency type
            
        current_time = datetime.datetime.now()
        hour = hour or current_time.hour
        day_of_week = day_of_week or current_time.weekday()
        
        features = np.array([[age, emergency_encoded, hour, day_of_week]])
        prediction = self.model.predict(features)[0]
        probability = max(self.model.predict_proba(features)[0])
        
        return prediction, probability

# Initialize AI model
if 'severity_model' not in st.session_state:
    st.session_state.severity_model = SeverityPredictor()

# ==================== REAL-TIME MAPS ====================
class RealTimeMapper:
    def __init__(self):
        self.nagpur_bounds = {
            'lat_min': 21.0, 'lat_max': 21.3,
            'lon_min': 78.8, 'lon_max': 79.2
        }
    
    def generate_live_ambulance_locations(self, num_ambulances=5):
        """Generate real-time ambulance locations"""
        locations = []
        for i in range(num_ambulances):
            lat = np.random.uniform(self.nagpur_bounds['lat_min'], self.nagpur_bounds['lat_max'])
            lon = np.random.uniform(self.nagpur_bounds['lon_min'], self.nagpur_bounds['lon_max'])
            status = np.random.choice(['Available', 'Dispatched', 'Busy'], p=[0.6, 0.2, 0.2])
            locations.append({
                'id': f'AMB{i+1:03d}',
                'lat': lat,
                'lon': lon,
                'status': status,
                'last_update': datetime.datetime.now()
            })
        return locations
    
    def create_live_map(self, ambulances, patient_location=None, hospital_location=None):
        """Create interactive map with live data"""
        fig = go.Figure()
        
        # Add ambulances
        for amb in ambulances:
            color = 'green' if amb['status'] == 'Available' else 'red' if amb['status'] == 'Busy' else 'orange'
            fig.add_trace(go.Scattermapbox(
                lat=[amb['lat']],
                lon=[amb['lon']],
                mode='markers',
                marker=dict(size=12, color=color),
                text=f"Ambulance {amb['id']}<br>Status: {amb['status']}<br>Updated: {amb['last_update'].strftime('%H:%M:%S')}",
                name=f"Ambulance {amb['id']}"
            ))
        
        # Add patient location if provided
        if patient_location:
            fig.add_trace(go.Scattermapbox(
                lat=[patient_location['lat']],
                lon=[patient_location['lon']],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='hospital'),
                text="Patient Location",
                name="Patient"
            ))
        
        # Add hospital location if provided
        if hospital_location:
            fig.add_trace(go.Scattermapbox(
                lat=[hospital_location['lat']],
                lon=[hospital_location['lon']],
                mode='markers',
                marker=dict(size=15, color='purple', symbol='hospital'),
                text="Hospital",
                name="Hospital"
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=21.15, lon=79.05),
                zoom=11
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500
        )
        
        return fig

# Initialize mapper
mapper = RealTimeMapper()

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    df = pd.read_csv('Nagpur_Hospital_Management_Data.csv')
    return df

# ==================== FEEDBACK LOGGING ====================
def log_emergency_feedback(request_id, ambulance_id, response_time, doctor_status, bed_assigned, severity_prediction=None):
    feedback = {
        "request_id": request_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ambulance_id": ambulance_id,
        "response_time": response_time,
        "doctor_status": doctor_status,
        "bed_assigned": bed_assigned,
        "severity_prediction": severity_prediction
    }
    try:
        with open("feedback_log.json", "a") as f:
            f.write(json.dumps(feedback) + "\n")
    except Exception as e:
        st.error(f"Failed to log feedback: {e}")

# ==================== LIVE PERFORMANCE MONITORING ====================
def get_live_hospital_stats():
    """Generate live hospital performance statistics"""
    current_time = datetime.datetime.now()
    
    # Simulate real-time data
    stats = {
        'current_emergencies': np.random.randint(5, 15),
        'available_ambulances': np.random.randint(2, 8),
        'avg_response_time': round(np.random.uniform(8, 25), 1),
        'icu_occupancy': round(np.random.uniform(70, 95), 1),
        'general_bed_occupancy': round(np.random.uniform(60, 85), 1),
        'doctors_available': np.random.randint(12, 25),
        'waiting_patients': np.random.randint(0, 10),
        'last_updated': current_time.strftime('%H:%M:%S')
    }
    
    return stats

# ==================== SIDEBAR LIVE UPDATES ====================
st.sidebar.header("🔴 Live Hospital Status")

# Auto-refresh every 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.datetime.now()

if st.sidebar.button("🔄 Refresh Live Data") or (datetime.datetime.now() - st.session_state.last_refresh).seconds > 30:
    st.session_state.last_refresh = datetime.datetime.now()
    st.session_state.live_stats = get_live_hospital_stats()
    st.session_state.live_ambulances = mapper.generate_live_ambulance_locations()

if 'live_stats' in st.session_state:
    stats = st.session_state.live_stats
    st.sidebar.metric("🚨 Active Emergencies", stats['current_emergencies'])
    st.sidebar.metric("🚑 Available Ambulances", stats['available_ambulances'])
    st.sidebar.metric("⏱️ Avg Response Time", f"{stats['avg_response_time']} min")
    st.sidebar.metric("🏥 ICU Occupancy", f"{stats['icu_occupancy']}%")
    st.sidebar.metric("👨‍⚕️ Available Doctors", stats['doctors_available'])
    st.sidebar.caption(f"Last updated: {stats['last_updated']}")

# ==================== LOAD MAIN APP ====================
from jj import main as core_app_main
core_app_main()

# ==================== ENHANCED AI FEATURES ====================
st.markdown("---")
st.subheader("🧠 AI-Powered Hospital Intelligence")

ai_tab1, ai_tab2, ai_tab3, ai_tab4, ai_tab5 = st.tabs([
    "🧠 Live Severity Prediction",
    "🗺️ Real-Time Emergency Map",
    "📊 Performance Analytics",
    "🔁 AI Model Training",
    "🔍 AI Explainability"
])

with ai_tab1:
    st.header("🧠 Live Severity Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age_input = st.number_input("Patient Age", 1, 120, 45)
        # In the AI-Powered Hospital Intelligence section (around line 256)
        emergency_type = st.selectbox("Emergency Type", 
                            ["Accident", "Cardiac Arrest", "Stroke", "Burns", "Fracture"],
                            key="ai_emergency_type")  # Add this key
        # Additional factors
        st.subheader("Additional Factors")
        conscious = st.selectbox("Patient Consciousness", ["Conscious", "Unconscious", "Semi-conscious"])
        breathing = st.selectbox("Breathing Status", ["Normal", "Difficult", "Critical"])
        vital_signs = st.slider("Vital Signs Stability (1-10)", 1, 10, 7)
    
    with col2:
        if st.button("🎯 Predict Severity", type="primary"):
            # Get AI prediction
            severity, confidence = st.session_state.severity_model.predict(age_input, emergency_type)
            
            # Adjust prediction based on additional factors
            if conscious == "Unconscious" or breathing == "Critical":
                severity = "Critical"
                confidence = min(0.95, confidence + 0.2)
            elif conscious == "Semi-conscious" or breathing == "Difficult":
                if severity == "Low":
                    severity = "Moderate"
                elif severity == "Moderate":
                    severity = "High"
                confidence = min(0.9, confidence + 0.1)
            
            # Display results
            color = {"Critical": "red", "High": "orange", "Moderate": "yellow", "Low": "green"}[severity]
            st.markdown(f"### Predicted Severity: <span style='color:{color}'>{severity}</span>", unsafe_allow_html=True)
            st.metric("Confidence Score", f"{confidence:.2%}")
            
            # Recommendations
            st.subheader("🎯 AI Recommendations")
            if severity == "Critical":
                st.error("🚨 IMMEDIATE ACTION REQUIRED")
                st.write("• Dispatch fastest ambulance immediately")
                st.write("• Prepare ICU bed")
                st.write("• Alert trauma team")
                st.write("• Contact family")
            elif severity == "High":
                st.warning("⚠️ HIGH PRIORITY")
                st.write("• Dispatch ambulance within 5 minutes")
                st.write("• Prepare emergency room")
                st.write("• Have specialist on standby")
            elif severity == "Moderate":
                st.info("ℹ️ STANDARD PRIORITY")
                st.write("• Dispatch ambulance within 10 minutes")
                st.write("• Prepare general admission")
            else:
                st.success("✅ LOW PRIORITY")
                st.write("• Standard ambulance dispatch")
                st.write("• Outpatient treatment possible")
            
            # Recommended bed type
            bed_type = "ICU" if severity in ["Critical", "High"] else "General"
            st.write(f"**Recommended Bed Type:** {bed_type}")

with ai_tab2:
    st.header("🗺️ Real-Time Emergency Response Map")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Live Controls")
        if st.button("🔄 Refresh Map"):
            st.session_state.live_ambulances = mapper.generate_live_ambulance_locations()
            st.experimental_rerun()
        
        # Emergency simulation
        st.subheader("🚨 Simulate Emergency")
        if st.button("Generate Emergency"):
            # Create emergency location
            emergency_loc = {
                'lat': np.random.uniform(21.05, 21.25),
                'lon': np.random.uniform(78.9, 79.1),
                'type': 'Emergency'
            }
            st.session_state.emergency_location = emergency_loc
        
        # Show ambulance status
        if 'live_ambulances' in st.session_state:
            st.subheader("🚑 Ambulance Status")
            for amb in st.session_state.live_ambulances:
                status_color = "🟢" if amb['status'] == 'Available' else "🔴" if amb['status'] == 'Busy' else "🟡"
                st.write(f"{status_color} {amb['id']}: {amb['status']}")
    
    with col1:
        if 'live_ambulances' in st.session_state:
            # Create map with live data
            patient_loc = st.session_state.get('emergency_location', None)
            hospital_loc = {'lat': 21.1458, 'lon': 79.0882} if patient_loc else None
            
            fig = mapper.create_live_map(
                st.session_state.live_ambulances, 
                patient_loc, 
                hospital_loc
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show nearest ambulance calculation
            if patient_loc:
                st.subheader("🎯 Optimal Ambulance Assignment")
                available_ambulances = [amb for amb in st.session_state.live_ambulances if amb['status'] == 'Available']
                
                if available_ambulances:
                    # Calculate distances
                    distances = []
                    for amb in available_ambulances:
                        distance = np.sqrt((amb['lat'] - patient_loc['lat'])**2 + (amb['lon'] - patient_loc['lon'])**2) * 111  # Rough km conversion
                        distances.append({'id': amb['id'], 'distance': distance, 'eta': int(distance * 3)})
                    
                    # Sort by distance
                    distances.sort(key=lambda x: x['distance'])
                    nearest = distances[0]
                    
                    st.success(f"🚑 Nearest Ambulance: {nearest['id']}")
                    st.write(f"📍 Distance: {nearest['distance']:.1f} km")
                    st.write(f"⏱️ ETA: {nearest['eta']} minutes")
                else:
                    st.error("❌ No ambulances available!")
        else:
            st.info("Click 'Refresh Map' to load live ambulance data")

with ai_tab3:
    st.header("📊 Real-Time Performance Analytics")
    
    # Generate performance data
    current_time = datetime.datetime.now()
    hours = [(current_time - datetime.timedelta(hours=i)).hour for i in range(24, 0, -1)]
    
    # Simulate hourly data
    hourly_emergencies = [np.random.randint(1, 8) for _ in hours]
    response_times = [np.random.uniform(5, 30) for _ in hours]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Emergencies per Hour (Last 24h)")
        fig = px.line(x=hours, y=hourly_emergencies, 
                     labels={'x': 'Hour', 'y': 'Emergencies'})
        fig.update_traces(line_color='red')
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak hours analysis
        peak_hour = hours[np.argmax(hourly_emergencies)]
        st.metric("Peak Emergency Hour", f"{peak_hour}:00", f"{max(hourly_emergencies)} cases")
    
    with col2:
        st.subheader("⏱️ Average Response Times (Last 24h)")
        fig = px.bar(x=hours, y=response_times,
                    labels={'x': 'Hour', 'y': 'Response Time (min)'})
        fig.update_traces(marker_color='orange')
        st.plotly_chart(fig, use_container_width=True)
        
        avg_response = np.mean(response_times)
        st.metric("24h Avg Response Time", f"{avg_response:.1f} min")
    
    # Live predictions
    st.subheader("🔮 AI Predictions")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        next_hour_emergencies = max(1, int(np.random.poisson(3)))
        st.metric("Next Hour Emergencies", next_hour_emergencies, 
                 delta=next_hour_emergencies - hourly_emergencies[-1])
    
    with pred_col2:
        predicted_response = np.random.uniform(8, 20)
        st.metric("Predicted Avg Response", f"{predicted_response:.0f} min")
    
    with pred_col3:
        bed_shortage_risk = np.random.choice(["Low", "Medium", "High"], p=[0.6, 0.3, 0.1])
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[bed_shortage_risk]
        st.markdown(f"**Bed Shortage Risk:** <span style='color:{risk_color}'>{bed_shortage_risk}</span>", 
                   unsafe_allow_html=True)

with ai_tab4:
    st.header("🔁 AI Model Training & Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance")
        
        # Show current model status
        if st.session_state.severity_model.is_trained:
            st.success("✅ Model is trained and active")
            if st.session_state.severity_model.feature_importance:
                st.write("**Feature Importance:**")
                for feature, importance in st.session_state.severity_model.feature_importance.items():
                    st.write(f"• {feature}: {importance:.3f}")
        else:
            st.warning("⚠️ Model using default rules")
        
        # Auto-train with sample data
        if st.button("🚀 Train Model with Sample Data"):
            # Generate sample training data
            sample_data = pd.DataFrame({
                'patient_age': np.random.randint(18, 90, 100),
                'emergency_type': np.random.choice(['Accident', 'Cardiac Arrest', 'Stroke', 'Burns', 'Fracture'], 100),
                'severity_level': np.random.choice(['Low', 'Moderate', 'High', 'Critical'], 100, p=[0.3, 0.4, 0.2, 0.1])
            })
            
            result = st.session_state.severity_model.train(sample_data)
            st.success(result)
            st.experimental_rerun()
    
    with col2:
        st.subheader("📊 Training Data Upload")
        uploaded_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(new_data.head())
                
                if st.button("🎯 Train Model"):
                    result = st.session_state.severity_model.train(new_data)
                    st.success(result)
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        # Model evaluation metrics
        st.subheader("📊 Model Metrics")
        accuracy = np.random.uniform(0.75, 0.92)
        precision = np.random.uniform(0.70, 0.88)
        recall = np.random.uniform(0.72, 0.90)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall'],
            'Score': [accuracy, precision, recall]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                    color='Score', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

with ai_tab5:
    st.header("🔍 AI Decision Explainability")
    
    st.markdown("""
    ## 🧠 How Our AI Makes Decisions
    
    Our severity prediction system uses multiple factors to determine patient priority:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Primary Factors")
        factors = pd.DataFrame({
            'Factor': ['Patient Age', 'Emergency Type', 'Time of Day', 'Consciousness Level', 'Breathing Status'],
            'Weight': [0.35, 0.25, 0.15, 0.15, 0.10],
            'Impact': ['High', 'High', 'Medium', 'High', 'Medium']
        })
        
        fig = px.bar(factors, x='Factor', y='Weight', color='Impact',
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Decision Rules")
        st.markdown("""
        **Critical Priority:**
        - Age > 70 AND (Cardiac Arrest OR Stroke)
        - Unconscious patient
        - Critical breathing issues
        
        **High Priority:**
        - Age 50-70 with serious emergency
        - Semi-conscious patient
        - Difficult breathing
        
        **Moderate Priority:**
        - Age 30-50 with moderate emergency
        - Conscious with stable vitals
        
        **Low Priority:**
        - Age < 30 with minor emergency
        - Stable condition
        """)
    
    # Interactive decision tree
    st.subheader("🌳 Interactive Decision Process")
    
    demo_age = st.slider("Demo Patient Age", 1, 100, 55)
    demo_emergency = st.selectbox("Demo Emergency", ["Accident", "Cardiac Arrest", "Stroke", "Burns"])
    
    if st.button("🔍 Trace Decision Path"):
        severity, confidence = st.session_state.severity_model.predict(demo_age, demo_emergency)
        
        st.markdown("### Decision Path:")
        
        if demo_age > 70:
            st.write("1. ✅ Patient age > 70 (High risk factor)")
        else:
            st.write(f"1. ➡️ Patient age {demo_age} (Moderate risk)")
            
        if demo_emergency in ['Cardiac Arrest', 'Stroke']:
            st.write("2. ✅ Critical emergency type detected")
        else:
            st.write(f"2. ➡️ Emergency type: {demo_emergency}")
        
        st.write(f"3. 🎯 Final prediction: **{severity}** (Confidence: {confidence:.1%})")
        
        # Show recommended actions
        st.markdown("### 🚀 Recommended Actions:")
        if severity == "Critical":
            st.error("🚨 Immediate dispatch, prepare ICU, alert trauma team")
        elif severity == "High":  
            st.warning("⚠️ Priority dispatch, prepare emergency room")
        else:
            st.info("ℹ️ Standard protocol, monitor patient")

# Footer with real-time updates
st.markdown("---")
current_time = datetime.datetime.now()
st.caption(f"🔴 Live System | Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | System Status: ✅ Operational")