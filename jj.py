import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import random
from geopy.distance import geodesic
import plotly.express as px
import matplotlib.pyplot as plt
# Load and prepare data
@st.cache_data
def load_data():
    # For demonstration, we'll use the provided CSV data
    # In a real application, you might load this from a database
    df = pd.read_csv('Nagpur_Hospital_Management_Data.csv')
    return df

# Define location coordinates (used by multiple components)
location_coords = {
    'Ambazari': (21.1334, 79.0502),
    'Civil Lines': (21.1458, 79.0882),
    'Dharampeth': (21.1387, 79.0568),
    'Medical Square': (21.1535, 79.1004),
    'Sadar': (21.1510, 79.0755),
    'Sitabuldi': (21.1462, 79.0780),
    'Wardhaman Nagar': (21.1692, 79.1078)
}

# AMBULANCE SYSTEM FUNCTIONS ================================================

# Function to calculate distance between two locations
def calculate_distance(loc1, loc2, location_coords):
    if loc1 in location_coords and loc2 in location_coords:
        return geodesic(location_coords[loc1], location_coords[loc2]).kilometers
    return 10  # Default distance if coordinates not found

# Function to find nearest available ambulance
def find_nearest_ambulance(patient_location, ambulances_df, location_coords):
    available_ambulances = ambulances_df[ambulances_df['ambulance_status'] == 'Available']
    
    if available_ambulances.empty:
        # If no ambulances are available, try to find one that's just completed a task
        available_ambulances = ambulances_df[ambulances_df['ambulance_status'] == 'Completed']
        
    if available_ambulances.empty:
        return None, None, None
    
    # Calculate distances from patient to each ambulance
    distances = []
    for _, ambulance in available_ambulances.iterrows():
        amb_location = ambulance['ambulance_location']
        distance = calculate_distance(patient_location, amb_location, location_coords)
        distances.append((ambulance['ambulance_id'], distance, ambulance['ambulance_location']))
    
    # Sort by distance
    if distances:
        distances.sort(key=lambda x: x[1])
        return distances[0]  # Return the closest ambulance
    
    return None, None, None

# Function to estimate response time
def estimate_response_time(distance):
    # Simple estimation: assume 2 minutes per km + 5 minutes base time
    return int(distance * 2 + 5)

# Function to estimate ETA
def estimate_eta(distance):
    # Assume average speed of 30 km/h in city traffic
    return int(distance * 2)  # minutes

# Function to dispatch ambulance
def dispatch_ambulance(ambulance_id, ambulances_df):
    # Update ambulance status
    ambulances_df.loc[ambulances_df['ambulance_id'] == ambulance_id, 'ambulance_status'] = 'Dispatched'
    return ambulances_df

# DOCTOR AGENT FUNCTIONS ===================================================

class DoctorManagementAgent:
    def __init__(self, data):
        self.data = data
        self.doctors = self.extract_doctors()
    
    def extract_doctors(self):
        # Extract unique doctors with their specialties and availability status
        doctors_df = self.data[['doctor_name', 'specialty', 'doctor_availability']].drop_duplicates()
        return doctors_df
    
    def get_available_doctors(self, specialty=None):
        """Filter doctors by specialty and availability"""
        doctors = self.doctors.copy()
        if specialty:
            doctors = doctors[doctors['specialty'] == specialty]
        # Get doctors that are not on leave
        available_docs = doctors[doctors['doctor_availability'] != 'On Leave']
        return available_docs
    
    def schedule_appointment(self, request_id, specialty, preferred_time=None):
        """Schedule an appointment with a doctor based on specialty and availability"""
        available_doctors = self.get_available_doctors(specialty)
        
        if available_doctors.empty:
            return {
                "status": "Pending",
                "message": f"No {specialty} available currently. Request placed on queue.",
                "doctor_name": None,
                "appointment_time": None,
                "eta": None
            }
        
        # Get doctor who is available (not busy if possible)
        if 'Available' in available_doctors['doctor_availability'].values:
            selected_doctor = available_doctors[available_doctors['doctor_availability'] == 'Available'].iloc[0]
        else:
            selected_doctor = available_doctors.iloc[0]  # Take busy doctor if no available one
        
        # Generate appointment time (would be more complex in real system)
        current_time = datetime.datetime.now()
        if preferred_time:
            appointment_time = preferred_time
        else:
            # Generate time in the next 3 hours
            appointment_time = (current_time + datetime.timedelta(hours=np.random.randint(1, 4))).strftime('%H:%M')
        
        # Get ETA if doctor is external (simulation)
        eta = np.random.randint(10, 30) if np.random.random() > 0.5 else None
        
        # Determine status (Confirmed, Rescheduled, Pending)
        status_options = ['Confirmed', 'Rescheduled'] if selected_doctor['doctor_availability'] == 'Available' else ['Pending']
        status = np.random.choice(status_options)
        
        return {
            "request_id": request_id,
            "status": status,
            "doctor_name": selected_doctor['doctor_name'],
            "specialty": selected_doctor['specialty'],
            "doctor_availability": selected_doctor['doctor_availability'],
            "appointment_time": appointment_time,
            "eta": eta
        }

# BED MANAGEMENT FUNCTIONS =================================================

def update_bed_status(bed_id, new_status, patient_name=None):
    df = st.session_state.data
    bed_index = df[df['bed_id'] == bed_id].index
    if len(bed_index) > 0:
        df.at[bed_index[0], 'bed_status'] = new_status
        if patient_name:
            st.session_state.bookings.append({
                "timestamp": datetime.datetime.now(),
                "bed_id": bed_id,
                "patient_name": patient_name,
                "status": new_status
            })
        return True
    return False

def get_metrics():
    df = st.session_state.data
    total = len(df)
    available = len(df[df['bed_status'] == 'Available'])
    occupied = total - available
    by_type = df.groupby('bed_type')['bed_id'].count().to_dict()
    available_by_type = df[df['bed_status'] == 'Available'].groupby('bed_type')['bed_id'].count().to_dict()
    
    return {
        'total': total,
        'available': available,
        'occupied': occupied,
        'by_type': by_type,
        'available_by_type': available_by_type,
        'occupancy_rate': round((occupied / total) * 100 if total > 0 else 0, 1)
    }

# MAIN APPLICATION =========================================================

def main():
    st.title("🏥 Hospital Management System")
    
    # Initialize session state if not already done
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
        st.session_state.bookings = []
        st.session_state.last_refresh = datetime.datetime.now()
    
    # Create tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Emergency Response", 
        "👨‍⚕️ Doctor Management", 
        "🛏️ Bed Management", 
        "📊 Analytics Dashboard"
    ])
    
    # EMERGENCY RESPONSE TAB ===============================================
    with tab1:
        st.header("Emergency Response System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            request_id = f"REQ{random.randint(101, 999)}"
            st.text_input("Request ID", request_id, disabled=True)
            
            patient_name = st.text_input("Patient Name", "")
            patient_age = st.number_input("Patient Age", 1, 120, 35)
            patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
            
            emergency_type = st.selectbox(
                "Emergency Type",
                ["Accident", "Cardiac Arrest", "Stroke", "Burns", "Fracture"]
            )
            
            patient_location = st.selectbox(
                "Patient Location",
                list(location_coords.keys())
            )
            
            severity_level = st.selectbox(
                "Severity Level",
                ["Critical", "High", "Moderate", "Low"]
            )
        
        with col2:
            nearest_hospital = st.selectbox(
                "Nearest Hospital",
                list(location_coords.keys())
            )
            
            # Optional preferred doctor
            preferred_doctor = st.text_input("Preferred Doctor (Optional)", "")
            
            # Additional information
            additional_info = st.text_area("Additional Information", "")
            
            # Contact information
            contact_number = st.text_input("Contact Number", "")
            
        # Submit button
        if st.button("Dispatch Emergency Response", type="primary"):
            with st.spinner("Coordinating emergency response..."):
                # Create a copy of the data for our working dataset
                ambulances_df = st.session_state.data[['ambulance_id', 'ambulance_status', 'ambulance_location']].copy()
                
                # 1. Find and dispatch ambulance
                ambulance_id, distance, amb_location = find_nearest_ambulance(
                    patient_location, ambulances_df, location_coords
                )
                
                if ambulance_id:
                    # Calculate response time and ETAs
                    response_time = estimate_response_time(distance)
                    eta_patient = estimate_eta(distance)
                    
                    # Calculate distance from patient to hospital
                    hospital_distance = calculate_distance(
                        patient_location, nearest_hospital, location_coords
                    )
                    eta_hospital = estimate_eta(hospital_distance)
                    
                    # Update ambulance status
                    ambulances_df = dispatch_ambulance(ambulance_id, ambulances_df)
                    
                    # 2. Schedule doctor appointment
                    doctor_agent = DoctorManagementAgent(st.session_state.data)
                    specialty = "Cardiologist" if emergency_type == "Cardiac Arrest" else "General Physician"
                    doctor_result = doctor_agent.schedule_appointment(request_id, specialty)
                    
                    # 3. Book a bed
                    bed_type = "ICU" if severity_level in ["Critical", "High"] else "General"
                    available_beds = st.session_state.data[
                        (st.session_state.data['bed_status'] == 'Available') & 
                        (st.session_state.data['bed_type'] == bed_type)
                    ]
                    bed_assigned = None
                    if not available_beds.empty:
                        bed_assigned = available_beds.iloc[0]['bed_id']
                        update_bed_status(bed_assigned, "Occupied", patient_name)
                    
                    # Display response
                    st.success("Emergency response coordinated successfully!")
                    
                    # Ambulance details
                    with st.expander("🚑 Ambulance Dispatch Details", expanded=True):
                        result_cols = st.columns(5)
                        result_cols[0].metric("Ambulance ID", ambulance_id)
                        result_cols[1].metric("ETA to Patient (min)", eta_patient)
                        result_cols[2].metric("ETA to Hospital (min)", eta_hospital)
                        result_cols[3].metric("Response Time (min)", response_time)
                        result_cols[4].metric("Status", "Dispatched")
                        
                        # Create and display a map of the route
                        st.subheader("Ambulance Route")
                        route_df = pd.DataFrame([
                            {"location": amb_location, "type": "Ambulance", "lat": location_coords[amb_location][0], "lon": location_coords[amb_location][1]},
                            {"location": patient_location, "type": "Patient", "lat": location_coords[patient_location][0], "lon": location_coords[patient_location][1]},
                            {"location": nearest_hospital, "type": "Hospital", "lat": location_coords[nearest_hospital][0], "lon": location_coords[nearest_hospital][1]}
                        ])
                        
                        fig = px.scatter_mapbox(
                            route_df, 
                            lat="lat", 
                            lon="lon", 
                            color="type",
                            size_max=15, 
                            zoom=12,
                            mapbox_style="carto-positron",
                            hover_name="location"
                        )
                        
                        # Add lines connecting the points
                        fig.add_trace(
                            px.line_mapbox(
                                lat=[location_coords[amb_location][0], location_coords[patient_location][0], location_coords[nearest_hospital][0]],
                                lon=[location_coords[amb_location][1], location_coords[patient_location][1], location_coords[nearest_hospital][1]],
                                mapbox_style="carto-positron"
                            ).data[0]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Doctor appointment details
                    with st.expander("👨‍⚕️ Doctor Appointment Details"):
                        if doctor_result['status'] == "Pending":
                            st.warning(f"Doctor appointment is {doctor_result['status']}")
                        else:
                            st.success(f"Doctor appointment is {doctor_result['status']}")
                        
                        doc_col1, doc_col2 = st.columns(2)
                        with doc_col1:
                            st.write("**Appointment Details:**")
                            st.write(f"Request ID: {doctor_result['request_id']}")
                            st.write(f"Doctor: {doctor_result['doctor_name']}")
                            st.write(f"Specialty: {doctor_result['specialty']}")
                        
                        with doc_col2:
                            st.write("**Schedule Information:**")
                            st.write(f"Status: {doctor_result['status']}")
                            st.write(f"Time: {doctor_result['appointment_time']}")
                            if doctor_result['eta']:
                                st.write(f"Doctor ETA: {doctor_result['eta']} minutes")
                            st.write(f"Doctor Status: {doctor_result['doctor_availability']}")
                    
                    # Bed assignment details
                    with st.expander("🛏️ Bed Assignment Details"):
                        if bed_assigned:
                            st.success(f"Bed {bed_assigned} assigned successfully!")
                            bed_info = st.session_state.data[st.session_state.data['bed_id'] == bed_assigned].iloc[0]
                            st.write(f"**Bed Type:** {bed_info['bed_type']}")
                            st.write(f"**Hospital:** {bed_info['nearest_hospital']}")
                            st.write(f"**Status:** Occupied")
                        else:
                            st.error("No beds currently available for this patient")
                
                else:
                    st.error("No ambulances currently available. Please try again in a few minutes.")
    
    # DOCTOR MANAGEMENT TAB ================================================
    with tab2:
        st.header("Doctor Management System")
        
        # Initialize agent
        doctor_agent = DoctorManagementAgent(st.session_state.data)
        
        # Display two sub-tabs
        subtab1, subtab2 = st.tabs(["Schedule Appointment", "Doctor Availability"])
        
        with subtab1:
            st.subheader("Schedule Doctor Appointment")
            
            # Form for appointment request
            with st.form("appointment_form"):
                request_id = st.text_input("Request ID", value="REQ" + str(np.random.randint(1000, 9999)))
                
                # Get unique specialties from the data
                specialties = sorted(doctor_agent.doctors['specialty'].unique())
                specialty = st.selectbox("Specialty Required", specialties)
                
                # Time preference (optional)
                use_preferred_time = st.checkbox("Set preferred time")
                preferred_time = None
                if use_preferred_time:
                    preferred_time = st.time_input("Preferred Time", value=datetime.datetime.now())
                    preferred_time = preferred_time.strftime('%H:%M')
                
                submit_button = st.form_submit_button("Schedule Appointment")
                
                if submit_button:
                    result = doctor_agent.schedule_appointment(request_id, specialty, preferred_time)
                    
                    # Display result in a nice format
                    if result['status'] == "Pending":
                        st.warning(f"Appointment {result['status']}!")
                    else:
                        st.success(f"Appointment {result['status']}!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Appointment Details:**")
                        st.write(f"Request ID: {result['request_id']}")
                        st.write(f"Doctor: {result['doctor_name']}")
                        st.write(f"Specialty: {result['specialty']}")
                    
                    with col2:
                        st.write("**Schedule Information:**")
                        st.write(f"Status: {result['status']}")
                        st.write(f"Time: {result['appointment_time']}")
                        if result['eta']:
                            st.write(f"Doctor ETA: {result['eta']} minutes")
                        st.write(f"Doctor Status: {result['doctor_availability']}")
        
        with subtab2:
            st.subheader("Doctor Availability Dashboard")
            
            # Filter options
            specialty_filter = st.selectbox(
                "Filter by Specialty", 
                ["All"] + sorted(doctor_agent.doctors['specialty'].unique()),
                key="doctor_specialty_filter"
            )
            
            availability_filter = st.multiselect(
                "Filter by Availability",
                ["Available", "Busy", "On Leave"],
                default=["Available", "Busy"],
                key="doctor_availability_filter"
            )
            
            # Filter doctors based on selection
            filtered_doctors = doctor_agent.doctors.copy()
            if specialty_filter != "All":
                filtered_doctors = filtered_doctors[filtered_doctors['specialty'] == specialty_filter]
            
            if availability_filter:
                filtered_doctors = filtered_doctors[filtered_doctors['doctor_availability'].isin(availability_filter)]
            
            # Display doctors
            if not filtered_doctors.empty:
                st.dataframe(
                    filtered_doctors,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "doctor_name": "Doctor Name",
                        "specialty": "Specialty",
                        "doctor_availability": st.column_config.SelectboxColumn(
                            "Status",
                            help="Current availability status",
                            width="medium",
                            options=["Available", "Busy", "On Leave"],
                            required=True,
                        )
                    }
                )
            else:
                st.warning("No doctors match the selected filters")
    
    # BED MANAGEMENT TAB ===================================================
    with tab3:
        st.header("Hospital Bed Management System")
        
        # Display sub-tabs for bed management
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "Dashboard", 
            "Bed Availability", 
            "Book a Bed", 
            "Bed Management"
        ])
        
        with subtab1:
            st.subheader("Bed Management Dashboard")
            metrics = get_metrics()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Beds", metrics['total'])
            col2.metric("Available Beds", metrics['available'])
            col3.metric("Occupied Beds", metrics['occupied'])
            col4.metric("Occupancy Rate", f"{metrics['occupancy_rate']}%")
            
            # Bed availability chart
            st.subheader("Bed Availability by Type")
            bed_types = list(metrics['by_type'].keys())
            avail_counts = [metrics['available_by_type'].get(bt, 0) for bt in bed_types]
            occupied_counts = [metrics['by_type'][bt] - metrics['available_by_type'].get(bt, 0) for bt in bed_types]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(bed_types, avail_counts, label='Available', color='green')
            ax.bar(bed_types, occupied_counts, bottom=avail_counts, label='Occupied', color='red')
            ax.set_ylabel('Number of Beds')
            ax.set_title('Bed Availability by Type')
            ax.legend()
            st.pyplot(fig)
            
            # Recent bookings
            st.subheader("Recent Activity")
            if st.session_state.bookings:
                bookings_df = pd.DataFrame(st.session_state.bookings[-5:])
                bookings_df['timestamp'] = bookings_df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                st.dataframe(bookings_df)
            else:
                st.info("No recent booking activity.")
        
        with subtab2:
            st.subheader("Current Bed Availability")
            df = st.session_state.data
            
            # Filters in one row
            col1, col2, col3 = st.columns(3)
            bed_type = col1.selectbox("Bed Type", ["All Types"] + sorted(df['bed_type'].unique().tolist()), key="bed_type_filter")
            status = col2.selectbox("Status", ["All Statuses", "Available", "Occupied"], key="bed_status_filter")
            hospital = col3.selectbox("Hospital", ["All Hospitals"] + sorted(df['nearest_hospital'].unique().tolist()), key="hospital_filter")
            
            # Filter the data
            filtered = df.copy()
            if bed_type != "All Types":
                filtered = filtered[filtered['bed_type'] == bed_type]
            if status != "All Statuses":
                filtered = filtered[filtered['bed_status'] == status]
            if hospital != "All Hospitals":
                filtered = filtered[filtered['nearest_hospital'] == hospital]
            
            # Display beds
            st.write(f"Showing {len(filtered)} beds")
            display_cols = ['bed_id', 'bed_type', 'bed_status', 'bed_Price', 'nearest_hospital']
            st.dataframe(filtered[display_cols], use_container_width=True)
            
            # Summary
            st.subheader("Summary")
            summary = filtered.groupby(['bed_type', 'bed_status']).size().unstack(fill_value=0)
            if 'Available' not in summary.columns:
                summary['Available'] = 0
            if 'Occupied' not in summary.columns:
                summary['Occupied'] = 0
            summary['Total'] = summary.sum(axis=1)
            summary['Available %'] = (summary['Available'] / summary['Total'] * 100).round(1)
            st.dataframe(summary)
        
        with subtab3:
            st.subheader("Book a Hospital Bed")
            
            # Simple booking form
            with st.form("booking_form"):
                col1, col2 = st.columns(2)
                patient_name = col1.text_input("Patient Name", key="bed_patient_name")
                patient_age = col1.number_input("Patient Age", 0, 120, 30, key="bed_patient_age")
                patient_gender = col2.selectbox("Gender", ["Male", "Female", "Other"], key="bed_patient_gender")
                emergency = col2.selectbox("Emergency Type", sorted(st.session_state.data['emergency_type'].unique()), key="bed_emergency_type")
                
                col1, col2 = st.columns(2)
                bed_type = col1.selectbox("Bed Type", sorted(st.session_state.data['bed_type'].unique()), key="bed_type_select")
                hospital = col1.selectbox("Hospital", ["Any"] + sorted(st.session_state.data['nearest_hospital'].unique()), key="bed_hospital_select")
                severity = col2.select_slider("Severity", ["Low", "Moderate", "High", "Critical"], key="bed_severity")
                insurance = col2.selectbox("Insurance", ["Yes", "No"], key="bed_insurance")
                
                search = st.form_submit_button("Search for Beds")
            
            if search and patient_name:
                # Find available beds
                available = st.session_state.data[(st.session_state.data['bed_status'] == 'Available') & 
                                                (st.session_state.data['bed_type'] == bed_type)]
                if hospital != "Any":
                    available = available[available['nearest_hospital'] == hospital]
                
                if len(available) > 0:
                    st.success(f"Found {len(available)} available {bed_type} beds")
                    
                    # Select bed
                    bed_options = [f"Bed {row['bed_id']} - {row['bed_type']} - ₹{row['bed_Price']} - {row['nearest_hospital']}" 
                                for _, row in available.sort_values('bed_Price').iterrows()]
                    selected = st.selectbox("Select a bed:", range(len(bed_options)), format_func=lambda i: bed_options[i], key="bed_selection")
                    selected_bed = available.sort_values('bed_Price').iloc[selected]
                    
                    # Book bed
                    if st.button("Confirm Booking", key="confirm_bed_booking"):
                        if update_bed_status(selected_bed['bed_id'], "Occupied", patient_name):
                            st.success(f"Bed {selected_bed['bed_id']} booked for {patient_name}")
                            
                            # Show booking details
                            booking = {
                                "Patient": patient_name,
                                "Age": patient_age,
                                "Bed ID": selected_bed['bed_id'],
                                "Hospital": selected_bed['nearest_hospital'],
                                "Price": f"₹{selected_bed['bed_Price']}",
                                "Booked On": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.json(booking)
                        else:
                            st.error("Failed to book the bed")
                else:
                    st.error(f"No {bed_type} beds available")
                    if hospital != "Any":
                        any_beds = st.session_state.data[(st.session_state.data['bed_status'] == 'Available') & 
                                                        (st.session_state.data['bed_type'] == bed_type)]
                        if len(any_beds) > 0:
                            st.info(f"{len(any_beds)} {bed_type} beds available in other hospitals")
                            if st.button("Show all available beds", key="show_all_beds"):
                                st.dataframe(any_beds[['bed_id', 'nearest_hospital', 'bed_Price']])
        
        with subtab4:
            st.subheader("Bed Management")
            
            subtab_a, subtab_b = st.tabs(["Update Status", "Booking History"])
            
            with subtab_a:
                col1, col2 = st.columns(2)
                bed_id = col1.selectbox("Bed ID", sorted(st.session_state.data['bed_id'].unique()), key="bed_id_select")
                new_status = col2.selectbox("New Status", ["Available", "Occupied", "Under Maintenance"], key="bed_new_status")
                
                # Current info
                bed_info = st.session_state.data[st.session_state.data['bed_id'] == bed_id].iloc[0]
                st.write(f"Current Status: **{bed_info['bed_status']}** | Type: {bed_info['bed_type']} | Hospital: {bed_info['nearest_hospital']}")
                
                if st.button("Update Status", key="update_bed_status"):
                    if update_bed_status(bed_id, new_status):
                        st.success(f"Bed {bed_id} status updated to {new_status}")
                    else:
                        st.error("Failed to update status")
            
            with subtab_b:
                if st.session_state.bookings:
                    bookings_df = pd.DataFrame(st.session_state.bookings)
                    bookings_df['timestamp'] = bookings_df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                    st.dataframe(bookings_df)
                else:
                    st.info("No booking history available")
    
    # ANALYTICS DASHBOARD TAB ==============================================
    with tab4:
        st.header("Hospital Analytics Dashboard")
        
        # Sample analytics from the dataset
        emergency_types = st.session_state.data['emergency_type'].value_counts()
        severity_distribution = st.session_state.data['severity_level'].value_counts()
        avg_response_time = st.session_state.data['response_time_minutes'].mean()
        
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Total Ambulances", len(st.session_state.data['ambulance_id'].unique()))
        metrics_cols[1].metric("Total Doctors", len(st.session_state.data['doctor_name'].unique()))
        metrics_cols[2].metric("Avg. Response Time (min)", f"{avg_response_time:.1f}")
        metrics_cols[3].metric("Critical Cases", len(st.session_state.data[st.session_state.data['severity_level'] == 'Critical']))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emergency Types")
            fig = px.bar(
                x=emergency_types.index,
                y=emergency_types.values,
                labels={'x': 'Emergency Type', 'y': 'Count'},
                color=emergency_types.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Severity Distribution")
            # Order severity levels from Critical to Low
            severity_order = ['Critical', 'High', 'Moderate', 'Low']
            sorted_severity = pd.Series({level: severity_distribution.get(level, 0) for level in severity_order})
            
            fig = px.pie(
                values=sorted_severity.values,
                names=sorted_severity.index,
                color=sorted_severity.index,
                color_discrete_map={
                    'Critical': '#FF5252',
                    'High': '#FFA726',
                    'Moderate': '#FFEE58',
                    'Low': '#66BB6A'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Response time by location
        st.subheader("Average Response Time by Location")
        response_by_location = st.session_state.data.groupby('patient_location')['response_time_minutes'].mean().reset_index()
        response_by_location = response_by_location.sort_values('response_time_minutes', ascending=False)
        
        fig = px.bar(
            response_by_location,
            x='patient_location',
            y='response_time_minutes',
            color='response_time_minutes',
            labels={'patient_location': 'Location', 'response_time_minutes': 'Avg. Response Time (min)'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Doctor availability
        st.subheader("Doctor Availability by Specialty")
        doctor_availability = st.session_state.data.groupby(['specialty', 'doctor_availability']).size().unstack(fill_value=0)
        fig = px.bar(
            doctor_availability,
            barmode='group',
            labels={'value': 'Count', 'specialty': 'Specialty'},
            color_discrete_map={
                'Available': 'green',
                'Busy': 'orange',
                'On Leave': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_main_app():
    main()
