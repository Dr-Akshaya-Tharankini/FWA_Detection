import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fwa_engine import FWAPredictionEngine
import io
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Healthcare FWA Detection - UAE",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'summary_stats' not in st.session_state:
    st.session_state.summary_stats = None

# Header
st.markdown('<p class="main-header">Healthcare Claims FWA Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Fraud, Waste & Abuse Detection for UAE Healthcare Claims</p>', unsafe_allow_html=True)
st.markdown("---")

# Function to generate realistic sample data
def generate_realistic_sample_data(n_claims=55):
    """Generate realistic healthcare claims data for UAE"""
    np.random.seed(42)
    
    # Base date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Provider IDs
    provider_ids = ['PR001', 'PR002', 'PR003', 'PR004', 'PR005', 'PR006', 'PR007', 'PR008']
    
    # Procedure codes with realistic distribution
    procedure_codes = ['P001', 'P002', 'P003', 'P004', 'P005', 'P010', 'P011', 
                       'P020', 'P021', 'P030', 'P031', 'P101', 'P102', 'P201', 'P202', 'P203']
    
    # Diagnosis codes
    diagnosis_codes = ['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008']
    
    # Specialties
    specialties = ['Cardiology', 'General', 'Surgery', 'Orthopedics', 'Dermatology', 
                   'Urology', 'Gynecology']
    
    # Patient IDs
    patient_ids = [f'PAT{str(i).zfill(3)}' for i in range(1, 41)]
    
    data = []
    
    for i in range(n_claims):
        claim_id = f'CLM{str(i+1).zfill(4)}'
        provider_id = np.random.choice(provider_ids)
        patient_id = np.random.choice(patient_ids)
        
        # Generate realistic date within 30 days
        days_ago = np.random.randint(0, 30)
        claim_date = end_date - timedelta(days=days_ago)
        
        # Select procedure and related data
        procedure_code = np.random.choice(procedure_codes)
        
        # Determine specialty based on procedure
        if procedure_code in ['P001', 'P010', 'P011']:
            specialty = 'Cardiology'
        elif procedure_code in ['P020', 'P021']:
            specialty = 'Dermatology'
        elif procedure_code in ['P030', 'P031']:
            specialty = 'Orthopedics'
        elif procedure_code in ['P101', 'P102']:
            specialty = 'Urology'
        elif procedure_code in ['P201', 'P202', 'P203']:
            specialty = 'Gynecology'
        elif procedure_code in ['P003', 'P005']:
            specialty = 'Surgery'
        else:
            specialty = 'General'
        
        # Generate claim amount based on procedure
        base_amounts = {
            'P001': 4500, 'P002': 2800, 'P003': 7500, 'P004': 1800, 'P005': 14000,
            'P010': 5500, 'P011': 6500, 'P020': 2200, 'P021': 3200, 'P030': 9500,
            'P031': 7500, 'P101': 11000, 'P102': 7500, 'P201': 8500, 'P202': 6500, 'P203': 3800
        }
        base_amount = base_amounts.get(procedure_code, 4000)
        
        # Add variation (some claims intentionally high for fraud detection)
        if np.random.random() < 0.15:  # 15% abnormally high
            claim_amount = base_amount * np.random.uniform(1.5, 2.5)
        else:
            claim_amount = base_amount * np.random.uniform(0.8, 1.2)
        
        # Patient demographics
        patient_age = np.random.randint(18, 85)
        
        # Gender logic (some mismatches for testing)
        if procedure_code in ['P101', 'P102']:  # Male procedures
            patient_gender = 'M' if np.random.random() > 0.05 else 'F'  # 5% mismatch
        elif procedure_code in ['P201', 'P202', 'P203']:  # Female procedures
            patient_gender = 'F' if np.random.random() > 0.05 else 'M'  # 5% mismatch
        else:
            patient_gender = np.random.choice(['M', 'F'])
        
        # Provider claims in 30 days (some providers with high frequency)
        if provider_id in ['PR001', 'PR005']:  # High frequency providers
            provider_claims_30days = np.random.randint(55, 85)
        else:
            provider_claims_30days = np.random.randint(20, 50)
        
        # Encounter dates
        encounter_start = claim_date
        encounter_end = claim_date + timedelta(days=np.random.randint(0, 3))
        
        diagnosis_code = np.random.choice(diagnosis_codes)
        
        data.append({
            'claim_id': claim_id,
            'provider_id': provider_id,
            'claim_amount': round(claim_amount, 2),
            'procedure_code': procedure_code,
            'diagnosis_code': diagnosis_code,
            'Encounter Start Date': encounter_start.strftime('%Y-%m-%d'),
            'Encounter End Date': encounter_end.strftime('%Y-%m-%d'),
            'provider_claims_30days': provider_claims_30days,
            'patient_id': patient_id,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'claim_date': claim_date.strftime('%Y-%m-%d'),
            'provider_specialty': specialty
        })
    
    # Add some intentional duplicates
    if n_claims >= 5:
        duplicate_indices = np.random.choice(range(len(data)), size=3, replace=False)
        for idx in duplicate_indices:
            duplicate_claim = data[idx].copy()
            duplicate_claim['claim_id'] = f"CLM{str(len(data)+1).zfill(4)}"
            data.append(duplicate_claim)
    
    # Add some same-day visits
    if n_claims >= 10:
        same_day_patient = data[5]['patient_id']
        same_day_date = data[5]['claim_date']
        for _ in range(2):
            same_day_claim = data[5].copy()
            same_day_claim['claim_id'] = f"CLM{str(len(data)+1).zfill(4)}"
            same_day_claim['procedure_code'] = np.random.choice(procedure_codes)
            data.append(same_day_claim)
    
    return pd.DataFrame(data)

# Sidebar
with st.sidebar:
    st.header("📤 Upload Claims Data")
    st.markdown("Upload a CSV file with the following columns:")
    st.code("""
- claim_id
- provider_id
- claim_amount
- procedure_code
- diagnosis_code
- Encounter Start Date
- Encounter End Date
- provider_claims_30days
- patient_id
- patient_age
- patient_gender
- claim_date
- provider_specialty
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    st.markdown("---")
    st.header("📋 System Rules")
    with st.expander("View FWA Detection Rules"):
        st.markdown("""
        **12 Active Rules:**
        1. High Claim Amount (30 pts)
        2. Frequent Billing >50 claims (25 pts)
        3. Unusual Combinations (35 pts)
        4. Duplicate Detection (40 pts)
        5. Outlier Billing >2σ (25 pts)
        6. Weekend/Holiday (10 pts)
        7. Demographic Mismatch (35 pts)
        8. Same-Day Visits (20 pts)
        9. Specialty Mismatch (30 pts)
        10. Excessive Drugs/Supplies (25 pts)
        11. Referral Pattern (15 pts)
        12. Reversal/Rebill (20 pts)
        
        **Risk Classification:**
        - Low (0-30): Auto-approve
        - Medium (31-60): Manual review
        - High (61+): Investigate for FWA
        """)

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully. Total records: {len(df):,}")
        
        # Show data preview
        with st.expander("👁️ Preview Uploaded Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Process button
        if st.button("🔍 Run FWA Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing claims for FWA indicators..."):
                # Initialize engine
                engine = FWAPredictionEngine()
                
                # Process claims
                processed_df = engine.calculate_fwa_score(df)
                
                # Get summary statistics
                summary_stats = engine.get_summary_statistics(processed_df)
                
                # Store in session state
                st.session_state.processed_df = processed_df
                st.session_state.summary_stats = summary_stats
                
                st.success("✅ Analysis complete!")
        
        # Display results if available
        if st.session_state.processed_df is not None:
            processed_df = st.session_state.processed_df
            summary_stats = st.session_state.summary_stats
            
            # Summary metrics
            st.markdown("## 📊 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Claims",
                    f"{summary_stats['total_claims']:,}",
                    help="Total number of claims analyzed"
                )
            
            with col2:
                st.metric(
                    "Total Amount",
                    f"{summary_stats['total_amount_aed']} AED",
                    help="Total claim amount in AED"
                )
            
            with col3:
                st.metric(
                    "Avg Risk Score",
                    summary_stats['avg_risk_score'],
                    help="Average risk score across all claims"
                )
            
            with col4:
                st.metric(
                    "High Risk Amount",
                    f"{summary_stats['high_risk_amount']} AED",
                    delta="Requires investigation",
                    delta_color="inverse",
                    help="Total amount flagged as high risk"
                )
            
            # Additional key metrics
            st.markdown("### 🔍 Key Detection Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Duplicates Detected", summary_stats['duplicate_detected_count'])
            with col2:
                st.metric("Incompatible Procedures", summary_stats['incompatible_procedures_count'])
            with col3:
                st.metric("Same-Day Visits", summary_stats['same_day_visits_total'])
            with col4:
                st.metric("Controlled Substances", summary_stats['controlled_substances_excessive_count'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Suspicious Referrals", summary_stats['suspicious_referral_pattern_count'])
            with col2:
                st.metric("Frequent Reversals", summary_stats['frequent_reversals_count'])
            with col3:
                st.metric("Providers Flagged", f"{summary_stats['providers_flagged']}/{summary_stats['total_providers']}")
            with col4:
                flagged_pct = (summary_stats['providers_flagged'] / summary_stats['total_providers'] * 100) if summary_stats['total_providers'] > 0 else 0
                st.metric("Provider Flag Rate", f"{flagged_pct:.1f}%")
            
            st.markdown("---")
            
            # Risk distribution
            st.markdown("## 📈 Risk Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level pie chart
                risk_dist = pd.DataFrame(
                    list(summary_stats['risk_distribution'].items()),
                    columns=['Risk Level', 'Count']
                )
                fig_risk = px.pie(
                    risk_dist,
                    values='Count',
                    names='Risk Level',
                    title='Claims by Risk Level',
                    color='Risk Level',
                    color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                # Action distribution pie chart
                action_dist = pd.DataFrame(
                    list(summary_stats['action_distribution'].items()),
                    columns=['Action', 'Count']
                )
                fig_action = px.pie(
                    action_dist,
                    values='Count',
                    names='Action',
                    title='Recommended Actions'
                )
                st.plotly_chart(fig_action, use_container_width=True)
            
            # Flag frequency
            st.markdown("## 🚩 FWA Flag Frequency")
            if summary_stats['flag_frequency']:
                flag_df = pd.DataFrame(
                    list(summary_stats['flag_frequency'].items()),
                    columns=['Flag', 'Count']
                ).sort_values('Count', ascending=False)
                
                fig_flags = px.bar(
                    flag_df,
                    x='Count',
                    y='Flag',
                    orientation='h',
                    title='Most Common FWA Flags',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_flags, use_container_width=True)
            else:
                st.info("No flags detected in the analyzed claims")
            
            st.markdown("---")
            
            # Detailed results
            st.markdown("## 📋 Detailed Results")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    options=['Low', 'Medium', 'High'],
                    default=['Low', 'Medium', 'High']
                )
            
            with col2:
                action_filter = st.multiselect(
                    "Filter by Action",
                    options=processed_df['recommended_action'].unique(),
                    default=processed_df['recommended_action'].unique()
                )
            
            with col3:
                min_score = st.number_input("Minimum Risk Score", min_value=0, value=0)
            
            # Apply filters
            filtered_df = processed_df[
                (processed_df['risk_level'].isin(risk_filter)) &
                (processed_df['recommended_action'].isin(action_filter)) &
                (processed_df['risk_score'] >= min_score)
            ]
            
            st.info(f"Showing {len(filtered_df):,} of {len(processed_df):,} claims")
            
            # Display filtered data
            display_columns = [
                'claim_id', 'provider_id', 'claim_amount', 'procedure_code', 'patient_id',
                'risk_score', 'risk_level', 'recommended_action', 'flags', 'flag_reasons',
                'duplicate_detected', 'incompatible_procedures', 'same_day_visits',
                'controlled_substances_excessive', 'suspicious_referral_pattern', 'frequent_reversals'
            ]
            
            # Ensure all columns exist
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[display_columns].sort_values('risk_score', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Download options
            st.markdown("## 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download full results
                csv_full = processed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv_full,
                    file_name="fwa_analysis_full.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download high risk only
                high_risk_df = processed_df[processed_df['risk_level'] == 'High']
                csv_high_risk = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="⚠️ Download High Risk Claims (CSV)",
                    data=csv_high_risk,
                    file_name="fwa_high_risk_claims.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Additional insights
            st.markdown("---")
            st.markdown("## 💡 Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk score distribution
                fig_score_dist = px.histogram(
                    processed_df,
                    x='risk_score',
                    nbins=30,
                    title='Risk Score Distribution',
                    color_discrete_sequence=['#3498db']
                )
                fig_score_dist.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Low/Medium")
                fig_score_dist.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Medium/High")
                st.plotly_chart(fig_score_dist, use_container_width=True)
            
            with col2:
                # Claims by provider specialty
                if 'provider_specialty' in processed_df.columns:
                    specialty_risk = processed_df.groupby('provider_specialty').agg({
                        'risk_score': 'mean',
                        'claim_id': 'count'
                    }).reset_index()
                    specialty_risk.columns = ['Specialty', 'Avg Risk Score', 'Claim Count']
                    
                    fig_specialty = px.scatter(
                        specialty_risk,
                        x='Claim Count',
                        y='Avg Risk Score',
                        size='Claim Count',
                        text='Specialty',
                        title='Average Risk Score by Provider Specialty'
                    )
                    st.plotly_chart(fig_specialty, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file contains all required columns and is properly formatted.")

else:
    # Welcome screen
    st.info("Please upload a CSV file to begin FWA analysis")
    
    st.markdown("## ℹ️ About This System")
    st.markdown("""
    This Healthcare Claims FWA (Fraud, Waste, and Abuse) Detection System analyzes claims data 
    using 12 rule-based checks designed for the UAE healthcare market.
    
    **Key Features:**
    - Real-time fraud detection across 12 rules
    - Risk scoring and classification (Low/Medium/High)
    - Automated recommendations (Auto-approve/Review/Investigate)
    - Comprehensive reporting and analytics
    - Export capabilities for further investigation
    
    **How It Works:**
    1. Upload your claims CSV file
    2. Click "Run FWA Analysis"
    3. Review risk distributions and flagged claims
    4. Export results for action
    
    All amounts are processed in UAE Dirhams (AED).
    """)
    
    # Sample data download
    st.markdown("## 📥 Sample Data")
    st.markdown("Download a realistic sample CSV file to test the system (55+ claims with various FWA patterns):")
    
    # Generate sample data
    sample_df = generate_realistic_sample_data(n_claims=55)
    csv_sample = sample_df.to_csv(index=False)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_sample,
            file_name="sample_claims_realistic.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        st.markdown("""
        **Sample includes:**
        - 55+ realistic healthcare claims
        - Multiple providers and patients
        - Intentional duplicates
        - Same-day visits
        - Gender mismatches
        - High-frequency providers
        - Weekend billings
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>Healthcare FWA Detection System | UAE Edition | Powered by Rule-Based AI</strong>
    </div>
    """,
    unsafe_allow_html=True
)