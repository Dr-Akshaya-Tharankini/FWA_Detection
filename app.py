import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fwa_engine import FWAPredictionEngine
import io

# Page configuration
st.set_page_config(
    page_title="Healthcare FWA Prediction - UAE",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'summary_stats' not in st.session_state:
    st.session_state.summary_stats = None

# Header
st.markdown('<p class="main-header">Healthcare Claims FWA Prediction System</p>', unsafe_allow_html=True)
st.markdown("**Fraud, Waste & Abuse Detection for UAE Healthcare Claims**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Upload Claims Data")
    st.markdown("Upload a CSV file with the following columns:")
    st.code("""
- claim_id
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
    st.header("System Rules")
    with st.expander("View FWA Detection Rules"):
        st.markdown("""
        **12 Active Rules:**
        1. High Claim Amount (30 pts)
        2. Frequent Billing (25 pts)
        3. Unusual Combinations (35 pts)
        4. Duplicate Detection (40 pts)
        5. Outlier Billing (25 pts)
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
        
        st.success(f"File uploaded successfully. Total records: {len(df):,}")
        
        # Show data preview
        with st.expander("Preview Uploaded Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Process button
        if st.button("Run FWA Analysis", type="primary", use_container_width=True):
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
                
                st.success("Analysis complete!")
        
        # Display results if available
        if st.session_state.processed_df is not None:
            processed_df = st.session_state.processed_df
            summary_stats = st.session_state.summary_stats
            
            # Summary metrics
            st.header("Analysis Summary")
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
            
            st.markdown("---")
            
            # Risk distribution
            st.header("Risk Distribution")
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
            st.header("FWA Flag Frequency")
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
            st.header("Detailed Results")
            
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
                'claim_id', 'claim_amount', 'procedure_code', 'patient_id',
                'risk_score', 'risk_level', 'recommended_action', 'flags', 'flag_reasons'
            ]
            
            # Ensure all columns exist
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[display_columns].sort_values('risk_score', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Download options
            st.header("Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download full results
                csv_full = processed_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results (CSV)",
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
                    label="Download High Risk Claims (CSV)",
                    data=csv_high_risk,
                    file_name="fwa_high_risk_claims.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Additional insights
            st.markdown("---")
            st.header("Additional Insights")
            
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
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file contains all required columns and is properly formatted.")

else:
    # Welcome screen
    st.info("Please upload a CSV file to begin FWA analysis")
    
    st.header("About This System")
    st.markdown("""
    This Healthcare Claims FWA (Fraud, Waste, and Abuse) Prediction System analyzes claims data 
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
    st.header("Sample Data")
    st.markdown("Download a sample CSV file to test the system:")
    
    sample_data = {
        'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
        'claim_amount': [3000, 8500, 2500, 15000, 4200],
        'procedure_code': ['P001', 'P003', 'P002', 'P005', 'P001'],
        'diagnosis_code': ['D001', 'D002', 'D003', 'D004', 'D001'],
        'Encounter Start Date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
        'Encounter End Date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
        'provider_claims_30days': [45, 65, 30, 55, 48],
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'patient_age': [45, 32, 67, 55, 41],
        'patient_gender': ['M', 'F', 'M', 'F', 'M'],
        'claim_date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
        'provider_specialty': ['Cardiology', 'General', 'Orthopedics', 'General', 'Cardiology']
    }
    
    sample_df = pd.DataFrame(sample_data)
    csv_sample = sample_df.to_csv(index=False)
    
    st.download_button(
        label="Download Sample CSV",
        data=csv_sample,
        file_name="sample_claims.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
    Healthcare FWA Prediction System | UAE Edition | Powered by Rule-Based AI
    </div>
    """,
    unsafe_allow_html=True
)