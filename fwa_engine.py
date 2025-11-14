import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FWAPredictionEngine:
    """
    Healthcare Claims FWA (Fraud, Waste, Abuse) Prediction Engine for UAE
    Implements 12 rule-based checks with scoring and flagging system
    """
    
    def __init__(self):
        # UAE procedure code thresholds (in AED)
        self.procedure_thresholds = {
            'P001': 5000,   # Cardiology
            'P002': 3000,   # General Consultation
            'P003': 8000,   # Surgery
            'P004': 2000,   # Lab Test
            'P005': 15000,  # Advanced Surgery
            'P010': 6000,   # Cardiac Catheterization
            'P011': 7000,   # Angioplasty
            'P020': 2500,   # Skin Biopsy
            'P021': 3500,   # Laser Treatment
            'P030': 10000,  # Hip Replacement
            'P031': 8000,   # Knee Surgery
            'P101': 12000,  # Prostate Surgery
            'P102': 8000,   # Vasectomy
            'P201': 9000,   # Hysterectomy
            'P202': 7000,   # C-Section
            'P203': 4000,   # Mammogram
            'DEFAULT': 5000
        }
        
        # Incompatible procedure combinations
        self.incompatible_procedures = [
            {'P001', 'P030'},  # Cardiology + Orthopedic major surgery
            {'P002', 'P005'},  # General consult + Advanced surgery
            {'P101', 'P201'},  # Male-specific + Female-specific
            {'P020', 'P030'},  # Dermatology + Orthopedic surgery
        ]
        
        # Gender-specific procedures
        self.male_only_procedures = ['P101', 'P102']  # Prostate surgery, Vasectomy
        self.female_only_procedures = ['P201', 'P202', 'P203']  # Hysterectomy, C-Section, Mammogram
        
        # UAE holidays and weekends (Friday-Saturday)
        self.weekend_days = [4, 5]  # Friday=4, Saturday=5
        
        # Specialty-procedure mapping
        self.specialty_procedures = {
            'Cardiology': ['P001', 'P010', 'P011'],
            'Dermatology': ['P020', 'P021'],
            'Orthopedics': ['P030', 'P031'],
            'General': ['P002', 'P004'],
            'Surgery': ['P003', 'P005'],
            'Urology': ['P101', 'P102'],
            'Gynecology': ['P201', 'P202', 'P203']
        }
        
        # Controlled substances (procedure codes that include drugs)
        self.controlled_substance_procedures = ['P004', 'P005', 'P011']
        
    def calculate_fwa_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to calculate FWA scores for all claims
        """
        df = df.copy()
        
        # Initialize score and flags columns
        df['risk_score'] = 0
        df['flags'] = ''
        df['flag_reasons'] = ''
        df['incompatible_procedures'] = False
        df['same_day_visits'] = 0
        df['controlled_substances_excessive'] = False
        df['suspicious_referral_pattern'] = False
        df['frequent_reversals'] = False
        df['duplicate_detected'] = False
        
        # Convert date columns
        date_columns = ['Encounter Start Date', 'Encounter End Date', 'claim_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Apply all rules
        df = self._rule_1_high_claim_amount(df)
        df = self._rule_2_frequent_billing(df)
        df = self._rule_3_unusual_combinations(df)
        df = self._rule_4_duplicate_detection(df)
        df = self._rule_5_outlier_billing(df)
        df = self._rule_6_weekend_holiday(df)
        df = self._rule_7_demographic_mismatch(df)
        df = self._rule_8_same_day_visits(df)
        df = self._rule_9_specialty_mismatch(df)
        df = self._rule_10_excessive_drugs(df)
        df = self._rule_11_referral_pattern(df)
        df = self._rule_12_reversal_rebill(df)
        
        # Classify risk level
        df['risk_level'] = df['risk_score'].apply(self._classify_risk)
        df['recommended_action'] = df['risk_level'].apply(self._recommend_action)
        
        return df
    
    def _add_flag(self, row, flag: str, reason: str, points: int):
        """Helper function to add flags and update score"""
        if row['flags']:
            row['flags'] += f", {flag}"
            row['flag_reasons'] += f" | {reason}"
        else:
            row['flags'] = flag
            row['flag_reasons'] = reason
        row['risk_score'] += points
        return row
    
    def _rule_1_high_claim_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 1: High Claim Amount"""
        def check_high_amount(row):
            threshold = self.procedure_thresholds.get(
                row['procedure_code'], 
                self.procedure_thresholds['DEFAULT']
            )
            if row['claim_amount'] > threshold:
                return self._add_flag(
                    row, 
                    'HIGH_AMOUNT', 
                    f"Claim amount {row['claim_amount']:,.0f} AED exceeds threshold {threshold:,.0f} AED",
                    30
                )
            return row
        
        return df.apply(check_high_amount, axis=1)
    
    def _rule_2_frequent_billing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 2: Frequent Billing (>50 claims in 30 days)"""
        def check_frequency(row):
            if row['provider_claims_30days'] > 50:
                return self._add_flag(
                    row,
                    'HIGH_FREQUENCY',
                    f"Provider submitted {row['provider_claims_30days']} claims in 30 days (threshold: 50)",
                    25
                )
            return row
        
        return df.apply(check_frequency, axis=1)
    
    def _rule_3_unusual_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 3: Unusual Procedure Combinations"""
        def check_combinations(row):
            proc_code = row['procedure_code']
            for incompatible_set in self.incompatible_procedures:
                if proc_code in incompatible_set:
                    row['incompatible_procedures'] = True
                    return self._add_flag(
                        row,
                        'INCOMPATIBLE_PROCEDURES',
                        f"Procedure {proc_code} is part of medically incompatible combination",
                        35
                    )
            return row
        
        return df.apply(check_combinations, axis=1)
    
    def _rule_4_duplicate_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 4: Duplicate Detection"""
        # Mark duplicates based on patient, provider, date, procedure
        df['is_duplicate'] = df.duplicated(
            subset=['patient_id', 'provider_id', 'claim_date', 'procedure_code'], 
            keep='first'
        )
        
        def check_duplicate(row):
            if row['is_duplicate']:
                row['duplicate_detected'] = True
                return self._add_flag(
                    row,
                    'DUPLICATE',
                    f"Duplicate claim: same patient, provider, date, and procedure",
                    40
                )
            return row
        
        df = df.apply(check_duplicate, axis=1)
        df.drop('is_duplicate', axis=1, inplace=True)
        return df
    
    def _rule_5_outlier_billing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 5: Outlier Billing Pattern"""
        if 'provider_specialty' not in df.columns:
            return df
        
        # Calculate mean and std by specialty
        specialty_stats = df.groupby('provider_specialty')['claim_amount'].agg(['mean', 'std']).reset_index()
        df = df.merge(specialty_stats, on='provider_specialty', how='left', suffixes=('', '_specialty'))
        
        def check_outlier(row):
            if pd.notna(row['mean']) and pd.notna(row['std']) and row['std'] > 0:
                z_score = (row['claim_amount'] - row['mean']) / row['std']
                if z_score > 2:
                    return self._add_flag(
                        row,
                        'OUTLIER_PROVIDER',
                        f"Claim amount is {z_score:.2f} standard deviations above specialty average ({row['mean']:,.0f} AED)",
                        25
                    )
            return row
        
        df = df.apply(check_outlier, axis=1)
        df.drop(['mean', 'std'], axis=1, inplace=True, errors='ignore')
        return df
    
    def _rule_6_weekend_holiday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 6: Weekend or Holiday Billing"""
        def check_weekend(row):
            if pd.notna(row['claim_date']):
                weekday = row['claim_date'].weekday()
                if weekday in self.weekend_days:
                    day_name = row['claim_date'].strftime('%A')
                    return self._add_flag(
                        row,
                        'UNUSUAL_DATE',
                        f"Elective procedure billed on {day_name} (UAE weekend)",
                        10
                    )
            return row
        
        return df.apply(check_weekend, axis=1)
    
    def _rule_7_demographic_mismatch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 7: Patient Age or Gender Mismatch"""
        def check_demographics(row):
            proc_code = row['procedure_code']
            gender = str(row['patient_gender']).upper()
            
            # Check gender-specific procedures
            if proc_code in self.male_only_procedures and gender == 'F':
                return self._add_flag(
                    row,
                    'DEMOGRAPHIC_MISMATCH',
                    f"Male-only procedure {proc_code} billed for female patient (ID: {row['patient_id']})",
                    35
                )
            elif proc_code in self.female_only_procedures and gender == 'M':
                return self._add_flag(
                    row,
                    'DEMOGRAPHIC_MISMATCH',
                    f"Female-only procedure {proc_code} billed for male patient (ID: {row['patient_id']})",
                    35
                )
            return row
        
        return df.apply(check_demographics, axis=1)
    
    def _rule_8_same_day_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 8: High Number of Same-Day Visits"""
        # Count same-day visits per patient
        same_day_counts = df.groupby(['patient_id', 'claim_date']).size().reset_index(name='same_day_count')
        df = df.merge(same_day_counts, on=['patient_id', 'claim_date'], how='left')
        
        def check_same_day(row):
            row['same_day_visits'] = row['same_day_count']
            if row['same_day_count'] > 1:
                return self._add_flag(
                    row,
                    'MULTIPLE_SAME_DAY',
                    f"Patient has {row['same_day_count']} claims on {row['claim_date'].strftime('%Y-%m-%d')}",
                    20
                )
            return row
        
        df = df.apply(check_same_day, axis=1)
        df.drop('same_day_count', axis=1, inplace=True, errors='ignore')
        return df
    
    def _rule_9_specialty_mismatch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 9: Provider Specialization Mismatch"""
        def check_specialty(row):
            if 'provider_specialty' in row and pd.notna(row['provider_specialty']):
                specialty = row['provider_specialty']
                proc_code = row['procedure_code']
                
                # Check if procedure is appropriate for specialty
                expected_procs = self.specialty_procedures.get(specialty, [])
                if expected_procs and proc_code not in expected_procs:
                    return self._add_flag(
                        row,
                        'SPECIALTY_MISMATCH',
                        f"Procedure {proc_code} not typical for {specialty} specialist (Provider: {row['provider_id']})",
                        30
                    )
            return row
        
        return df.apply(check_specialty, axis=1)
    
    def _rule_10_excessive_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 10: Excessive High-Risk Drugs or Supplies"""
        def check_drugs(row):
            # Check if claim amount is very high for controlled substance procedures
            if row['procedure_code'] in self.controlled_substance_procedures:
                if row['claim_amount'] > 20000:
                    row['controlled_substances_excessive'] = True
                    return self._add_flag(
                        row,
                        'DRUG_SUPPLY_OUTLIER',
                        f"Exceptionally high amount {row['claim_amount']:,.0f} AED for controlled substance procedure {row['procedure_code']}",
                        25
                    )
            # Check any very high claim
            elif row['claim_amount'] > 25000:
                row['controlled_substances_excessive'] = True
                return self._add_flag(
                    row,
                    'DRUG_SUPPLY_OUTLIER',
                    f"Very high claim amount {row['claim_amount']:,.0f} AED may indicate excessive drugs/supplies",
                    25
                )
            return row
        
        return df.apply(check_drugs, axis=1)
    
    def _rule_11_referral_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 11: Suspicious Referral Pattern"""
        # Count referrals per provider
        if 'provider_id' in df.columns:
            provider_counts = df.groupby('provider_id').size().reset_index(name='provider_claim_count')
            df = df.merge(provider_counts, on='provider_id', how='left')
            
            def check_referral(row):
                # If provider has many claims with same specialty pattern
                if row['provider_claim_count'] > 15:
                    row['suspicious_referral_pattern'] = True
                    return self._add_flag(
                        row,
                        'REFERRAL_PATTERN',
                        f"Provider {row['provider_id']} has {row['provider_claim_count']} claims showing potential referral pattern",
                        15
                    )
                return row
            
            df = df.apply(check_referral, axis=1)
            df.drop('provider_claim_count', axis=1, inplace=True, errors='ignore')
        
        return df
    
    def _rule_12_reversal_rebill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 12: Reversal and Rebill Patterns"""
        # Check for frequent claims from same provider (potential reversal pattern)
        if 'provider_id' in df.columns:
            def check_reversal(row):
                # If provider has very high claim frequency, flag for potential reversal/rebill
                if row['provider_claims_30days'] > 70:
                    row['frequent_reversals'] = True
                    return self._add_flag(
                        row,
                        'REVERSE_REBILL',
                        f"Provider {row['provider_id']} has {row['provider_claims_30days']} claims in 30 days, indicating potential reversal/rebill pattern",
                        20
                    )
                return row
            
            df = df.apply(check_reversal, axis=1)
        
        return df
    
    def _classify_risk(self, score: int) -> str:
        """Classify risk level based on total score"""
        if score <= 30:
            return 'Low'
        elif score <= 60:
            return 'Medium'
        else:
            return 'High'
    
    def _recommend_action(self, risk_level: str) -> str:
        """Recommend action based on risk level"""
        actions = {
            'Low': 'Auto-approve',
            'Medium': 'Manual review',
            'High': 'Investigate for FWA'
        }
        return actions.get(risk_level, 'Unknown')
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the analysis"""
        total_claims = len(df)
        total_amount = df['claim_amount'].sum()
        
        risk_summary = df['risk_level'].value_counts().to_dict()
        action_summary = df['recommended_action'].value_counts().to_dict()
        
        # Flag frequency
        all_flags = []
        for flags in df['flags']:
            if flags:
                all_flags.extend(flags.split(', '))
        
        from collections import Counter
        flag_counts = Counter(all_flags)
        
        # Additional metrics - safely handle missing columns
        incompatible_count = df['incompatible_procedures'].sum() if 'incompatible_procedures' in df.columns else 0
        duplicate_count = df['duplicate_detected'].sum() if 'duplicate_detected' in df.columns else 0
        controlled_substances_count = df['controlled_substances_excessive'].sum() if 'controlled_substances_excessive' in df.columns else 0
        suspicious_referral_count = df['suspicious_referral_pattern'].sum() if 'suspicious_referral_pattern' in df.columns else 0
        frequent_reversals_count = df['frequent_reversals'].sum() if 'frequent_reversals' in df.columns else 0
        same_day_visits_total = df['same_day_visits'].sum() if 'same_day_visits' in df.columns else 0
        
        # Provider metrics
        providers_flagged = df[df['risk_score'] > 0]['provider_id'].nunique() if 'provider_id' in df.columns else 0
        total_providers = df['provider_id'].nunique() if 'provider_id' in df.columns else 0
        
        summary = {
            'total_claims': total_claims,
            'total_amount_aed': f"{total_amount:,.2f}",
            'risk_distribution': risk_summary,
            'action_distribution': action_summary,
            'flag_frequency': dict(flag_counts),
            'avg_risk_score': f"{df['risk_score'].mean():.2f}",
            'high_risk_amount': f"{df[df['risk_level'] == 'High']['claim_amount'].sum():,.2f}",
            'incompatible_procedures_count': int(incompatible_count),
            'duplicate_detected_count': int(duplicate_count),
            'controlled_substances_excessive_count': int(controlled_substances_count),
            'suspicious_referral_pattern_count': int(suspicious_referral_count),
            'frequent_reversals_count': int(frequent_reversals_count),
            'same_day_visits_total': int(same_day_visits_total),
            'providers_flagged': int(providers_flagged),
            'total_providers': int(total_providers)
        }
        
        return summary