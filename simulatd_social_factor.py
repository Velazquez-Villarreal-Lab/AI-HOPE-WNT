import pandas as pd
import numpy as np

def categorize_samples(df):
    """Categorize samples into groups A, B, and C based on OS_MONTHS."""
    threshold_A = df['OS_MONTHS'].quantile(0.75)
    threshold_C = df['OS_MONTHS'].quantile(0.25)
    
    def assign_group(os_months):
        if os_months >= threshold_A:
            return 'A'
        elif os_months <= threshold_C:
            return 'C'
        else:
            return 'B'
    
    df['Group'] = df['OS_MONTHS'].apply(assign_group)
    return df

def assign_social_factor(df, dict_A, dict_B, dict_C,factor_name):
    """Assign social factor values based on sampling from provided dictionaries."""
    def sample_factor(group):
        if group == 'A':
            return np.random.choice(list(dict_A.keys()), p=list(dict_A.values()))
        elif group == 'B':
            return np.random.choice(list(dict_B.keys()), p=list(dict_B.values()))
        elif group == 'C':
            return np.random.choice(list(dict_C.keys()), p=list(dict_C.values()))
    
    df[factor_name] = df['Group'].apply(sample_factor)
    return df

def main():
    # Load data
    df = pd.read_csv("data/TCGA_COAD/pt_xena_metadata.tsv", sep="\t")
    
    # Ensure OS_MONTHS is numeric
    df['OS_MONTHS'] = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
    
    # Categorize samples into groups
    df = categorize_samples(df)
    ######################
    ##  alcohol_use 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Never": 0.5, "Rarely":0.3 , "Occasionally": 0.1, "Frequently": 0.1 , "Heavy":0.0}
    dict_B = {"Never": 0.2, "Rarely":0.2 , "Occasionally": 0.2, "Frequently": 0.2 , "Heavy":0.2}
    dict_C = {"Never": 0.0, "Rarely":0.1 , "Occasionally": 0.1, "Frequently": 0.3 , "Heavy":0.5}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "alcohol_use")
    
    ######################
    ##  marital_status 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Married": 0.5, "Single": 0.1, "Divorced": 0.1, "Widowed":0.1, "Separated":0.1, "Other":0.1}
    dict_B = {"Married": 0.2, "Single": 0.2, "Divorced": 0.2, "Widowed":0.2, "Separated":0.1, "Other":0.1}
    dict_C = {"Married": 0.1, "Single": 0.1, "Divorced": 0.5, "Widowed":0.1, "Separated":0.1, "Other":0.1}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "marital_status")

    ######################
    ##  financial_strain 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"No_financial_issues":0.5, "Mild_strain":0.3, "Moderate_strain":0.1, "Severe_strain":0.1, "Unable_to_afford_care":0}
    dict_B = {"No_financial_issues":0.2, "Mild_strain":0.2, "Moderate_strain":0.2, "Severe_strain":0.2, "Unable_to_afford_care":0.2}
    dict_C = {"No_financial_issues":0, "Mild_strain":0.1, "Moderate_strain":0.1, "Severe_strain":0.3, "Unable_to_afford_care":0.5}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "financial_strain")


    ######################
    ##  social_support 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Strong":0.5, "Moderate":0.3, "Limited":0.2, "No_support":0}
    dict_B = {"Strong":0.25, "Moderate":0.25, "Limited":0.25, "No_support":0.25}
    dict_C = {"Strong":0.0, "Moderate":0.2, "Limited":0.3, "No_support":0.5}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "social_support")

    ######################
    ##  social_isolation 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Never":0.5, "Rarely":0.3, "Sometimes":0.2, "Often":0, "Always":0}
    dict_B = {"Never":0.2, "Rarely":0.2, "Sometimes":0.2, "Often":0.2, "Always":0.2}
    dict_C = {"Never":0.0, "Rarely":0.0, "Sometimes":0.2, "Often":0.3, "Always":0.5}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "social_isolation")


    ######################
    ##  food_insecurity 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"No_issues":0.7, "Sometimes_insufficient":0.3, "Often_insufficient":0}
    dict_B = {"No_issues":0.3, "Sometimes_insufficient":0.4, "Often_insufficient":0.3}
    dict_C = {"No_issues":0.0, "Sometimes_insufficient":0.3, "Often_insufficient":0.7}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "food_insecurity")

    ######################
    ##  race_ethnicity 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Non_Hispanic_White":0.5, "Hispanic_Latino":0.1, "Black_African_American":0.1, "Asian":0.1, "Native_American":0.1, "Pacific_Islander":0.1, "Other":0}
    dict_B = {"Non_Hispanic_White":0.2, "Hispanic_Latino":0.2, "Black_African_American":0.2, "Asian":0.2, "Native_American":0.2, "Pacific_Islander":0.0, "Other":0}
    dict_C = {"Non_Hispanic_White":0.2, "Hispanic_Latino":0.2, "Black_African_American":0.2, "Asian":0.2, "Native_American":0.2, "Pacific_Islander":0.0, "Other":0}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "race_ethnicity")

    ######################
    ##  gender 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Male":0.2, "Female":0.7, "Non_binary":0.05, "Prefer_not_to_say":0.05}
    dict_B = {"Male":0.5, "Female":0.4, "Non_binary":0.05, "Prefer_not_to_say":0.05}
    dict_C = {"Male":0.7, "Female":0.2, "Non_binary":0.05, "Prefer_not_to_say":0.05}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "gender")

    
    ######################
    ##  screening_adherence 
    ######################

    # Define dictionaries with probability distributions
    dict_A = {"Completed_on_time":0.7, "Delayed":0.2, "Never_completed":0.1}
    dict_B = {"Completed_on_time":0.5, "Delayed":0.25, "Never_completed":0.25}
    dict_C = {"Completed_on_time":0.2, "Delayed":0.3, "Never_completed":0.5}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "screening_adherence")
    
    ######################
    ##  employment_status 
    ######################

    # Define dictionaries with probability distributions
    dict_A = { "Employed_full_time":0.6, "Employed_part_time":0.2, "Unemployed":0.1, "Retired":0.1, "Disabled":0, "Student":0}
    dict_B = {"Employed_full_time":0.4, "Employed_part_time":0.2, "Unemployed":0.2, "Retired":0.1, "Disabled":0.05, "Student":0.05}
    dict_C = {"Employed_full_time":0.2, "Employed_part_time":0.2, "Unemployed":0.4, "Retired":0.1, "Disabled":0.05, "Student":0.05}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "employment_status")

    ######################
    ##  healthcare_access 
    ######################

    # Define dictionaries with probability distributions
    dict_A = { "Easy_access":0.7, "Moderate_difficulty":0.2, "Significant_difficulty":0.1, "No_access":0}
    dict_B = {"Easy_access":0.4, "Moderate_difficulty":0.2, "Significant_difficulty":0.2, "No_access":0.2}
    dict_C = {"Easy_access":0.2, "Moderate_difficulty":0.2, "Significant_difficulty":0.2, "No_access":0.4}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "healthcare_access")


    ######################
    ##  pain_burden 
    ######################

    # Define dictionaries with probability distributions
    dict_A = { "No":0.6, "Mild":0.3, "Moderate":0.1, "Severe":0}
    dict_B = {"No":0.3, "Mild":0.3, "Moderate":0.3, "Severe":0.1}
    dict_C = {"No":0.1, "Mild":0.1, "Moderate":0.3, "Severe":0.5}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "pain_burden")

    ######################
    ##  health_literacy 
    ######################

    # Define dictionaries with probability distributions
    dict_A = { "High":0.7, "Moderate":0.3, "Low":0}
    dict_B = {"High":0.4, "Moderate":0.3, "Low":0.3}
    dict_C = {"High":0.1, "Moderate":0.3, "Low":0.6}
    
    df = assign_social_factor(df, dict_A, dict_B, dict_C, "health_literacy")

    # Save the modified data
    df.to_csv("data/SocialFactor_COAD/pt_xena_metadata.tsv", sep="\t", index=False)
    print("Updated file saved as data/SocialFactor_COAD/pt_xena_metadata.tsv")


if __name__ == "__main__":
    main()
