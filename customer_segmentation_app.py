import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
from dateutil import parser
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Personality Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .insight-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<div class='main-header'>Customer Personality Analysis</div>", unsafe_allow_html=True)
st.markdown("""
This application performs customer segmentation through unsupervised learning techniques. 
It identifies distinct customer groups based on demographic and behavioral attributes, 
helping businesses understand their customer base and tailor their marketing strategies accordingly.
""")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Customer Data CSV", type=['csv'])

# Function to preprocess the data
def preprocess_data(df):
    """Preprocess the customer data for clustering analysis"""
    # Create a copy of the dataframe
    data = df.copy()
    
    # Data cleaning and missing values handling
    st.markdown("<div class='section-header'>Data Cleaning</div>", unsafe_allow_html=True)
    initial_shape = data.shape
    st.write(f"Initial dataset shape: {initial_shape}")
    
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    st.write(f"Missing values found: {missing_values}")
    
    if missing_values > 0:
        # Handle missing values
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        
        st.write("Missing values have been handled.")
    
    # Feature Engineering
    st.markdown("<div class='section-header'>Feature Engineering</div>", unsafe_allow_html=True)
    
    # Calculate Age from Year_Birth
    current_year = datetime.datetime.now().year
    data['Age'] = current_year - data['Year_Birth']
    
    # Extract customer tenure in days and years
    if 'Dt_Customer' in data.columns:
        # Robust date parsing to handle multiple formats
        data['Dt_Customer'] = pd.to_datetime(
            data['Dt_Customer'], 
            errors='coerce', 
            dayfirst=True,  # Try day-first parsing for European-style dates
            infer_datetime_format=True
        )
        # Optionally, try to parse any remaining NaT values with a different approach
        if data['Dt_Customer'].isnull().any():
            data.loc[data['Dt_Customer'].isnull(), 'Dt_Customer'] = pd.to_datetime(
                data.loc[data['Dt_Customer'].isnull(), 'Dt_Customer'], 
                errors='coerce', 
                dayfirst=False, 
                infer_datetime_format=True
            )
        data['Customer_Days'] = (pd.to_datetime('today') - data['Dt_Customer']).dt.days
        data['Customer_Years'] = data['Customer_Days'] / 365.25
    
    # Create family size feature
    if 'Kidhome' in data.columns and 'Teenhome' in data.columns:
        data['Family_Size'] = 1 + data['Kidhome'] + data['Teenhome']  # +1 for the customer
        data['Children'] = data['Kidhome'] + data['Teenhome']
    
    # Total spending across all categories
    spending_cols = [col for col in data.columns if col.startswith('Mnt')]
    if spending_cols:
        data['Total_Spending'] = data[spending_cols].sum(axis=1)
        
        # Spending ratios
        for col in spending_cols:
            category = col[3:]  # Remove 'Mnt' prefix
            data[f'Ratio_{category}'] = data[col] / data['Total_Spending'].replace(0, np.nan)
            data[f'Ratio_{category}'] = data[f'Ratio_{category}'].fillna(0)
    
    # Total number of purchases
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    if all(col in data.columns for col in purchase_cols):
        data['Total_Purchases'] = data[purchase_cols].sum(axis=1)
        
        # Purchase channel preferences
        for col in purchase_cols:
            channel = col[3:-9]  # Extract channel name
            data[f'Ratio_{channel}'] = data[col] / data['Total_Purchases'].replace(0, np.nan)
            data[f'Ratio_{channel}'] = data[f'Ratio_{channel}'].fillna(0)
    
    # Campaign success ratio
    campaign_cols = [col for col in data.columns if col.startswith('Accepted')]
    if campaign_cols:
        data['Campaign_Success'] = data[campaign_cols].sum(axis=1)
        data['Campaign_Success_Ratio'] = data['Campaign_Success'] / len(campaign_cols)
    
    # Income per family member
    if 'Income' in data.columns and 'Family_Size' in data.columns:
        data['Income_per_person'] = data['Income'] / data['Family_Size']
    
    # Average spending per purchase
    if 'Total_Spending' in data.columns and 'Total_Purchases' in data.columns:
        data['Avg_Spending_per_Purchase'] = data['Total_Spending'] / data['Total_Purchases'].replace(0, np.nan)
        data['Avg_Spending_per_Purchase'] = data['Avg_Spending_per_Purchase'].fillna(0)
    
    # Education level mapping for ordinal encoding
    education_mapping = {
        'Basic': 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    
    # Apply education mapping if education column exists and contains these values
    if 'Education' in data.columns:
        if data['Education'].nunique() <= 10:  # Reasonable number of categories
            # Check if the education values match our expected mapping
            known_values = set(education_mapping.keys())
            actual_values = set(data['Education'].unique())
            
            if actual_values.issubset(known_values):
                data['Education_Level'] = data['Education'].map(education_mapping)
            else:
                # If values don't match, create dummy variables
                data = pd.get_dummies(data, columns=['Education'], prefix='Edu')
    
    # Create dummy variables for categorical columns
    categorical_cols = ['Marital_Status']
    categorical_cols = [col for col in categorical_cols if col in data.columns]
    
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Log transformation for skewed variables
    skewed_features = ['Income', 'Total_Spending']
    skewed_features = [col for col in skewed_features if col in data.columns]
    
    for feature in skewed_features:
        if (data[feature] > 0).all():  # Only apply log to positive values
            data[f'{feature}_Log'] = np.log1p(data[feature])
    
    st.write(f"After feature engineering, dataset shape: {data.shape}")
    
    # Display the list of new features
    new_features = set(data.columns) - set(df.columns)
    st.write(f"New features created: {', '.join(new_features)}")
    
    return data

# Function to perform clustering
def perform_clustering(data, n_clusters):
    """Perform K-means clustering on the preprocessed data"""
    # Select features for clustering
    features = []
    
    # Demographic features
    demographic_cols = ['Age', 'Income', 'Family_Size', 'Children']
    features.extend([col for col in demographic_cols if col in data.columns])
    
    # Spending features
    spending_cols = [col for col in data.columns if col.startswith('Mnt') or col.startswith('Ratio_')]
    features.extend([col for col in spending_cols if col in data.columns])
    
    # Purchase behavior
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                     'NumDealsPurchases', 'NumWebVisitsMonth']
    features.extend([col for col in purchase_cols if col in data.columns])
    
    # Campaign response
    campaign_cols = [col for col in data.columns if col.startswith('Accepted') or col == 'Response']
    features.extend([col for col in campaign_cols if col in data.columns])
    
    # Customer relationship
    relationship_cols = ['Recency', 'Customer_Years', 'Complain']
    features.extend([col for col in relationship_cols if col in data.columns])
    
    # Additional engineered features
    additional_cols = ['Campaign_Success_Ratio', 'Avg_Spending_per_Purchase', 'Total_Spending']
    features.extend([col for col in additional_cols if col in data.columns])
    
    # Keep only numeric columns
    features = [f for f in features if data[f].dtype in ['int64', 'float64']]
    
    # Handle potential duplicates in features list
    features = list(dict.fromkeys(features))
    
    # Extract the feature matrix
    X = data[features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette = round(silhouette_score(X_scaled, data['Cluster']), 3)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]
    
    return data, kmeans, features, silhouette, pca.explained_variance_ratio_

# Function to determine optimal number of clusters
def find_optimal_clusters(data, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    # Select features for clustering (same as in perform_clustering)
    features = []
    
    # Demographic features
    demographic_cols = ['Age', 'Income', 'Family_Size', 'Children']
    features.extend([col for col in demographic_cols if col in data.columns])
    
    # Spending features
    spending_cols = [col for col in data.columns if col.startswith('Mnt') or col.startswith('Ratio_')]
    features.extend([col for col in spending_cols if col in data.columns])
    
    # Purchase behavior
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                     'NumDealsPurchases', 'NumWebVisitsMonth']
    features.extend([col for col in purchase_cols if col in data.columns])
    
    # Campaign response
    campaign_cols = [col for col in data.columns if col.startswith('Accepted') or col == 'Response']
    features.extend([col for col in campaign_cols if col in data.columns])
    
    # Customer relationship
    relationship_cols = ['Recency', 'Customer_Years', 'Complain']
    features.extend([col for col in relationship_cols if col in data.columns])
    
    # Additional engineered features
    additional_cols = ['Campaign_Success_Ratio', 'Avg_Spending_per_Purchase', 'Total_Spending']
    features.extend([col for col in additional_cols if col in data.columns])
    
    # Keep only numeric columns
    features = [f for f in features if data[f].dtype in ['int64', 'float64']]
    
    # Handle potential duplicates in features list
    features = list(dict.fromkeys(features))
    
    # Extract the feature matrix
    X = data[features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    return range(2, max_clusters + 1), silhouette_scores

# Function to visualize clusters
def visualize_clusters(data, features):
    """Create visualizations for the clustered data"""
    # Create a figure for cluster visualization using PCA
    fig_pca = px.scatter(
        data, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        color_continuous_scale=px.colors.qualitative.Bold,
        hover_data=['Age', 'Income', 'Total_Spending'],
        title='Customer Segments (PCA Visualization)'
    )
    fig_pca.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Show cluster distribution
    fig_dist = px.histogram(
        data, 
        x='Cluster', 
        title='Cluster Size Distribution',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Cluster profiles
    st.markdown("<div class='section-header'>Cluster Profiles</div>", unsafe_allow_html=True)
    
    # Select key metrics for profiling
    profile_metrics = [
        'Age', 'Income', 'Family_Size', 'Children', 'Total_Spending',
        'Campaign_Success_Ratio', 'Recency', 'Total_Purchases', 'NumWebVisitsMonth'
    ]
    
    profile_metrics = [metric for metric in profile_metrics if metric in data.columns]
    
    # Create cluster profile summary
    profile_summary = data.groupby('Cluster')[profile_metrics].mean()
    
    # Scale the summary for radar chart
    scaler = StandardScaler()
    profile_summary_scaled = pd.DataFrame(
        scaler.fit_transform(profile_summary),
        index=profile_summary.index,
        columns=profile_summary.columns
    )
    
    # Display the profile summary
    st.write("Cluster Profile Summary (Mean Values):")
    st.dataframe(profile_summary.style.highlight_max(axis=0))
    
    # Create radar charts for each cluster
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending distribution by cluster
        spending_cols = [col for col in data.columns if col.startswith('Mnt')]
        if spending_cols:
            spending_by_cluster = data.groupby('Cluster')[spending_cols].mean()
            
            # Rename columns for better readability
            renamed_cols = {col: col[3:].replace('Products', '').replace('Prods', '') for col in spending_cols}
            spending_by_cluster = spending_by_cluster.rename(columns=renamed_cols)
            
            fig_spending = px.bar(
                spending_by_cluster.reset_index().melt(id_vars='Cluster'),
                x='Cluster',
                y='value',
                color='variable',
                barmode='group',
                title='Average Spending by Category Across Clusters'
            )
            st.plotly_chart(fig_spending, use_container_width=True)
    
    with col2:
        # Purchase channel preferences by cluster
        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        if all(col in data.columns for col in purchase_cols):
            purchases_by_cluster = data.groupby('Cluster')[purchase_cols].mean()
            
            # Rename columns for better readability
            renamed_cols = {
                'NumWebPurchases': 'Web', 
                'NumCatalogPurchases': 'Catalog', 
                'NumStorePurchases': 'Store'
            }
            purchases_by_cluster = purchases_by_cluster.rename(columns=renamed_cols)
            
            fig_purchases = px.bar(
                purchases_by_cluster.reset_index().melt(id_vars='Cluster'),
                x='Cluster',
                y='value',
                color='variable',
                barmode='group',
                title='Purchase Channel Preferences by Cluster'
            )
            st.plotly_chart(fig_purchases, use_container_width=True)
    
    # Campaign response by cluster
    campaign_cols = [col for col in data.columns if col.startswith('Accepted') or col == 'Response']
    if campaign_cols:
        campaign_response = data.groupby('Cluster')[campaign_cols].mean()
        
        fig_campaign = px.imshow(
            campaign_response,
            title='Campaign Response Rate by Cluster',
            color_continuous_scale='Blues',
            labels=dict(x='Campaign', y='Cluster', color='Response Rate')
        )
        st.plotly_chart(fig_campaign, use_container_width=True)
    
    # Feature importance for each cluster
    st.markdown("<div class='section-header'>Cluster Characteristics</div>", unsafe_allow_html=True)
    
    # Select meaningful features for comparison
    comparison_features = [
        'Age', 'Income', 'Family_Size', 'Total_Spending', 
        'Avg_Spending_per_Purchase', 'Campaign_Success_Ratio',
        'NumWebVisitsMonth', 'Recency'
    ]
    comparison_features = [f for f in comparison_features if f in data.columns]
    
    # Calculate z-scores for each cluster compared to overall average
    overall_means = data[comparison_features].mean()
    overall_stds = data[comparison_features].std()
    
    cluster_z_scores = {}
    for cluster in data['Cluster'].unique():
        cluster_means = data[data['Cluster'] == cluster][comparison_features].mean()
        z_scores = (cluster_means - overall_means) / overall_stds
        cluster_z_scores[cluster] = z_scores
    
    z_score_df = pd.DataFrame(cluster_z_scores).T
    
    # Create heatmap of z-scores
    fig_heatmap = px.imshow(
        z_score_df,
        title='Cluster Characteristics (Z-scores relative to overall average)',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        labels=dict(x='Feature', y='Cluster', color='Z-score')
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Create cluster narratives
    st.markdown("<div class='section-header'>Cluster Narratives</div>", unsafe_allow_html=True)
    
    for cluster in sorted(data['Cluster'].unique()):
        # Get key features for this cluster
        cluster_data = data[data['Cluster'] == cluster]
        cluster_size = len(cluster_data)
        cluster_pct = round(100 * cluster_size / len(data), 1)
        
        # Extract key metrics
        metrics = {}
        metrics['Age'] = round(cluster_data['Age'].mean(), 1) if 'Age' in cluster_data.columns else 'N/A'
        metrics['Income'] = f"${round(cluster_data['Income'].mean(), 2):,.2f}" if 'Income' in cluster_data.columns else 'N/A'
        metrics['Family_Size'] = round(cluster_data['Family_Size'].mean(), 1) if 'Family_Size' in cluster_data.columns else 'N/A'
        metrics['Total_Spending'] = f"${round(cluster_data['Total_Spending'].mean(), 2):,.2f}" if 'Total_Spending' in cluster_data.columns else 'N/A'
        metrics['Campaign_Success'] = f"{round(100 * cluster_data['Campaign_Success_Ratio'].mean(), 1)}%" if 'Campaign_Success_Ratio' in cluster_data.columns else 'N/A'
        
        # Find distinctive features (highest absolute z-scores)
        if cluster in z_score_df.index:
            cluster_means = cluster_data[comparison_features].mean()  # <-- Add this line
            z_scores = z_score_df.loc[cluster].abs().sort_values(ascending=False)
            top_features = z_scores.head(3).index.tolist()
            distinctive_traits = []
            for feature in top_features:
                original_z = (cluster_means[feature] - overall_means[feature]) / overall_stds[feature]
                if original_z > 0:
                    distinctive_traits.append(f"High {feature} (z={original_z:.2f})")
                else:
                    distinctive_traits.append(f"Low {feature} (z={original_z:.2f})")
        else:
            distinctive_traits = ["No distinctive traits identified"]
        
        # Create an expandable section for each cluster
        with st.expander(f"Cluster {cluster} ({cluster_size} customers, {cluster_pct}% of total)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Key Metrics")
                for metric, value in metrics.items():
                    st.write(f"**{metric}:** {value}")
            
            with col2:
                st.markdown("##### Distinctive Traits")
                for trait in distinctive_traits:
                    st.write(f"- {trait}")
            
            # Generate narrative description
            st.markdown("##### Cluster Description")
            
            # This is a simplified version - in a real application, this would be more sophisticated
            description = generate_cluster_narrative(cluster, cluster_data, z_score_df, overall_means, overall_stds)
            st.markdown(f"<div class='insight-box'>{description}</div>", unsafe_allow_html=True)
            
            # Marketing recommendations
            st.markdown("##### Marketing Recommendations")
            recommendations = generate_marketing_recommendations(cluster, cluster_data, z_score_df)
            
            for rec in recommendations:
                st.write(f"- {rec}")

# Function to generate cluster narratives
def generate_cluster_narrative(cluster, cluster_data, z_score_df, overall_means, overall_stds):
    """Generate a narrative description for a cluster based on its characteristics"""
    if cluster not in z_score_df.index:
        return "Insufficient data to generate a description for this cluster."
    
    z_scores = z_score_df.loc[cluster]
    narrative = []
    
    # Age description
    if 'Age' in z_scores:
        age_z = z_scores['Age']
        age_mean = cluster_data['Age'].mean()
        
        if age_z > 1:
            narrative.append(f"This segment consists of older customers (average age: {age_mean:.1f} years).")
        elif age_z < -1:
            narrative.append(f"This segment represents younger customers (average age: {age_mean:.1f} years).")
        else:
            narrative.append(f"This segment includes middle-aged customers (average age: {age_mean:.1f} years).")
    
    # Income and spending description
    if 'Income' in z_scores and 'Total_Spending' in z_scores:
        income_z = z_scores['Income']
        spending_z = z_scores['Total_Spending']
        income_mean = cluster_data['Income'].mean()
        spending_mean = cluster_data['Total_Spending'].mean()
        
        if income_z > 0.7 and spending_z > 0.7:
            narrative.append(f"They are high-income customers (${income_mean:,.2f}) with high spending (${spending_mean:,.2f}).")
        elif income_z < -0.7 and spending_z < -0.7:
            narrative.append(f"They are budget-conscious customers with lower income (${income_mean:,.2f}) and lower spending (${spending_mean:,.2f}).")
        elif income_z > 0.7 and spending_z < -0.7:
            narrative.append(f"Despite having higher income (${income_mean:,.2f}), they spend relatively little (${spending_mean:,.2f}), suggesting they are savers.")
        elif income_z < -0.7 and spending_z > 0.7:
            narrative.append(f"Although they have lower income (${income_mean:,.2f}), they show higher spending (${spending_mean:,.2f}), possibly financially stretched.")
        else:
            narrative.append(f"They have moderate income (${income_mean:,.2f}) and spending patterns (${spending_mean:,.2f}).")
    
    # Family composition
    if 'Family_Size' in z_scores and 'Children' in z_scores:
        family_z = z_scores['Family_Size']
        children_z = z_scores['Children']
        family_mean = cluster_data['Family_Size'].mean()
        children_mean = cluster_data['Children'].mean()
        
        if family_z > 0.7:
            narrative.append(f"They typically have larger families (average size: {family_mean:.1f}).")
        elif family_z < -0.7:
            narrative.append(f"They typically have smaller households (average size: {family_mean:.1f}).")
    
    # Campaign responsiveness
    if 'Campaign_Success_Ratio' in z_scores:
        campaign_z = z_scores['Campaign_Success_Ratio']
        campaign_mean = cluster_data['Campaign_Success_Ratio'].mean()
        
        if campaign_z > 0.7:
            narrative.append(f"This segment is highly responsive to marketing campaigns ({campaign_mean*100:.1f}% acceptance rate).")
        elif campaign_z < -0.7:
            narrative.append(f"This segment shows low responsiveness to marketing campaigns ({campaign_mean*100:.1f}% acceptance rate).")
    
    # Web behavior
    if 'NumWebVisitsMonth' in z_scores:
        web_z = z_scores['NumWebVisitsMonth']
        web_mean = cluster_data['NumWebVisitsMonth'].mean()
        
        if web_z > 0.7:
            narrative.append(f"They are active online with frequent website visits ({web_mean:.1f} visits per month).")
        elif web_z < -0.7:
            narrative.append(f"They rarely visit the company website ({web_mean:.1f} visits per month).")
    
    # Purchase recency
    if 'Recency' in z_scores:
        recency_z = z_scores['Recency']
        recency_mean = cluster_data['Recency'].mean()
        
        if recency_z > 0.7:
            narrative.append(f"It's been longer since their last purchase ({recency_mean:.1f} days), suggesting they may be at risk of churn.")
        elif recency_z < -0.7:
            narrative.append(f"They've made purchases recently ({recency_mean:.1f} days since last purchase), indicating they are active customers.")
    
    # Channel preferences
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    if all(col in z_scores for col in purchase_cols):
        web_mean = cluster_data['NumWebPurchases'].mean()
        catalog_mean = cluster_data['NumCatalogPurchases'].mean()
        store_mean = cluster_data['NumStorePurchases'].mean()
        
        max_channel = max(web_mean, catalog_mean, store_mean)
        
        if max_channel == web_mean and z_scores['NumWebPurchases'] > 0.5:
            narrative.append(f"They prefer online shopping ({web_mean:.1f} web purchases on average).")
        elif max_channel == catalog_mean and z_scores['NumCatalogPurchases'] > 0.5:
            narrative.append(f"They prefer catalog shopping ({catalog_mean:.1f} catalog purchases on average).")
        elif max_channel == store_mean and z_scores['NumStorePurchases'] > 0.5:
            narrative.append(f"They prefer in-store shopping ({store_mean:.1f} store purchases on average).")
    
    # Product preferences
    product_cols = [col for col in z_scores.index if col.startswith('Mnt')]
    if product_cols:
        product_z_scores = {col: z_scores[col] for col in product_cols}
        preferred_products = [col[3:].replace('Products', '').replace('Prods', '') 
                             for col in product_cols if z_scores[col] > 0.7]
        
        if preferred_products:
            narrative.append(f"They show a preference for {', '.join(preferred_products)} products.")
    
    # Join all parts of the narrative
    return " ".join(narrative)

# Function to generate marketing recommendations
def generate_marketing_recommendations(cluster, cluster_data, z_score_df):
    """Generate marketing recommendations based on cluster characteristics"""
    if cluster not in z_score_df.index:
        return ["Insufficient data to generate recommendations for this cluster."]
    
    z_scores = z_score_df.loc[cluster]
    recommendations = []
    
    # Campaign responsiveness recommendations
    if 'Campaign_Success_Ratio' in z_scores:
        campaign_z = z_scores['Campaign_Success_Ratio']
        
        if campaign_z > 0.7:
            recommendations.append("Prioritize this segment for new marketing campaigns due to high response rates.")
            recommendations.append("Test more frequent campaigns to maximize engagement.")
        elif campaign_z < -0.7:
            recommendations.append("Reduce marketing campaign frequency for this segment.")
            recommendations.append("Test radically different campaign approaches to find what resonates.")
    
    # Income and spending based recommendations
    if 'Income' in z_scores and 'Total_Spending' in z_scores:
        income_z = z_scores['Income']
        spending_z = z_scores['Total_Spending']
        
        if income_z > 0.7 and spending_z > 0.7:
            recommendations.append("Focus on premium/luxury products and exclusive offers.")
            recommendations.append("Create loyalty rewards that recognize their high spending.")
        elif income_z < -0.7 and spending_z < -0.7:
            recommendations.append("Emphasize value and affordability in messaging.")
            recommendations.append("Introduce special discounts and budget-friendly options.")
        elif income_z > 0.7 and spending_z < -0.7:
            recommendations.append("Highlight quality and investment value in products.")
            recommendations.append("Create special offers to convert disposable income into purchases.")
    
    # Age-based recommendations
    if 'Age' in z_scores:
        age_z = z_scores['Age']
        
        if age_z > 1:
            recommendations.append("Use traditional media channels alongside digital ones.")
            recommendations.append("Focus on comfort, reliability, and familiarity in messaging.")
        elif age_z < -1:
            recommendations.append("Prioritize mobile and social media marketing channels.")
            recommendations.append("Emphasize innovation and trends in product messaging.")
    
    # Family composition recommendations
    if 'Family_Size' in z_scores and 'Children' in z_scores:
        family_z = z_scores['Family_Size']
        children_z = z_scores['Children']
        
        if children_z > 0.7:
            recommendations.append("Develop family-oriented promotions and bundles.")
            recommendations.append("Time campaigns around back-to-school or holiday seasons.")
        elif family_z < -0.7 and children_z < -0.7:
            recommendations.append("Focus on individual-centered messaging and products.")
            recommendations.append("Highlight convenience and personal enjoyment aspects.")
    
    # Web behavior recommendations
    if 'NumWebVisitsMonth' in z_scores:
        web_z = z_scores['NumWebVisitsMonth']
        
        if web_z > 0.7:
            recommendations.append("Increase digital advertising and website promotions.")
            recommendations.append("Implement personalized website recommendations.")
        elif web_z < -0.7:
            recommendations.append("Focus on non-digital channels like direct mail or in-store promotions.")
            recommendations.append("Create compelling reasons to visit the website (exclusive online offers).")
    
    # Purchase recency recommendations
    if 'Recency' in z_scores:
        recency_z = z_scores['Recency']
        
        if recency_z > 0.7:
            recommendations.append("Implement a win-back campaign with special incentives.")
            recommendations.append("Conduct surveys to understand reasons for reduced engagement.")
        elif recency_z < -0.7:
            recommendations.append("Send follow-up offers related to recent purchases.")
            recommendations.append("Implement cross-selling strategies while engagement is high.")
    
    # Channel preference recommendations
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    if all(col in z_scores for col in purchase_cols):
        web_z = z_scores['NumWebPurchases']
        catalog_z = z_scores['NumCatalogPurchases']
        store_z = z_scores['NumStorePurchases']
        
        if web_z > 0.5:
            recommendations.append("Optimize the online shopping experience for this segment.")
            recommendations.append("Implement web-exclusive promotions and early access.")
        elif catalog_z > 0.5:
            recommendations.append("Continue investing in quality catalog materials for this segment.")
            recommendations.append("Create catalog-exclusive offers and early previews.")
        elif store_z > 0.5:
            recommendations.append("Focus on enhancing the in-store experience.")
            recommendations.append("Create in-store exclusive events or promotions.")
    
    # Deal sensitivity recommendations
    if 'NumDealsPurchases' in z_scores:
        deals_z = z_scores['NumDealsPurchases']
        
        if deals_z > 0.7:
            recommendations.append("Regularly offer discounts and deals to this price-sensitive segment.")
            recommendations.append("Create a special deals newsletter or notification system.")
        elif deals_z < -0.7:
            recommendations.append("Focus less on discounts and more on quality/exclusivity messaging.")
            recommendations.append("Test premium pricing strategies for this segment.")
    
    # Limit to top recommendations
    return recommendations[:5]
            



if uploaded_file is not None:
    # Load and process the data
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully loaded.")
        
        # Show raw data sample
        with st.expander("View Raw Data Sample"):
            st.write(df.head())
            st.write(f"Dataset shape: {df.shape}")
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Sidebar for clustering parameters
        st.sidebar.markdown("## Clustering Parameters")
        
        # Option to automatically find optimal clusters
        auto_clusters = st.sidebar.checkbox("Find Optimal Number of Clusters", value=False)
        
        if auto_clusters:
            st.sidebar.info("Calculating optimal number of clusters...")
            max_clusters = st.sidebar.slider("Maximum Clusters to Consider", min_value=2, max_value=15, value=10)
            n_clusters_range, silhouette_scores = find_optimal_clusters(processed_data, max_clusters)
            
            # Plot silhouette scores
            fig_silhouette = px.line(
                x=n_clusters_range, 
                y=silhouette_scores, 
                markers=True,
                title='Silhouette Scores for Different Numbers of Clusters',
                labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'}
            )
            st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Find optimal number of clusters
            optimal_clusters = n_clusters_range[silhouette_scores.index(max(silhouette_scores))]
            st.sidebar.success(f"Optimal number of clusters: {optimal_clusters}")
            
            # Set the default value for the slider to the optimal number
            n_clusters = st.sidebar.slider(
                "Number of Clusters (K)",
                min_value=2,
                max_value=15,
                value=optimal_clusters
            )
        else:
            # Manual selection of number of clusters
            n_clusters = st.sidebar.slider(
                "Number of Clusters (K)",
                min_value=2,
                max_value=15,
                value=4
            )
        
        # Random state for reproducibility
        random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42)
        
        # Button to perform clustering
        if st.sidebar.button("Perform Clustering"):
            with st.spinner("Performing customer segmentation..."):
                # Perform clustering
                clustered_data, kmeans_model, features_used, silhouette_score, variance_explained = perform_clustering(processed_data, n_clusters)
                
                # Display results
                st.markdown("<div class='section-header'>Clustering Results</div>", unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Number of Segments", n_clusters)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Silhouette Score", silhouette_score)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    variance_pct = sum(variance_explained) * 100
                    st.metric("PCA Variance Explained", f"{variance_pct:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display features used
                with st.expander("Features Used for Clustering"):
                    st.write(features_used)
                
                # Create visualizations
                visualize_clusters(clustered_data, features_used)
                
                # Download clustered data
                csv = clustered_data.to_csv(index=False)
                st.download_button(
                    label="Download Segmented Data",
                    data=csv,
                    file_name="customer_segments.csv",
                    mime="text/csv",
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data format and try again.")

else:
    # Show sample data option if no file is uploaded
    use_sample_data = st.checkbox("Use Sample Data")
    
    if use_sample_data:
        st.info("Loading sample customer data...")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 500
        
        # Create sample data frame
        sample_data = pd.DataFrame({
            'ID': range(1, n_samples + 1),
            'Year_Birth': np.random.randint(1940, 2000, n_samples),
            'Education': np.random.choice(['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'], n_samples),
            'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Together', 'Widow'], n_samples),
            'Income': np.random.lognormal(10, 0.5, n_samples),
            'Kidhome': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'Teenhome': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
            'Recency': np.random.randint(0, 100, n_samples),
            'Complain': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'MntWines': np.random.lognormal(4, 1, n_samples),
            'MntFruits': np.random.lognormal(3, 1, n_samples),
            'MntMeatProducts': np.random.lognormal(4, 0.8, n_samples),
            'MntFishProducts': np.random.lognormal(3, 1, n_samples),
            'MntSweetProducts': np.random.lognormal(2, 1, n_samples),
            'MntGoldProds': np.random.lognormal(3, 1, n_samples),
            'NumDealsPurchases': np.random.randint(0, 10, n_samples),
            'NumWebPurchases': np.random.randint(0, 20, n_samples),
            'NumCatalogPurchases': np.random.randint(0, 15, n_samples),
            'NumStorePurchases': np.random.randint(0, 25, n_samples),
            'NumWebVisitsMonth': np.random.randint(0, 20, n_samples),
        })
        
        # Generate campaign response variables
        for i in range(1, 6):
            sample_data[f'AcceptedCmp{i}'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        sample_data['Response'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        
        # Generate dates between 2018 and 2020
        start_date = pd.to_datetime('2018-01-01')
        end_date = pd.to_datetime('2020-12-31')
        
        # Generate random dates
        days_range = (end_date - start_date).days
        random_days = np.random.randint(0, days_range, n_samples)
        random_dates = start_date + pd.to_timedelta(random_days, unit='D')
        
        sample_data['Dt_Customer'] = random_dates.strftime('%Y-%m-%d')
        
        # Show sample data
        with st.expander("View Sample Data"):
            st.write(sample_data.head())
        
        # Process sample data
        processed_data = preprocess_data(sample_data)
        
        # Sidebar for clustering parameters
        st.sidebar.markdown("## Clustering Parameters")
        
        # Option to automatically find optimal clusters
        auto_clusters = st.sidebar.checkbox("Find Optimal Number of Clusters", value=False)
        
        if auto_clusters:
            st.sidebar.info("Calculating optimal number of clusters...")
            max_clusters = st.sidebar.slider("Maximum Clusters to Consider", min_value=2, max_value=15, value=10)
            n_clusters_range, silhouette_scores = find_optimal_clusters(processed_data, max_clusters)
            
            # Plot silhouette scores
            fig_silhouette = px.line(
                x=n_clusters_range, 
                y=silhouette_scores, 
                markers=True,
                title='Silhouette Scores for Different Numbers of Clusters',
                labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'}
            )
            st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Find optimal number of clusters
            optimal_clusters = n_clusters_range[silhouette_scores.index(max(silhouette_scores))]
            st.sidebar.success(f"Optimal number of clusters: {optimal_clusters}")
            
            # Set the default value for the slider to the optimal number
            n_clusters = st.sidebar.slider(
                "Number of Clusters (K)",
                min_value=2,
                max_value=15,
                value=optimal_clusters
            )
        else:
            # Manual selection of number of clusters
            n_clusters = st.sidebar.slider(
                "Number of Clusters (K)",
                min_value=2,
                max_value=15,
                value=4
            )
        
        # Random state for reproducibility
        random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42)
        
        # Button to perform clustering
        if st.sidebar.button("Perform Clustering"):
            with st.spinner("Performing customer segmentation..."):
                # Perform clustering
                clustered_data, kmeans_model, features_used, silhouette_score, variance_explained = perform_clustering(processed_data, n_clusters)
                
                # Display results
                st.markdown("<div class='section-header'>Clustering Results</div>", unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Number of Segments", n_clusters)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Silhouette Score", silhouette_score)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    variance_pct = sum(variance_explained) * 100
                    st.metric("PCA Variance Explained", f"{variance_pct:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Create visualizations
                visualize_clusters(clustered_data, features_used)
    else:
        # Instructions for uploading data
        st.markdown("""
        ## Getting Started
        
        To begin your customer segmentation analysis:
        
        1. Upload your customer data CSV file using the sidebar.
        2. The file should contain customer attributes similar to those described in the data attributes document.
        3. After uploading, configure the clustering parameters in the sidebar.
        4. Click "Perform Clustering" to generate customer segments.
        
        ### Expected Data Format
        
        Your data should include demographic information (age, income, family size), 
        purchase behavior (spending amounts, channel preferences), and campaign responses.
        
        You can also try the "Use Sample Data" option to explore the tool's capabilities.
        """)
        
        # Show data attribute information
        with st.expander("View Expected Data Attributes"):
            st.markdown("""
            ### People
            - ID: Customer's unique identifier
            - Year_Birth: Customer's birth year
            - Education: Customer's education level
            - Marital_Status: Customer's marital status
            - Income: Customer's yearly household income
            - Kidhome: Number of children in customer's household
            - Teenhome: Number of teenagers in customer's household
            - Dt_Customer: Date of customer's enrollment with the company
            - Recency: Number of days since customer's last purchase
            - Complain: 1 if customer complained in the last 2 years, 0 otherwise
            
            ### Products
            - MntWines: Amount spent on wine in last 2 years
            - MntFruits: Amount spent on fruits in last 2 years
            - MntMeatProducts: Amount spent on meat in last 2 years
            - MntFishProducts: Amount spent on fish in last 2 years
            - MntSweetProducts: Amount spent on sweets in last 2 years
            - MntGoldProds: Amount spent on gold in last 2 years
            
            ### Promotion
            - NumDealsPurchases: Number of purchases made with a discount
            - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
            - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
            - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
            - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
            - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
            - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
            
            ### Place
            - NumWebPurchases: Number of purchases made through the company's website
            - NumCatalogPurchases: Number of purchases made using a catalogue
            - NumStorePurchases: Number of purchases made directly in stores
            - NumWebVisitsMonth: Number of visits to company's website in the last month
            """)

# Add information about the methodology
with st.sidebar.expander("About the Methodology"):
    st.markdown("""
    ### Customer Segmentation Methodology
    
    This app uses K-means clustering, an unsupervised machine learning algorithm, to identify distinct customer segments.
    
    **Key Steps:**
    1. **Data Preprocessing**: Handling missing values, creating meaningful features
    2. **Feature Engineering**: Calculating derived attributes like total spending, campaign response rates
    3. **Standardization**: Scaling features to have mean=0 and std=1
    4. **K-means Clustering**: Grouping customers based on similarity
    5. **Visualization**: PCA for dimensionality reduction
    
    **Evaluation**: Silhouette score measures how similar each customer is to their own cluster compared to other clusters.
    
    **Data Privacy Note**: All data is processed locally in your browser and is not stored or shared.
    """)

# Add footer
st.markdown("""
---
### Customer Segmentation Analysis Application

This application helps businesses identify meaningful customer segments based on demographic, purchasing, and behavioral data.
Use the insights for targeted marketing campaigns and personalized customer experiences.
""")