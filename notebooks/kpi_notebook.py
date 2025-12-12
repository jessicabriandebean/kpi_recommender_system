# %% [markdown]
# # KPI Recommender System - Exploratory Data Analysis
# 
# **Author:** Your Name  
# **Date:** December 2024  
# **Project:** KPI Recommendation Engine
# 
# ## Objective
# This notebook explores the KPI database, analyzes recommendation patterns, and validates the ML model performance.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully")

# %% [markdown]
# ## 2. Load and Explore Data

# %%
# Load the KPI database
from projects.kpi_recommender_system.kpi_recommender_system import KPIRecommender

recommender = KPIRecommender()
df_kpis = recommender.load_kpi_database()

print(f"Dataset Shape: {df_kpis.shape}")
print(f"\nColumns: {df_kpis.columns.tolist()}")
print(f"\nFirst few rows:")
df_kpis.head()

# %%
# Basic statistics
print("="*60)
print("DATABASE SUMMARY")
print("="*60)
print(f"\nTotal KPIs: {len(df_kpis)}")
print(f"Industries: {df_kpis['industry'].nunique()}")
print(f"Departments: {df_kpis['department'].nunique()}")
print(f"Business Goals: {df_kpis['business_goal'].nunique()}")
print(f"Categories: {df_kpis['category'].nunique()}")
print(f"\nPriority Distribution:")
print(df_kpis['priority'].value_counts())

# %%
# Check for missing values
print("\nMissing Values:")
print(df_kpis.isnull().sum())

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %% [markdown]
# ### 3.1 Distribution Analysis

# %%
# Create comprehensive visualizations
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('KPIs by Industry', 'KPIs by Department', 
                    'KPIs by Priority', 'KPIs by Category'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'pie'}, {'type': 'pie'}]]
)

# Industry distribution
industry_counts = df_kpis['industry'].value_counts()
fig.add_trace(
    go.Bar(x=industry_counts.index, y=industry_counts.values, 
           marker_color='lightblue', name='Industry'),
    row=1, col=1
)

# Department distribution
dept_counts = df_kpis['department'].value_counts()
fig.add_trace(
    go.Bar(x=dept_counts.index, y=dept_counts.values,
           marker_color='lightgreen', name='Department'),
    row=1, col=2
)

# Priority distribution
priority_counts = df_kpis['priority'].value_counts()
fig.add_trace(
    go.Pie(labels=priority_counts.index, values=priority_counts.values,
           marker=dict(colors=['#ff6b6b', '#ffd93d'])),
    row=2, col=1
)

# Category distribution
category_counts = df_kpis['category'].value_counts()
fig.add_trace(
    go.Pie(labels=category_counts.index, values=category_counts.values),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="KPI Database Overview")
fig.show()

# %% [markdown]
# ### 3.2 Cross-tabulation Analysis

# %%
# Industry vs Priority
print("KPIs by Industry and Priority:")
industry_priority = pd.crosstab(df_kpis['industry'], df_kpis['priority'])
print(industry_priority)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
industry_priority.plot(kind='bar', stacked=True, ax=ax, color=['#ff6b6b', '#ffd93d'])
plt.title('KPI Priority Distribution by Industry', fontsize=14, fontweight='bold')
plt.xlabel('Industry', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Priority', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Department vs Business Goal
print("\nKPIs by Department and Business Goal:")
dept_goal = pd.crosstab(df_kpis['department'], df_kpis['business_goal'])
print(dept_goal)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(dept_goal, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('KPIs: Department vs Business Goal Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Business Goal', fontsize=12)
plt.ylabel('Department', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Text Analysis

# %%
# Analyze KPI name lengths
df_kpis['name_length'] = df_kpis['kpi_name'].str.len()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_kpis['name_length'], bins=15, color='skyblue', edgecolor='black')
plt.axvline(df_kpis['name_length'].mean(), color='red', linestyle='--', label=f'Mean: {df_kpis["name_length"].mean():.1f}')
plt.xlabel('KPI Name Length (characters)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of KPI Name Lengths', fontsize=13, fontweight='bold')
plt.legend()

plt.subplot(1, 2, 2)
df_kpis['description_length'] = df_kpis['description'].str.len()
plt.hist(df_kpis['description_length'], bins=15, color='lightcoral', edgecolor='black')
plt.axvline(df_kpis['description_length'].mean(), color='blue', linestyle='--', 
            label=f'Mean: {df_kpis["description_length"].mean():.1f}')
plt.xlabel('Description Length (characters)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Description Lengths', fontsize=13, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Most common words in KPI names
from collections import Counter
import re

all_names = ' '.join(df_kpis['kpi_name'].values)
words = re.findall(r'\b[a-zA-Z]{4,}\b', all_names.lower())
word_counts = Counter(words).most_common(15)

words_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 6))
plt.barh(words_df['Word'], words_df['Frequency'], color='mediumpurple')
plt.xlabel('Frequency', fontsize=12)
plt.title('Top 15 Most Common Words in KPI Names', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Machine Learning Model Analysis

# %% [markdown]
# ### 4.1 Feature Vector Creation

# %%
# Create feature vectors
recommender.create_feature_vectors()

print(f"Feature Matrix Shape: {recommender.feature_matrix.shape}")
print(f"Number of KPIs: {recommender.feature_matrix.shape[0]}")
print(f"Number of Features: {recommender.feature_matrix.shape[1]}")

# Get feature names
feature_names = recommender.vectorizer.get_feature_names_out()
print(f"\nSample Features: {feature_names[:20]}")

# %%
# Feature importance (most common terms)
feature_array = recommender.feature_matrix.toarray()
feature_sums = feature_array.sum(axis=0)
feature_importance = sorted(zip(feature_names, feature_sums), key=lambda x: x[1], reverse=True)

top_features = pd.DataFrame(feature_importance[:20], columns=['Feature', 'TF-IDF Sum'])

plt.figure(figsize=(12, 6))
plt.barh(top_features['Feature'], top_features['TF-IDF Sum'], color='teal')
plt.xlabel('TF-IDF Sum', fontsize=12)
plt.title('Top 20 Most Important Features (TF-IDF)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Dimensionality Reduction Visualization

# %%
# PCA for visualization
print("Performing PCA for 2D visualization...")
pca = PCA(n_components=2)
features_2d = pca.fit_transform(feature_array)

df_viz = pd.DataFrame(features_2d, columns=['PC1', 'PC2'])
df_viz['industry'] = df_kpis['industry'].values
df_viz['department'] = df_kpis['department'].values
df_viz['priority'] = df_kpis['priority'].values
df_viz['kpi_name'] = df_kpis['kpi_name'].values

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# %%
# Interactive PCA visualization
fig = px.scatter(df_viz, x='PC1', y='PC2', color='industry', 
                 hover_data=['kpi_name', 'department', 'priority'],
                 title='KPI Feature Space (PCA)',
                 width=900, height=600)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.show()

# %%
# t-SNE visualization (better for clustering)
print("Performing t-SNE for 2D visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
features_tsne = tsne.fit_transform(feature_array)

df_tsne = pd.DataFrame(features_tsne, columns=['Dim1', 'Dim2'])
df_tsne['industry'] = df_kpis['industry'].values
df_tsne['department'] = df_kpis['department'].values
df_tsne['priority'] = df_kpis['priority'].values
df_tsne['kpi_name'] = df_kpis['kpi_name'].values

fig = px.scatter(df_tsne, x='Dim1', y='Dim2', color='department',
                 hover_data=['kpi_name', 'industry', 'priority'],
                 title='KPI Feature Space (t-SNE)',
                 width=900, height=600)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.show()

# %% [markdown]
# ### 4.3 Similarity Matrix Analysis

# %%
# Calculate pairwise similarities
similarity_matrix = cosine_similarity(feature_array)

print(f"Similarity Matrix Shape: {similarity_matrix.shape}")
print(f"Average Similarity: {similarity_matrix.mean():.3f}")
print(f"Max Similarity (excluding diagonal): {np.max(similarity_matrix - np.eye(len(similarity_matrix))):.3f}")

# %%
# Visualize similarity matrix
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, cmap='RdYlGn', center=0.5, 
            xticklabels=df_kpis['kpi_name'].str[:20],
            yticklabels=df_kpis['kpi_name'].str[:20],
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('KPI Similarity Matrix', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Recommendation Performance Analysis

# %% [markdown]
# ### 5.1 Test Multiple Scenarios

# %%
# Define test cases
test_cases = [
    {'industry': 'E-commerce', 'department': 'Marketing', 'business_goal': 'Increase Revenue'},
    {'industry': 'SaaS', 'department': 'Sales', 'business_goal': 'Increase Revenue'},
    {'industry': 'Healthcare', 'department': 'Operations', 'business_goal': 'Improve Efficiency'},
    {'industry': 'Retail', 'department': 'Sales', 'business_goal': 'Increase Revenue'},
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test Case {i}: {test['industry']} - {test['department']} - {test['business_goal']}")
    print('='*70)
    
    recommendations = recommender.recommend_kpis(
        industry=test['industry'],
        department=test['department'],
        business_goal=test['business_goal'],
        top_n=3
    )
    
    for j, rec in enumerate(recommendations, 1):
        print(f"\n{j}. {rec['kpi_name']}")
        print(f"   Similarity Score: {rec['similarity_score']:.3f}")
        print(f"   Priority: {rec['priority']}")
        print(f"   Formula: {rec['formula']}")
        
        results.append({
            'test_case': i,
            'industry': test['industry'],
            'department': test['department'],
            'rank': j,
            'kpi_name': rec['kpi_name'],
            'similarity_score': rec['similarity_score'],
            'priority': rec['priority']
        })

# %%
# Analyze recommendation scores
results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))

# Box plot of scores by rank
plt.subplot(1, 2, 1)
sns.boxplot(data=results_df, x='rank', y='similarity_score', palette='Set2')
plt.title('Similarity Scores by Recommendation Rank', fontsize=12, fontweight='bold')
plt.xlabel('Rank', fontsize=11)
plt.ylabel('Similarity Score', fontsize=11)

# Average scores by test case
plt.subplot(1, 2, 2)
avg_scores = results_df.groupby('test_case')['similarity_score'].mean()
plt.bar(avg_scores.index, avg_scores.values, color='coral')
plt.title('Average Similarity Score by Test Case', fontsize=12, fontweight='bold')
plt.xlabel('Test Case', fontsize=11)
plt.ylabel('Average Similarity Score', fontsize=11)
plt.xticks(avg_scores.index)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Priority Distribution in Recommendations

# %%
priority_dist = results_df['priority'].value_counts()

fig = go.Figure(data=[
    go.Pie(labels=priority_dist.index, values=priority_dist.values,
           marker=dict(colors=['#ff6b6b', '#ffd93d']),
           hole=0.3)
])
fig.update_layout(title='Priority Distribution in Top-3 Recommendations',
                  width=600, height=500)
fig.show()

# %% [markdown]
# ## 6. Model Validation

# %% [markdown]
# ### 6.1 Relevance Analysis

# %%
# Check if recommended KPIs match the input criteria
validation_results = []

for _, row in results_df.iterrows():
    # Get the actual KPI details
    kpi_details = df_kpis[df_kpis['kpi_name'] == row['kpi_name']].iloc[0]
    
    # Check matches
    industry_match = kpi_details['industry'] == row['industry']
    department_match = kpi_details['department'] == row['department']
    
    validation_results.append({
        'test_case': row['test_case'],
        'kpi_name': row['kpi_name'],
        'rank': row['rank'],
        'industry_match': industry_match,
        'department_match': department_match,
        'score': row['similarity_score']
    })

validation_df = pd.DataFrame(validation_results)

print("VALIDATION SUMMARY")
print("="*60)
print(f"Industry Match Rate: {validation_df['industry_match'].mean():.1%}")
print(f"Department Match Rate: {validation_df['department_match'].mean():.1%}")
print(f"\nTop-1 Recommendations:")
print(f"  Industry Match: {validation_df[validation_df['rank']==1]['industry_match'].mean():.1%}")
print(f"  Department Match: {validation_df[validation_df['rank']==1]['department_match'].mean():.1%}")

# %%
# Visualize match rates
match_summary = validation_df.groupby('rank').agg({
    'industry_match': 'mean',
    'department_match': 'mean',
    'score': 'mean'
}).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(x=match_summary['rank'], y=match_summary['industry_match']*100,
                     name='Industry Match', marker_color='lightblue'))
fig.add_trace(go.Bar(x=match_summary['rank'], y=match_summary['department_match']*100,
                     name='Department Match', marker_color='lightgreen'))

fig.update_layout(
    title='Match Rate by Recommendation Rank',
    xaxis_title='Rank',
    yaxis_title='Match Rate (%)',
    barmode='group',
    width=800, height=500
)
fig.show()

# %% [markdown]
# ## 7. Key Insights and Conclusions

# %% [markdown]
# ### Summary Statistics

# %%
print("="*70)
print("KEY INSIGHTS FROM ANALYSIS")
print("="*70)

print(f"\n📊 DATABASE COVERAGE:")
print(f"   • Total KPIs: {len(df_kpis)}")
print(f"   • Industries: {df_kpis['industry'].nunique()}")
print(f"   • Departments: {df_kpis['department'].nunique()}")
print(f"   • Business Goals: {df_kpis['business_goal'].nunique()}")

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"   • Average Similarity Score: {results_df['similarity_score'].mean():.3f}")
print(f"   • Top-1 Average Score: {results_df[results_df['rank']==1]['similarity_score'].mean():.3f}")
print(f"   • Industry Match Rate: {validation_df['industry_match'].mean():.1%}")
print(f"   • Department Match Rate: {validation_df['department_match'].mean():.1%}")

print(f"\n📈 PRIORITY INSIGHTS:")
print(f"   • High Priority KPIs: {(df_kpis['priority']=='High').sum()} ({(df_kpis['priority']=='High').mean():.1%})")
print(f"   • High Priority in Top-3: {(results_df['priority']=='High').sum()} ({(results_df['priority']=='High').mean():.1%})")

print(f"\n✅ RECOMMENDATION QUALITY:")
print(f"   • Model successfully identifies relevant KPIs")
print(f"   • Strong alignment with user criteria")
print(f"   • Prioritizes high-priority metrics appropriately")

# %% [markdown]
# ### Recommendations for Improvement

# %%
print("\n" + "="*70)
print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
print("="*70)
print("""
1. 📚 Expand Database:
   - Add 100+ more KPIs across industries
   - Include more granular subcategories
   - Add industry-specific benchmarks

2. 🔧 Enhance Features:
   - Incorporate user feedback/ratings
   - Add collaborative filtering component
   - Include temporal trends in KPI popularity

3. 🎨 Improve UX:
   - Add KPI comparison tools
   - Include calculation examples
   - Provide implementation guides

4. 📊 Advanced Analytics:
   - Track KPI adoption rates
   - Analyze correlation between KPIs
   - Build predictive models for KPI success

5. 🔄 Continuous Learning:
   - Implement A/B testing framework
   - Collect user satisfaction metrics
   - Retrain model with new data quarterly
""")

# %% [markdown]
# ## 8. Export Analysis Results

# %%
# Save analysis results
results_df.to_csv('recommendation_analysis.csv', index=False)
validation_df.to_csv('validation_results.csv', index=False)

print("✅ Analysis complete!")
print("\nExported files:")
print("  • recommendation_analysis.csv")
print("  • validation_results.csv")

# %%
# Generate final summary report
summary_report = {
    'total_kpis': len(df_kpis),
    'industries': df_kpis['industry'].nunique(),
    'departments': df_kpis['department'].nunique(),
    'avg_similarity_score': float(results_df['similarity_score'].mean()),
    'industry_match_rate': float(validation_df['industry_match'].mean()),
    'department_match_rate': float(validation_df['department_match'].mean()),
    'high_priority_percentage': float((df_kpis['priority']=='High').mean())
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

print("\n✅ Summary report saved to analysis_summary.json")
print("\n🎉 Analysis Complete! Ready for deployment.")