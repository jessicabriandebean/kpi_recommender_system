# streamlit_app.py
"""
KPI Recommender System - Streamlit Web Application
Deploy with: streamlit run streamlit_kpi_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from same directory
try:
    from kpi_recommender_system import KPIRecommender
except ImportError:
    # If that doesn't work, try adding current directory to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from kpi_recommender_system import KPIRecommender

# Page configuration
st.set_page_config(
    page_title="KPI Recommender System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .priority-high {
        background-color: #fee;
        border-left-color: #dc3545;
    }
    .priority-medium {
        background-color: #fff8e1;
        border-left-color: #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = KPIRecommender()
    st.session_state.recommender.load_kpi_database()
    st.session_state.recommender.create_feature_vectors()

# Header
st.markdown('<p class="main-header">🎯 KPI Recommender System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get data-driven KPI recommendations tailored to your business needs</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get unique values for dropdowns
    industries = st.session_state.recommender.df_kpis['industry'].unique().tolist()
    departments = st.session_state.recommender.df_kpis['department'].unique().tolist()
    goals = st.session_state.recommender.df_kpis['business_goal'].unique().tolist()
    
    industry = st.selectbox("Industry", [""] + industries)
    department = st.selectbox("Department", [""] + departments)
    business_goal = st.selectbox("Business Goal", [""] + goals)
    
    st.divider()
    
    top_n = st.slider("Number of Recommendations", 1, 10, 5)
    
    st.divider()
    
    show_analytics = st.checkbox("Show Database Analytics", value=False)

# Main content area
if industry and department and business_goal:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Recommended KPIs")
        
        # Get recommendations
        recommendations = st.session_state.recommender.recommend_kpis(
            industry=industry,
            department=department,
            business_goal=business_goal,
            top_n=top_n
        )
        
        # Display context
        st.info(f"**Context:** Showing recommendations for {industry} companies in {department} department focusing on '{business_goal}'")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            priority_class = f"priority-{rec['priority'].lower()}"
            
            # Better contrast colors
            if rec['priority'] == 'High':
                bg_color = '#fee2e2'  # Light red background
                border_color = '#dc2626'  # Dark red border
                priority_bg = '#dc2626'  # Dark red badge
            else:  # Medium
                bg_color = '#fef3c7'  # Light yellow background
                border_color = '#f59e0b'  # Dark orange border
                priority_bg = '#f59e0b'  # Dark orange badge
            
            with st.container():
                st.markdown(f"""
                    <div style="
                        background-color: {bg_color};
                        border-left: 5px solid {border_color};
                        padding: 1.5rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                    ">
                        <h3 style="color: #1f2937; margin-bottom: 0.5rem;">#{i} {rec['kpi_name']}</h3>
                        <p style="margin: 0.5rem 0;">
                            <strong style="color: #374151;">Priority:</strong> 
                            <span style="
                                background-color: {priority_bg}; 
                                color: white; 
                                padding: 0.2rem 0.5rem; 
                                border-radius: 5px;
                                font-weight: 600;
                            ">{rec['priority']}</span>
                        </p>
                        <p style="color: #374151; margin: 0.5rem 0;">
                            <strong>Similarity Score:</strong> {rec['similarity_score']:.2%}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Expandable details with good contrast
                with st.expander("View Details"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Formula:**  \n`{rec['formula']}`")
                        st.markdown(f"**Benchmark:**  \n{rec['benchmark']}")
                    with col_b:
                        st.markdown(f"**Category:**  \n{rec['category']}")
                        st.markdown(f"**Description:**  \n{rec['description']}")
        
        # Export button
        if st.button("📥 Export Recommendations"):
            st.session_state.recommender.export_recommendations(recommendations)
            st.success("Recommendations exported to kpi_recommendations.json")
    
    with col2:
        st.header("📊 Visualization")
                
        # Priority distribution - FIXED
        priority_data = [rec['priority'] for rec in recommendations]
        priority_counts = pd.Series(priority_data).value_counts()
        
        # Convert to DataFrame properly
        priority_df = pd.DataFrame({
            'Priority': priority_counts.index,
            'Count': priority_counts.values
        })
        
        fig_priority = px.pie(
            priority_df,
            values='Count',
            names='Priority',
            title="Recommendations by Priority",
            color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107'}
        )
        st.plotly_chart(fig_priority, use_container_width=True)
        
        # Similarity scores - FIXED
        similarity_data = pd.DataFrame({
            'KPI': [f"KPI {i+1}" for i in range(len(recommendations))],
            'Score': [rec['similarity_score'] for rec in recommendations]
        })
        
        fig_similarity = go.Figure(data=[
            go.Bar(
                x=similarity_data['KPI'],
                y=similarity_data['Score'],
                marker_color='#1f77b4'
            )
        ])
        fig_similarity.update_layout(
            title="Similarity Scores",
            yaxis_title="Score",
            xaxis_title="KPI"
        )
        st.plotly_chart(fig_similarity, use_container_width=True)
        
        # Implementation checklist
        st.subheader("✅ Implementation Checklist")
        st.checkbox("Set up data collection infrastructure", value=False)
        st.checkbox("Define baseline measurements", value=False)
        st.checkbox("Create automated dashboards", value=False)
        st.checkbox("Establish reporting cadence", value=False)
        st.checkbox("Align team incentives with KPIs", value=False)

else:
    # Welcome screen
    st.info("👆 Please select your Industry, Department, and Business Goal from the sidebar to get started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total KPIs", len(st.session_state.recommender.df_kpis))
    with col2:
        st.metric("Industries", st.session_state.recommender.df_kpis['industry'].nunique())
    with col3:
        st.metric("Departments", st.session_state.recommender.df_kpis['department'].nunique())
    
    st.divider()
    
    # Feature highlights
    st.subheader("🌟 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Content-Based Recommendations**
        - Machine learning-powered suggestions
        - Industry-specific metrics
        - Department-aligned KPIs
        """)
        
        st.markdown("""
        **Comprehensive KPI Database**
        - E-commerce, SaaS, Healthcare, Retail
        - Marketing, Sales, Operations, Finance
        - Priority rankings and benchmarks
        """)
    
    with col2:
        st.markdown("""
        **Interactive Visualizations**
        - Priority distribution charts
        - Similarity score analysis
        - Performance tracking tools
        """)
        
        st.markdown("""
        **Export & Implementation**
        - JSON export functionality
        - Implementation checklists
        - Best practice guidance
        """)

# Analytics section (sidebar toggle)
if show_analytics:
    st.divider()
    st.header("📈 Database Analytics")
    
    analytics = st.session_state.recommender.get_kpi_analytics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By Industry")
        df_industry = pd.DataFrame(analytics['industries'].items(), columns=['Industry', 'Count'])
        fig = px.bar(df_industry, x='Industry', y='Count', color='Industry')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("By Priority")
        df_priority = pd.DataFrame(analytics['priorities'].items(), columns=['Priority', 'Count'])
        fig = px.pie(df_priority, values='Count', names='Priority', 
                    color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("By Department")
    df_dept = pd.DataFrame(analytics['departments'].items(), columns=['Department', 'Count'])
    fig = px.bar(df_dept, x='Department', y='Count', color='Department')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit | KPI Recommender System v1.0</p>
        <p>📧 Questions? Contact: your.email@example.com</p>
    </div>
""", unsafe_allow_html=True)