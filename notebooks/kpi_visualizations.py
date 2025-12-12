"""
Advanced Visualizations for KPI Recommender System
Creates publication-ready charts and interactive dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from projects.kpi_recommender_system.kpi_recommender_system import KPIRecommender
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)


class KPIVisualizer:
    """Create advanced visualizations for KPI analysis"""
    
    def __init__(self):
        self.recommender = KPIRecommender()
        self.recommender.load_kpi_database()
        self.recommender.create_feature_vectors()
        self.df = self.recommender.df_kpis
        
    def create_dashboard(self):
        """Create comprehensive dashboard with multiple views"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'KPIs by Industry',
                'Priority Distribution',
                'Department Coverage',
                'Category Breakdown',
                'Industry-Priority Matrix',
                'Top KPI Categories'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'sunburst'}],
                [{'type': 'heatmap'}, {'type': 'treemap'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. KPIs by Industry (Bar)
        industry_counts = self.df['industry'].value_counts()
        fig.add_trace(
            go.Bar(
                x=industry_counts.index,
                y=industry_counts.values,
                marker_color='lightblue',
                name='Industry',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Priority Distribution (Pie)
        priority_counts = self.df['priority'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=priority_counts.index,
                values=priority_counts.values,
                marker=dict(colors=['#FF6B6B', '#FFD93D']),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 3. Department Coverage (Bar)
        dept_counts = self.df['department'].value_counts()
        fig.add_trace(
            go.Bar(
                x=dept_counts.values,
                y=dept_counts.index,
                orientation='h',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Category Breakdown (Sunburst)
        sunburst_data = self.df.groupby(['category', 'priority']).size().reset_index(name='count')
        fig.add_trace(
            go.Sunburst(
                labels=sunburst_data['category'].tolist() + sunburst_data['priority'].tolist(),
                parents=[''] * len(sunburst_data['category']) + sunburst_data['category'].tolist(),
                values=sunburst_data['count'].tolist() + sunburst_data['count'].tolist(),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Industry-Priority Heatmap
        heatmap_data = pd.crosstab(self.df['industry'], self.df['priority'])
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                showscale=True
            ),
            row=3, col=1
        )
        
        # 6. Treemap of Categories
        treemap_data = self.df.groupby(['category', 'industry']).size().reset_index(name='count')
        fig.add_trace(
            go.Treemap(
                labels=treemap_data['industry'].tolist(),
                parents=treemap_data['category'].tolist(),
                values=treemap_data['count'].tolist(),
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="KPI Recommender System - Comprehensive Dashboard",
            title_font_size=20,
            height=1200,
            showlegend=True
        )
        
        return fig
    
    def plot_recommendation_comparison(self, test_cases):
        """Compare recommendations across different scenarios"""
        results = []
        
        for test in test_cases:
            recs = self.recommender.recommend_kpis(
                industry=test['industry'],
                department=test['department'],
                business_goal=test['business_goal'],
                top_n=5
            )
            
            for i, rec in enumerate(recs, 1):
                results.append({
                    'scenario': f"{test['industry']}-{test['department']}",
                    'rank': i,
                    'kpi': rec['kpi_name'][:30],  # Truncate for display
                    'score': rec['similarity_score'],
                    'priority': rec['priority']
                })
        
        df_results = pd.DataFrame(results)
        
        # Create grouped bar chart
        fig = px.bar(
            df_results,
            x='scenario',
            y='score',
            color='priority',
            pattern_shape='rank',
            title='Recommendation Scores Across Different Scenarios',
            labels={'score': 'Similarity Score', 'scenario': 'Scenario'},
            color_discrete_map={'High': '#FF6B6B', 'Medium': '#FFD93D'},
            height=600
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            legend_title_text='Priority'
        )
        
        return fig
    
    def plot_kpi_network(self):
        """Create network graph showing KPI relationships"""
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(
            self.recommender.feature_matrix.toarray()
        )
        
        # Create edge list for highly similar KPIs
        edges = []
        threshold = 0.5  # Only show strong connections
        
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                if similarity_matrix[i, j] > threshold:
                    edges.append({
                        'source': self.df.iloc[i]['kpi_name'],
                        'target': self.df.iloc[j]['kpi_name'],
                        'weight': similarity_matrix[i, j],
                        'industry': self.df.iloc[i]['industry']
                    })
        
        df_edges = pd.DataFrame(edges)
        
        # Get node information
        nodes = []
        for _, row in self.df.iterrows():
            nodes.append({
                'id': row['kpi_name'],
                'industry': row['industry'],
                'department': row['department'],
                'priority': row['priority']
            })
        
        df_nodes = pd.DataFrame(nodes)
        
        # Create network visualization using plotly
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes
        for _, node in df_nodes.iterrows():
            G.add_node(node['id'], 
                      industry=node['industry'],
                      department=node['department'],
                      priority=node['priority'])
        
        # Add edges
        for _, edge in df_edges.iterrows():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        # Get positions
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create edge traces
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        color_map = {
            'E-commerce': '#FF6B6B',
            'SaaS': '#4ECDC4',
            'Healthcare': '#95E1D3',
            'Retail': '#FFD93D'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(color_map.get(G.nodes[node]['industry'], '#888'))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=15,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title='KPI Relationship Network (Similarity > 0.5)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800
        )
        
        return fig
    
    def plot_priority_funnel(self):
        """Create funnel chart showing KPI priorities"""
        priority_counts = self.df['priority'].value_counts()
        
        fig = go.Figure(go.Funnel(
            y=['High Priority', 'Medium Priority'],
            x=priority_counts.values,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=['#FF6B6B', '#FFD93D'])
        ))
        
        fig.update_layout(
            title='KPI Priority Funnel',
            height=500
        )
        
        return fig
    
    def plot_kpi_radar(self, industry, department):
        """Create radar chart comparing KPI categories"""
        # Filter for specific context
        df_filtered = self.df[
            (self.df['industry'] == industry) & 
            (self.df['department'] == department)
        ]
        
        if len(df_filtered) == 0:
            print(f"No KPIs found for {industry} - {department}")
            return None
        
        # Count by category
        category_counts = df_filtered['category'].value_counts()
        
        categories = category_counts.index.tolist()
        values = category_counts.values.tolist()
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'{industry} - {department}',
            marker_color='#4ECDC4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) + 1]
                )
            ),
            title=f'KPI Category Distribution: {industry} - {department}',
            height=600
        )
        
        return fig
    
    def create_animated_timeline(self):
        """Create animated visualization (simulated timeline)"""
        # Simulate KPI adoption over time
        np.random.seed(42)
        
        timeline_data = []
        for month in range(1, 13):
            for industry in self.df['industry'].unique():
                count = len(self.df[self.df['industry'] == industry])
                # Simulate gradual increase
                timeline_data.append({
                    'month': month,
                    'industry': industry,
                    'kpis': int(count * (0.3 + 0.7 * month / 12))
                })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        fig = px.line(
            df_timeline,
            x='month',
            y='kpis',
            color='industry',
            title='Simulated KPI Adoption Timeline',
            labels={'month': 'Month', 'kpis': 'Number of KPIs'},
            markers=True
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            height=600
        )
        
        return fig
    
    def create_3d_scatter(self):
        """Create 3D scatter plot of KPIs"""
        from sklearn.decomposition import PCA
        
        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(
            self.recommender.feature_matrix.toarray()
        )
        
        df_3d = pd.DataFrame(
            features_3d,
            columns=['PC1', 'PC2', 'PC3']
        )
        df_3d['kpi_name'] = self.df['kpi_name'].values
        df_3d['industry'] = self.df['industry'].values
        df_3d['priority'] = self.df['priority'].values
        df_3d['department'] = self.df['department'].values
        
        fig = px.scatter_3d(
            df_3d,
            x='PC1',
            y='PC2',
            z='PC3',
            color='industry',
            symbol='priority',
            hover_data=['kpi_name', 'department'],
            title='KPI Feature Space (3D PCA)',
            height=800
        )
        
        fig.update_traces(marker=dict(size=8))
        
        return fig
    
    def export_all_visualizations(self, output_dir='visualizations'):
        """Export all visualizations to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualizations...")
        
        # 1. Dashboard
        print("  1. Dashboard...")
        fig = self.create_dashboard()
        fig.write_html(f'{output_dir}/dashboard.html')
        
        # 2. Comparison
        print("  2. Recommendation comparison...")
        test_cases = [
            {'industry': 'E-commerce', 'department': 'Marketing', 'business_goal': 'Increase Revenue'},
            {'industry': 'SaaS', 'department': 'Sales', 'business_goal': 'Increase Revenue'},
            {'industry': 'Healthcare', 'department': 'Operations', 'business_goal': 'Improve Efficiency'}
        ]
        fig = self.plot_recommendation_comparison(test_cases)
        fig.write_html(f'{output_dir}/comparison.html')
        
        # 3. Network
        print("  3. Network graph...")
        fig = self.plot_kpi_network()
        fig.write_html(f'{output_dir}/network.html')
        
        # 4. Funnel
        print("  4. Priority funnel...")
        fig = self.plot_priority_funnel()
        fig.write_html(f'{output_dir}/funnel.html')
        
        # 5. 3D Scatter
        print("  5. 3D scatter plot...")
        fig = self.create_3d_scatter()
        fig.write_html(f'{output_dir}/3d_scatter.html')
        
        # 6. Timeline
        print("  6. Timeline animation...")
        fig = self.create_animated_timeline()
        fig.write_html(f'{output_dir}/timeline.html')
        
        print(f"\n✅ All visualizations saved to '{output_dir}/' directory!")
        print(f"   Open the HTML files in your browser to view them.")


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("KPI RECOMMENDER SYSTEM - ADVANCED VISUALIZATIONS")
    print("="*70)
    
    # Initialize visualizer
    viz = KPIVisualizer()
    
    # Create and show individual visualizations
    print("\n1. Creating comprehensive dashboard...")
    fig = viz.create_dashboard()
    fig.show()
    
    print("\n2. Creating recommendation comparison...")
    test_cases = [
        {'industry': 'E-commerce', 'department': 'Marketing', 'business_goal': 'Increase Revenue'},
        {'industry': 'SaaS', 'department': 'Sales', 'business_goal': 'Increase Revenue'},
        {'industry': 'Healthcare', 'department': 'Operations', 'business_goal': 'Improve Efficiency'}
    ]
    fig = viz.plot_recommendation_comparison(test_cases)
    fig.show()
    
    print("\n3. Creating KPI network graph...")
    fig = viz.plot_kpi_network()
    fig.show()
    
    print("\n4. Creating 3D scatter plot...")
    fig = viz.create_3d_scatter()
    fig.show()
    
    # Export all
    print("\n" + "="*70)
    print("EXPORTING ALL VISUALIZATIONS")
    print("="*70)
    viz.export_all_visualizations()
    
    print("\n🎉 Visualization generation complete!")
    print("All charts are interactive and can be embedded in presentations or reports.")