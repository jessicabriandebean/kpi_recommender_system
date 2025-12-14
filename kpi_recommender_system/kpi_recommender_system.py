# kpi_recommender_system.py
"""
KPI Recommender System - Data Science Project
A machine learning-based recommendation system for suggesting relevant KPIs
based on industry, department, and business goals.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

class KPIRecommender:
    """
    A content-based recommendation system that suggests KPIs based on 
    industry context, department focus, and business objectives.
    """
    
    def __init__(self):
        self.kpi_database = None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
        self.feature_matrix = None
        self.df_kpis = None
        
    def load_kpi_database(self):
        """Load and structure KPI database"""
        kpis = [
            # E-commerce Marketing KPIs
            {
                'kpi_name': 'Customer Acquisition Cost (CAC)',
                'industry': 'E-commerce',
                'department': 'Marketing',
                'business_goal': 'Increase Revenue',
                'priority': 'High',
                'formula': 'Total Marketing Spend / New Customers',
                'benchmark': '$50-200',
                'description': 'Measures efficiency of customer acquisition efforts',
                'tags': 'acquisition cost marketing efficiency customer',
                'category': 'Financial'
            },
            {
                'kpi_name': 'Return on Ad Spend (ROAS)',
                'industry': 'E-commerce',
                'department': 'Marketing',
                'business_goal': 'Increase Revenue',
                'priority': 'High',
                'formula': 'Revenue from Ads / Ad Spend',
                'benchmark': '4:1 or higher',
                'description': 'Directly measures marketing ROI',
                'tags': 'roi advertising revenue marketing',
                'category': 'Financial'
            },
            {
                'kpi_name': 'Conversion Rate',
                'industry': 'E-commerce',
                'department': 'Marketing',
                'business_goal': 'Increase Revenue',
                'priority': 'High',
                'formula': 'Conversions / Total Visitors × 100',
                'benchmark': '2-3%',
                'description': 'Shows effectiveness of traffic monetization',
                'tags': 'conversion sales funnel optimization',
                'category': 'Performance'
            },
            # SaaS KPIs
            {
                'kpi_name': 'Monthly Recurring Revenue (MRR)',
                'industry': 'SaaS',
                'department': 'Sales',
                'business_goal': 'Increase Revenue',
                'priority': 'High',
                'formula': 'Sum of Monthly Subscriptions',
                'benchmark': 'Growing 10%+ MoM',
                'description': 'Core revenue metric for SaaS businesses',
                'tags': 'revenue subscription recurring growth',
                'category': 'Financial'
            },
            {
                'kpi_name': 'Customer Churn Rate',
                'industry': 'SaaS',
                'department': 'Sales',
                'business_goal': 'Increase Revenue',
                'priority': 'High',
                'formula': 'Lost Customers / Total Customers × 100',
                'benchmark': '<5% monthly',
                'description': 'Critical for sustainable growth',
                'tags': 'retention churn customer loss',
                'category': 'Customer'
            },
            {
                'kpi_name': 'Net Promoter Score (NPS)',
                'industry': 'SaaS',
                'department': 'Product',
                'business_goal': 'Improve Experience',
                'priority': 'Medium',
                'formula': '% Promoters - % Detractors',
                'benchmark': '30+',
                'description': 'Indicates customer satisfaction and loyalty',
                'tags': 'satisfaction loyalty customer feedback',
                'category': 'Customer'
            },
            # Healthcare KPIs
            {
                'kpi_name': 'Patient Wait Time',
                'industry': 'Healthcare',
                'department': 'Operations',
                'business_goal': 'Improve Efficiency',
                'priority': 'High',
                'formula': 'Avg Time from Check-in to Appointment',
                'benchmark': '<15 minutes',
                'description': 'Critical for patient satisfaction',
                'tags': 'wait time patient satisfaction efficiency',
                'category': 'Operational'
            },
            {
                'kpi_name': 'Bed Occupancy Rate',
                'industry': 'Healthcare',
                'department': 'Operations',
                'business_goal': 'Improve Efficiency',
                'priority': 'High',
                'formula': 'Occupied Beds / Total Beds × 100',
                'benchmark': '85-90%',
                'description': 'Measures resource utilization',
                'tags': 'capacity utilization resources hospital',
                'category': 'Operational'
            },
            # Retail KPIs
            {
                'kpi_name': 'Same-Store Sales Growth',
                'industry': 'Retail',
                'department': 'Sales',
                'business_goal': 'Increase Revenue',
                'priority': 'High',
                'formula': '(Current Period Sales - Prior Period Sales) / Prior Period Sales × 100',
                'benchmark': '3-5% annually',
                'description': 'Core retail performance metric',
                'tags': 'revenue growth sales performance',
                'category': 'Financial'
            },
            {
                'kpi_name': 'Inventory Turnover',
                'industry': 'Retail',
                'department': 'Operations',
                'business_goal': 'Improve Efficiency',
                'priority': 'High',
                'formula': 'COGS / Average Inventory',
                'benchmark': '6-12 times/year',
                'description': 'Critical for retail profitability',
                'tags': 'inventory efficiency stock management',
                'category': 'Operational'
            }
        ]
        
        self.df_kpis = pd.DataFrame(kpis)
        self.kpi_database = kpis
        return self.df_kpis
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    def create_feature_vectors(self):
        """Create TF-IDF feature vectors for content-based filtering"""
        # Combine relevant text features into one string per KPI
        self.df_kpis['combined_features'] = (
            self.df_kpis['industry'].fillna('') + ' ' +
            self.df_kpis['department'].fillna('') + ' ' +
            self.df_kpis['business_goal'].fillna('') + ' ' +
            self.df_kpis['tags'].fillna('') + ' ' +
            self.df_kpis['description'].fillna('')
        )

        # Initialize vectorizer if not already set
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer()

        # Create TF-IDF matrix
        self.feature_matrix = self.vectorizer.fit_transform(
            self.df_kpis['combined_features']
        )

        return self.feature_matrix
    
    def _generate_recommendations(self, industry, department, business_goal):
        """
        Internal helper: compute similarity scores and return all KPIs with scores.
        """
        query = f"{industry} {department} {business_goal}"
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.feature_matrix)[0]

        recommendations = []
        for idx, score in enumerate(similarity_scores):
            kpi = self.df_kpis.iloc[idx].to_dict()
            kpi["similarity_score"] = float(score)
            recommendations.append(kpi)

        # Sort descending by similarity
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations
    
    def recommend_kpis(self, industry, department, business_goal, top_n=10):
        """
        Public method: return top_n KPIs for the given query.
        """
        # Guard clause
        if not isinstance(top_n, int) or top_n <= 0:
            return []

        # Use helper
        recommendations = self._generate_recommendations(industry, department, business_goal)

        # Slice to top_n
        return recommendations[:top_n]
    
    # def recommend_kpis(self, industry, department, business_goal, top_n=5):
    #     """
    #     Recommend KPIs based on user input using content-based filtering
    
    #     Parameters:
    #     -----------
    #     industry : str
    #     User's industry (e.g., 'E-commerce', 'SaaS')
    #     department : str
    #     User's department (e.g., 'Marketing', 'Sales')
    #     business_goal : str
    #     Business objective (e.g., 'Increase Revenue')
    #     top_n : int
    #     Number of recommendations to return
        
    #     Returns:
    #     --------
    #     list : Recommended KPIs with scores
    #     """
    #     # ✅ Guard clause: handle invalid top_n values first
    #     if not isinstance(top_n, int) or top_n <= 0:
    #         return []
        
        # Create query vector
        query = f"{industry} {department} {business_goal}"
        query_vector = self.vectorizer.transform([query])
    
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, self.feature_matrix)[0]
    
        # Get top recommendations
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
        recommendations = []
        for idx in top_indices:
            kpi = self.df_kpis.iloc[idx].to_dict()
            kpi['similarity_score'] = float(similarity_scores[idx])
            recommendations.append(kpi)
    
        return recommendations
    
    def filter_by_criteria(self, industry=None, department=None, 
                          business_goal=None, priority=None):
        """
        Filter KPIs by exact criteria match
        
        Parameters:
        -----------
        industry : str, optional
        department : str, optional
        business_goal : str, optional
        priority : str, optional
            
        Returns:
        --------
        DataFrame : Filtered KPIs
        """
        df_filtered = self.df_kpis.copy()
        
        if industry:
            df_filtered = df_filtered[df_filtered['industry'] == industry]
        if department:
            df_filtered = df_filtered[df_filtered['department'] == department]
        if business_goal:
            df_filtered = df_filtered[df_filtered['business_goal'] == business_goal]
        if priority:
            df_filtered = df_filtered[df_filtered['priority'] == priority]
            
        return df_filtered
    
    def get_kpi_analytics(self):
        """Generate analytics about the KPI database"""
        analytics = {
            'total_kpis': len(self.df_kpis),
            'industries': self.df_kpis['industry'].value_counts().to_dict(),
            'departments': self.df_kpis['department'].value_counts().to_dict(),
            'priorities': self.df_kpis['priority'].value_counts().to_dict(),
            'categories': self.df_kpis['category'].value_counts().to_dict()
        }
        return analytics
    
    def export_recommendations(self, recommendations, filename='kpi_recommendations.json'):
        """Export recommendations to JSON file"""
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'recommendations': recommendations
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Recommendations exported to {filename}")


# Example usage and testing
    if __name__ == "__main__":
        # Initialize recommender
        recommender = KPIRecommender()
        
        # Load KPI database
        print("Loading KPI database...")
        df_kpis = recommender.load_kpi_database()
        print(f"Loaded {len(df_kpis)} KPIs")
        print("\nKPI Database Preview:")
        print(df_kpis[['kpi_name', 'industry', 'department', 'priority']].head(10))
        
        # Create feature vectors
        print("\nCreating feature vectors for recommendations...")
        recommender.create_feature_vectors()
        
        # Test recommendations
        print("\n" + "="*60)
        print("TESTING RECOMMENDATIONS")
        print("="*60)
        
        # Test case 1: E-commerce Marketing
        print("\nTest Case 1: E-commerce Marketing - Increase Revenue")
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=3
        )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['kpi_name']} (Score: {rec['similarity_score']:.3f})")
            print(f"   Priority: {rec['priority']}")
            print(f"   Formula: {rec['formula']}")
            print(f"   Benchmark: {rec['benchmark']}")
        
        # Test case 2: SaaS Sales
        print("\n" + "-"*60)
        print("\nTest Case 2: SaaS Sales - Increase Revenue")
        recommendations = recommender.recommend_kpis(
            industry='SaaS',
            department='Sales',
            business_goal='Increase Revenue',
            top_n=3
        )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['kpi_name']} (Score: {rec['similarity_score']:.3f})")
            print(f"   Priority: {rec['priority']}")
            print(f"   Formula: {rec['formula']}")
        
        # Display analytics
        print("\n" + "="*60)
        print("KPI DATABASE ANALYTICS")
        print("="*60)
        analytics = recommender.get_kpi_analytics()
        print(f"\nTotal KPIs: {analytics['total_kpis']}")
        print(f"\nBy Industry: {analytics['industries']}")
        print(f"\nBy Department: {analytics['departments']}")
        print(f"\nBy Priority: {analytics['priorities']}")
        
        # Export example
        print("\n" + "="*60)
        recommender.export_recommendations(recommendations)
        print("Recommendations exported successfully!")
