"""
Unit Tests for KPI Recommender System
Run with: pytest test_recommender.py -v
Or: python -m pytest test_recommender.py --cov=kpi_recommender_system
"""

import pytest
import pandas as pd
import numpy as np
from kpi_recommender_system import KPIRecommender
import json
import os

class KPIRecommender:
    def recommend_kpis(self, industry, department, business_goal, top_n=10):
        # Guard clause: invalid top_n values return empty list
        if not isinstance(top_n, int) or top_n <= 0:
            return []

        # Your existing recommendation logic
        recommendations = self._generate_recommendations(industry, department, business_goal)

        # Slice to top_n
        return recommendations[:top_n]

class TestKPIRecommender:
    """Test suite for KPIRecommender class"""
    
    @pytest.fixture
    def recommender(self):
        """Fixture to create a fresh recommender instance for each test"""
        rec = KPIRecommender()
        rec.load_kpi_database()   # ✅ matches your class definition
        rec.create_feature_vectors()
        return rec
    
    # ========== Database Loading Tests ==========
    
    def test_database_loading(self, recommender):
        """Test that database loads correctly"""
        assert recommender.df_kpis is not None
        assert len(recommender.df_kpis) > 0
        assert isinstance(recommender.df_kpis, pd.DataFrame)
    
    def test_database_columns(self, recommender):
        """Test that all required columns exist"""
        required_columns = [
            'kpi_name', 'industry', 'department', 'business_goal',
            'priority', 'formula', 'benchmark', 'description', 
            'tags', 'category'
        ]
        for col in required_columns:
            assert col in recommender.df_kpis.columns, f"Missing column: {col}"
    
    def test_no_missing_values(self, recommender):
        """Test that there are no missing values in critical columns"""
        critical_columns = ['kpi_name', 'industry', 'department', 'formula']
        for col in critical_columns:
            assert recommender.df_kpis[col].notna().all(), f"Missing values in {col}"
    
    def test_valid_priorities(self, recommender):
        """Test that all priorities are valid"""
        valid_priorities = ['High', 'Medium', 'Low']
        priorities = recommender.df_kpis['priority'].unique()
        for priority in priorities:
            assert priority in valid_priorities, f"Invalid priority: {priority}"
    
    def test_valid_categories(self, recommender):
        """Test that all categories are valid"""
        valid_categories = ['Financial', 'Operational', 'Customer', 'Performance']
        categories = recommender.df_kpis['category'].unique()
        for category in categories:
            assert category in valid_categories, f"Invalid category: {category}"
    
    # ========== Feature Vector Tests ==========
    
    def test_feature_matrix_creation(self, recommender):
        """Test that feature matrix is created correctly"""
        assert recommender.feature_matrix is not None
        assert recommender.feature_matrix.shape[0] == len(recommender.df_kpis)
        assert recommender.feature_matrix.shape[1] > 0
    
    def test_vectorizer_initialization(self, recommender):
        """Test that TF-IDF vectorizer is initialized"""
        assert recommender.vectorizer is not None
        assert hasattr(recommender.vectorizer, 'get_feature_names_out')
    
    def test_combined_features_column(self, recommender):
        """Test that combined features column is created"""
        assert 'combined_features' in recommender.df_kpis.columns
        assert recommender.df_kpis['combined_features'].notna().all()
    
    # ========== Recommendation Tests ==========
    
    def test_basic_recommendation(self, recommender):
        """Test basic recommendation functionality"""
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=5
        )
        assert len(recommendations) == 5
        assert all(isinstance(rec, dict) for rec in recommendations)
    
    def test_recommendation_structure(self, recommender):
        """Test that recommendations have correct structure"""
        recommendations = recommender.recommend_kpis(
            industry='SaaS',
            department='Sales',
            business_goal='Increase Revenue',
            top_n=3
        )
        
        required_keys = [
            'kpi_name', 'industry', 'department', 'business_goal',
            'priority', 'formula', 'benchmark', 'description',
            'similarity_score'
        ]
        
        for rec in recommendations:
            for key in required_keys:
                assert key in rec, f"Missing key: {key}"
    
    def test_recommendation_scores_descending(self, recommender):
        """Test that recommendations are sorted by score (descending)"""
        recommendations = recommender.recommend_kpis(
            industry='Healthcare',
            department='Operations',
            business_goal='Improve Efficiency',
            top_n=5
        )
        
        scores = [rec['similarity_score'] for rec in recommendations]
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"
    
    def test_similarity_score_range(self, recommender):
        """Test that similarity scores are in valid range [0, 1]"""
        recommendations = recommender.recommend_kpis(
            industry='Retail',
            department='Sales',
            business_goal='Increase Revenue',
            top_n=5
        )
        
        for rec in recommendations:
            assert 0 <= rec['similarity_score'] <= 1, \
                f"Invalid similarity score: {rec['similarity_score']}"
    
    def test_top_n_parameter(self, recommender):
        """Test that top_n parameter works correctly"""
        for n in [1, 3, 5, 10]:
            recommendations = recommender.recommend_kpis(
                industry='E-commerce',
                department='Marketing',
                business_goal='Increase Revenue',
                top_n=n
            )
            assert len(recommendations) == min(n, len(recommender.df_kpis))
    
    def test_different_industries(self, recommender):
        """Test recommendations for different industries"""
        industries = recommender.df_kpis['industry'].unique()
        
        for industry in industries:
            # Get a valid department for this industry
            valid_depts = recommender.df_kpis[
                recommender.df_kpis['industry'] == industry
            ]['department'].unique()
            
            if len(valid_depts) > 0:
                recommendations = recommender.recommend_kpis(
                    industry=industry,
                    department=valid_depts[0],
                    business_goal='Increase Revenue',
                    top_n=3
                )
                assert len(recommendations) > 0, f"No recommendations for {industry}"
    
    # ========== Filter Tests ==========
    
    def test_filter_by_industry(self, recommender):
        """Test filtering by industry"""
        filtered = recommender.filter_by_criteria(industry='E-commerce')
        assert all(filtered['industry'] == 'E-commerce')
        assert len(filtered) > 0
    
    def test_filter_by_department(self, recommender):
        """Test filtering by department"""
        filtered = recommender.filter_by_criteria(department='Marketing')
        assert all(filtered['department'] == 'Marketing')
        assert len(filtered) > 0
    
    def test_filter_by_priority(self, recommender):
        """Test filtering by priority"""
        filtered = recommender.filter_by_criteria(priority='High')
        assert all(filtered['priority'] == 'High')
        assert len(filtered) > 0
    
    def test_filter_multiple_criteria(self, recommender):
        """Test filtering by multiple criteria"""
        filtered = recommender.filter_by_criteria(
            industry='SaaS',
            department='Sales',
            priority='High'
        )
        assert all(filtered['industry'] == 'SaaS')
        assert all(filtered['department'] == 'Sales')
        assert all(filtered['priority'] == 'High')
    
    def test_filter_returns_dataframe(self, recommender):
        """Test that filter returns a DataFrame"""
        filtered = recommender.filter_by_criteria(industry='Healthcare')
        assert isinstance(filtered, pd.DataFrame)
    
    # ========== Analytics Tests ==========
    
    def test_analytics_structure(self, recommender):
        """Test that analytics returns correct structure"""
        analytics = recommender.get_kpi_analytics()
        
        required_keys = ['total_kpis', 'industries', 'departments', 
                        'priorities', 'categories']
        for key in required_keys:
            assert key in analytics, f"Missing analytics key: {key}"
    
    def test_analytics_total_kpis(self, recommender):
        """Test that total KPIs count is correct"""
        analytics = recommender.get_kpi_analytics()
        assert analytics['total_kpis'] == len(recommender.df_kpis)
    
    def test_analytics_counts_sum(self, recommender):
        """Test that analytics counts sum correctly"""
        analytics = recommender.get_kpi_analytics()
        
        # Sum of industry counts should equal total
        industry_sum = sum(analytics['industries'].values())
        assert industry_sum == analytics['total_kpis']
        
        # Sum of priority counts should equal total
        priority_sum = sum(analytics['priorities'].values())
        assert priority_sum == analytics['total_kpis']
    
    # ========== Export Tests ==========
    
    def test_export_recommendations(self, recommender, tmp_path):
        """Test exporting recommendations to JSON"""
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=3
        )
        
        # Export to temporary file
        output_file = tmp_path / "test_recommendations.json"
        recommender.export_recommendations(recommendations, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert 'generated_at' in data
        assert 'recommendations' in data
        assert len(data['recommendations']) == 3
    
    # ========== Edge Case Tests ==========
    
    def test_empty_query(self, recommender):
        """Test recommendation with empty query strings"""
        recommendations = recommender.recommend_kpis(
            industry='',
            department='',
            business_goal='',
            top_n=5
        )
        # Should still return recommendations (may not be very relevant)
        assert len(recommendations) > 0
    
    
    def test_invalid_top_n(self, recommender):
        """Test with invalid top_n values"""
        # Zero
        recommendations = recommender.recommend_kpis(
        industry='E-commerce',
        department='Marketing',
        business_goal='Increase Revenue',
        top_n=0
        )
        assert len(recommendations) == 0

    # Negative (should be handled gracefully)
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=-1
        )
        assert len(recommendations) == 0
    
    def test_very_large_top_n(self, recommender):
        """Test with top_n larger than database"""
        large_n = len(recommender.df_kpis) + 100
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=large_n
        )
        assert len(recommendations) <= len(recommender.df_kpis)
    
    # ========== Data Quality Tests ==========
    
    def test_no_duplicate_kpis(self, recommender):
        """Test that there are no duplicate KPI names"""
        duplicates = recommender.df_kpis['kpi_name'].duplicated()
        assert not duplicates.any(), "Found duplicate KPI names"
    
    def test_formula_not_empty(self, recommender):
        """Test that all formulas are non-empty"""
        assert all(recommender.df_kpis['formula'].str.len() > 0)
    
    def test_benchmark_not_empty(self, recommender):
        """Test that all benchmarks are non-empty"""
        assert all(recommender.df_kpis['benchmark'].str.len() > 0)
    
    def test_tags_contain_keywords(self, recommender):
        """Test that tags contain relevant keywords"""
        for _, row in recommender.df_kpis.iterrows():
            tags = row['tags'].lower()
            # Tags should contain at least one word from the KPI name
            kpi_words = set(row['kpi_name'].lower().split())
            tag_words = set(tags.split())
            # Allow some flexibility - at least some overlap
            assert len(kpi_words & tag_words) > 0 or len(tags) > 10


class TestIntegration:
    """Integration tests for end-to-end workflows"""
    
    @pytest.fixture
    def recommender(self):
        """Fixture for integration tests"""
        rec = KPIRecommender()
        rec.load_kpi_database()
        rec.create_feature_vectors()
        return rec
    
    def test_complete_workflow(self, recommender):
        """Test complete workflow from init to export"""
        # 1. Get recommendations
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=5
        )
        
        assert len(recommendations) == 5
        
        # 2. Get analytics
        analytics = recommender.get_kpi_analytics()
        assert analytics['total_kpis'] > 0
        
        # 3. Filter KPIs
        filtered = recommender.filter_by_criteria(
            industry='E-commerce',
            priority='High'
        )
        assert len(filtered) > 0
        
        # 4. Export (to temp location)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            recommender.export_recommendations(recommendations, temp_file)
            assert os.path.exists(temp_file)
            
            with open(temp_file, 'r') as f:
                data = json.load(f)
            assert len(data['recommendations']) == 5
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_recommendation_relevance(self, recommender):
        """Test that recommendations are relevant to input"""
        # E-commerce Marketing recommendations should mostly be E-commerce
        recommendations = recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=5
        )
        
        # At least top 3 should match industry or department
        top_3 = recommendations[:3]
        matches = sum(
            1 for rec in top_3 
            if rec['industry'] == 'E-commerce' or rec['department'] == 'Marketing'
        )
        assert matches >= 2, "Top recommendations not relevant enough"


class TestPerformance:
    """Performance tests"""
    
    @pytest.fixture
    def recommender(self):
        rec = KPIRecommender()
        rec.load_kpi_database()
        rec.create_feature_vectors()
        return rec
    
    def test_recommendation_speed(self, recommender):
        """Test that recommendations are generated quickly"""
        import time
        
        start = time.time()
        recommender.recommend_kpis(
            industry='E-commerce',
            department='Marketing',
            business_goal='Increase Revenue',
            top_n=5
        )
        elapsed = time.time() - start
        
        # Should complete in under 1 second
        assert elapsed < 1.0, f"Recommendation took {elapsed:.2f}s (too slow)"
    
    def test_multiple_recommendations_performance(self, recommender):
        """Test performance of multiple recommendation calls"""
        import time
        
        start = time.time()
        for _ in range(10):
            recommender.recommend_kpis(
                industry='E-commerce',
                department='Marketing',
                business_goal='Increase Revenue',
                top_n=5
            )
        elapsed = time.time() - start
        
        # 10 calls should complete in under 3 seconds
        assert elapsed < 3.0, f"10 recommendations took {elapsed:.2f}s (too slow)"


# ========== Pytest Configuration ==========

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--cov=kpi_recommender_system", 
                 "--cov-report=html", "--cov-report=term"])