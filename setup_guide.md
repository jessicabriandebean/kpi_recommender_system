# 🚀 KPI Recommender System - Complete Setup Guide

This guide will walk you through setting up the KPI Recommender System from scratch, including all dependencies, testing, and deployment.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Installation Steps](#installation-steps)
4. [Running Tests](#running-tests)
5. [Running the Notebook](#running-the-notebook)
6. [Running the Application](#running-the-application)
7. [Creating Visualizations](#creating-visualizations)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Required Software
- **Python 3.8+** (3.9 or 3.10 recommended)
- **pip** (Python package manager)
- **Git** (for version control)
- **Virtual environment** (venv or conda)

### Check Your Python Version
```bash
python --version
# or
python3 --version
```

Should output: `Python 3.8.x` or higher

---

## 📁 Project Structure

Create the following directory structure:

```
kpi-recommender-system/
│
├── kpi_recommender_system.py       # Core recommendation engine
├── streamlit_app.py                 # Streamlit web application
├── advanced_visualizations.py       # Visualization module
├── test_recommender.py              # Unit tests
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── SETUP_GUIDE.md                   # This file
├── .gitignore                       # Git ignore file
│
├── notebooks/
│   └── analysis.ipynb              # Jupyter notebook
│
├── data/
│   └── (optional data files)
│
├── tests/
│   └── __init__.py
│   └── test_recommender.py         # Symlink or copy
│
├── visualizations/
│   └── (generated HTML files)
│
└── screenshots/
    └── (app screenshots for README)
```

---

## 📥 Installation Steps

### Step 1: Clone or Create Repository

**Option A: Clone from GitHub**
```bash
git clone https://github.com/yourusername/kpi-recommender-system.git
cd kpi-recommender-system
```

**Option B: Create New Project**
```bash
mkdir kpi-recommender-system
cd kpi-recommender-system
git init
```

### Step 2: Create Virtual Environment

**Using venv (Python built-in):**
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n kpi-recommender python=3.9
conda activate kpi-recommender
```

### Step 3: Create requirements.txt

Create a file named `requirements.txt` with the following content:

```txt
# Core dependencies
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Web application
streamlit==1.28.0

# Jupyter notebook
jupyter==1.0.0
notebook==7.0.3
ipykernel==6.25.2

# Testing
pytest==7.4.2
pytest-cov==4.1.0

# Network graphs
networkx==3.1

# Optional but recommended
python-dotenv==1.0.0
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This may take a few minutes. You should see output like:
```
Collecting pandas==2.0.3
  Downloading pandas-2.0.3-cp39-cp39-...
Successfully installed pandas-2.0.3 numpy-1.24.3 ...
```

### Step 5: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, streamlit, plotly; print('All packages installed successfully!')"
```

Should output: `All packages installed successfully!`

### Step 6: Create Project Files

Copy the code from the artifacts into these files:
1. `kpi_recommender_system.py` - Core library
2. `streamlit_app.py` - Streamlit app
3. `test_recommender.py` - Unit tests
4. `advanced_visualizations.py` - Visualization module
5. `README.md` - Documentation

### Step 7: Create .gitignore

Create `.gitignore` file:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
*.csv
*.json
!requirements.txt
visualizations/*.html
.pytest_cache/
htmlcov/
.coverage

# OS
.DS_Store
Thumbs.db
```

---

## 🧪 Running Tests

### Run All Tests
```bash
pytest test_recommender.py -v
```

Expected output:
```
test_recommender.py::TestKPIRecommender::test_database_loading PASSED
test_recommender.py::TestKPIRecommender::test_database_columns PASSED
...
==================== 45 passed in 2.34s ====================
```

### Run Tests with Coverage
```bash
pytest test_recommender.py --cov=kpi_recommender_system --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`

### Run Specific Test Classes
```bash
# Test only the recommender class
pytest test_recommender.py::TestKPIRecommender -v

# Test only integration tests
pytest test_recommender.py::TestIntegration -v

# Test only performance
pytest test_recommender.py::TestPerformance -v
```

---

## 📓 Running the Notebook

### Step 1: Start Jupyter
```bash
jupyter notebook
```

This opens your browser to `http://localhost:8888`

### Step 2: Create New Notebook
1. Navigate to `notebooks/` folder
2. Click "New" → "Python 3"
3. Copy the code from the analysis notebook artifact
4. Save as `analysis.ipynb`

### Step 3: Run All Cells
- Click "Cell" → "Run All"
- Or press `Shift + Enter` to run cells individually

### Convert Notebook to HTML
```bash
jupyter nbconvert --to html notebooks/analysis.ipynb
```

---

## 🌐 Running the Application

### Option 1: Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

The app will automatically open in your browser!

### Option 2: Core Library Only

Create a test script `test_run.py`:
```python
from kpi_recommender_system import KPIRecommender

# Initialize
recommender = KPIRecommender()
recommender.load_kpi_database()
recommender.create_feature_vectors()

# Get recommendations
recommendations = recommender.recommend_kpis(
    industry='E-commerce',
    department='Marketing',
    business_goal='Increase Revenue',
    top_n=5
)

# Display
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['kpi_name']} - Score: {rec['similarity_score']:.3f}")
```

Run it:
```bash
python test_run.py
```

---

## 📊 Creating Visualizations

### Generate All Visualizations
```bash
python advanced_visualizations.py
```

This will:
1. Create interactive charts
2. Save them to `visualizations/` folder
3. Open them in your browser

### View Generated Visualizations
```bash
# On macOS
open visualizations/dashboard.html

# On Windows
start visualizations/dashboard.html

# On Linux
xdg-open visualizations/dashboard.html
```

### Generate Individual Visualizations

Create a custom script:
```python
from advanced_visualizations import KPIVisualizer

viz = KPIVisualizer()

# Create specific visualization
fig = viz.create_dashboard()
fig.show()
```

---

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/kpi-recommender-system.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click "New app"
- Select your repository
- Set main file to `streamlit_app.py`
- Click "Deploy"!

Your app will be live at: `https://yourusername-kpi-recommender-system.streamlit.app`

### Deploy to Heroku

1. **Create Procfile**
```bash
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

2. **Create setup.sh**
```bash
cat > setup.sh << 'EOF'
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
EOF
```

3. **Deploy**
```bash
heroku create your-kpi-app
git push heroku main
heroku open
```

### Deploy with Docker

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and Run**
```bash
docker build -t kpi-recommender .
docker run -p 8501:8501 kpi-recommender
```

Open `http://localhost:8501`

---

## 🔍 Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'xxx'`
**Solution:**
```bash
pip install xxx
# or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

#### Issue: `streamlit: command not found`
**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall streamlit
pip install streamlit
```

#### Issue: Port 8501 already in use
**Solution:**
```bash
# Run on different port
streamlit run streamlit_app.py --server.port 8502
```

#### Issue: Visualizations not showing in Jupyter
**Solution:**
```bash
# Install plotly extension
pip install "notebook>=5.3" "ipywidgets>=7.5"
jupyter nbextension enable --py widgetsnbextension
```

#### Issue: Tests failing
**Solution:**
```bash
# Make sure you're in the project root
cd kpi-recommender-system

# Run tests with verbose output
pytest test_recommender.py -v -s

# Check specific failing test
pytest test_recommender.py::TestKPIRecommender::test_name -v
```

#### Issue: Import errors in notebook
**Solution:**
```python
# Add project root to Python path (in notebook cell)
import sys
sys.path.append('..')
from kpi_recommender_system import KPIRecommender
```

### Getting Help

If you encounter issues:

1. Check the error message carefully
2. Search for the error on Google/Stack Overflow
3. Check GitHub Issues (if applicable)
4. Ask on Reddit r/Python or r/datascience
5. Contact the maintainer (see README)

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] Tests pass (`pytest test_recommender.py`)
- [ ] Core library works (`python test_run.py`)
- [ ] Streamlit app runs (`streamlit run streamlit_app.py`)
- [ ] Jupyter notebook opens (`jupyter notebook`)
- [ ] Visualizations generate (`python advanced_visualizations.py`)
- [ ] Git repository initialized (`git status`)

---

## 📚 Next Steps

After successful setup:

1. ✅ Run the Jupyter notebook to understand the analysis
2. ✅ Explore the Streamlit app features
3. ✅ Generate and review visualizations
4. ✅ Add more KPIs to the database
5. ✅ Customize the UI/styling
6. ✅ Deploy to Streamlit Cloud
7. ✅ Add to your portfolio/resume
8. ✅ Share on LinkedIn/GitHub

---

## 🎉 Congratulations!

You now have a fully functional KPI Recommender System ready for your portfolio!

**What you've built:**
- ✅ Machine learning recommendation engine
- ✅ Interactive web application
- ✅ Comprehensive test suite
- ✅ Data analysis notebook
- ✅ Advanced visualizations
- ✅ Production-ready deployment

**Share your project:**
- GitHub: Add README, screenshots, and demos
- LinkedIn: Post about your project with screenshots
- Portfolio: Add to your personal website
- Resume: List as a featured project

Good luck with your data science journey! 🚀