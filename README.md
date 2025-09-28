# Monte Carlo Climate Risk Prediction System

A comprehensive system for predicting climate-related disaster risks using Monte Carlo simulations and machine learning models.

## 🎯 Project Overview

This project implements a Monte Carlo simulation system to predict the probability and impact of climate-related disasters on human life, property, and infrastructure. The system uses historical NOAA storm events data and climate data to model risk scenarios for different disaster categories across US states.

### Key Features

- **Risk Prediction**: Predict probabilities of deaths, injuries, property damage, and crop damage
- **State-Level Analysis**: Provide state-specific risk assessments and comparative analysis  
- **Category-Specific Models**: Separate prediction models for different disaster types
- **Monte Carlo Simulation**: Probabilistic modeling with uncertainty quantification
- **Interactive Visualizations**: Comprehensive EDA and risk mapping

## 📁 Project Structure

```
monte_carlo_climate_risk/
├── 01_config/              # ⚙️ Configuration files
├── 02_data_scripts/        # 📥 Data download scripts
├── 03_data/                # 📊 Raw datasets
├── 04_cleaning_scripts/    # 🧹 Data preprocessing
├── 05_modelling_scripts/   # �� ML models & simulations
├── 06_test_scripts/        # ✅ Testing & validation
├── 07_results/             # 📈 Output files & reports
├── climate_risk_env/       # 🐍 Python virtual environment
├── run_pipeline.py         # 🚀 Main execution script
├── requirements.txt        # 📦 Dependencies
└── [documentation files...]
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd /Users/ritubansal/personal_projects/monte_carlo_climate_risk
source climate_risk_env/bin/activate
```

### 2. Download Data
```bash
python 02_data_scripts/01_download_noaa.py
python 02_data_scripts/02_download_storms.py
```

### 3. Test Installation
```bash
python 06_test_scripts/simple_test.py
```

### 4. Run Full Pipeline
```bash
python run_pipeline.py
```

## 📊 Data Sources

- **NOAA Climate Data**: County-level climate metrics (1950-2025)
- **Storm Events Database**: Historical storm events with impact data
- **Billion-Dollar Events**: Major weather and climate disasters

## 🔧 Configuration

The `01_config/pipeline_config.json` file contains all configurable parameters:

- Data cleaning settings
- Feature engineering options
- Model training parameters
- Monte Carlo simulation settings
- Visualization preferences

## 🎯 Workflow

1. **Download**: Acquire raw climate and storm data
2. **Clean**: Preprocess and validate datasets
3. **Engineer**: Create temporal and geographic features
4. **Model**: Train statistical models for each disaster category
5. **Simulate**: Run Monte Carlo simulations for risk assessment
6. **Analyze**: Generate reports and visualizations
7. **Validate**: Test and verify results

## 📈 Output & Results

The system generates:
- **Risk probability distributions** for different disaster scenarios
- **Interactive visualizations** of state-level risk assessments
- **Statistical model performance** metrics
- **Comprehensive reports** with confidence intervals
- **Temporal trend analysis** for long-term risk forecasting

## 🛠️ Customization

### Adding New Disaster Categories
1. Update `04_cleaning_scripts/feature_engineering.py`
2. Add category-specific features
3. Train new models in `05_modelling_scripts/`

### Modifying Risk Metrics
1. Edit `01_config/pipeline_config.json`
2. Update impact calculation formulas
3. Adjust simulation parameters

## 📚 Documentation

- `plan.md` - Comprehensive implementation plan
- `PROJECT_STRUCTURE.md` - Detailed project organization guide
- `ENVIRONMENT_README.md` - Environment management instructions

## 🔍 Key Components

### Data Processing Pipeline
- **Data Cleaning**: Handle missing values, data types, and validation
- **Feature Engineering**: Create temporal, geographic, and impact features
- **Category Classification**: Organize events into disaster categories

### Modeling Framework
- **Statistical Models**: Machine learning models for impact prediction
- **Monte Carlo Engine**: Probabilistic risk simulation with uncertainty
- **Risk Assessment**: Calculate probabilities and confidence intervals

### Analysis & Visualization
- **State-Level EDA**: Exploratory analysis by geographic region
- **Category Analysis**: Compare disaster types and patterns
- **Interactive Dashboards**: Web-based result visualization

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python 06_test_scripts/simple_test.py    # Basic functionality
python 06_test_scripts/test_pipeline.py  # Full pipeline test
```

## 📝 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- plotly, matplotlib, seaborn
- jupyter, streamlit, dash
- geopy, pyarrow, fastparquet

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

1. Follow the logical workflow order
2. Add tests for new functionality
3. Update documentation as needed
4. Use the established configuration system

## 📄 License

This project uses publicly available NOAA climate and storm data for research and analysis purposes.

---

*Built with ❤️ for climate risk assessment and disaster preparedness*
