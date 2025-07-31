# EV Forecasting Streamlit Application

An interactive web application for forecasting Electric Vehicle (EV) adoption trends using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

Or alternatively:
```bash
python -m streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── forecasting_ev_model.pkl    # Pre-trained ML model
├── preprocessed_ev_data.csv    # Processed EV data by county
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Features

- **Interactive County Selection**: Choose from available counties to analyze
- **Historical Data Visualization**: View EV adoption trends over time
- **ML-Powered Forecasting**: Generate multi-month EV adoption predictions
- **Comprehensive Analytics**: Detailed statistics and growth metrics
- **Modern UI**: Responsive design with smooth animations

## 📊 Usage

1. **Select a County**: Choose from the dropdown menu
2. **Review Historical Data**: Examine past EV adoption trends
3. **Generate Forecasts**: Set forecast parameters and create predictions
4. **Analyze Results**: View interactive charts and detailed metrics

## 🛠️ Technical Details

- **Framework**: Streamlit
- **ML Model**: Pre-trained Random Forest model
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Caching**: Optimized with Streamlit caching decorators

## 📋 Requirements

See `requirements.txt` for complete list of dependencies:
- streamlit>=1.28.0
- pandas>=1.5.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- joblib>=1.3.0
- matplotlib>=3.6.0

## 🎯 Application Flow

1. **Data Loading**: Automatically loads county data and ML model
2. **County Selection**: Interactive dropdown for county selection
3. **Data Display**: Shows historical trends and statistics
4. **Forecast Generation**: Creates multi-month predictions
5. **Results Visualization**: Interactive charts and detailed analytics

## 🔍 Troubleshooting

If you encounter issues:

1. **Missing Files**: Ensure all required files are present
2. **Dependencies**: Install requirements with `pip install -r requirements.txt`
3. **Port Conflicts**: Use `--server.port XXXX` to specify a different port
4. **Memory Issues**: Close other applications to free up system resources

## 📈 Data Sources

The application uses preprocessed EV adoption data by county, including:
- Historical EV totals
- EV adoption percentages
- Battery Electric Vehicles (BEVs)
- Plug-in Hybrid Electric Vehicles (PHEVs)

---

**Ready to explore EV adoption trends? Run `streamlit run app.py` to get started!**