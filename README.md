# 🎓 ScoreNet


<div align="center">

**A Multi-Algorithm Performance Prediction Framework that predicts student performance based on demographic and academic factors using advanced regression techniques and a user-friendly Flask web interface.**

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge&logo=vercel&logoColor=white)]()

</div>



## 🌟 Features

- **Multi-Algorithm Model Training**: Implements 10 different regression algorithms with hyperparameter tuning
- **High Performance**: Achieves **88.06%** accuracy with Ridge Regression model
- **Automated Model Selection**: Automatically selects the best performing model based on R² score
- **Robust Data Pipeline**: Complete ETL pipeline with data ingestion, transformation, and model training
- **Web Interface**: User-friendly Flask application for real-time predictions
- **Advanced Preprocessing**: Handles both numerical and categorical features with appropriate scaling and encoding
- **Comprehensive Logging**: Detailed logging system for monitoring and debugging
- **Custom Exception Handling**: Robust error handling throughout the pipeline

## 🏗️ Architecture

```
src/
├── components/
│   ├── data_ingestion.py   
│   ├── data_transformation.py  
│   └── model_trainer.py
├── pipeline/
│   └── predict_pipeline.py
├── utils.py
├── exception.py    
└── logger.py   

artifacts/                     
├── data.csv 
├── train.csv     
├── test.csv     
├── preprocessor.pkl     
└── model.pkl          

templates/               
├── index.html  
└── home.html           
```

## 🔧 Machine Learning Pipeline

### 1. Data Ingestion
- Loads student performance dataset
- Performs 80-20 train-test split
- Saves processed data to artifacts directory

### 2. Data Transformation
- **Numerical Features**: `writing_score`, `reading_score`
  - Missing value imputation using median
  - Standard scaling normalization
- **Categorical Features**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
  - Missing value imputation using most frequent value
  - One-hot encoding
  - Standard scaling (without mean centering for sparse matrices)

### 3. Model Training & Selection
The system trains and evaluates multiple regression algorithms with comprehensive hyperparameter tuning:

| Algorithm | Hyperparameter Tuning | R² Score |
|-----------|----------------------|----------|
| **Ridge Regression** | `alpha`: [0.1, 1.0, 10.0, 100.0] | **0.880593**  |
| **Linear Regression** | Default parameters | 0.880433 |
| **Random Forest** | `n_estimators`: [8, 16, 32, 64, 128, 256] | 0.851712 |
| **CatBoost** | `depth`: [6, 8, 10], `learning_rate`: [0.01, 0.05, 0.1], `iterations`: [30, 50, 100] | 0.851632 |
| **AdaBoost** | `learning_rate`: [0.1, 0.01, 0.5, 0.001], `n_estimators`: [8-256] | 0.847580 |
| **Lasso Regression** | `alpha`: [0.001, 0.01, 0.1, 1.0, 10.0] | 0.825320 |
| **XGBoost** | `learning_rate`: [0.1, 0.01, 0.05, 0.001], `n_estimators`: [8-256] | 0.821221 |
| **K-Neighbors** | `n_neighbors`: [3, 5, 7, 9], `weights`: ['uniform', 'distance'] | 0.783770 |
| **Decision Tree** | `criterion`: ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'] | 0.762306 |
| **Support Vector Regression** | `C`: [0.1, 1, 10], `kernel`: ['linear', 'rbf', 'poly'] | 0.728600 |

**🏆 Best Model**: Ridge Regression achieved the highest R² score of **0.880593**

- Uses **GridSearchCV** with 3-fold cross-validation for hyperparameter optimization
- Automatically selects the best model based on R² score on test data
- Saves the best model and preprocessing pipeline for deployment

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd student-performance-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Place your student dataset as `notebook/data/stud.csv`
   - Ensure it contains the required columns (see Data Format section)

4. **Train the model**
```bash
python src/components/data_ingestion.py
```

5. **Run the Flask application**
```bash
python app.py
```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Use the web interface to make predictions

## 📊 Data Format

The system expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `gender` | Categorical | Student's gender |
| `race/ethnicity` | Categorical | Student's racial/ethnic background |
| `parental level of education` | Categorical | Highest education level of parents |
| `lunch` | Categorical | Lunch program participation |
| `test preparation course` | Categorical | Test prep course completion |
| `reading score` | Numerical | Reading assessment score |
| `writing score` | Numerical | Writing assessment score |
| `math score` | Numerical | **Target variable** - Math assessment score |

## 🖥️ Web Interface

### Home Page (`/`)
- Welcome page with project overview
- Navigation to prediction interface

### Prediction Interface (`/predictdata`)
- **GET**: Displays the prediction form
- **POST**: Processes form data and returns math score prediction

### Form Fields:
- Gender selection
- Race/ethnicity dropdown
- Parental education level
- Lunch program status
- Test preparation course completion
- Reading score input
- Writing score input

## 📈 Model Performance

The system evaluates 10 different regression algorithms and automatically selects the best performer:

### 🏆 Performance Rankings:

1. **Ridge Regression**: 88.06% R² Score ⭐ **(Selected Model)**
2. **Linear Regression**: 88.04% R² Score  
3. **Random Forest**: 85.17% R² Score
4. **CatBoost**: 85.16% R² Score
5. **AdaBoost**: 84.76% R² Score

**Key Metrics:**
- **Primary Metric**: R² (coefficient of determination)
- **Best Performance**: 88.06% with Ridge Regression
- **Minimum Threshold**: 60% R² score required for deployment
- **Validation**: 3-fold cross-validation during hyperparameter tuning
- **Selection Criteria**: Highest R² score on test set

## 🛠️ Configuration

### Logging
- Automatic log file generation with timestamps
- Format: `[timestamp] line_number module_name - log_level - message`
- Stored in: `logs/MM_DD_YYYY_HH_MM_SS.log`

### Artifacts Storage
All trained components are saved in the `artifacts/` directory:
- `data.csv`: Original dataset
- `train.csv`, `test.csv`: Split datasets
- `preprocessor.pkl`: Trained preprocessing pipeline
- `model.pkl`: Best performing model

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | Landing page |
| `/predictdata` | GET | Prediction form |
| `/predictdata` | POST | Submit prediction request |

## 🧪 Usage Examples

### Programmatic Prediction
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create custom data instance
data = CustomData(
    gender='male',
    race_ethnicity='group A',
    parental_level_of_education='some college',
    lunch='standard',
    test_preparation_course='completed',
    reading_score=85,
    writing_score=82
)

# Generate prediction
pipeline = PredictPipeline()
df = data.get_data_as_data_frame()
prediction = pipeline.predict(df)
print(f"Predicted Math Score: {prediction[0]}")
```

### Web Interface Usage
1. Navigate to `http://localhost:5000/predictdata`
2. Fill out the student information form
3. Click "Predict Math Score"
4. View the predicted score on the results page

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install in development mode
pip install -e .
```

**File Path Issues**
- Ensure `notebook/data/stud.csv` exists
- Check that `artifacts/` directory is writable

**Model Performance Issues**
- Verify data quality and completeness
- Check for sufficient training samples
- Review feature distributions

### Debug Mode
Enable Flask debug mode for development:
```python
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 🤝 Acknowledgments

- **Scikit-learn** for machine learning algorithms and utilities
- **Flask** for the web framework
- **XGBoost & CatBoost** for advanced gradient boosting implementations
- **Pandas & NumPy** for data manipulation and numerical computing



---

 **Last Updated**: January 2025