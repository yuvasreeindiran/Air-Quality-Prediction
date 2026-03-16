from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Updated HTML template without dataset size selection
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Air Quality Prediction System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(45deg, #2c3e50, #3498db);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            padding: 2rem;
        }

        .section {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border: 1px solid #e0e6ed;
        }

        .section h2 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.4rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            color: #555;
            font-weight: 600;
        }

        input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            outline: none;
        }

        button {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 1rem;
            transition: all 0.3s ease;
        }

        button:hover {
            background: linear-gradient(45deg, #2980b9, #21618c);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .results {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            border-left: 4px solid #3498db;
        }

        .results h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }

        .metric-card.best {
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        }

        .metric-card h4 {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }

        .metric-card .value {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }

        .metric-card .sub-value {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .best-model {
            background: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        }

        .best-model p {
            margin-bottom: 0.5rem;
        }

        .prediction-result {
            margin-top: 1rem;
        }

        .aqi-display {
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }

        .aqi-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .aqi-status {
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .aqi-label {
            font-size: 1rem;
            opacity: 0.9;
        }

        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .aqi-good { background: linear-gradient(135deg, #00e400dd, #00e400aa) !important; }
        .aqi-moderate { background: linear-gradient(135deg, #ffff00dd, #ffff00aa) !important; color: #333 !important; }
        .aqi-unhealthy-sensitive { background: linear-gradient(135deg, #ff7e00dd, #ff7e00aa) !important; }
        .aqi-unhealthy { background: linear-gradient(135deg, #ff0000dd, #ff0000aa) !important; }
        .aqi-very-unhealthy { background: linear-gradient(135deg, #8f3f97dd, #8f3f97aa) !important; }
        .aqi-hazardous { background: linear-gradient(135deg, #7e0023dd, #7e0023aa) !important; }

        .sample-data-btn {
            background: linear-gradient(45deg, #f39c12, #e67e22) !important;
            margin-top: 0.5rem !important;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            #featureImportance div {
                flex-direction: column;
            }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Air Quality Prediction System</h1>
            <p>Predict Benzene Concentration (C6H6) using Real Air Quality Data</p>
        </div>

        <div class="main-content">
            <!-- Model Training Section -->
            <div class="section">
                <h2> Model Training & Evaluation</h2>
                <div class="form-group">
                    <label>Training Dataset:</label>
                </div>
                <button id="trainBtn" onclick="trainModel()"> Train Model</button>

                <div id="trainingResults" class="results" style="display: none;">
                    <h3>Model Performance Metrics</h3>
                    <div class="metrics-grid" id="metricsGrid">
                        <!-- Metrics will be populated here -->
                    </div>
                    <div class="best-model">
                        <p><strong>Model:</strong> <span id="bestModel">Random Forest</span></p>
                        <p><strong>Dataset Size:</strong> <span id="datasetInfo">--</span></p>
                    </div>
                </div>
            </div>

            <!-- Prediction Section -->
            <div class="section">
                <h2>🔮 Predict Air Quality (C6H6)</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="co">CO (mg/m³):</label>
                        <input type="number" id="co" min="0" max="100" step="0.1" value="2.0" required>
                    </div>
                    <div class="form-group">
                        <label for="pt08_s1">PT08.S1 (CO):</label>
                        <input type="number" id="pt08_s1" min="0" max="3000" step="1" value="1000" required>
                    </div>
                    <div class="form-group">
                        <label for="nmhc">NMHC (µg/m³):</label>
                        <input type="number" id="nmhc" min="0" max="1000" step="1" value="100" required>
                    </div>
                    <div class="form-group">
                        <label for="nox">NOx (ppb):</label>
                        <input type="number" id="nox" min="0" max="1000" step="1" value="100" required>
                    </div>
                    <div class="form-group">
                        <label for="pt08_s3">PT08.S3 (NOx):</label>
                        <input type="number" id="pt08_s3" min="0" max="3000" step="1" value="1000" required>
                    </div>
                    <div class="form-group">
                        <label for="no2">NO2 (µg/m³):</label>
                        <input type="number" id="no2" min="0" max="500" step="1" value="100" required>
                    </div>
                    <div class="form-group">
                        <label for="pt08_s4">PT08.S4 (NO2):</label>
                        <input type="number" id="pt08_s4" min="0" max="3000" step="1" value="1000" required>
                    </div>
                    <div class="form-group">
                        <label for="pt08_s5">PT08.S5 (O3):</label>
                        <input type="number" id="pt08_s5" min="0" max="3000" step="1" value="1000" required>
                    </div>
                    <div class="form-group">
                        <label for="temperature">Temperature (°C):</label>
                        <input type="number" id="temperature" min="-50" max="60" step="0.1" value="20" required>
                    </div>
                    <div class="form-group">
                        <label for="rh">Relative Humidity (%):</label>
                        <input type="number" id="rh" min="0" max="100" step="0.1" value="50" required>
                    </div>
                    <div class="form-group">
                        <label for="ah">Absolute Humidity:</label>
                        <input type="number" id="ah" min="0" max="2" step="0.0001" value="0.8" required>
                    </div>

                    <button type="submit" id="predictBtn"> Predict Air Quality </button>
                    <button type="button" class="sample-data-btn" onclick="fillSampleData()"> Load Sample Data</button>
                </form>

                <div id="predictionResult" class="prediction-result" style="display: none;">
                    <div class="aqi-display" id="aqiDisplay">
                        <div class="aqi-value" id="aqiValue">--</div>
                        <div class="aqi-status" id="aqiStatus">--</div>
                        <div class="aqi-label">Benzene Concentration (µg/m³)</div>
                    </div>
                </div>
            </div>

            <section class="section full-width">
                <h2> Feature Importance Analysis</h2>
                <button onclick="showFeatureImportance()"> Show Feature Importance</button>
                <div id="featureImportance" class="feature-importance" style="display: none;">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1;">
                            <h3>Bar Chart</h3>
                            <canvas id="featureImportanceBarChart" width="400" height="200"></canvas>
                        </div>
                        <div style="flex: 1;">
                            <h3>Line Chart</h3>
                            <canvas id="featureImportanceLineChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <!-- Loading Overlay -->
        <div id="loadingOverlay" class="loading">
            <div class="spinner"></div>
            <p id="loadingMessage">Processing...</p>
        </div>
    </div>

    <script>
        let isModelTrained = false;

        function showLoading(message = 'Processing...') {
            document.getElementById('loadingMessage').textContent = message;
            document.getElementById('loadingOverlay').style.display = 'flex';
        }

        function hideLoading() {
            document.getElementById('loadingOverlay').style.display = 'none';
        }

        function showError(message) {
            alert('Error: ' + message);
        }

        async function trainModel() {
            const trainBtn = document.getElementById('trainBtn');
            
            try {
                trainBtn.disabled = true;
                showLoading('Training Random Forest model...');
                
                const response = await fetch('/api/train', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayTrainingResults(data);
                    isModelTrained = true;
                } else {
                    showError(data.message);
                }
                
            } catch (error) {
                showError('Failed to train model: ' + error.message);
            } finally {
                trainBtn.disabled = false;
                hideLoading();
            }
        }

        function displayTrainingResults(data) {
            const resultsDiv = document.getElementById('trainingResults');
            const metricsGrid = document.getElementById('metricsGrid');
            const bestModelSpan = document.getElementById('bestModel');
            const datasetInfo = document.getElementById('datasetInfo');
            
            metricsGrid.innerHTML = '';
            
            const metrics = data.results.random_forest;
            const modelCard = document.createElement('div');
            modelCard.className = 'metric-card best';
            modelCard.innerHTML = `
                <h4>Random Forest</h4>
                <div class="value">R²: ${metrics.r2_score}</div>
                <div class="sub-value">MAE: ${metrics.mae}</div>
                <div class="sub-value">RMSE: ${metrics.rmse}</div>
                <div class="sub-value">Time: ${metrics.training_time}s</div>
            `;
            metricsGrid.appendChild(modelCard);
            
            bestModelSpan.textContent = 'Random Forest';
            datasetInfo.textContent = `${data.dataset_size} samples`;
            
            resultsDiv.style.display = 'block';
        }

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const predictBtn = document.getElementById('predictBtn');
            
            if (!isModelTrained) {
                alert('Please train the model first!');
                return;
            }
            
            try {
                predictBtn.disabled = true;
                showLoading('Making prediction...');
                
                const formData = {
                    co: parseFloat(document.getElementById('co').value),
                    pt08_s1: parseFloat(document.getElementById('pt08_s1').value),
                    nmhc: parseFloat(document.getElementById('nmhc').value),
                    nox: parseFloat(document.getElementById('nox').value),
                    pt08_s3: parseFloat(document.getElementById('pt08_s3').value),
                    no2: parseFloat(document.getElementById('no2').value),
                    pt08_s4: parseFloat(document.getElementById('pt08_s4').value),
                    pt08_s5: parseFloat(document.getElementById('pt08_s5').value),
                    temperature: parseFloat(document.getElementById('temperature').value),
                    rh: parseFloat(document.getElementById('rh').value),
                    ah: parseFloat(document.getElementById('ah').value)
                };
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayPredictionResult(data);
                } else {
                    showError(data.message);
                }
                
            } catch (error) {
                showError('Failed to make prediction: ' + error.message);
            } finally {
                predictBtn.disabled = false;
                hideLoading();
            }
        });

        function displayPredictionResult(data) {
            const resultDiv = document.getElementById('predictionResult');
            const aqiValue = document.getElementById('aqiValue');
            const aqiStatus = document.getElementById('aqiStatus');
            const aqiDisplay = document.getElementById('aqiDisplay');
            
            aqiValue.textContent = data.prediction.toFixed(1);
            aqiStatus.textContent = data.aqi_status;
            
            aqiDisplay.className = 'aqi-display';
            
            const statusClasses = {
                'Good': 'aqi-good',
                'Moderate': 'aqi-moderate',
                'Unhealthy for Sensitive Groups': 'aqi-unhealthy-sensitive',
                'Unhealthy': 'aqi-unhealthy',
                'Very Unhealthy': 'aqi-very-unhealthy',
                'Hazardous': 'aqi-hazardous'
            };
            
            if (statusClasses[data.aqi_status]) {
                aqiDisplay.classList.add(statusClasses[data.aqi_status]);
            }
            
            resultDiv.style.display = 'block';
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        async function showFeatureImportance() {
            try {
                showLoading('Loading feature importance...');
                
                const response = await fetch('/api/metrics');
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayFeatureImportance(data.feature_importance);
                } else {
                    showError(data.message);
                }
                
            } catch (error) {
                showError('Failed to load feature importance: ' + error.message);
            } finally {
                hideLoading();
            }
        }

        function displayFeatureImportance(featureImportance) {
            const container = document.getElementById('featureImportance');
            
            if (!featureImportance || Object.keys(featureImportance).length === 0) {
                container.innerHTML = '<p>No feature importance data available. Please train the Random Forest model first.</p>';
                container.style.display = 'block';
                return;
            }
            
            const importance = featureImportance.random_forest;
            const labels = Object.keys(importance).map(feature => feature.replace('_', ' '));
            const values = Object.values(importance).map(value => value * 100); // Convert to percentage for visibility
            
            // Bar Chart
            const barCtx = document.getElementById('featureImportanceBarChart').getContext('2d');
            if (window.featureImportanceBarChart && typeof window.featureImportanceBarChart.destroy === 'function') {
                window.featureImportanceBarChart.destroy();
            }
            window.featureImportanceBarChart = new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feature Importance (%)',
                        data: values,
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                            '#FF9F40', '#66FF66', '#FF66B3', '#6666FF', '#FFB366', '#66CCCC'
                        ],
                        borderColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                            '#FF9F40', '#66FF66', '#FF66B3', '#6666FF', '#FFB366', '#66CCCC'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Importance (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Features'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true
                        },
                        title: {
                            display: true,
                            text: 'Feature Importance (Bar Chart)'
                        }
                    }
                }
            });

            // Line Chart
            const lineCtx = document.getElementById('featureImportanceLineChart').getContext('2d');
            if (window.featureImportanceLineChart && typeof window.featureImportanceLineChart.destroy === 'function') {
                window.featureImportanceLineChart.destroy();
            }
            window.featureImportanceLineChart = new Chart(lineCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feature Importance (%)',
                        data: values,
                        fill: false,
                        borderColor: '#36A2EB',
                        backgroundColor: '#36A2EB',
                        tension: 0.1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Importance (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Features'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true
                        },
                        title: {
                            display: true,
                            text: 'Feature Importance (Line Chart)'
                        }
                    }
                }
            });
            
            container.style.display = 'block';
        }

        function fillSampleData() {
            const samples = [
                { co: 2.6, pt08_s1: 1360, nmhc: 150, nox: 166, pt08_s3: 1056, no2: 113, pt08_s4: 1692, pt08_s5: 1268, temperature: 13.6, rh: 48.9, ah: 0.7578 },
                { co: 2.0, pt08_s1: 1292, nmhc: 112, nox: 103, pt08_s3: 1174, no2: 92, pt08_s4: 1559, pt08_s5: 972, temperature: 13.3, rh: 47.7, ah: 0.7255 },
                { co: 2.2, pt08_s1: 1402, nmhc: 88, nox: 131, pt08_s3: 1140, no2: 114, pt08_s4: 1555, pt08_s5: 1074, temperature: 11.9, rh: 54.0, ah: 0.7502 }
            ];
            
            const sample = samples[Math.floor(Math.random() * samples.length)];
            
            document.getElementById('co').value = sample.co;
            document.getElementById('pt08_s1').value = sample.pt08_s1;
            document.getElementById('nmhc').value = sample.nmhc;
            document.getElementById('nox').value = sample.nox;
            document.getElementById('pt08_s3').value = sample.pt08_s3;
            document.getElementById('no2').value = sample.no2;
            document.getElementById('pt08_s4').value = sample.pt08_s4;
            document.getElementById('pt08_s5').value = sample.pt08_s5;
            document.getElementById('temperature').value = sample.temperature;
            document.getElementById('rh').value = sample.rh;
            document.getElementById('ah').value = sample.ah;
        }
    </script>
</body>
</html>
'''

class AirQualityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_metrics = {}
        self.feature_importance = {}
        self.features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'NOx(GT)', 'PT08.S3(NOx)', 
                         'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

    def load_data(self):
        """Load and preprocess data from Air Quality.csv"""
        try:
            df = pd.read_csv('Air Quality.csv')
            print(f"Loaded dataset with {len(df)} rows initially")
        except FileNotFoundError:
            raise ValueError("Air Quality.csv file not found in the current directory.")
        
        # Replace -200 with NaN and fill missing values with column means
        df = df.replace(-200, np.nan)
        df = df[self.features + ['C6H6(GT)']]  # Select relevant columns
        df = df.fillna(df.mean())
        print(f"After filling NaN values, {len(df)} rows remain")
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after processing. Check the dataset.")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        X = df[self.features]
        y = df['C6H6(GT)']
        
        if X.empty or y.empty:
            raise ValueError("Empty dataset after feature selection. Check data loading.")
        
        X_scaled = self.scaler.fit_transform(X)
        print(f"Preprocessed data shape: {X_scaled.shape}")
        
        return X_scaled, y
    
    def train_model(self, df):
        """Train the Random Forest model and return its metrics"""
        X_scaled, y = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest...")
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = self.model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.model_metrics = {
            'random_forest': {
                'r2_score': round(r2, 4),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'training_time': round(training_time, 2)
            }
        }
        
        self.feature_importance = {
            'random_forest': dict(zip(self.features, self.model.feature_importances_))
        }
        
        return self.model_metrics, len(df)
    
    def predict(self, features):
        """Make prediction using the trained model"""
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("No model trained yet!")
        
        feature_array = np.array([[
            features['co'],
            features['pt08_s1'],
            features['nmhc'],
            features['nox'],
            features['pt08_s3'],
            features['no2'],
            features['pt08_s4'],
            features['pt08_s5'],
            features['temperature'],
            features['rh'],
            features['ah']
        ]])
        
        feature_scaled = self.scaler.transform(feature_array)
        prediction = self.model.predict(feature_scaled)[0]
        
        return max(0, round(prediction, 1))

# Initialize predictor
predictor = AirQualityPredictor()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        df = predictor.load_data()
        results, dataset_size_used = predictor.train_model(df)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'best_model': 'random_forest',
            'dataset_size': dataset_size_used
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not hasattr(predictor.model, 'estimators_'):
            return jsonify({'status': 'error', 'message': 'No trained model found. Please train a model first.'}), 400
        
        features = {
            'co': float(data['co']),
            'pt08_s1': float(data['pt08_s1']),
            'nmhc': float(data['nmhc']),
            'nox': float(data['nox']),
            'pt08_s3': float(data['pt08_s3']),
            'no2': float(data['no2']),
            'pt08_s4': float(data['pt08_s4']),
            'pt08_s5': float(data['pt08_s5']),
            'temperature': float(data['temperature']),
            'rh': float(data['rh']),
            'ah': float(data['ah'])
        }
        
        prediction = predictor.predict(features)
        
        # Define benzene concentration categories (approximated based on typical air quality standards)
        if prediction <= 5:
            status = 'Good'
        elif prediction <= 10:
            status = 'Moderate'
        elif prediction <= 15:
            status = 'Unhealthy for Sensitive Groups'
        elif prediction <= 20:
            status = 'Unhealthy'
        elif prediction <= 30:
            status = 'Very Unhealthy'
        else:
            status = 'Hazardous'
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'aqi_status': status
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        if not predictor.feature_importance:
            return jsonify({'status': 'error', 'message': 'No feature importance data available. Please train the Random Forest model first.'}), 400
        
        return jsonify({
            'status': 'success',
            'feature_importance': predictor.feature_importance
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🌟 Starting Air Quality Prediction System...")
    print("📊 Open your browser and go to: http://127.0.0.1:5000")
    print("✨ Predicting Benzene Concentration using Random Forest and real dataset!")
    app.run(debug=True)