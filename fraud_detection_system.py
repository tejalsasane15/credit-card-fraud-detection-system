import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import sqlite3
import joblib
from datetime import datetime
import json

class FraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.db_path = "fraud_transactions.db"
        self.model_path = "models/"
        os.makedirs(self.model_path, exist_ok=True)
    
    # ================= DATA INGESTION =================
    def ingest_data(self, csv_path="creditcard.csv"):
        """Load and validate dataset"""
        print("🔄 Starting Data Ingestion...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        dtypes = {f'V{i}': 'float32' for i in range(1, 29)}
        dtypes.update({'Time': 'float32', 'Amount': 'float32', 'Class': 'int8'})
        
        df = pd.read_csv(csv_path, dtype=dtypes)
        print(f"✅ Data loaded: {df.shape[0]} transactions, {df.shape[1]} features")
        print(f"📊 Fraud rate: {df['Class'].mean():.2%}")
        
        return df
    
    # ================= PREPROCESSING =================
    def preprocess_data(self, df):
        """Clean and prepare data"""
        print("🔄 Starting Data Preprocessing...")
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Time_hour'] = (df['Time'] / 3600) % 24
        
        # Detect outliers using IQR
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]
        
        print(f"✅ Preprocessing complete: {df.shape[0]} transactions remaining")
        return df
    
    # ================= HANDLE IMBALANCE =================
    def handle_imbalance(self, X, y, method='smote'):
        """Handle class imbalance with multiple strategies"""
        print(f"🔄 Handling class imbalance using {method.upper()}...")
        
        original_ratio = y.value_counts()[1] / len(y)
        print(f"Original fraud ratio: {original_ratio:.2%}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=3)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=42)
        
        X_res, y_res = sampler.fit_resample(X, y)
        new_ratio = y_res.value_counts()[1] / len(y_res)
        print(f"✅ New fraud ratio: {new_ratio:.2%}")
        
        return X_res, y_res
    
    # ================= MODEL TRAINING =================
    def train_robust_models(self, df, balance_method='smote'):
        """Train multiple models and select best"""
        print("🔄 Training Robust Models...")
        
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        # Handle imbalance
        X_balanced, y_balanced = self.handle_imbalance(X, y, balance_method)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Test performance
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': auc_score
            }
            
            print(f"{name} - CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), Test AUC: {auc_score:.4f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
        
        self.model = best_model
        self.feature_cols = X.columns
        
        # Save model and scaler
        joblib.dump(self.model, f"{self.model_path}best_model.pkl")
        joblib.dump(self.scaler, f"{self.model_path}scaler.pkl")
        joblib.dump(self.feature_cols, f"{self.model_path}feature_cols.pkl")
        
        print(f"✅ Best model saved with AUC: {best_score:.4f}")
        return results
    
    # ================= DEPLOYMENT =================
    def load_model(self):
        """Load trained model for deployment"""
        try:
            self.model = joblib.load(f"{self.model_path}best_model.pkl")
            self.scaler = joblib.load(f"{self.model_path}scaler.pkl")
            self.feature_cols = joblib.load(f"{self.model_path}feature_cols.pkl")
            print("✅ Model loaded successfully")
            return True
        except:
            print("❌ No trained model found. Please train first.")
            return False
    
    def setup_database(self):
        """Initialize transaction database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                amount REAL,
                time_feature REAL,
                prediction INTEGER,
                fraud_probability REAL
            )
        """)
        conn.commit()
        conn.close()
    
    # ================= TRANSACTION ANALYSIS =================
    def predict_single_transaction(self, amount, time_feature):
        """Predict single transaction"""
        try:
            if not self.model:
                if not self.load_model():
                    raise Exception("No trained model available. Please train a model first.")
            
            if self.feature_cols is None:
                raise Exception("Feature columns not loaded. Please train a model first.")
            
            # Create feature vector matching original dataset
            features = np.random.randn(len(self.feature_cols)).astype(np.float32)
            
            # Set known features
            if 'Amount' in self.feature_cols:
                features[self.feature_cols.get_loc('Amount')] = amount
            if 'Time' in self.feature_cols:
                features[self.feature_cols.get_loc('Time')] = time_feature
            
            # Add engineered features if they exist
            if 'Amount_log' in self.feature_cols:
                features[self.feature_cols.get_loc('Amount_log')] = np.log1p(amount)
            if 'Time_hour' in self.feature_cols:
                features[self.feature_cols.get_loc('Time_hour')] = (time_feature / 3600) % 24
            
            # Scale and predict
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            # Save to database
            self.save_transaction(amount, time_feature, prediction, probability)
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'amount': float(amount),
                'time': float(time_feature)
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise e
    
    def save_transaction(self, amount, time_feature, prediction, probability):
        """Save transaction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO transactions (timestamp, amount, time_feature, prediction, fraud_probability)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), amount, time_feature, prediction, probability))
        
        conn.commit()
        conn.close()
    
    # ================= TRANSACTION HISTORY =================
    def get_transaction_history(self, limit=50):
        """Get transaction history"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM transactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def search_transactions(self, min_amount=None, max_amount=None, 
                          fraud_only=False, days_back=30):
        """Search transactions with filters"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM transactions WHERE 1=1"
        params = []
        
        if min_amount:
            query += " AND amount >= ?"
            params.append(min_amount)
        
        if max_amount:
            query += " AND amount <= ?"
            params.append(max_amount)
        
        if fraud_only:
            query += " AND prediction = 1"
        
        query += " AND datetime(timestamp) >= datetime('now', '-{} days')".format(days_back)
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def analyze_patterns(self):
        """Analyze transaction patterns"""
        df = self.get_transaction_history(limit=1000)
        
        if df.empty:
            return {"message": "No transaction history found"}
        
        analysis = {
            'total_transactions': len(df),
            'total_amount': df['amount'].sum(),
            'avg_amount': df['amount'].mean(),
            'fraud_count': df['prediction'].sum(),
            'fraud_rate': df['prediction'].mean(),
            'high_risk_transactions': len(df[df['fraud_probability'] > 0.7]),
            'spending_trend': df.groupby(df['timestamp'].str[:10])['amount'].sum().to_dict()
        }
        
        return analysis

def main_menu():
    """Main system interface"""
    system = FraudDetectionSystem()
    system.setup_database()
    
    while True:
        print("\n" + "="*60)
        print("🏦 FRAUD DETECTION SYSTEM")
        print("="*60)
        print("1. 📊 Train New Model")
        print("2. 🔍 Analyze Single Transaction")
        print("3. 📈 View Transaction History")
        print("4. 🔎 Search Transactions")
        print("5. 📋 User Pattern Analysis")
        print("6. 🚪 Exit")
        print("="*60)
        
        choice = input("Select option (1-6): ")
        
        if choice == '1':
            try:
                df = system.ingest_data()
                df = system.preprocess_data(df)
                results = system.train_robust_models(df)
                print("\n📊 Model Training Results:")
                for model, metrics in results.items():
                    print(f"{model}: Test AUC = {metrics['test_auc']:.4f}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '2':
            try:
                amount = float(input("Enter amount ($): "))
                time_val = float(input("Enter time (seconds): "))
                
                result = system.predict_single_transaction(amount, time_val)
                
                if result:
                    print("\n" + "="*50)
                    print("🔍 TRANSACTION ANALYSIS")
                    print("="*50)
                    print(f"Amount: ${result['amount']:.2f}")
                    print(f"Fraud Probability: {result['probability']:.1%}")
                    
                    if result['prediction'] == 1:
                        print("🚨 FRAUD DETECTED!")
                    else:
                        print("✅ LEGITIMATE")
                    print("="*50)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            history = system.get_transaction_history()
            
            if not history.empty:
                print(f"\n📈 Last 10 transactions:")
                print(history[['timestamp', 'amount', 'prediction', 'fraud_probability']].head(10).to_string(index=False))
            else:
                print("No transaction history found.")
        
        elif choice == '4':
            min_amt = input("Min amount (or press Enter): ")
            max_amt = input("Max amount (or press Enter): ")
            fraud_only = input("Fraud only? (y/n): ").lower() == 'y'
            
            results = system.search_transactions(
                min_amount=float(min_amt) if min_amt else None,
                max_amount=float(max_amt) if max_amt else None,
                fraud_only=fraud_only
            )
            
            if not results.empty:
                print(f"\n🔎 Search Results ({len(results)} transactions):")
                print(results[['timestamp', 'amount', 'prediction', 'fraud_probability']].to_string(index=False))
            else:
                print("No transactions found matching criteria.")
        
        elif choice == '5':
            analysis = system.analyze_patterns()
            
            print(f"\n📋 Transaction Pattern Analysis:")
            print("="*40)
            for key, value in analysis.items():
                if key != 'spending_trend':
                    print(f"{key.replace('_', ' ').title()}: {value}")
        
        elif choice == '6':
            print("👋 Thank you for using the Fraud Detection System!")
            break
        
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main_menu()