import pandas as pd
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
from datetime import datetime
from data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class SalesDataset(torch.utils.data.Dataset):
    """Custom dataset for transformer model"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class PlayStationSalesModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.preprocessor = DataPreprocessor(data_path)
        
    def prepare_data(self):
        """Prepare data for all models"""
        print("="*60)
        print("PREPARING DATASET")
        print("="*60)
        
        # Process data
        df = self.preprocessor.process_all()
        
        # Get features and target
        X_traditional = self.preprocessor.get_ml_features()
        X_text = self.preprocessor.get_text_features()
        y = self.preprocessor.get_target()
        
        # Normalize target for better training (log transform)
        y_normalized = np.log1p(y)
        
        # Split data
        self.X_train_trad, self.X_test_trad, self.y_train, self.y_test = train_test_split(
            X_traditional, y_normalized, test_size=0.2, random_state=42
        )
        
        self.X_train_text, self.X_test_text, self.y_train_text, self.y_test_text = train_test_split(
            X_text, y_normalized, test_size=0.2, random_state=42
        )
        
        # Store original scale targets for evaluation
        _, _, self.y_train_orig, self.y_test_orig = train_test_split(
            X_traditional, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {len(self.X_train_trad)}")
        print(f"Test set size: {len(self.X_test_trad)}")
        
    def train_deberta(self, epochs=3, batch_size=16):
        """Train DeBERTa transformer model"""
        print("\n" + "="*60)
        print("TRAINING DEBERTA TRANSFORMER MODEL")
        print("="*60)
        
        try:
            # Use smaller DeBERTa model for efficiency
            model_name = "microsoft/deberta-v3-small"
            print(f"Loading tokenizer and model: {model_name}")
            
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            
            # For regression, we use sequence classification with 1 output
            model = DebertaV2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                problem_type="regression"
            )
            
            # Create datasets
            train_dataset = SalesDataset(
                self.X_train_text, 
                self.y_train_text, 
                tokenizer
            )
            test_dataset = SalesDataset(
                self.X_test_text, 
                self.y_test_text, 
                tokenizer
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./models/deberta_checkpoints',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                report_to="none"
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset
            )
            
            # Train
            print("\nStarting DeBERTa training...")
            trainer.train()
            
            # Evaluate
            print("\nEvaluating DeBERTa model...")
            predictions = trainer.predict(test_dataset)
            y_pred_normalized = predictions.predictions.flatten()
            y_pred = np.expm1(y_pred_normalized)  # Inverse log transform
            y_true = self.y_test_orig
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            accuracy = (1 - mae / np.mean(y_true)) * 100
            
            self.results['DeBERTa'] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Accuracy': accuracy
            }
            
            # Save model
            model.save_pretrained('./models/deberta_model')
            tokenizer.save_pretrained('./models/deberta_model')
            
            print(f"\n✓ DeBERTa Model Trained Successfully!")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            
            self.models['DeBERTa'] = (model, tokenizer)
            
        except Exception as e:
            print(f"\n✗ DeBERTa training failed: {str(e)}")
            print("Continuing with other models...")
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        print("Training XGBoost...")
        model = xgb.XGBRegressor(**params)
        model.fit(self.X_train_trad, self.y_train, verbose=False)
        
        # Predictions
        y_pred_normalized = model.predict(self.X_test_trad)
        y_pred = np.expm1(y_pred_normalized)
        y_true = self.y_test_orig
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        accuracy = (1 - mae / np.mean(y_true)) * 100
        
        self.results['XGBoost'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        }
        
        # Save model
        with open('./models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✓ XGBoost Model Trained Successfully!")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        
        self.models['XGBoost'] = model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Random Forest parameters
        params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        print("Training Random Forest...")
        model = RandomForestRegressor(**params)
        model.fit(self.X_train_trad, self.y_train)
        
        # Predictions
        y_pred_normalized = model.predict(self.X_test_trad)
        y_pred = np.expm1(y_pred_normalized)
        y_true = self.y_test_orig
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        accuracy = (1 - mae / np.mean(y_true)) * 100
        
        self.results['Random Forest'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        }
        
        # Save model
        with open('./models/random_forest_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✓ Random Forest Model Trained Successfully!")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        
        self.models['Random Forest'] = model
    
    def display_summary(self):
        """Display training summary with all model results"""
        print("\n" + "="*60)
        print("TRAINING COMPLETE - MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(2)
        
        print("\n" + results_df.to_string())
        
        # Find best model
        best_model = results_df['Accuracy'].idxmax()
        best_accuracy = results_df['Accuracy'].max()
        
        print(f"\n{'='*60}")
        print(f"🏆 BEST MODEL: {best_model}")
        print(f"   Accuracy: {best_accuracy:.2f}%")
        print(f"   R² Score: {results_df.loc[best_model, 'R2']:.4f}")
        print(f"{'='*60}\n")
        
        # Efficiency metrics
        print("EFFICIENCY METRICS:")
        for model_name in self.results.keys():
            efficiency = (self.results[model_name]['R2'] + self.results[model_name]['Accuracy']/100) / 2 * 100
            print(f"  {model_name}: {efficiency:.2f}% efficient")
        
        # Save results
        results_df.to_csv('./models/model_comparison.csv')
        print("\n✓ Results saved to ./models/model_comparison.csv")
    
    def train_all(self):
        """Train all models"""
        start_time = datetime.now()
        print("\n" + "="*60)
        print("PLAYSTATION SALES PREDICTION - MODEL TRAINING")
        print("="*60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Prepare data
        self.prepare_data()
        
        # Train models
        self.train_xgboost()
        self.train_random_forest()
        self.train_deberta(epochs=3, batch_size=16)
        
        # Display summary
        self.display_summary()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\nTotal training time: {duration:.2f} seconds")
        print("="*60)


if __name__ == "__main__":
    # Create models directory
    os.makedirs('./models', exist_ok=True)
    
    # Train all models
    trainer = PlayStationSalesModelTrainer("attached_assets/PlayStation Sales and Metadata (PS3PS4PS5) (Oct 2025)_1762234789301.csv")
    trainer.train_all()
