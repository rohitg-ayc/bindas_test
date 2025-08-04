import pandas as pd
import numpy as np
import re
import pickle
import logging
from typing import Dict, List, Union, Tuple
from scipy.stats import entropy, skew, kurtosis
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from backend.utils.response_generator import success_response, error_response


#Configure logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnClassifier:
    """
    Class to detect the types of each column using ML model.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the classifier with a pre-trained model.
        
        Args:
            model_path: Path to the pickled XGBoost model file
        """
        self.model_path = model_path
        self.model = None
        
        # Class name mapping
        self.class_mapping = {
            0: "Categorical Dimension",
            1: "Categorical Indicator",
            2: "Continuous Dimension",
            3: "Continuous Indicator",
            4: "Time Based Indicator",
            5: "Time Based Dimension"
        }
        
        # Term lists for column name analysis
        self.time_related_terms = [
            'date', 'time', 'day', 'month', 'year', 'quarter', 'week', 'hour', 
            'minute', 'second', 'timestamp', 'dt', 'period', 'datetime', 'created', 
            'modified', 'updated', 'start', 'end', 'from', 'to', 'since', 'until', 
            'expiry', 'exp', 'deadline', 'birth', 'dob', 'joined', 'left', 'duration', 
            'elapsed', 'schedule', 'calendar', 'event_time', 'log_time', 'run_time', 
            'recorded', 'registered', 'entry', 'exit', 'anniversary', 'epoch', 
            'instantevent', 'session', 'timeline', 'occurrence'
        ]

        self.indicator_terms = [
            'is_', 'has_', 'flag', 'indicator', 'status', 'active', 'enabled',
            'count', 'total', 'sum', 'avg', 'mean', 'amount', 'balance', 'score',
            'ratio', 'rate', 'percentage', 'percent', 'number', 'num', 'cnt',
            'check', 'valid', 'invalid', 'exists', 'present', 'missing', 'deleted',
            'error', 'success', 'failure', 'complete', 'incomplete', 'approved', 
            'rejected', 'verified', 'confirmed', 'locked', 'unlocked', 'returned',
            'used', 'unused', 'opted', 'optout', 'selected', 'chosen', 'cancelled',
            'min', 'max', 'median', 'stddev', 'variance', 'threshold', 'index',
            'level', 'delta', 'diff', 'gap', 'spread', 'weight', 'scorecard',
            'load', 'frequency', 'utilization', 'consumption', 'rateof', 'density',
            'revenue', 'profit', 'loss', 'margin', 'cost', 'expense', 'budget',
            'roi', 'investment', 'valuation', 'premium', 'penalty', 'fee', 'credit', 'debit',
            'signal', 'status_code', 'metric', 'reading', 'measure', 'value', 
            'trend', 'health', 'quality', 'grade', 'rating', 'tier', 'risk'
        ]

        self.dimension_terms = [
            'id', 'category', 'type', 'group', 'segment', 'class', 'tier',
            'level', 'grade', 'rank', 'name', 'code', 'description', 'desc',
            'label', 'tag', 'bucket', 'cluster', 'zone', 'division', 'region',
            'location', 'state', 'area', 'district', 'province', 'country',
            'department', 'team', 'unit', 'section', 'batch', 'line',
            'brand', 'model', 'series', 'variant', 'sku', 'item', 'product',
            'service', 'module', 'feature', 'component', 'asset', 'portfolio',
            'user', 'customer', 'client', 'account', 'profile', 'subscriber',
            'role', 'membership', 'plan', 'tier', 'level', 'access',
            'industry', 'sector', 'category_code', 'class_code', 'license',
            'make', 'build', 'vehicle', 'version', 'edition',
            'tag', 'dimension', 'factor', 'variable', 'attribute', 'field',
            'indicator', 'reference', 'taxonomy', 'key', 'option'
        ]
        
        # Load model
        self._load_model()
            
    def _load_model(self) -> None:
        """Load the XGBoost model from the specified path."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            # print(f"Successfully loaded model from {self.model_path}")
            logger.info(f"Successfully loaded Column classifier model")
        except FileNotFoundError:
            logger.error(f"Column classifier model file not found at {self.model_path}")
            # print(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading Column classifier model: {str(e)}")
            # print(f"Error loading model: {str(e)}")
            raise
    
    def extract_column_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract features from DataFrame columns for classification.
        
        Args:
            df: Input DataFrame to extract features from
            
        Returns:
            Tuple containing:
            - DataFrame with extracted features for each column
            - List of column names in the same order as features
        """
        if df.empty:
            # print("Empty DataFrame provided for feature extraction")
            logger.info("Empty Dataframe provided for feature extraction in column classifier.")
            return pd.DataFrame(), []
            
        features = []
        column_names = []
        # print(f"Extracting features from {len(df.columns)} columns")

        for col in df.columns:
            try:
                col_data = df[col]
                col_data_no_na = col_data.dropna()
                
                if len(col_data_no_na) == 0:
                    continue

                # Base stats
                dtype = col_data.dtype
                is_numeric = pd.api.types.is_numeric_dtype(col_data)
                is_object = pd.api.types.is_object_dtype(col_data)
                is_bool = pd.api.types.is_bool_dtype(col_data)
                is_datetime = pd.api.types.is_datetime64_any_dtype(col_data)

                n = len(col_data)
                nunique = col_data.nunique(dropna=True)
                unique_ratio = nunique / n if n > 0 else 0
                
                # Handle edge case where all values are NaN
                most_freq_ratio = 0
                if nunique > 0:
                    value_counts = col_data.value_counts(normalize=True)
                    if not value_counts.empty:
                        most_freq_ratio = value_counts.iloc[0]

                # Entropy (with error handling)
                ent = 0
                if nunique > 1:
                    try:
                        value_counts = col_data.value_counts(normalize=True)
                        ent = entropy(value_counts, base=2) if len(value_counts) > 0 else 0
                    except Exception as e:
                        # print(f"Error calculating entropy for column '{col}': {str(e)}")
                        ent = 0

                # Binary
                is_binary = nunique == 2

                # Text features
                avg_str_len = 0
                if is_object:
                    try:
                        avg_str_len = col_data_no_na.astype(str).str.len().mean()
                    except Exception as e:
                        logger.error(f"Error calculating string length for columns '{col}': {str(e)}")
                        # print(f"Error calculating string length for column '{col}': {str(e)}")

                # Name-based signals
                col_name_lower = col.lower()
                name_signals = {
                    'contains_time_term': any(term in col_name_lower for term in self.time_related_terms),
                    'contains_indicator_term': any(term in col_name_lower for term in self.indicator_terms),
                    'contains_dimension_term': any(term in col_name_lower for term in self.dimension_terms)
                }

                # Date signals (if not already parsed)
                contains_year = False
                if is_object:
                    try:
                        # Only check a sample to save processing time
                        sample = col_data_no_na.iloc[:min(20, len(col_data_no_na))]
                        contains_year = any(re.search(r'\b(19|20)\d{2}\b', str(x)) for x in sample)
                    except Exception as e:
                        logger.error(f"Eror checking year patterns for column '{col}': {str(e)}")
                        # print(f"Error checking year patterns for column '{col}': {str(e)}")

                # Calculate statistical features for numeric columns
                skew_val, kurt_val, std_val, mean_val = 0, 0, 0, 0
                
                if is_numeric and len(col_data_no_na) > 2:
                    try:
                        std_val = col_data_no_na.std()
                        mean_val = col_data_no_na.mean()
                        
                        if std_val > 1e-8:  # Avoid division by zero issues
                            skew_val = skew(col_data_no_na)
                            kurt_val = kurtosis(col_data_no_na)
                    except Exception as e:
                        logger.error(f"Error calculating statistics for column '{col}': {str(e)}")
                        # print(f"Error calculating statistics for column '{col}': {str(e)}")

                numeric_stats = {
                    'mean': mean_val,
                    'std': std_val,
                    'skew': skew_val,
                    'kurtosis': kurt_val
                }

                feature_row = {
                    'is_numeric': is_numeric,
                    'is_object': is_object,
                    'is_bool': is_bool,
                    'is_datetime': is_datetime,
                    'nunique': nunique,
                    'unique_ratio': unique_ratio,
                    'entropy': ent,
                    'most_freq_ratio': most_freq_ratio,
                    'is_binary': is_binary,
                    'avg_str_len': avg_str_len,
                    'contains_year_in_values': contains_year,
                }

                feature_row.update(name_signals)
                feature_row.update(numeric_stats)
                features.append(feature_row)
                column_names.append(col)
                
            except Exception as e:
                logger.error(f"Error processing column '{col}': {str(e)}")

                # Add a placeholder row with default values to maintain structure
                features.append({
                    'is_numeric': False,
                    'is_object': False,
                    'is_bool': False,
                    'is_datetime': False,
                    'nunique': 0,
                    'unique_ratio': 0,
                    'entropy': 0,
                    'most_freq_ratio': 0,
                    'is_binary': False,
                    'avg_str_len': 0,
                    'contains_year_in_values': False,
                    'contains_time_term': False,
                    'contains_indicator_term': False,
                    'contains_dimension_term': False,
                    'mean': 0,
                    'std': 0,
                    'skew': 0,
                    'kurtosis': 0
                })
                column_names.append(col)

        return pd.DataFrame(features), column_names

    def preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess extracted features for use in the model.
        
        Args:
            features_df: DataFrame containing features extracted from columns
            
        Returns:
            Preprocessed DataFrame ready for model input
        """
        if features_df.empty:
            logger.info("Empty features DataFrames provided for preprocessing.")
            return pd.DataFrame()
            
        try:
            # Handle problematic values
            for col in features_df.select_dtypes(include='number').columns:
                features_df[col] = features_df[col].fillna(0)
                features_df[col] = features_df[col].replace([np.inf, -np.inf], 0)
            
            # Split features by type for preprocessing
            numeric_cols = [col for col in features_df.select_dtypes(include='number').columns]
            non_numeric_cols = [col for col in features_df.select_dtypes(exclude='number').columns]
            
            # Create preprocessors
            preprocessor = ColumnTransformer([
                ('scale_numeric', MinMaxScaler(), numeric_cols),
                ('encode_non_numeric', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
                 non_numeric_cols) if non_numeric_cols else ('no_op', 'passthrough', [])
            ])
            
            # Apply preprocessing
            processed_features = preprocessor.fit_transform(features_df)
            
            # Convert back to DataFrame to maintain column names
            processed_df = pd.DataFrame(
                processed_features, 
                columns=numeric_cols + non_numeric_cols if non_numeric_cols else numeric_cols
            )
            return processed_df
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Make predictions for all columns in the DataFrame.
        
        Args:
            df: Input DataFrame with columns to classify
            
        Returns:
            Dictionary with predictions for each column
        """
        if df.empty:
            logger.info("Empty DataFrame provided for prediction")
            return {}
            
        try:
            # Extract features and column names
            features_df, column_names = self.extract_column_features(df)
            
            if features_df.empty: 
                logger.info("No features extracted for prediction")
                return {}
                
            # Preprocess features
            processed_features = self.preprocess_features(features_df)
              
            if processed_features.empty:
                logger.info("No processed features available for prediction")
                return {}
                
            # Make predictions
            predictions = self.model.predict(processed_features)
            probabilities = self.model.predict_proba(processed_features)
            
            results = {}
            for i, col in enumerate(column_names):
                pred_class_index = int(predictions[i])
                pred_class_name = self.class_mapping.get(pred_class_index, "Unknown")

                prob = probabilities[i][pred_class_index]

                results[col] = {
                    "prediction" : pred_class_name,
                    "confidence" : float(prob)
                }
                
            return {
                "status": "success",
                "message": "Columns classified successfully",
                "data": results
        }
            
        except Exception as e:
            logger.info(f"Error making predictions: {str(e)}")
            return error_response(message=f"Error making predictions: {str(e)}")

# classifier = ColumnClassifier(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\Bindas-2.1\server\python_modules\Relation_Detector\Column_classifier (1).pkl")
# df = pd.read_csv("C:\\Users\\aniketd\\Downloads\\apple_quality.csv")
# result = classifier.predict(df)
# print(result)