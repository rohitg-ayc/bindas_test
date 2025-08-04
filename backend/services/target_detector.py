import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from backend.utils.response_generator import success_response, error_response
import logging



#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The data should exclude id like columns to get faster results.
class TargetColumnDetector:
    """
    A class to detect most likely target column in a dataset.
    """
  
    def __init__(self, model_path, sample_size=30000, n_jobs=-1):
        """
        Initialize the TargetColumnDetector.
        
        Parameters:
        model_path : A trained machine learning model path.
        
        sample_size : int, default=30000
            Maximum number of rows to sample to extract metadata to improve performance & also used to calculate stats_score.
            [sample_size=30000 is used at model training]
            
        n_jobs : int, default=-1
            Number of parallel jobs to run. -1 means using all processors.
        """
        self.model_path = model_path
        self.sample_size = sample_size
        self.n_jobs = n_jobs
        self.ml_model = None

        # Load model
        self._load_model()

    def _load_model(self):
        try:
            self.ml_model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded Target detector model")
        except FileNotFoundError:
            logger.error(f"Target detector model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading Target detector model: {str(e)}")
            raise
    
    def _process_column_metadata(self, df_sampled, col, valid_columns, encoded_columns, column_types):
        """
        Process metadata for a single column - helper function for parallelization.
        """
        metadata = {}
        non_null_values = df_sampled[col].dropna()
        metadata['n_unique'] = non_null_values.nunique()
        metadata['unique_ratio'] = metadata['n_unique'] / len(non_null_values) if len(non_null_values) > 0 else 0
        metadata['is_binary'] = metadata['n_unique'] == 2

        try:
            value_counts = non_null_values.value_counts(normalize=True)
            metadata['entropy'] = stats.entropy(value_counts)
        except Exception:
            metadata['entropy'] = 0

        if column_types[col] == 'numeric':
            metadata['std_dev'] = non_null_values.std()
            metadata['skewness'] = stats.skew(non_null_values) if len(non_null_values) > 2 else 0
            metadata['kurtosis'] = stats.kurtosis(non_null_values) if len(non_null_values) > 3 else 0
            metadata['zero_fraction'] = (non_null_values == 0).sum() / len(non_null_values) if len(non_null_values) > 0 else 0
        else:
            metadata['std_dev'] = 0
            metadata['skewness'] = 0
            metadata['kurtosis'] = 0
            metadata['zero_fraction'] = 0

        # Initialize advanced metrics
        if encoded_columns[col] is None:
            for key in ['avg_mutual_info_from_others', 'avg_r2_from_others', 'avg_accuracy_from_others',
                        'avg_correlation_from_others', 'avg_correlation', 'mean_cv_score_as_feature',
                        'mean_cv_score_as_target', 'max_pairwise_corr_with_others',
                        'num_high_corr_with_others', 'cv_score_diff_as_target_vs_feature',
                        'gini_importance_sum_from_others', 'num_nonzero_importances_from_others']:
                metadata[key] = 0
            return col, metadata

        y = encoded_columns[col]
        target_is_classification = column_types[col] != 'numeric' or metadata['is_binary']

        mi_scores = []
        correlation_scores = []
        high_corr_count = 0
        max_corr = 0
        cv_scores_as_feature = []
        cv_scores_as_target = []
        gini_importance = 0
        nonzero_importances = 0

        # Pre-calculate to avoid repeated operations
        target_is_numeric = column_types[col] == 'numeric'

        for feature_col in valid_columns:
            if feature_col == col or encoded_columns[feature_col] is None:
                continue

            X = encoded_columns[feature_col].reshape(-1, 1)
            feature_is_classification = column_types[feature_col] != 'numeric' or (
                feature_col in metadata and metadata.get('is_binary', False)
            )
            feature_is_numeric = column_types[feature_col] == 'numeric'

            # Mutual Info - vectorized approach
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mi = mutual_info_classif(X, y, random_state=42)[0] if target_is_classification \
                        else mutual_info_regression(X, y, random_state=42)[0]
                if not np.isnan(mi) and not np.isinf(mi):
                    mi_scores.append(mi)
            except Exception:
                pass

            # Correlation - only calculate if both are numeric
            if target_is_numeric and feature_is_numeric:
                try:
                    # Use boolean indexing for better performance
                    valid_idx = df_sampled[col].notna() & df_sampled[feature_col].notna()
                    if valid_idx.sum() > 1:
                        # Use numpy directly for correlation calculation
                        x_vals = df_sampled.loc[valid_idx, feature_col].values
                        y_vals = df_sampled.loc[valid_idx, col].values
                        corr = abs(np.corrcoef(x_vals, y_vals)[0, 1])
                        if not np.isnan(corr):
                            correlation_scores.append(corr)
                            if corr > max_corr:
                                max_corr = corr
                            if corr > 0.8:
                                high_corr_count += 1
                except Exception:
                    pass

            # CV score (target → feature) - only calculate if enough samples
            try:
                if len(y) >= 10:
                    model = (RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, n_jobs=1) 
                             if feature_is_classification 
                             else RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=1))
                    score_type = 'accuracy' if feature_is_classification else 'r2'
                    scores = cross_val_score(model, y.reshape(-1, 1), encoded_columns[feature_col], 
                                            cv=min(3, len(y) // 3), scoring=score_type)
                    scores = [s for s in scores if not np.isnan(s) and (s > 0 if score_type == 'r2' else True)]
                    if scores:
                        cv_scores_as_feature.append(np.mean(scores))
            except Exception:
                pass

        # CV score (others → target)
        valid_feature_cols = [fcol for fcol in valid_columns if fcol != col and encoded_columns[fcol] is not None]
        if valid_feature_cols and len(y) >= 10:
            try:
                # Create feature matrix once
                X_all = np.column_stack([encoded_columns[fcol] for fcol in valid_feature_cols])
                X_all = StandardScaler().fit_transform(X_all)
                model = (RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, n_jobs=1) 
                         if target_is_classification 
                         else RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=1))
                score_type = 'accuracy' if target_is_classification else 'r2'
                scores = cross_val_score(model, X_all, y, cv=min(3, len(y) // 3), scoring=score_type)
                scores = [s for s in scores if not np.isnan(s) and (s > 0 if score_type == 'r2' else True)]
                if scores:
                    cv_scores_as_target.append(np.mean(scores))

                # Feature importances
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_all, y)
                    importances = model.feature_importances_
                    gini_importance = np.sum(importances)
                    nonzero_importances = np.count_nonzero(importances)

            except Exception:
                pass

        metadata['avg_mutual_info_from_others'] = np.mean(mi_scores) if mi_scores else 0
        metadata['avg_correlation'] = np.mean(correlation_scores) if correlation_scores else 0
        metadata['max_pairwise_corr_with_others'] = max_corr
        metadata['num_high_corr_with_others'] = high_corr_count
        metadata['mean_cv_score_as_feature'] = np.mean(cv_scores_as_feature) if cv_scores_as_feature else 0
        metadata['mean_cv_score_as_target'] = np.mean(cv_scores_as_target) if cv_scores_as_target else 0
        metadata['cv_score_diff_as_target_vs_feature'] = (
            metadata['mean_cv_score_as_target'] - metadata['mean_cv_score_as_feature']
        )
        metadata['gini_importance_sum_from_others'] = gini_importance
        metadata['num_nonzero_importances_from_others'] = nonzero_importances

        return col, metadata

    def _process_stats_score_column(self, df, candidate_col, threshold, score_weight):
        """
        Process stats score for a single column - helper function for parallelization.
        """
        y = df[candidate_col]
        X = df.drop(columns=[candidate_col])
        candidate_score = 0
        n_valid_tests = 0
        
        # Use vectorized operations where possible
        y_is_numeric = pd.api.types.is_numeric_dtype(y)
        
        unique_y = None
        if not y_is_numeric:
            unique_y = pd.Series(y.dropna()).unique()
        
        for feature_col in X.columns:
            x = df[feature_col]
            try:
                mask = y.notna() & x.notna()
                if mask.sum() < 3:
                    continue
                    
                x_clean = x[mask]
                y_clean = y[mask]
                
                is_x_numeric = pd.api.types.is_numeric_dtype(x_clean)
                
                # Case 1: Categorical target, Categorical feature → Chi-squared
                if not y_is_numeric and not is_x_numeric:
                    # Use vectorized crosstab
                    contingency = pd.crosstab(y_clean, x_clean)
                    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        if p < threshold:
                            candidate_score += score_weight
                            n_valid_tests += 1
                
                # Case 2: Categorical target, Numeric feature → ANOVA
                elif not y_is_numeric and is_x_numeric:
                    # More efficient grouping
                    groups = []
                    for cat in unique_y:
                        group_vals = x_clean[y_clean == cat].values
                        if len(group_vals) > 1:
                            groups.append(group_vals)
                            
                    if len(groups) > 1:
                        f, p = stats.f_oneway(*groups)
                        if p < threshold:
                            candidate_score += score_weight
                            n_valid_tests += 1
                
                # Case 3: Numeric target, Categorical feature → ANOVA
                elif y_is_numeric and not is_x_numeric:
                    unique_x = pd.Series(x_clean).unique()
                    groups = []
                    for cat in unique_x:
                        group_vals = y_clean[x_clean == cat].values
                        if len(group_vals) > 1:
                            groups.append(group_vals)
                            
                    if len(groups) > 1:
                        f, p = stats.f_oneway(*groups)
                        if p < threshold:
                            candidate_score += score_weight
                            n_valid_tests += 1
                
                # Case 4: Numeric target, Numeric feature → Pearson correlation
                elif y_is_numeric and is_x_numeric:
                    # Convert to numpy arrays for faster calculation
                    x_array = x_clean.values
                    y_array = y_clean.values
                    r, p = stats.pearsonr(x_array, y_array)
                    if p < threshold and abs(r) > 0.2:  # Only count strong correlations
                        candidate_score += score_weight * abs(r)  # Weight by correlation strength
                        n_valid_tests += 1
                
            except Exception:
                continue
        
        # Calculate final score
        if n_valid_tests > 0:
            avg_score = candidate_score / n_valid_tests
            # Add complexity bonus for columns with more relationships
            complexity_bonus = min(1.0, n_valid_tests / max(1, len(X.columns)))
            final_score = avg_score * (1 + complexity_bonus)
        else:
            final_score = 0
            
        return {
            'column': candidate_col, 
            'score': final_score
        }

    def extract_column_metadata(self, df):
        """
        Extracts metadata features for each column in a DataFrame.
        This metadata features are further passed to ML model to predict proba of target.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame to analyze
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing metadata features for each column
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if df.empty:
            return pd.DataFrame()

        # Sample data
        if len(df) > self.sample_size:
            df_sampled = df.sample(n=self.sample_size, random_state=42)
        else:
            df_sampled = df.copy()

        # Pre-compute column types and encodings - vectorized operations
        valid_columns = [col for col in df.columns if df[col].notna().any()]
        column_types = {}
        encoders = {}
        encoded_columns = {}
        
        # Batch calculation of column types
        for col in valid_columns:
            column_types[col] = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'

        # Perform encoding - use vectorized operations where possible
        for col in valid_columns:
            if column_types[col] == 'numeric':
                # Use values for efficiency
                col_values = df_sampled[col].values
                median_val = np.nanmedian(col_values)
                encoded_columns[col] = np.nan_to_num(col_values, nan=median_val)
            else:
                try:
                    le = LabelEncoder()
                    # Convert to string once and handle missing values efficiently
                    filled = df_sampled[col].fillna('missing_value').astype(str).values
                    encoded_columns[col] = le.fit_transform(filled)
                    encoders[col] = le
                except Exception:
                    encoded_columns[col] = None

        # Use parallel processing for column metadata
        metadata = {}
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=self._get_n_jobs()) as executor:
            # Process each column in parallel
            process_column_fn = partial(
                self._process_column_metadata,
                df_sampled, 
                valid_columns=valid_columns,
                encoded_columns=encoded_columns,
                column_types=column_types
            )
            
            futures = {executor.submit(process_column_fn, col): col for col in valid_columns}
            
            for future in as_completed(futures):
                col, col_metadata = future.result()
                metadata[col] = col_metadata

        result_df = pd.DataFrame.from_dict(metadata, orient='index')
        return result_df.fillna(0)

    def stats_score(self, df, threshold=0.05, score_weight=1.0):
        """
        Performs statistical analysis to identify potential target columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame to analyze
        threshold : float, default=0.05
            Statistical significance threshold for p-values
        score_weight : float, default=1.0
            Weight to apply to each significant relationship
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing top 5 potential target columns with scores
        """
        # Sample data
        if len(df) > self.sample_size:
            df_sampled = df.sample(n=self.sample_size, random_state=42)
        else:
            df_sampled = df
        
        scores = []
        
        # Parallelize column processing
        with ThreadPoolExecutor(max_workers=self._get_n_jobs()) as executor:
            # Process each column in parallel
            process_column_fn = partial(
                self._process_stats_score_column,
                df_sampled,
                threshold=threshold,
                score_weight=score_weight
            )
            
            # Submit all columns for processing
            futures = {executor.submit(process_column_fn, col): col for col in df.columns}
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    scores.append(result)
                except Exception as e:
                    logger.error(f"Error processing column {futures[future]}: {str(e)}")
                    # print(f"Error processing column {futures[future]}: {str(e)}")
                    
        result_df = pd.DataFrame(scores).sort_values(by='score', ascending=False)
        return result_df.head(5)

    def _get_n_jobs(self):
        """Helper method to determine number of jobs for parallel processing"""
        import multiprocessing
        if self.n_jobs == -1:
            return multiprocessing.cpu_count()
        return min(max(1, self.n_jobs), multiprocessing.cpu_count())


    def detect_target_column(self, df, threshold=0.156):
        """
        Combines machine learning and statistical methods to detect the most likely target column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame to analyze
        threshold : float, default=0.156
            Threshold for considering a column as a potential target
            
        Returns:
        --------
        str
            Name of the detected target column
        """
        try:
            if df.empty:
                return {
                    "status": "fail",
                    "target_column": None,
                    "confidence": None,
                    "message": "Input DataFrame is empty."
            }

            if df.shape[1] == 0:
                {
                "status": "fail",
                "target_column": None,
                "confidence": None,
                "message": "Input DataFrame has no columns."
            }
            # Extract ML features for each column
            X_features = self.extract_column_metadata(df)
            
            # Get statistical scores
            stat_scores_df = self.stats_score(df)
            
            if self.ml_model is None:
                # If no ML model provided, return the top statistical result
                target_col = stat_scores_df.iloc[0]['column']
                return {
                    "status" : "success",
                    "target_column" : target_col,
                    "confidence" : None,
                    "message" : "Target column detected using statistical method" 
                }
            
            # Get ML model predictions
            proba_scores = self.ml_model.predict_proba(X_features)[:, 1]
            
            columns = X_features.index.tolist()
            ml_ranked = [(col, prob) for col, prob in zip(columns, proba_scores) if prob >= threshold]
            ml_ranked = sorted(ml_ranked, key=lambda x: x[1], reverse=True)
            
            stat_score_dict = dict(zip(stat_scores_df['column'], stat_scores_df['score']))
            stat_ranked = sorted(stat_score_dict.items(), key=lambda x: x[1], reverse=True)
    
            ml_top = {col for col, _ in ml_ranked}
            stat_top = {col for col, _ in stat_ranked}
            intersection = ml_top & stat_top
    
            if intersection:
                # Return column with highest combined ranking
                combined_sorted = sorted(
                        intersection,
                        key=lambda col: (
                            next(i for i, (c, _) in enumerate(ml_ranked) if c == col) +
                            next(i for i, (c, _) in enumerate(stat_ranked) if c == col)
                        )
                    )
                final_col = combined_sorted[0]
                final_prob = dict(ml_ranked).get(final_col, None)
                return {
                    "status" : "success",
                    "target_column" : final_col,
                    "confidence" : final_prob,
                    "message" : "Target column detetcted using ML + statistical intersection."
                }
            elif ml_ranked:
                final_col, final_prob = ml_ranked[0]
                return {
                    "status" : "success",
                    "target_column" : final_col,
                    "confidence" : final_prob,
                    "message" : "Target column detetcted using ML model."
                }
    
            
            elif stat_ranked:
                final_col = stat_ranked[0][0]
                return {
                    "status" : "success",
                    "target_column" : final_col,
                    "confidence" : None,
                    "message" : "Target column detetcted using statistical method."
                } 
                
            else:
                return {
                    "status" : "fail",
                    "target_column" : None,
                    "confidence" : None,
                    "message" : "Could not detect a target column."
                }
            
        except Exception as e:
            return {
                "status" : "fail",
                "target_column" : None,
                "confidence" : None,
                "message" : f"Unexpected error: {str(e)}"
            }


    # def detect_target_column(self, df, threshold=0.156):
    #     """
    #     Combines machine learning and statistical methods to detect the most likely target column.
        
    #     Parameters:
    #     -----------
    #     df : pandas.DataFrame
    #         The input DataFrame to analyze
    #     threshold : float, default=0.156
    #         Threshold for considering a column as a potential target
            
    #     Returns:
    #     --------
    #     str
    #         Name of the detected target column
    #     """
    #     # Extract ML features for each column
    #     X_features = self.extract_column_metadata(df)
        
    #     # Get statistical scores
    #     stat_scores_df = self.stats_score(df)
        
    #     if self.ml_model is None:
    #         # If no ML model provided, return the top statistical result
    #         return stat_scores_df.iloc[0]['column']
    #     # Get ML model predictions
    #     proba_scores = self.ml_model.predict_proba(X_features)[:, 1]        
    #     columns = X_features.index.tolist()
    #     ml_ranked = [(col, prob) for col, prob in zip(columns, proba_scores) if prob >= threshold]
    #     ml_ranked = sorted(ml_ranked, key=lambda x: x[1], reverse=True)
    
    #     stat_score_dict = dict(zip(stat_scores_df['column'], stat_scores_df['score']))
    #     stat_ranked = sorted(stat_score_dict.items(), key=lambda x: x[1], reverse=True)

    #     ml_top = {col for col, _ in ml_ranked}
    #     stat_top = {col for col, _ in stat_ranked}
    #     intersection = ml_top & stat_top

    #     if intersection:
    #         # Return column with highest combined ranking
    #         return success_response(
    #             message="Target column detected successfully", 
    #             data = sorted(
    #                         intersection,
    #                         key=lambda col: (
    #                             next(i for i, (c, _) in enumerate(ml_ranked) if c == col) +
    #                             next(i for i, (c, _) in enumerate(stat_ranked) if c == col)
    #                         )
    #                     )[0]
    #             )
    #     elif ml_ranked:
    #         return success_response(message="Target column detected successfully", data=ml_ranked[0][0])
    #     elif stat_ranked:
    #         return success_response(message="Target column detected successfully", data=stat_ranked[0][0])
    #     else:
    #         return success_response(message="Could not detect a target column")
        #     return sorted(
        #         intersection,
        #         key=lambda col: (
        #             next(i for i, (c, _) in enumerate(ml_ranked) if c == col) +
        #             next(i for i, (c, _) in enumerate(stat_ranked) if c == col)
        #         )
        #     )[0]
        # elif ml_ranked:
        #     return ml_ranked[0][0]
        # elif stat_ranked:
        #     return stat_ranked[0][0]
        # else:
        #     return "Could not detect a target column."
        

# detector = TargetColumnDetector(model_path='C:\\Users\\aniketd\\Documents\\AYC\\BINDAS APP\\python_modules\\modules_by_aniket\\ML_models\\Target_detector.pkl')
# df = pd.read_csv("C:\\Aniket\\Bindas\\Datasets for target column\\housing_price_dataset.csv")
# target_column = detector.detect_target_column(df)
# print(target_column)
