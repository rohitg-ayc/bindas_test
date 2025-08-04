# Global Import
import pandas as pd
import numpy as np
import re
import logging
from scipy import stats
from backend.core.duckdb_project_manager import DuckDBProjectManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean data to make it ready for Exploratory Data Analysis.

    Parameters:
    - df: Pandas DataFrame. 
    - target_column (str, optional): The target column or Dependant variable in dataset.

    Methods:
    -detect_data_types ()..
    - fill_missing_value()..
    - remove_duplicates()..
    - detect_and_handle_outliers()..
    """
    def __init__(self, df=None, target_column=None,db_manager=None, table_name=None):

        __version__ = '0.1'
        if df is None and target_column is None:
            logger.error("Initialization failed: Both DataFrame and target_column are None.")
            raise ValueError("Provide Valid Data Frame and Target column")
            
        elif df is not None and target_column is None:
            logger.warning("No target column specified. Defaulting to None.")
            self.df = df
            self.target_column = None
            self.df_original = df.copy()
            
        elif df is not None and target_column is not None:
            if target_column in df.columns:
                self.df = df
                self.target_column = target_column 
                self.df_original = df.copy()
            else:
                logger.error("Target column '%s' not found in DataFrame columns: %s", target_column, df.columns.tolist())
                raise ValueError(f"Target column '{target_column}' does not match any column in the DataFrame.")
        else:
            logger.error("Initialization failed: DataFrame is None but target_column is provided.")
            raise ValueError("DataFrame cannot be None when target_column is provided.")
        
        # 💡 Logging Integration
        self.db_manager = db_manager
        self.table_name = table_name
        # self.sequence = 1  # Used to track step sequence
        
        logger.info("DataCleaner initialized with target column: %s", self.target_column)
    
    
    def detect_data_types(self, date_threshold=0.75, num_threshold=0.8, cat_threshold=0.1, ident_threshold=0.6, inplace=False):
        """
        Detect data into different subcategories based on data types and performs explicit data type conversion if needed.
        e.g., datetime_column, boolean_column, categorical_column, continuous_column, identifier_column.
    
        Parameters:
        - date_threshold (float): Limit for converting datetime columns if above the threshold (default=0.75).
        - num_threshold (float): Limit for converting numerical columns if above the threshold (default=0.0.8).
        - cat_threshold (float): Limit for converting object columns to categorical if below the threshold (default=0.1).
        - ident_threshold (float): Limit for Identifying Identifier column (default=0.6). 
        - inplace: Boolean to specify if self.df should be modified in place (default=False).
    
        Returns:
        An updated DataFrame and a dictionary containing lists of columns categorized by their data type if inplace = False, 
        if inplace = True then only dictionary containing lists of columns.
        """
        
    
        # Local variables for each column type
        datetime_cols = []                            
        boolean_cols = []
        category_cols = []
        continuous_cols = []
        identifier_cols = []

        df = self.df
        target_column = self.target_column
    
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("No data loaded into detect_data_types method.")
            raise ValueError("No data loaded into Detect Data Types Method.")
            
        try:    
            for col in df.columns:
                logger.debug(f"Processing column: {col}")
                # Handle datetime columns
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                    logger.info(f"Identified datetime column: {col}")
    
                # Handle boolean columns
                elif pd.api.types.is_bool_dtype(df[col]):
                    boolean_cols.append(col)
                    logger.info(f"Identified boolean column: {col}")
                # Handle numeric columns
                elif pd.api.types.is_numeric_dtype(df[col]):
                    continuous_cols.append(col)
                    logger.info(f"Identified continuous column: {col}")
                    # Check for Identifier column
                    if pd.api.types.is_integer_dtype(df[col]):
                        if df[col].nunique(dropna=True) / len(df[col].dropna()) > ident_threshold or df[col].nunique(dropna=True) == len(df[col].dropna()):
                            identifier_cols.append(col)   
                            logger.info(f"Identified identifier column: {col}")
 
    
                # Handle object/string/categorical columns
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    unique_values = df[col].dropna().unique()
                    total_num = len(df[col].dropna())
    
                    # Define regex patterns for date-related keywords
                    date_patterns = re.compile(r'(date|day|time|year|month|period)', re.IGNORECASE)
    
                    # Try to detect datetime columns based on regex and format detection
                    # Attempt to parse dates with multiple formats
                    date_formats = ["%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d"]
                    
                    if date_patterns.search(col) or df[col].str.contains(r'\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{1,2}-\d{1,2}').any():
                        for fmt in date_formats:
                            try:
                                temp_col = pd.to_datetime(df[col], format=fmt, errors='coerce')
                                non_nat_ratio_date = temp_col.notna().mean()
                                if non_nat_ratio_date > date_threshold:
                                    df[col] = temp_col
                                    datetime_cols.append(col)
                                    logger.info(f"Converted to datetime: {col} using format {fmt}")
                                    break  # Exit the loop if successful
                            except:
                                logger.debug(f"Failed to convert {col} using format {fmt}: {e}")
                                continue
                        else:
                            # If none of the formats worked, try generic parsing
                            temp_col = pd.to_datetime(df[col], errors='coerce')
                            non_nat_ratio_date = temp_col.notna().mean()
                            if non_nat_ratio_date > date_threshold:
                                df[col] = temp_col
                                datetime_cols.append(col)
                                logger.info(f"Converted to datetime: {col} using generic parser")
                            else:
                                df[col] = df[col].astype('string')
                                category_cols.append(col)
                                logger.info(f"Column {col} kept as category (string)")
                    
                    else:
                        # Check if the column can be converted to numeric
                        converted_col_num = pd.to_numeric(df[col], errors='coerce')
                        non_nat_ratio_num = converted_col_num.notna().mean()
    
                        if non_nat_ratio_num > num_threshold:
                            df[col] = converted_col_num
                            continuous_cols.append(col)
                            logger.info(f"Converted to numeric: {col}")
    
                        # Handle boolean-like columns
                        
                        elif len(unique_values) == 2 and set(unique_values).issubset(
                                {True, False, 'True', 'False', 'Yes', 'No', 'Y', 'N', '1', '0', 1, 0}):
                            df[col] = df[col].replace(
                                {'True': True, 'False': False,
                                 'Yes': True, 'No': False, 'Y': True, 'N': False,
                                 '1': True, '0': False, 1: True, 0: False}).astype('bool')
                            boolean_cols.append(col)
                            logger.info(f"Converted to boolean: {col}")
    
                        # Handle identifier columns
                        elif len(unique_values)/total_num > ident_threshold or len(unique_values) == total_num:
                            if df[col].dropna().apply(lambda x: str(x).isalnum() and not str(x).isalpha() and not str(x).isdigit()).all():
                                df[col] = df[col].astype('string')
                                identifier_cols.append(col)
                                logger.info(f"Identified identifier string column: {col}")
    
                        # Handle categorical columns based on threshold
                        elif len(unique_values) / total_num < cat_threshold:
                            df[col] = df[col].astype('category')
                            category_cols.append(col)
                            logger.info(f"Converted to category: {col}")
    
                        else:
                            df[col] = df[col].astype('string')
                            category_cols.append(col)
                            logger.info(f"Converted to string (fallback): {col}")
    
                else:
                    df[col] = df[col].astype('string')
                    category_cols.append(col)
                    logger.info(f"Defaulted to string: {col}")
            
            if inplace:
                self.df = df
                logger.info("Data type detection completed and changes saved to self.df.")

                # 🔍 Log applied step if db_manager is provided
                if self.db_manager:
                    self.db_manager.log_applied_step(
                        table_name=self.table_name,
                        operation="detect_data_types",
                        parameters={
                            "date_threshold": date_threshold,
                            "num_threshold": num_threshold,
                            "cat_threshold": cat_threshold,
                            "ident_threshold": ident_threshold,
                            "results": {
                                'datetime': datetime_cols,
                                'boolean': boolean_cols,
                                'categorical': category_cols,
                                'continuous': continuous_cols,
                                'identifier': identifier_cols
                            }
                        },
                        # sequence=self.sequence
                    )
                    # self.sequence += 1
                return {'data_types': {
                        'datetime': datetime_cols,
                        'boolean': boolean_cols,
                        'categorical': category_cols,
                        'continuous': continuous_cols,
                        'identifier': identifier_cols}
                       }
            else:
                logger.warning("Data type detection completed but changes not saved (inplace=False).")
                return df, {'data_types': {
                        'datetime': datetime_cols,
                        'boolean': boolean_cols,
                        'categorical': category_cols,
                        'continuous': continuous_cols,
                        'identifier': identifier_cols}
                       }
    
        except Exception as e:
            logger.error(f"An error occurred while detecting the data types: {e}")
            raise RuntimeError(f"An error occurred while detecting the data types: {e}")
        


    
    def fill_missing_value(self, numeric_strategy='median', 
                       categorical_strategy='mode', 
                       datetime_strategy='ffill', 
                       threshold=0.8, knn_neighbors=5, default_date=None, 
                       inplace=False, custom_suggestions=None, 
                       remove_missing_values_in_custom_cols=False):
        """
        Check for missing values and drop columns where the missing value ratio is above the threshold.
        Impute missing values based on the specified strategy for different data types. Allows for removing 
        missing values specifically for custom-suggested columns if remove_missing_values_in_custom_cols is True.
        
        Parameters:
        - numeric_strategy (str): Strategy for handling missing values in numeric columns.
              Choices: ['mean', 'median', 'mode', 'knn', 'zero'] (default: 'median').
        - categorical_strategy (str): Strategy for handling missing values in categorical columns.
              Choices: ['mode', 'ffill', 'bfill'] (default: 'mode').
        - datetime_strategy (str): Strategy for handling missing values in datetime columns.
              Choices: ['ffill', 'bfill', 'default'] (default: 'ffill').
        - threshold (float): Minimum fraction of non-missing values required to keep the column (default=0.8).
        - knn_neighbors (int): Number of neighbors to use for KNN imputer (only applies to numeric columns) (default=5).
        - default_date (str): Specific date to use if datetime_strategy='default'.
        - inplace (bool): If True, changes will be saved in self.df, else returns a new DataFrame (default=False).
        - custom_suggestions: Dictionary with custom strategies for specific columns. Format: {'column_name': 'strategy',...}.
        - remove_missing_values_in_custom_cols (bool): If True, remove missing values only from columns in custom_suggestions.

        Note: Input data can be (DataFrame, Series, or NumPy array).
        
        Returns:
        - An updated DataFrame if inplace is False else None.
        """

        logger.info("Starting fill_missing_value method.")

        # Import Instance variable
        df = self.df
        target_column = self.target_column

         # Convert input data to DataFrame if it's a Series or NumPy array
        if isinstance(df, pd.Series):
            logger.debug("Converting Series to DataFrame.")
            df = df.to_frame()  # Convert Series to DataFrame
        elif isinstance(df, np.ndarray):
            logger.debug("Converting NumPy array to DataFrame.")
            if df.ndim == 1:
                df = pd.DataFrame(df, columns=['Array_Column'])  # Handle 1D arrays
            else:
                df = pd.DataFrame(df)  # Handle 2D arrays
        elif isinstance(df, pd.DataFrame):
            logger.debug("Input is already a DataFrame.")
            df = df
        else:
            logger.error("Input type is invalid.")
            raise ValueError("Input must be a Pandas DataFrame, Series, or NumPy array.")
    
        try:
            #if not isinstance(df, pd.DataFrame) or df.empty:
                #raise ValueError("No data loaded into Fill Missing Value.")
            if df.empty:
                logger.error("Input DataFrame is empty.")
                raise ValueError("No data loaded into Fill Missing Value.")
            
            # Drop columns with missing values exceeding the threshold
            threshold_count = len(df) * threshold                                  
            cols_to_drop = [col for col in df.columns if df[col].notnull().sum() < threshold_count and col != target_column]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                logger.info(f'Columns dropped due to missing value threshold:\n{cols_to_drop}')
            else:
                logger.info("No columns dropped due to threshold.")
            # Log for filled columns and their strategies
            fill_log = []
            
            # If custom_suggestions are not provided, initialize an empty dictionary
            if custom_suggestions is None:
                logger.debug("No custom suggestions provided, initializing as empty dictionary.")
                custom_suggestions = {}
    
            # Function to apply strategy based on column type and custom suggestions
            def apply_strategy(col, col_type, default_strategy):
                strategy_to_apply = custom_suggestions.get(col, default_strategy)
                logger.debug(f"Applying strategy '{strategy_to_apply}' to column '{col}'.")

                if strategy_to_apply == 'mean' and col_type == 'numeric':
                    mean_value = df[col].mean()
                    if pd.api.types.is_integer_dtype(df[col]):
                        mean_value = int(mean_value)
                    df[col].fillna(mean_value, inplace=True)
                    return 'mean'
                elif strategy_to_apply == 'median' and col_type == 'numeric':
                    median_value = df[col].median()
                    if pd.api.types.is_integer_dtype(df[col]):
                        median_value = int(median_value)  
                    df[col].fillna(median_value, inplace=True)
                    return 'median'
                elif strategy_to_apply == 'mode':
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)
                    return 'mode'
                elif strategy_to_apply == 'zero' and col_type == 'numeric':
                    df[col].fillna(0, inplace=True)
                    return 'zero'
                elif strategy_to_apply in ['ffill', 'bfill']:
                    df[col].fillna(method=strategy_to_apply, inplace=True)
                    return strategy_to_apply
                elif strategy_to_apply == 'remove':
                    df.dropna(subset=[col], inplace=True)
                    return 'remove'
                else:
                    logger.error(f"Invalid strategy '{strategy_to_apply}' for column {col}.")
                    raise ValueError(f"Invalid strategy '{strategy_to_apply}' for column {col}.")
    
            # Handle missing values based on remove_missing_values_in_custom_cols
            columns_to_process = custom_suggestions.keys() if remove_missing_values_in_custom_cols else df.columns
    
            # Handle missing values for numerical columns
            # for col in df.select_dtypes(include=['number']).columns:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in columns_to_process or col == target_column:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values == 0:
                    logger.debug(f"No missing values in numeric column '{col}'. Skipping.")
                    continue  # Skip columns with no missing values
                applied_strategy = apply_strategy(col, 'numeric', numeric_strategy)
                fill_log.append(f'Missing values: {missing_values} in column {col} filled using {applied_strategy} strategy.')
    
            # Handle missing values for categorical and boolean columns
            for col in df.select_dtypes(include=['boolean', 'category', 'object', 'string']).columns:
                if col not in columns_to_process or col == target_column:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values == 0:
                    logger.debug(f"No missing values in categorical column '{col}'. Skipping.")
                    continue  # Skip columns with no missing values
                applied_strategy = apply_strategy(col, 'categorical', categorical_strategy)
                fill_log.append(f'Missing values: {missing_values} in column {col} filled using {applied_strategy} strategy.')
            
            # Handle missing values for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            for col in datetime_cols:
                if col not in columns_to_process or col == target_column:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values == 0:
                    logger.debug(f"No missing values in datetime column '{col}'. Skipping.")
                    continue  # Skip columns with no missing values
                strategy_to_apply = custom_suggestions.get(col, datetime_strategy)
                logger.debug(f"Applying datetime strategy '{strategy_to_apply}' to column '{col}'.")
                if strategy_to_apply == 'ffill':
                    df[col].fillna(method='ffill', inplace=True)
                elif strategy_to_apply == 'bfill':
                    df[col].fillna(method='bfill', inplace=True)
                elif strategy_to_apply == 'default' and default_date:
                    df[col].fillna(pd.to_datetime(default_date), inplace=True)
                else:
                    logger.error(f"Invalid datetime_strategy '{strategy_to_apply}' for column '{col}'.")
                    raise ValueError(f"Invalid datetime_strategy {strategy_to_apply} for column {col}.")
                fill_log.append(f'Missing values: {missing_values} in column {col} filled using {strategy_to_apply} strategy.')
    
            # Print log and return results
            logger.info("\n--- Missing Value Imputation Log ---")
            for col_log in fill_log:
                logger.info(col_log)
            
            if inplace:
                self.df = df
                logger.info(f"Missing values filled with specified strategies, changes saved.")
                
                if self.db_manager:
                    self.db_manager.log_applied_step(
                        table_name=self.table_name,
                        operation="fill_missing_value",
                        parameters={
                            "numeric_strategy": numeric_strategy,
                            "categorical_strategy": categorical_strategy,
                            "datetime_strategy": datetime_strategy,
                            "threshold": threshold,
                            "custom_suggestions": custom_suggestions,
                            "columns_dropped": cols_to_drop,
                            "fill_log": fill_log
                        },
                        # sequence=self.sequence
                    )
                    # self.sequence += 1
                    return None
            
            else:
                logger.info(f"Missing values filled with specified strategies, but changes not saved (inplace=False).")
                return df
        
        except Exception as e:
            logger.exception("Exception occurred in fill_missing_value.")
            raise RuntimeError(f"An error occurred while filling missing values: {str(e)}")


    def remove_duplicates(self, inplace=False):
        """
        Remove duplicate rows from the DataFrame.

        Parameters:
        - inplace (bool): If True, changes will be saved in self.df, else returns a new DataFrame (default=False).

        Note: Input data can be (DataFrame, NumPy multidimensional array).
    
        Returns:
        - The DataFrame with duplicates removed if inplace is False, None if inplace is True.
        """
        logger.info("Starting duplicate removal process.")
        df = self.df

        # Convert input data to DataFrame if it's a Series or NumPy array
        if isinstance(df, np.ndarray):
            logger.debug("Converting NumPy array to DataFrame.")
            if df.ndim == 1:
                logger.error("1D NumPy array is not supported.")
                raise ValueError("Input must be a Pandas DataFrame or two dimensional NumPy array.")
            else:
                df = pd.DataFrame(df)  # Handle 2D arrays
        elif isinstance(df, pd.DataFrame):
            logger.debug("Input is already a DataFrame.")
            df = df
        else:
            logger.error("Input type is invalid.")
            raise ValueError("Input must be a Pandas DataFrame, Series, or NumPy array.")
        
        try:
            # Ensure the DataFrame is valid
            if df.empty:
                logger.error("DataFrame is empty. Cannot remove duplicates.")
                raise ValueError("No valid DataFrame provided for removing duplicates.")
    
            initial_shape = df.shape
            logger.debug(f"Initial DataFrame shape: {initial_shape}")
    
            # Check if there's a datetime column and adjust the strategy
            keep_strategy = 'last' if any(df.dtypes == 'datetime') else 'first'
            logger.debug(f"Using keep strategy: '{keep_strategy}'")
    
            # Create a boolean mask for duplicates and drop them
            duplicates_mask = df.duplicated(keep=keep_strategy)
            num_duplicates = duplicates_mask.sum()
    
            # Inform the user if no duplicates are found
            if num_duplicates == 0:
                logger.info("No duplicates found.")
                return None
    
            # Drop duplicates
            df = df[~duplicates_mask].reset_index(drop=True)
    
            final_shape = df.shape
            
            logger.info(f"Removed {num_duplicates} duplicate rows.")
            logger.debug(f"New DataFrame shape after duplicate removal: {final_shape}")
    
            # Handle in-place modification
            if inplace:
                self.df = df
                logger.info("Duplicate rows have been removed and changes saved.")
                if self.db_manager:
                    self.db_manager.log_applied_step(
                        table_name=self.table_name,
                        operation="remove_duplicates",
                        parameters={
                            "initial_shape": initial_shape,
                            "final_shape": final_shape,
                            "duplicates_removed": num_duplicates
                        },
                        # sequence=self.sequence
                    )
                    # self.sequence += 1
                return None
            else:
                logger.info("Duplicate rows removed but changes not saved (inplace=False).")
                return df
    
        except Exception as e:
            logger.exception("An error occurred while removing duplicates.")
            raise RuntimeError(f"An error occurred while removing duplicates: {e}")


    def detect_and_handle_outliers(self, outlier_detection_method='IQR', z_thresh=3,
                               dbscan_eps=0.5, dbscan_min_samples=5, 
                               outlier_handling_method='remove', inplace=False, 
                               custom_strategy=None, remove_outliers_in_custom_cols=False):
        """
        Detect and handle outliers using the specified method or custom strategy.
    
        Parameters:
        - outlier_detection_method (str): ['IQR', 'z_score', 'auto', 'dbscan'] (default: 'IQR').
        - z_thresh (float): Threshold for Z-score method (default: 3).
        - dbscan_eps (float): DBSCAN eps parameter (default: 0.5).
        - dbscan_min_samples (int): DBSCAN min_samples parameter (default: 5).
        - outlier_handling_method (str): How to handle outliers ['remove', 'mean', 'median', 'mode', 'adjust'] (default: 'remove').
        - inplace (bool): Modify DataFrame in place (default: False).
        - custom_strategy: Dictionary with custom strategies for specific columns.
                           eg. {col_name:{outlier_detection_method: value, outlier_handling_method: value}}
        - remove_outliers_in_custom_cols (bool): If True, only handle outliers in the columns mentioned in the custom strategy (default: False).

        Note: Input data can be (DataFrame, Series, or NumPy array).
    
        Returns:
        - DataFrame with outliers handled or None if inplace=True.
        """

        logger.info("Starting outlier detection and handling.")
        

        df = self.df
        target_column = self.target_column
        
        # Convert input data to DataFrame if it's a Series or NumPy array
        if isinstance(df, pd.Series):
            logger.debug("Input is a Series. Converting to DataFrame.")
            df = df.to_frame()  # Convert Series to DataFrame
        elif isinstance(df, np.ndarray):
            logger.debug("Input is a NumPy array. Converting to DataFrame.")
            if df.ndim == 1:
                df = pd.DataFrame(df, columns=['Array_Column'])  # Handle 1D arrays
            else:
                df = pd.DataFrame(df)  # Handle 2D arrays
        elif isinstance(df, pd.DataFrame):
            logger.debug("Input is already a DataFrame.")
            df = df
        else:
            logger.error(" Invalid input type for outlier detection.")
            raise ValueError("Input must be a Pandas DataFrame, Series, or NumPy array.")
    
        try:
            # Check if DataFrame is valid
            if df.empty:
                logger.error(" The DataFrame is empty or invalid.")
                raise ValueError("The DataFrame is empty or invalid.")
            
            logger.info(f" DataFrame shape before outlier handling: {df.shape}")
    
            # Exclude target column from outlier detection
            columns_to_check = [col for col in df.columns if col != target_column]
    
            # Handle numerical columns and ignore non-numeric, datetime, and boolean columns
            numerical_cols = df[columns_to_check].select_dtypes(include=['number']).columns.tolist()
            datetime_cols = df[columns_to_check].select_dtypes(include=['datetime']).columns.tolist()
            boolean_cols = df[columns_to_check].select_dtypes(include=['bool']).columns.tolist()

            logger.debug(f"Numerical columns detected: {numerical_cols}")
            logger.debug(f"Datetime columns detected (ignored): {datetime_cols}")
            logger.debug(f"Boolean columns detected (ignored): {boolean_cols}")


            # Ensure numerical columns exist
            if not numerical_cols:
                logger.warning(" No numerical columns found for outlier detection.")
                raise ValueError("No numerical columns available for outlier detection.")
    
            # If only handling outliers in custom-specified columns
            if remove_outliers_in_custom_cols:
                if not custom_strategy:
                    logger.error("Custom strategy required when remove_outliers_in_custom_cols is True.")
                    raise ValueError("Custom strategy must be provided if remove_outliers_in_custom_cols is set to True.")
                numerical_cols = [col for col in numerical_cols if col in custom_strategy]
                logger.debug(f"Restricting outlier detection to custom strategy columns: {numerical_cols}")
    
            # Initialize variables for logging
            outliers_removed = 0
            outlier_log = []
    
            # Process each numerical column for other outlier detection methods
            for col in numerical_cols:
                col_method = outlier_detection_method
                col_replace_with = outlier_handling_method
    
                # Apply custom strategy if provided for the column
                if custom_strategy and col in custom_strategy:
                    col_method = custom_strategy[col].get('outlier_detection_method', col_method)
                    col_replace_with = custom_strategy[col].get('outlier_handling_method', col_replace_with)
                logger.debug(f"Processing column: {col} | Method: {col_method} | Handling: {col_replace_with}")

                try:
                    # Detect outliers based on the specified method
                    if col_method == 'IQR':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                        logger.debug(f"{col} | IQR bounds: ({lower_bound}, {upper_bound})")
    
                    elif col_method == 'z_score':
                        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                        outliers = z_scores > z_thresh
                        logger.debug(f"{col} | z-score threshold: {z_thresh}")

    
                    elif col_method == 'auto':
                        # Check normality for large or small samples
                        if len(df[col].dropna()) > 5000:
                            from scipy.stats import kstest
                            stat, p_value = kstest(df[col].dropna(), 'norm')
                        else:
                            from scipy.stats import shapiro
                            stat, p_value = shapiro(df[col].dropna())
    
                        normality = 'Normal' if p_value > 0.05 else 'Not Normal'
                        logger.debug(f"{col} | Normality test p-value: {p_value} | Assumed: {normality}")
                        if normality == 'Normal':
                            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                            outliers = z_scores > z_thresh
                        else:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    
                    else:
                        logger.error(f"❌ Invalid outlier detection method '{col_method}' for column '{col}'.")
                        raise ValueError(f"Invalid outlier detection method '{col_method}' for column '{col}'.")
    
                    # Count outliers
                    num_outliers = outliers.sum()
                    logger.info(f"✅ Column '{col}' | Outliers detected: {num_outliers}")
    
                    # Skip if no outliers found
                    if num_outliers == 0:
                        continue
    
                    # Log the details for outlier handling
                    outlier_log.append(f"Column '{col}': {num_outliers} outliers detected using {col_method} method and handled using {col_replace_with}.")
    
                    # Handle outliers based on the provided handling method
                    if col_replace_with == 'remove':
                        df = df[~outliers]
                        outliers_removed += num_outliers
    
                    elif col_replace_with == 'mean':
                        df.loc[outliers, col] = df[col].mean()
    
                    elif col_replace_with == 'median':
                        df.loc[outliers, col] = df[col].median()
    
                    elif col_replace_with == 'mode':
                        df.loc[outliers, col] = df[col].mode()[0]
    
                    elif col_replace_with == 'adjust':
                        if col_method == 'IQR':
                            logger.info(f"Adjusting values for column '{col}' to IQR bounds.")
                            df.loc[df[col] < lower_bound, col] = lower_bound
                            df.loc[df[col] > upper_bound, col] = upper_bound
                            logger.info(f"column: {col}, lower bound: {lower_bound} and upper bound: {upper_bound}")
                        elif col_method == 'z_score':
                            df.loc[outliers, col] = df[col].median()
    
                    else:
                        logger.error(f" Invalid handling method '{col_replace_with}' for column '{col}'.")
                        raise ValueError(f"Invalid handling method '{col_replace_with}' for column '{col}'.")
    
                except Exception as e:
                    logger.error(f"Error handling outliers for column '{col}': {e}")
    
            logger.info("\n--- Outliers Handling Log ---")
            for log_entry in outlier_log:
                  logger.info(log_entry)
    
            if inplace:
                self.df = df
                logger.info("Outliers have been handled, and changes are saved (inplace=True).")
                if self.db_manager:
                    self.db_manager.log_applied_step(
                        table_name=self.table_name,
                        operation="detect_and_handle_outliers",
                       parameters={
                        "columns_processed": numerical_cols,
                        "outlier_detection_method": outlier_detection_method,
                        "outlier_handling_method": outlier_handling_method,
                        "outliers_removed": outliers_removed,
                        "custom_strategy": custom_strategy
                    },
                        # sequence=self.sequence
                    )
                    # self.sequence += 1
                return None
            else:
                logger.info("Outliers have been handled but changes are not saved (inplace=False).")
                return df
    
        except Exception as e:
                logger.info(f"An unexpected error occurred while handling outliers: {e}")

