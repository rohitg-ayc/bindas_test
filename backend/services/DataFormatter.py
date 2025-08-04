# Global Import
import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from backend.core.duckdb_project_manager import DuckDBProjectManager
from sklearn.metrics import mean_squared_error

# # Setup logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     filename='data_formatter.log',  # Optional: save to file
#     filemode='a'
# )

# logger = logging.getLogger("DataFormatter")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Provides different data formatting options.

    Parameters:
        - df: Pandas DataFrame to be formatted.
        - target_column (str, optional): target/dependant variable in dataframe (default=None).

    Methods:
    - change_column_headers()
    - duplicate_column()
    - delete_columns()
    - convert_data_type() -- routes
    - generate_datetime_columns()
    - combine_date_and_time()
    - combine_columns()
    - split_column()
    - remove_special_characters()
    - add_suffix_or_prefix()
    - format_string_columns()
    - pad_zeros()
    - convert_timeseries_pred_to_dataframe() (staticmethod)
    - calculate_time_series_metrics() (staticmethod)
    """
    
    def __init__(self, df=None, target_column=None,db_manager=None, table_name=None):

        __version__ = '0.1'
        logger.info("Initializing DataCleaner instance.")
        if df is None and target_column is None:
            logger.error("Both DataFrame and target_column are None.")
            raise ValueError("Provide Valid Data Frame and Target column")
            
        elif df is not None and target_column is None:
            self.df = df
            self.target_column = None
            self.df_original = df.copy(deep=True)
            logger.info("DataFrame initialized without a target column.")
            
        elif df is not None and target_column is not None:
            if target_column in df.columns:
                self.df = df
                self.target_column = target_column 
                self.df_original = df.copy(deep=True)
                logger.info(f"DataFrame and target column '{target_column}' initialized.")
            else:
                logger.error(f"Target column '{target_column}' not found in DataFrame.")
                raise ValueError(f"Target column '{target_column}' does not match any column in the DataFrame.")
        else:
            raise ValueError("DataFrame cannot be None when target_column is provided.")

        if (not isinstance(df, pd.DataFrame)) or df.empty:
            logger.error("Invalid DataFrame object or empty DataFrame.")
            raise ValueError("The provided object is either not a Pandas DataFrame or is empty.")
        self.df = df
        self.df_original = df.copy()
        self.db_manager = db_manager
        self.table_name = table_name
        # self.sequence = 1  # For logging step sequence

        if self.db_manager and self.table_name:
            logger.info(f"DataFormatter initialized for table: {self.table_name}")    

    def change_column_headers(self, column_mapping, inplace=False):
        """
        Changes the column headers of the DataFrame.
    
        Parameters:
        - column_mapping (dict): A dictionary where keys are the original column names,
                                 and values are the new column names.
        Example: {'old_name1': 'new_name1', 'old_name2': 'new_name2'}
        - inplace (bool): Set True to save changes to self.df. If False, returns a new DataFrame.
                          Default is False.
        
        Returns:
        - If inplace=False, returns a DataFrame with the updated column names.
        - If inplace is set to True, the method updates the existing DataFrame 
          and does not return anything.
        """
        # Check if all original columns exist in the DataFrame
        for original_col in column_mapping.keys():
            if original_col not in self.df.columns:
                logger.error(f"Column '{original_col}' does not exist in the DataFrame.")
                raise ValueError(f"Column '{original_col}' does not exist in the DataFrame")
        
        # Rename the columns based on the provided dictionary
        if inplace:
            self.df.rename(columns=column_mapping, inplace=True)
            logger.info(f"Columns renamed successfully: {column_mapping}")
        else:
            new_df = self.df.rename(columns=column_mapping)
            logger.warning('Warning: inplace=False, changes will not be saved')
            logger.info(f"Columns renamed successfully: {column_mapping}")
            return new_df
        
    
    def duplicate_column(self, original_col, new_col=None, inplace=False):
        """
        Creates a copy of a column in the DataFrame.
    
        Parameters:
        - original_col (str): The name of the column to be copied.
        - new_col (str): The name of the new column that will store the copy (default=None).
        - inplace (bool): Set True to modify the DataFrame in place. 
                          If False, return a new DataFrame with the copied column (default=false).
        
        Returns:
        - If inplace=False, returns a new DataFrame with the copied column.
        
        Example:
        copy_column('old_column', 'new_column_copy', inplace=True)
        """
        # Check if the original column exists in the DataFrame
        if original_col not in self.df.columns:
            logger.error(f"Column '{original_col}' does not exist in the DataFrame.")
            raise ValueError(f"Column '{original_col}' does not exist in the DataFrame")
        
        # Set default column name if not provided
        if new_col is None:
            new_col = f'{original_col}_copy'
        
        # Check if the new column name already exists
        if new_col in self.df.columns:
            logger.error(f"Column '{new_col}' already exists in the DataFrame.")
            raise ValueError(f"Column '{new_col}' already exists in the DataFrame")
    
        # Copy the column
        if inplace:
            self.df[new_col] = self.df[original_col].copy(deep=True)
            logger.info(f"Column '{original_col}' copied to '{new_col}' successfully.")
        else:
            new_df = self.df.copy()
            new_df[new_col] = self.df[original_col].copy(deep=True)
            logger.info(f"Column '{original_col}' copied to '{new_col}' successfully (returned new DataFrame).")
            return new_df

    
    def delete_column(self, columns_to_delete_list, inplace=False):
        """
        Delete specified columns from the DataFrame.
    
        Parameters:
        - columns_to_delete_list (list): List of column names to be deleted.
                             eg. ['column1', 'column2', ...]
        - inplace (bool): Set True to save changes (default=False.)
    
        Returns:
        Updated DataFrame if inplace=False, None if inplace=True.
        """
        if not isinstance(columns_to_delete_list, list):
            logger.error("Input is not a list.")
            raise TypeError("Input must be a list with list of column names.")
            
        # Check if columns_to_delete_list is not an empty list
        if not columns_to_delete_list:
            logger.error("Empty list of columns provided for deletion.")
            raise ValueError("No columns provided for deletion. Please provide at least one column name.")
        
        try:
            # Identify columns that exist in the DataFrame
            existing_columns = [col for col in columns_to_delete_list if col in self.df.columns and col != self.target_column]
            
            if existing_columns:
                # Drop the columns from the DataFrame
                df = self.df.drop(columns=existing_columns)
                logger.info(f"Columns {existing_columns} deleted successfully.")
            else:
                # Identify which columns do not exist in the DataFrame
                not_existing_columns = [col for col in columns_to_delete_list if col not in self.df.columns]
                logger.error(f"None of the specified columns {not_existing_columns} found in DataFrame.")
                raise ValueError(f"None of the specified columns {not_existing_columns} were found in the DataFrame.")
    
            # Apply the deletion
            if inplace:
                self.df = df
            else:
                logger.warning('Warning: inplace=False, changes will not be saved')
                return df
    
        except Exception as e:
            logger.exception("An error occurred while deleting columns.")
            raise RuntimeError(f"An error occurred while deleting columns: {str(e)}")

    
    def convert_data_type(self, columns_data_types_dict, inplace=False):
        """ 
        Explicitly convert multiple columns to specified data types.
        
        Parameters:
        - columns_data_types_dict (dict): A dictionary where keys are column names and values are target data types.
                    Example: {'column_name1': 'data_type', 'column_name2': 'data_type', ...}
                    Supported types are ('int', 'float', 'datetime', 'category', 'string', 'boolean').
        - inplace (bool): Set True to save changes (default=False.)
        
        Returns:
        Updated DataFrame if inplace=False, None if inplace=True.
        """
        
        # Check if the input is a dictionary
        if not isinstance(columns_data_types_dict, dict):
            logger.error("Provided data types mapping is not a dictionary.")
            raise TypeError("Input must be a dictionary where keys are column names and values are target data types.")
        
        # Check if the dictionary is not empty
        if not columns_data_types_dict:
            logger.error("Empty dictionary provided for data type conversion.")
            raise ValueError("Input dictionary cannot be empty. Please provide column names and corresponding data types.")
        
        # Iterate over each column and its target data type
        for column_name, data_type in columns_data_types_dict.items():
            data_type = data_type.lower() 
        
            # Check if the column exists in the DataFrame
            if column_name not in self.df.columns:
                logger.warning(f"Column '{column_name}' not found in DataFrame. Skipping.")
                raise ValueError(f"Warning: Column '{column_name}' not found in DataFrame. Skipping.")
                #print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping.")
                #continue
    
            try:
                # Detect if the current data type is already the target type
                if self.df[column_name].dtype == data_type:
                    logger.warning(f"Column '{column_name}' is already of type {data_type}. Skipping conversion.")
                    raise ValueError(f"Column '{column_name}' is already of type {data_type}. Skipping conversion.")
                    #print(f"Column '{column_name}' is already of type {data_type}. Skipping conversion.")
                    #continue
    
                # Handle boolean conversion
                if data_type == 'boolean':
                    unique_values = self.df[column_name].nunique()
                    if unique_values != 2:
                        logger.warning(f"Column '{column_name}' does not have exactly two unique values. Skipping boolean conversion.")
                        raise ValueError(f"Column '{column_name}' is already of type {data_type}. Skipping conversion.")
                        #print(f"Warning: Column '{column_name}' does not have exactly two unique values, skipping boolean conversion.")
                        #continue
                    converted_column = self.df[column_name].map({True: True, False: False, 'True': True, 'False': False, '1': True, '0': False}).astype('boolean')
    
                # Handle datetime conversion
                elif data_type == 'datetime':
                    converted_column = pd.to_datetime(self.df[column_name], errors='coerce')
    
                # Handle categorical and string conversion
                elif data_type in ['category', 'string']:
                    if pd.api.types.is_datetime64_any_dtype(self.df[column_name]):
                        converted_column = self.df[column_name].dt.strftime('%Y-%m-%d %H:%M:%S')  
                    else:
                        converted_column = self.df[column_name].astype(data_type)
    
                # Handle numerical conversion
                elif data_type in ['int', 'float']:
                    converted_column = pd.to_numeric(self.df[column_name], errors='coerce')
    
                else:
                    logger.warning(f"Unsupported data type '{data_type}' for column '{column_name}'. Skipping.")
                    raise ValueError(f"Warning: Unsupported data type '{data_type}' for column '{column_name}'. Skipping.")
                    #print(f"Warning: Unsupported data type '{data_type}' for column '{column_name}'. Skipping.")
                    #continue
    
                # Check for NaN values introduced by conversion
                # if converted_column.isnull().any():
                #     print(f"Warning: Conversion may have introduced NaN values in column '{column_name}'.")
    
                # Apply the conversion
                if inplace:
                    self.df[column_name] = converted_column
                    logger.info(f"Column '{column_name}' successfully converted to {data_type}.")
                    # print(f"Column '{column_name}' successfully converted to {data_type}.")
                else:
                    logger.info(f"Column '{column_name}' converted to {data_type}, but not saved (inplace=False).")
                    # pass
                    # print(f"Column '{column_name}' successfully converted to {data_type}. Warning: inplace=False, changes will not be saved.")
    
            except Exception as e:
                logger.exception(f"Error converting column '{column_name}' to {data_type}.")
                #print(f"Error converting column '{column_name}' to {data_type}: {e}")
                raise ValueError(f"Error converting column '{column_name}' to {data_type}: {e}")
        
        return self.df if not inplace else None
            

    def generate_datetime_columns(self, datetime_columns_dict, inplace=False, keep_original=True):
        """
        Generate specified datetime components (year, month, day, etc.) from multiple datetime columns.
    
        Parameters:
        - datetime_columns_dict (dict): Dictionary where keys are datetime column names, and values are lists of components to create.
          eg. {'column_name1'=['month', 'day'], 'column_name2'=['week', 'day_of_week', 'quarter'], ...}
          Supported components: 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'day_of_week', 'quarter', 'day_of_year', 'is_leap_year'.
        - inplace (bool): Set True to save changes (default=False).
        - keep_original (bool): If False, removes the original datetime columns after generating the components (default=True).
    
        Returns:
        Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting generate_datetime_columns method.")

        # Check if the input is a dictionary
        if not isinstance(datetime_columns_dict, dict):
            logger.error("Input is not a dictionary.")
            raise TypeError("Input must be a dictionary where keys are column names and values are lists of components to create.")
        
        # Check if the dictionary is not empty
        if not datetime_columns_dict:
            logger.error("Input dictionary is empty.")
            raise ValueError("Input dictionary cannot be empty. Please provide column names and corresponding list of components to create.")
        
        valid_components = {
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'week', 'day_of_week', 'quarter', 'day_of_year', 'is_leap_year'
        }
    
        # This will store columns that are being modified
        modified_columns = pd.DataFrame()
    
        for column_name, components in datetime_columns_dict.items():
            if column_name not in self.df.columns:
                logger.error(f"Column '{column_name}' not found.")
                raise ValueError(f"Warning: Column '{column_name}' not found in DataFrame.")
                #print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping.")
                #continue
    
            # Check if provided components are valid
            invalid_components = set(components) - valid_components
            if invalid_components:
                logger.error(f"Invalid components {invalid_components} for column '{column_name}'")
                raise ValueError(f"Warning: Invalid components {invalid_components} for column '{column_name}'")
                # print(f"Warning: Invalid components {invalid_components} for column '{column_name}'. Skipping these.")
                # components = set(components) - invalid_components
    
            try:
                # Convert column to datetime if it is not already
                if not pd.api.types.is_datetime64_any_dtype(self.df[column_name]):
                    logger.info(f"Converting column '{column_name}' to datetime.")
                    self.df[column_name] = pd.to_datetime(self.df[column_name], errors='coerce')
    
                # Detect if the datetime column has time information
                has_time_info = not self.df[column_name].dt.time.eq(pd.Timestamp('00:00:00').time).all()
    
                # Generate specified datetime components
                for component in components:
                    logger.debug(f"Generating component '{component}' for column '{column_name}'.")
                    if component == 'year':
                        modified_columns[f'{column_name}_Year'] = self.df[column_name].dt.year
                    elif component == 'month':
                        modified_columns[f'{column_name}_Month'] = self.df[column_name].dt.month
                    elif component == 'day':
                        modified_columns[f'{column_name}_Day'] = self.df[column_name].dt.day
                    elif component == 'hour' and has_time_info:
                        modified_columns[f'{column_name}_Hour'] = self.df[column_name].dt.hour
                    elif component == 'minute' and has_time_info:
                        modified_columns[f'{column_name}_Minute'] = self.df[column_name].dt.minute
                    elif component == 'second' and has_time_info:
                        modified_columns[f'{column_name}_Second'] = self.df[column_name].dt.second
                    elif component == 'week':
                        modified_columns[f'{column_name}_Week'] = self.df[column_name].dt.isocalendar().week
                    elif component == 'day_of_week':
                        modified_columns[f'{column_name}_DayOfWeek'] = self.df[column_name].dt.dayofweek
                    elif component == 'quarter':
                        modified_columns[f'{column_name}_Quarter'] = self.df[column_name].dt.quarter
                    elif component == 'day_of_year':
                        modified_columns[f'{column_name}_DayOfYear'] = self.df[column_name].dt.dayofyear
                    elif component == 'is_leap_year':
                        modified_columns[f'{column_name}_IsLeapYear'] = self.df[column_name].dt.is_leap_year
                    else:
                        pass
                        #print(f"Skipping time components ('{component}') for '{column_name}' as it has no time information.")
    
                #print(f"Generated {components} from '{column_name}'.")
    
            except Exception as e:
                logger.exception(f"Error generating datetime components from '{column_name}': {e}")
                raise ValueError(f"Error generating datetime components from '{column_name}': {e}")
                #print(f"Error generating datetime components from '{column_name}': {e}")
    
        # Inplace 
        if inplace:
            logger.info("Applying changes inplace.")
            self.df = pd.concat([self.df, modified_columns], axis=1)
            
            # Remove original datetime columns if keep_original=False
            if not keep_original:
                self.df.drop(list(datetime_columns_dict.keys()), axis=1, inplace=True)
                logger.info(f"Removed original columns: {list(datetime_columns_dict.keys())}")
                #print(f"Removed original columns: {list(datetime_columns_dict.keys())} after generating new components.")
    
            #print("Changes applied to the DataFrame.")
        else:
            # Create a new DataFrame if inplace=False
            result_df = pd.concat([self.df, modified_columns], axis=1)
    
            # Remove original datetime columns if keep_original=False
            if not keep_original:
                result_df.drop(list(datetime_columns_dict.keys()), axis=1, inplace=True)
                logger.info(f"Removed original columns: {list(datetime_columns_dict.keys())}")
            logger.info("Returning modified DataFrame.")
                #print(f"Removed original columns: {list(datetime_columns_dict.keys())} after generating new components in the new DataFrame.")
    
            #print("Displaying the changes without saving them:")
            return result_df


    def combine_date_and_time(self, date_col, time_col, new_col_name=None, inplace=False, keep_original=True):
        """
        Combine separate date and time columns into one datetime column.
    
        Parameters:
        - date_col (str): Column name for the date.
        - time_col (str): Column name for the time or another datetime column.
        - new_col_name (str): Name for the new combined column.
        - inplace (bool): If True, modifies self.df in place, otherwise returns a new DataFrame.
        - keep_original (bool): If False, removes the original date_col and time_col after combining (default=True).
    
        Returns:
        - Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting combine_date_and_time method.")
        
        if date_col not in self.df.columns:
            logger.error(f"Date column '{date_col}' not found.")
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
        
        if time_col not in self.df.columns:
            logger.error(f"Time column '{time_col}' not found.")
            raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

        # Set default new column name if not provided
        if new_col_name is None:
            new_col_name = f'{date_col}_{time_col}'
    
        # Convert the date column to datetime if it's not already in correct format
        try:
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                logger.info(f"Converting date column '{date_col}' to datetime.")
                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        except Exception as e:
            logger.exception(f"Error converting '{date_col}' to datetime: {e}")
            raise ValueError(f"Error converting '{date_col}' to datetime: {e}")
    
        # for case where the time_col is another datetime or just a time column
        try:
            if pd.api.types.is_datetime64_any_dtype(self.df[time_col]):
                # extract the time part and combine it with the date column
                time_str = self.df[time_col].dt.strftime('%H:%M:%S')
            else:
                # If time_col is not a datetime column, treat it as a time string
                time_str = self.df[time_col].astype(str)
    
            # Combine the date and time into a new datetime column
            combined_datetime = pd.to_datetime(self.df[date_col].astype(str) + ' ' + time_str, errors='coerce')
    
            if inplace:
                self.df[new_col_name] = combined_datetime
                logger.info(f"Created new datetime column '{new_col_name}' inplace.")
                #print(f"Combined '{date_col}' and '{time_col}' into '{new_col_name}' in the original DataFrame.")
    
                if not keep_original:
                    self.df.drop([date_col, time_col], axis=1, inplace=True)
                    logger.info(f"Dropped original columns: '{date_col}', '{time_col}'")
                    #print(f"Removed original columns '{date_col}' and '{time_col}' after combining.")
    
                return None
            else:
                result_df = self.df.copy()
                result_df[new_col_name] = combined_datetime
    
                if not keep_original:
                    result_df.drop([date_col, time_col], axis=1, inplace=True)
                    logger.info(f"Dropped original columns: '{date_col}', '{time_col}' from copied DataFrame")
                    
                #print("Displaying the changes without saving them:")
                return result_df
    
        except Exception as e:
            logger.exception(f"Error combining '{date_col}' and '{time_col}': {e}")
            raise ValueError(f"Error combining '{date_col}' and '{time_col}': {e}")


    def combine_columns(self, col1, col2, operation, new_col_name=None, keep_original=True, inplace=False):
        """
        Combine two columns using specified operations (concatenation for strings, arithmetic for numbers).
        
        Parameters:
        - col1 (str): First column name.
        - col2 (str): Second column name.
        - operation (str): Operation to perform ('add', 'subtract', 'multiply', 'divide', 'concat').
        - new_col_name (str): Name for the new combined column. If not provided, a default name will be generated.
        - keep_original (bool): If False, remove the original columns after combining (default=True).
        - inplace (bool): If True, modifies self.df in place, otherwise returns a new DataFrame (default=False).

        Note: Based o operation this method can be applied on,
            for 'concat' supports 'string', 'category' and 'object' data type columns.
            for others supports 'int' and 'float' data type columns.
        
        Returns:
        - Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting combine_columns method.")
    
        if col1 not in self.df.columns:
            logger.error(f"Column '{col1}' not found.")
            raise ValueError(f"Column '{col1}' not found in the DataFrame.")
        if col2 not in self.df.columns:
            logger.error(f"Column '{col2}' not found.")
            raise ValueError(f"Column '{col2}' not found in the DataFrame.")
    
        # Set default new column name if not provided
        if new_col_name is None:
            new_col_name = f'{col1}_{operation}_{col2}'
    
        col1_dtype = self.df[col1].dtype
        col2_dtype = self.df[col2].dtype
    
        # Handling operations based on data types
        try:
            if operation == 'concat':
                # Concatenate only if both columns are string, object, or categorical
                if pd.api.types.is_string_dtype(col1_dtype) or pd.api.types.is_object_dtype(col1_dtype) or pd.api.types.is_categorical_dtype(col1_dtype):
                    if pd.api.types.is_string_dtype(col2_dtype) or pd.api.types.is_object_dtype(col2_dtype) or pd.api.types.is_categorical_dtype(col2_dtype):
                        result = self.df[col1].astype(str) +" "+ self.df[col2].astype(str)
                    else:
                        logger.error(f"Cannot concatenate non-string column '{col2}' with string column '{col1}'.")
                        raise ValueError(f"Cannot concatenate non-string column '{col2}' with string column '{col1}'.")
                else:
                    logger.error(f"Cannot concatenate non-string column '{col1}' with column '{col2}'.")
                    raise ValueError(f"Cannot concatenate non-string column '{col1}' with column '{col2}'.")
            
            elif operation == 'add':
                # Addition for numeric columns
                if pd.api.types.is_numeric_dtype(col1_dtype) and pd.api.types.is_numeric_dtype(col2_dtype):
                    result = self.df[col1] + self.df[col2]
                else:
                    logger.error("Addition operation is only allowed for numeric columns.")
                    raise ValueError("Addition operation is only allowed for numeric columns.")
            
            elif operation == 'subtract':
                # Subtraction for numeric columns
                if pd.api.types.is_numeric_dtype(col1_dtype) and pd.api.types.is_numeric_dtype(col2_dtype):
                    result = self.df[col1] - self.df[col2]
                else:
                    logger.error("Subtraction operation is only allowed for numeric columns.")
                    raise ValueError("Subtraction operation is only allowed for numeric columns.")
            
            elif operation == 'multiply':
                # Multiplication for numeric columns
                if pd.api.types.is_numeric_dtype(col1_dtype) and pd.api.types.is_numeric_dtype(col2_dtype):
                    result = self.df[col1] * self.df[col2]
                else:
                    logger.error("Multiplication operation is only allowed for numeric columns.")
                    raise ValueError("Multiplication operation is only allowed for numeric columns.")
            
            elif operation == 'divide':
                # Division for numeric columns
                if pd.api.types.is_numeric_dtype(col1_dtype) and pd.api.types.is_numeric_dtype(col2_dtype):
                    result = self.df[col1] / self.df[col2]
                else:
                    logger.error("Division operation is only allowed for numeric columns.")
                    raise ValueError("Division operation is only allowed for numeric columns.")
            
            else:
                logger.error(f"Invalid operation '{operation}'.")
                raise ValueError(f"Invalid operation '{operation}'. Supported operations: 'add', 'subtract', 'multiply', 'divide', 'concat'.")
    
            if inplace:
                self.df[new_col_name] = result
    
                if not keep_original:
                    self.df.drop([col1, col2], axis=1, inplace=True)
                    logger.info(f"Dropped original columns: '{col1}', '{col2}'")
                logger.info(f"Created new column '{new_col_name}' inplace.")
                    #print(f"Original columns '{col1}' and '{col2}' were removed after combining.")
                
                #print(f"Combined '{col1}' and '{col2}' using '{operation}' into '{new_col_name}' in the original DataFrame.")
                return None
            else:
                result_df = self.df.copy()
                result_df[new_col_name] = result
    
                if not keep_original:
                    result_df.drop([col1, col2], axis=1, inplace=True)
                    logger.info(f"Dropped original columns: '{col1}', '{col2}' from copied DataFrame")
                logger.info(f"Returning new DataFrame with column '{new_col_name}'.")
                    #print(f"Original columns '{col1}' and '{col2}' were removed after combining in the new DataFrame.")
                
                #print("Displaying the changes without saving them:")
                #print(f"Displaying combined result '{new_col_name}' in the new DataFrame.")
                return result_df
        
        except Exception as e:
            logger.exception(f"Error combining columns '{col1}' and '{col2}': {e}")
            raise ValueError(f"Error combining columns '{col1}' and '{col2}': {e}")


    def split_column(self, col_name, delimiter=None, new_col_names=None, split_type='delimiter', keep_original=True, inplace=False):
        """
        Split a column into two new columns based on a delimiter (for strings) or by splitting numerical values.
    
        Parameters:
        - col_name (str): The name of the column to split.
        - delimiter (str): The delimiter to use for splitting (for string/object columns). Default is None.
                          eg. '@', '-', '.',...
        - new_col_names (list): List of two names for the new columns. If not provided, default names will be generated.
        - split_type (str): Type of split to perform. Options are:
            - 'delimiter': Split string/object columns based on a delimiter.
            - 'half': Split numeric columns into two halves.
        - keep_original (bool): If False, remove the original column after splitting (default=True).
        - inplace (bool): If True, modifies self.df in place. If False, returns a new DataFrame (default=False).
    
        Note: Based on split type this method can be applied on,
              for 'delimiter' supports 'string', 'object' and category data type columns.
              for 'half' supports numeric data type columns.
              
        Returns:
        - Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting split_column method.")
        
        if col_name not in self.df.columns:
            logger.error(f"Column '{col_name}' not found.")
            raise ValueError(f"Column '{col_name}' not found in the DataFrame.")
        
        # Set default new column names
        if new_col_names is None:
            new_col_names = [f'{col_name}_Part1', f'{col_name}_Part2']
        elif len(new_col_names) != 2:
            logger.error("You must provide exactly two new column names.")
            raise ValueError("You must provide exactly two new column names.")
        elif new_col_names is not None:
            for name in new_col_names:
                if name in self.df.columns:
                    logger.error(f"{name} header already exists.")
                    raise ValueError(f"{name} header already exist in the data, please provide different header")
        
        col_dtype = self.df[col_name].dtype
    
        try:
            if split_type == 'delimiter':
                # Ensure the column is a string, object, or categorical type
                if pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype):
                    if delimiter is None:
                        logger.error("Delimiter must be provided for string split.")
                        raise ValueError("Delimiter must be provided for string split.")
                    
                    # Split column based on the delimiter
                    split_data = self.df[col_name].str.split(delimiter, n=1, expand=True)
                    split_data.columns = new_col_names
                    result = split_data
                    #print('split_data', split_data)
                    
                    # Check if the split was successful
                    if split_data.shape[1] != 2:
                        logger.error(f"Column '{col_name}' could not be split correctly.")
                        raise ValueError(f"Column '{col_name}' could not be split into exactly two parts using delimiter '{delimiter}'.")
                    
                    # # Assign split data to new columns
                    # result = pd.DataFrame(split_data, columns=new_col_names)
                    # print('result', result)
                
                else:
                    logger.error(f"Column '{col_name}' must be a string, object, or categorical type for delimiter-based split.")
                    raise ValueError(f"Column '{col_name}' must be a string, object, or categorical type for delimiter-based split.")
            
            elif split_type == 'half':
                # Ensure the column is numeric
                if pd.api.types.is_numeric_dtype(col_dtype):
                    # Split numeric column into two equal parts
                    half_index = len(self.df[col_name]) // 2
                    part1 = self.df[col_name].iloc[:half_index]
                    part2 = self.df[col_name].iloc[half_index:]
                    
                    # Create two new columns
                    result = pd.DataFrame({new_col_names[0]: part1, new_col_names[1]: part2})
                
                else:
                    logger.error("Split by 'half' is only allowed for numeric columns.")
                    raise ValueError("Split by 'half' is only allowed for numeric columns.")
            
            else:
                logger.error(f"Invalid split_type '{split_type}'.")
                raise ValueError(f"Invalid split_type '{split_type}'. Supported types: 'delimiter', 'half'.")
            
            if inplace:
                self.df[new_col_names[0]] = result[new_col_names[0]]
                self.df[new_col_names[1]] = result[new_col_names[1]]
                
                if not keep_original:
                    self.df.drop([col_name], axis=1, inplace=True)
                    logger.info(f"Dropped original column: '{col_name}'")
                    #print(f"Original column '{col_name}' was removed after splitting.")
                
                #print(f"Split '{col_name}' into '{new_col_names[0]}' and '{new_col_names[1]}' in the original DataFrame.")
                return None
            
            else:
                # Return a new DataFrame with the split columns
                result_df = self.df.copy()
                result_df[new_col_names[0]] = result[new_col_names[0]]
                result_df[new_col_names[1]] = result[new_col_names[1]]
                
                # Remove original column if keep_original=False
                if not keep_original:
                    result_df.drop([col_name], axis=1, inplace=True)
                    logger.info(f"Dropped original column: '{col_name}' from copied DataFrame")
                    #print(f"Original column '{col_name}' was removed after splitting in the new DataFrame.")

                #print("Displaying the changes without saving them:")
                #print(f"Displaying split result '{new_col_names[0]}' and '{new_col_names[1]}' in the new DataFrame.")
                return result_df
    
        except Exception as e:
            logger.exception(f"Error splitting column '{col_name}': {e}")
            raise ValueError(f"Error splitting column '{col_name}': {e}")
            
            
    def remove_special_characters(self, column_dict, inplace=False):
        """
        Remove special characters from multiple columns based on a dictionary input.
        
        Parameters:
        - column_dict (dict): A dictionary where:
            - key (str): The column name.
            - value (dict): Contains:
                - 'characters_to_remove' (list or 'all'): List of custom characters to remove, or 'all' to remove all special characters.
                - 'replace_with' (str): String to replace removed characters with (default: '').
        - inplace (bool): If True, modifies self.df in place. If False, returns a new DataFrame (default=False).
        
        Returns:
        - Updated DataFrame if inplace=False, None if inplace=True.
        
        Notes:
        - This method only supports columns of 'string', 'object', and 'category' data types.
        """
        logger.info("Starting remove_special_characters method.")

        if not isinstance(column_dict, dict):
            logger.error("column_dict must be a dictionary.")
            raise TypeError("column_dict must be a dictionary with column names as keys and values as dictionaries of 'characters_to_remove' and 'replace_with'.")
    
        # Copy the DataFrame if not inplace
        if not inplace:
            result_df = self.df.copy()
        else:
            result_df = self.df
    
        for column_name, options in column_dict.items():
            # Check if column exists
            if column_name not in result_df.columns:
                logger.error(f"Column '{column_name}' not found.")
                raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
            # Ensure the column is of string, object, or categorical type
            col_dtype = result_df[column_name].dtype
            if not (pd.api.types.is_string_dtype(col_dtype) or 
                    pd.api.types.is_object_dtype(col_dtype) or 
                    pd.api.types.is_categorical_dtype(col_dtype)):
                logger.error(f"Column '{column_name}' is not string-compatible.")
                raise ValueError(f"Column '{column_name}' must be a string, object, or categorical type for special character removal.")
            
            # Get options for characters to remove and replacement string
            characters_to_remove = options.get('characters_to_remove', 'all')
            replace_with = options.get('replace_with', '')
    
            # Handle special case where 'all' special characters should be removed
            if characters_to_remove == 'all':
                chars_to_remove = r'[^\w\s]'  # Regex for all special characters
            elif isinstance(characters_to_remove, list):
                # Create a regex pattern for the custom characters
                custom_chars_pattern = ''.join(map(re.escape, characters_to_remove))
                chars_to_remove = f'[{custom_chars_pattern}]'
            else:
                logger.error(f"Invalid characters_to_remove for column '{column_name}'.")
                raise ValueError(f"Invalid value for 'characters_to_remove' in column '{column_name}'. Expected 'all' or a list of characters.")
    
            try:
                # Perform the replacement
                result_df[column_name] = result_df[column_name].str.replace(chars_to_remove, replace_with, regex=True)
                #print(f"Removed special characters from '{column_name}'.")
                logger.info(f"Removed special characters from '{column_name}'.")
    
            except Exception as e:
                logger.exception(f"Error removing special characters from '{column_name}': {e}")
                raise ValueError(f"Error removing special characters from '{column_name}': {e}")
    
        # Return the modified DataFrame if inplace is False
        if not inplace:
            #print("Displaying the changes without saving them:")
            return result_df
        else:
            #print("Special characters removed and changes are saved.")
            return None


    def add_suffix_or_prefix(self, text_to_add_dict, inplace=False):
        """
        Add a suffix or prefix to the specified columns in the DataFrame with customizable separators.
        
        Parameters:
        - inplace (bool): If True, modifies the original DataFrame in place. If False, returns a modified copy (Default = False).
        - text_to_add_dict (dict): Dictionary where keys are column names, and values are tuples in the format (text, position, separator).
                                eg. {'column_name1'=('add_this_text', 'suffix', '-'), 'column_name2'=('text_to_add', 'preffix', '|'),...}
            - text: The string to add as a suffix or prefix.
            - position: Either 'suffix' or 'prefix', determining where to add the text.
            - separator: The separator between the original text and the suffix/prefix.

        Note: This method can be used only on 'string', 'object' and 'category' data type columns.
        
        Returns:
        - Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting add_suffix_or_prefix method.")
        
        result_df = self.df if inplace else self.df.copy()
    
        for column_name, (text, position, separator) in text_to_add_dict.items():
            if column_name not in self.df.columns:
                logger.error(f"Column '{column_name}' not found.")
                raise ValueError(f"Column '{column_name}' not found in the DataFrame. Skipping.")
                # print(f"Column '{column_name}' not found in the DataFrame. Skipping.")
                # continue
    
            # Check if the text is a valid input
            if not isinstance(text, str):
                logger.error(f"Text to add is not a string for column '{column_name}'.")
                raise ValueError(f"Invalid text '{text}' for column '{column_name}'. Must be a string. Skipping.")
                # print(f"Invalid text '{text}' for column '{column_name}'. Must be a string. Skipping.")
                # continue
    
            # Ensure the column can be converted to string
            col_dtype = self.df[column_name].dtype
            if not (pd.api.types.is_string_dtype(col_dtype) or 
                    pd.api.types.is_object_dtype(col_dtype) or 
                    pd.api.types.is_categorical_dtype(col_dtype)):
                logger.error(f"Invalid column type for '{column_name}'.")
                raise ValueError(f"Column '{column_name}' must be a string, object, or categorical type. Skipping.")
                # print(f"Column '{column_name}' must be a string, object, or categorical type. Skipping.")
                # continue
    
            # Validate that separator is a string
            if not isinstance(separator, str):
                logger.error(f"Separator is not a string for column '{column_name}'.")
                raise ValueError(f"Invalid separator '{separator}' for column '{column_name}'. Must be a string. Skipping.")
                # print(f"Invalid separator '{separator}' for column '{column_name}'. Must be a string. Skipping.")
                # continue
            
            try:
                # Apply suffix or prefix based on the position argument
                if position == 'suffix':
                    result_df[column_name] = result_df[column_name].astype(str) + separator + text
                    logger.info(f"Added suffix '{text}' to column '{column_name}'.")
                    #print(f"Added '{text}' as a suffix to '{column_name}' with separator '{separator}'.")
                elif position == 'prefix':
                    result_df[column_name] = text + separator + result_df[column_name].astype(str)
                    logger.info(f"Added prefix '{text}' to column '{column_name}'.")
                    #print(f"Added '{text}' as a prefix to '{column_name}' with separator '{separator}'.")
                else:
                    logger.warning(f"Invalid position '{position}' for column '{column_name}'. Skipping.")
                    # pass
            
            except Exception as e:
                logger.exception(f"Error adding {position} to '{column_name}': {e}")
                raise ValueError(f"Error adding {position} to '{column_name}': {e}")
        
        if not inplace:
            return result_df
        else:
            return None


    def format_string_columns(self, column_name_operation_dict, inplace=False):
        """
        Perform basic string manipulation operations on multiple specified columns of a DataFrame.
        
        Parameters:
        - inplace (bool): If True, modifies the original DataFrame in place. If False, returns a modified copy (Default = False).
        - column_name_operation_dict (dict): Dictinary where keys are column names, and values are string operations to perform.
                                            eg. {'column_name1'='strip', 'column_name2'='lower', ...}
            Supported operations for each column are:
                - 'strip': Remove leading and trailing whitespace.
                - 'stripall': Remove all whitespace.
                - 'capitalize': Capitalize the first letter of each string.
                - 'lower': Convert all characters to lowercase.
                - 'upper': Convert all characters to uppercase.
        
        Returns:
        - Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting format_string_columns method.")
        # Create a copy of the DataFrame
        result_df = self.df if inplace else self.df.copy()
    
        for column_name, operation in column_name_operation_dict.items():
        
            if column_name not in self.df.columns:
                logger.warning(f"Column '{column_name}' not found in the DataFrame. Skipping.")
                continue
    
            if not (pd.api.types.is_string_dtype(self.df[column_name]) or 
                    pd.api.types.is_object_dtype(self.df[column_name]) or 
                    pd.api.types.is_categorical_dtype(self.df[column_name])):
                logger.warning(f"Column '{column_name}' is not of string type. Skipping.")
                continue
            
            try:
                modified_column = self.df[column_name].copy()
    
                # Perform the specified operation
                if operation == 'strip':
                    modified_column = modified_column.str.strip()
                if operation == 'stripall':
                    modified_column = modified_column.fillna('').str.replace(r'\s+', '', regex=True)
                elif operation == 'capitalize':
                    modified_column = modified_column.str.capitalize()
                elif operation == 'lower':
                    modified_column = modified_column.str.lower()
                elif operation == 'upper':
                    modified_column = modified_column.str.upper()
                else:
                    logger.warning(f"Invalid operation '{operation}' for column '{column_name}'. Skipping.")
                    continue
    
                # Update the DataFrame
                result_df[column_name] = modified_column
                logger.info(f"Applied '{operation}' operation on column '{column_name}'.")
    
            except Exception as e:
                logger.exception(f"Error formatting string column '{column_name}': {e}")
                raise ValueError(f"Error formatting string column '{column_name}': {e}")
    
        if not inplace:
            return result_df
        else:
            return None
            

    def pad_zeros(self, column_names, decimal_places=2, round_values=False, inplace=False):
        """
            Pad zeros in a numeric column to a specified number of decimal places.
            
            Parameters:
            - column_name (list): A list of column names to pad zeros.
                               eg. column_names = ['column_name1', 'column_name2']
            - decimal_places (int): Number of decimal places to pad to (Default = 2).
            - round_values (bool): If True, the values will be rounded to the specified number of decimal places before padding (Default = False.).
            - inplace (bool): If True, modifies the original DataFrame in place. If False, returns the modified column (Default = False).

            Note: Can be applied only on numerical columns.
            
            Returns:
            - Updated DataFrame if inplace=False, None if inplace=True.
        """
        logger.info("Starting pad_zeros function with column_names=%s, decimal_places=%d, round_values=%s, inplace=%s",
                column_names, decimal_places, round_values, inplace)

        if not isinstance(column_names, list):
            logger.error("'column_names' must be a list of column names. Got %s instead.", type(column_names))
            raise ValueError(f"'column_names' must be a list of column names. Got {type(column_names)} instead.")
        
        # Check if decimal_places is a non-negative integer
        if not isinstance(decimal_places, int) or decimal_places < 0:
            logger.error("'decimal_places' must be a non-negative integer. Got %s instead.", decimal_places)
            raise ValueError(f"'decimal_places' must be a non-negative integer. Got {decimal_places} instead.")
        
        # Create a copy of the DataFrame if inplace is False
        result_df = self.df if inplace else self.df.copy()
        logger.debug("Created DataFrame copy for inplace=%s", inplace)
    
        for column_name in column_names:
            if column_name not in self.df.columns:
                logger.warning("Column '%s' not found in the DataFrame. Skipping.", column_name)
                raise ValueError(f"Column '{column_name}' not found in the DataFrame. Skipping.")
                
                # print(f"Column '{column_name}' not found in the DataFrame. Skipping.")
                # continue
            
            # Check if the column is numeric
            if not pd.api.types.is_numeric_dtype(self.df[column_name]):
                logger.error("Column '%s' must be numeric for zero padding.", column_name)
                raise ValueError(f"Column '{column_name}' must be numeric for zero padding.")
                # print(f"Column '{column_name}' must be numeric for zero padding. Skipping.")
                # continue
            
            try:
                # Copy the column for modification
                modified_column = self.df[column_name].copy()
                logger.debug("Processing column '%s'", column_name)
    
                # Round values if round_values is True
                if round_values:
                    modified_column = modified_column.round(decimal_places)
                    logger.debug("Rounded values in column '%s' to %d decimal places", column_name, decimal_places)
    
                # Apply padding to the specified number of decimal places
                modified_column = modified_column.apply(lambda x: f"{x:.{decimal_places}f}")
                logger.debug("Applied zero padding to column '%s' with %d decimal places", column_name, decimal_places)
    
                # Update the DataFrame
                result_df[column_name] = modified_column.astype(float)
                logger.info("Successfully padded zeros in '%s' to %d decimal places.", column_name, decimal_places)
                #print(f"Padded zeros in '{column_name}' to {decimal_places} decimal places.")
            
            except Exception as e:
                logger.error("Error padding zeros in numeric column '%s': %s", column_name, str(e))
                raise ValueError(f"Error padding zeros in numeric column '{column_name}': {e}")
        
        if not inplace:
            return result_df
        else:
            logger.info("Modified DataFrame in place, returning None")
            return None


    @staticmethod
    def convert_timeseries_pred_to_dataframe(hierarchical_columns, predicted_column, data):
        """
        Convert output from time series to DataFrame (project-specific).
        
        Parameters:
        - hierarchical_columns (list): The list of column names to be set for the multi-index.
        - predicted_column (list): List of predicted column names.
        - data (dict): Output data that needs to be converted to a DataFrame, with hierarchical keys and values.
        
        Returns:
        - A hierarchical DataFrame or raises an error for invalid inputs.
        """
        logger.info("Starting convert_timeseries_pred_to_dataframe with hierarchical_columns=%s, predicted_column=%s",
                hierarchical_columns, predicted_column)
        
        # Validate input types
        if not isinstance(hierarchical_columns, list):
            logger.error("Expected 'hierarchical_columns' to be a list, but got %s", type(hierarchical_columns).__name__)
            raise TypeError(f"Expected 'hierarchical_columns' to be a list, but got {type(hierarchical_columns).__name__}")
        if not isinstance(predicted_column, list):
            logger.error("Expected 'predicted_column' to be a list, but got %s", type(predicted_column).__name__)
            raise TypeError(f"Expected 'predicted_column' to be a list, but got {type(predicted_column).__name__}")
        if not isinstance(data, dict):
            logger.error("Expected 'data' to be a dictionary, but got %s", type(data).__name__)
            raise TypeError(f"Expected 'data' to be a dictionary, but got {type(data).__name__}")
    
        if not data:
            logger.error("Input 'data' is empty")
            raise ValueError("Input 'data' is empty.")
    
        # Local variables
        index_tuples = []
        values = []
    
        # Ensure hierarchical_columns length matches the actual hierarchy of keys in data
        for key, series in data.items():
            if not isinstance(key, tuple):
                logger.error("Expected keys in 'data' to be tuples, but found %s", type(key).__name__)
                raise ValueError(f"Expected keys in 'data' to be tuples, but found {type(key).__name__}")
            if len(key) != len(hierarchical_columns):
                logger.error("Length of 'hierarchical_columns' (%d) does not match the number of elements in key '%s' (%d)",
                         len(hierarchical_columns), key, len(key))
                raise ValueError(f"Length of 'hierarchical_columns' ({len(hierarchical_columns)}) does not match the number of elements in key '{key}' ({len(key)})")
    
            # Ensure series is iterable
            if not hasattr(series, '__iter__'):
                logger.error("Expected value corresponding to key %s to be iterable, but got %s", key, type(series).__name__)
                raise TypeError(f"Expected value corresponding to key {key} to be iterable, but got {type(series).__name__}")
    
            for value in series:
                index_tuples.append(key)
                values.append(value)
    
        # Check that the length of predicted_column matches the number of values (or handle single-column case)
        if len(predicted_column) != 1:
            logger.error("Expected 'predicted_column' to have exactly 1 column name, but got %d", len(predicted_column))
            raise ValueError(f"Expected 'predicted_column' to have exactly 1 column name, but got {len(predicted_column)}")
    
        # Create the DataFrame
        logger.debug("Creating DataFrame with MultiIndex")
        df_out = pd.DataFrame(values, 
                          index=pd.MultiIndex.from_tuples(index_tuples, names=hierarchical_columns), 
                          columns=predicted_column)
        
        logger.info("Successfully created DataFrame with %d rows", len(df_out))
        return df_out


    @staticmethod
    def calculate_Time_series_metrics(y_true, y_pred):
        """
        Calculate time series metrics such as MAPE, SMAPE, WMAPE, and RMSE.
        
        Parameters:
        - y_true: Array of actual values.
        - y_pred: Array of predicted values.
        
        Returns:
        - A dictionary containing MAPE, SMAPE, WMAPE, and RMSE.
        """
        logger.info("Starting calculate_time_series_metrics with y_true length=%d, y_pred length=%d",
                len(y_true), len(y_pred))
        
        # Ensure inputs are numpy arrays 
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        logger.debug("Converted inputs to numpy arrays")

        # Remove zero values from y_true to avoid division by zero issues
        mask = y_true != 0
        y_true_nonzero = y_true[mask]
        y_pred_nonzero = y_pred[mask]
        logger.debug("Removed zero values from y_true; %d non-zero values remain", len(y_true_nonzero))

        # MAPE (Mean Absolute Percentage Error)
        def mape(y_true, y_pred):
            mape_value = np.mean(np.abs((y_true - y_pred) / y_true))
            # Cap the MAPE between 0 and 1
            # return np.clip(mape_value, 0, 1)
            mape_value = np.clip(mape_value, 0, 1)
            logger.debug("Calculated MAPE: %f", mape_value)
            return mape_value

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        def smape(y_true, y_pred):
            smape_value = np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
            logger.debug("Calculated SMAPE: %f", smape_value)
            return smape_value

            # return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

        # WMAPE (Weighted Mean Absolute Percentage Error)
        def wmape(y_true, y_pred):
            wmape_value = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
            logger.debug("Calculated WMAPE: %f", wmape_value)
            return wmape_value
            # return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

        # RMSE (Root Mean Squared Error)
        def rmse(y_true, y_pred):
            rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
            logger.debug("Calculated RMSE: %f", rmse_value)
            return rmse_value
            # return np.sqrt(mean_squared_error(y_true, y_pred))

        # Calculate metrics
        metrics = {
            'MAPE': mape(y_true_nonzero, y_pred_nonzero),
            'SMAPE': smape(y_true_nonzero, y_pred_nonzero),
            'WMAPE': wmape(y_true_nonzero, y_pred_nonzero),
            'RMSE': rmse(y_true, y_pred)
        }

        logger.info("Metrics calculated: %s", metrics)

        return metrics

    
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
        logger.info("Starting fill_missing_value with numeric_strategy=%s, categorical_strategy=%s, datetime_strategy=%s, threshold=%f, knn_neighbors=%d, inplace=%s",
                numeric_strategy, categorical_strategy, datetime_strategy, threshold, knn_neighbors, inplace)

        # Import Instance variable
        df = self.df
        target_column = self.target_column
        logger.debug("Imported instance variables: df with %d rows, target_column=%s", len(df), target_column)


         # Convert input data to DataFrame if it's a Series or NumPy array
        if isinstance(df, pd.Series):
            df = df.to_frame()  # Convert Series to DataFrame
            logger.debug("Converted Series to DataFrame")
        elif isinstance(df, np.ndarray):
            if df.ndim == 1:
                df = pd.DataFrame(df, columns=['Array_Column'])  # Handle 1D arrays
                logger.debug("Converted 1D NumPy array to DataFrame")
            else:
                df = pd.DataFrame(df)  # Handle 2D arrays
                logger.debug("Converted 2D NumPy array to DataFrame")
        elif isinstance(df, pd.DataFrame):
            logger.debug("Input is already a DataFrame")
            df = df
        else:
            logger.error("Input must be a Pandas DataFrame, Series, or NumPy array. Got %s", type(df).__name__)
            raise ValueError("Input must be a Pandas DataFrame, Series, or NumPy array.")
    
        try:
            #if not isinstance(df, pd.DataFrame) or df.empty:
                #raise ValueError("No data loaded into Fill Missing Value.")
            if df.empty:
                logger.error("No data loaded into Fill Missing Value")
                raise ValueError("No data loaded into Fill Missing Value.")
            
            # Drop columns with missing values exceeding the threshold
            threshold_count = len(df) * threshold                                  
            cols_to_drop = [col for col in df.columns if df[col].notnull().sum() < threshold_count and col != target_column]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                logger.info("Dropped columns due to missing value threshold: %s", cols_to_drop)
                #print(f'Columns dropped due to missing value threshold:\n{cols_to_drop}')
            
            # Log for filled columns and their strategies
            fill_log = []
            
            # If custom_suggestions are not provided, initialize an empty dictionary
            if custom_suggestions is None:
                custom_suggestions = {}
                logger.debug("Initialized empty custom_suggestions dictionary")
    
            # Function to apply strategy based on column type and custom suggestions
            def apply_strategy(col, col_type, default_strategy):
                strategy_to_apply = custom_suggestions.get(col, default_strategy)
                logger.debug("Applying strategy '%s' to column '%s' (type: %s)", strategy_to_apply, col, col_type)
                if strategy_to_apply == 'mean' and col_type == 'numeric':
                    mean_value = df[col].mean()
                    if pd.api.types.is_integer_dtype(df[col]):
                        mean_value = int(mean_value)
                    df[col].fillna(mean_value, inplace=True)
                    logger.debug("Filled column '%s' with mean value: %s", col, mean_value)
                    return 'mean'
                elif strategy_to_apply == 'median' and col_type == 'numeric':
                    median_value = df[col].median()
                    if pd.api.types.is_integer_dtype(df[col]):
                        median_value = int(median_value)  
                    df[col].fillna(median_value, inplace=True)
                    logger.debug("Filled column '%s' with median value: %s", col, median_value)
                    return 'median'
                elif strategy_to_apply == 'mode':
                    mode_value = df[col].mode().iloc[0]
                    df[col].fillna(mode_value, inplace=True)
                    logger.debug("Filled column '%s' with mode value: %s", col, mode_value)
                    return 'mode'
                elif strategy_to_apply == 'zero' and col_type == 'numeric':
                    df[col].fillna(0, inplace=True)
                    logger.debug("Filled column '%s' with zero", col)
                    return 'zero'
                elif strategy_to_apply in ['ffill', 'bfill']:
                    df[col].fillna(method=strategy_to_apply, inplace=True)
                    logger.debug("Filled column '%s' with %s method", col, strategy_to_apply)
                    return strategy_to_apply
                elif strategy_to_apply == 'remove':
                    df.dropna(subset=[col], inplace=True)
                    logger.debug("Removed rows with missing values in column '%s'", col)
                    return 'remove'
                else:
                    logger.error("Invalid strategy '%s' for column '%s'", strategy_to_apply, col)
                    raise ValueError(f"Invalid strategy '{strategy_to_apply}' for column {col}.")
    
            # Handle missing values based on remove_missing_values_in_custom_cols
            columns_to_process = custom_suggestions.keys() if remove_missing_values_in_custom_cols else df.columns
            logger.debug("Processing columns: %s", columns_to_process)

            # Handle missing values for numerical columns
            for col in df.select_dtypes(include=['number']).columns:
                if col not in columns_to_process or col == target_column:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values == 0:
                    continue  # Skip columns with no missing values
                applied_strategy = apply_strategy(col, 'numeric', numeric_strategy)
                fill_log.append(f'Missing values: {missing_values} in column {col} filled using {applied_strategy} strategy.')
                logger.info("Filled %d missing values in numeric column '%s' using %s strategy", missing_values, col, applied_strategy)

            # Handle missing values for categorical and boolean columns
            for col in df.select_dtypes(include=['boolean', 'category', 'object', 'string']).columns:
                if col not in columns_to_process or col == target_column:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values == 0:
                    continue  # Skip columns with no missing values
                applied_strategy = apply_strategy(col, 'categorical', categorical_strategy)
                fill_log.append(f'Missing values: {missing_values} in column {col} filled using {applied_strategy} strategy.')
                logger.info("Filled %d missing values in categorical column '%s' using %s strategy", missing_values, col, applied_strategy)

            # Handle missing values for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            for col in datetime_cols:
                if col not in columns_to_process or col == target_column:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values == 0:
                    logger.debug("No missing values in datetime column '%s'", col)
                    continue  # Skip columns with no missing values
                strategy_to_apply = custom_suggestions.get(col, datetime_strategy)
                logger.debug("Applying datetime strategy '%s' to column '%s'", strategy_to_apply, col)
                if strategy_to_apply == 'ffill':
                    df[col].fillna(method='ffill', inplace=True)
                    logger.debug("Filled datetime column '%s' with ffill", col)
                elif strategy_to_apply == 'bfill':
                    df[col].fillna(method='bfill', inplace=True)
                    logger.debug("Filled datetime column '%s' with bfill", col)
                elif strategy_to_apply == 'default' and default_date:
                    df[col].fillna(pd.to_datetime(default_date), inplace=True)
                    logger.debug("Filled datetime column '%s' with default date: %s", col, default_date)
                else:
                    logger.error("Invalid datetime_strategy '%s' for column '%s'", strategy_to_apply, col)
                    raise ValueError(f"Invalid datetime_strategy {strategy_to_apply} for column {col}.")
                fill_log.append(f'Missing values: {missing_values} in column {col} filled using {strategy_to_apply} strategy.')
                logger.info("Filled %d missing values in datetime column '%s' using %s strategy", missing_values, col, strategy_to_apply)
            
            # Print log and return results
            logger.info("Missing value imputation log:\n%s", "\n".join(fill_log))
            # print("\n--- Missing Value Imputation Log ---")
            # for col_log in fill_log:
            #     print(col_log)
            
            if inplace:
                self.df = df
                logger.info("Changes saved to self.df (inplace=True)")
                # print(f"Missing values filled with specified strategies, changes saved.")
                return None
            else:
                logger.info("Returning modified DataFrame (inplace=False)")
                # print(f"Missing values filled with specified strategies, but changes not saved (inplace=False).")
                return df
        
        except Exception as e:
            logger.error("Error occurred while filling missing values: %s", str(e))
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
        logger.info("Starting remove_duplicates method.")
        df = self.df

        # Convert input data to DataFrame if it's a Series or NumPy array
        if isinstance(df, np.ndarray):
            if df.ndim == 1:
                logger.error("Input array is one-dimensional.")
                raise ValueError("Input must be a Pandas DataFrame or two dimensional NumPy array.")
            else:
                df = pd.DataFrame(df)  # Handle 2D arrays
        elif isinstance(df, pd.DataFrame):
            df = df
        else:
            logger.error("Invalid input type.")
            raise ValueError("Input must be a Pandas DataFrame, Series, or NumPy array.")
        
        try:
            # Ensure the DataFrame is valid
            if df.empty:
                logger.warning("The DataFrame is empty.")
                raise ValueError("No valid DataFrame provided for removing duplicates.")
    
            initial_shape = df.shape
    
            # Check if there's a datetime column and adjust the strategy
            keep_strategy = 'last' if any(df.dtypes == 'datetime') else 'first'
    
            # Create a boolean mask for duplicates and drop them
            duplicates_mask = df.duplicated(keep=keep_strategy)
            num_duplicates = duplicates_mask.sum()
    
            # Inform the user if no duplicates are found
            if num_duplicates == 0:
                logger.info("No duplicates found.")
                return None
    
            # Drop duplicates
            df = df[~duplicates_mask].reset_index(drop=True)
            logger.info(f"Removed {num_duplicates} duplicate rows.")
    
            final_shape = df.shape
            logger.info(f"Removed {num_duplicates} duplicate rows.")
    
            # Handle in-place modification
            if inplace:
                self.df = df
                logger.info("Duplicate rows have been removed and changes saved (inplace=True).")
                return None
            else:
                logger.info("Duplicate rows removed but changes not saved (inplace=False).")
                return df
    
        except Exception as e:
            logger.exception(f"An error occurred while removing duplicates: {e}")
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
        logger.info("Starting detect_and_handle_outliers method.")
        from scipy import stats

        df = self.df
        target_column = self.target_column
        
        # Convert input data to DataFrame if it's a Series or NumPy array
        if isinstance(df, pd.Series):
            df = df.to_frame()  # Convert Series to DataFrame
        elif isinstance(df, np.ndarray):
            if df.ndim == 1:
                df = pd.DataFrame(df, columns=['Array_Column'])  # Handle 1D arrays
            else:
                df = pd.DataFrame(df)  # Handle 2D arrays
        elif isinstance(df, pd.DataFrame):
            df = df
        else:
            logger.error("Invalid input type for DataFrame.")
            raise ValueError("Input must be a Pandas DataFrame, Series, or NumPy array.")
    
        try:
            # Check if DataFrame is valid
            if df.empty:
                logger.warning("The DataFrame is empty.")
                raise ValueError("The DataFrame is empty or invalid.")
    
            # Exclude target column from outlier detection
            columns_to_check = [col for col in df.columns if col != target_column]
    
            # Handle numerical columns and ignore non-numeric, datetime, and boolean columns
            numerical_cols = df[columns_to_check].select_dtypes(include=['number']).columns.tolist()
            datetime_cols = df[columns_to_check].select_dtypes(include=['datetime']).columns.tolist()
            boolean_cols = df[columns_to_check].select_dtypes(include=['bool']).columns.tolist()
    
            # Ensure numerical columns exist
            if not numerical_cols:
                logger.error("No numerical columns available for outlier detection.")
                raise ValueError("No numerical columns available for outlier detection.")
    
            # If only handling outliers in custom-specified columns
            if remove_outliers_in_custom_cols:
                if not custom_strategy:
                    logger.error("Custom strategy is required for selected columns.")
                    raise ValueError("Custom strategy must be provided if remove_outliers_in_custom_cols is set to True.")
                numerical_cols = [col for col in numerical_cols if col in custom_strategy]
    
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
    
                try:
                    # Detect outliers based on the specified method
                    if col_method == 'IQR':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    
                    elif col_method == 'z_score':
                        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                        outliers = z_scores > z_thresh
    
                    elif col_method == 'auto':
                        # Check normality for large or small samples
                        if len(df[col].dropna()) > 5000:
                            from scipy.stats import kstest
                            stat, p_value = kstest(df[col].dropna(), 'norm')
                        else:
                            from scipy.stats import shapiro
                            stat, p_value = shapiro(df[col].dropna())
    
                        normality = 'Normal' if p_value > 0.05 else 'Not Normal'
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
                        logger.error(f"Invalid outlier detection method '{col_method}' for column '{col}'.")
                        raise ValueError(f"Invalid outlier detection method '{col_method}' for column '{col}'.")
    
                    # Count outliers
                    num_outliers = outliers.sum()
    
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
                            df.loc[df[col] < lower_bound, col] = lower_bound
                            df.loc[df[col] > upper_bound, col] = upper_bound
                            print(f"column: {col}, lower bound: {lower_bound} and upper bound: {upper_bound}")
                        elif col_method == 'z_score':
                            df.loc[outliers, col] = df[col].median()
    
                    else:
                        logger.error(f"Invalid handling method '{col_replace_with}' for column '{col}'.")
                        raise ValueError(f"Invalid handling method '{col_replace_with}' for column '{col}'.")
    
                except Exception as e:
                    logger.exception(f"Error handling outliers for column '{col}': {e}")
                    # print(f"Error handling outliers for column '{col}': {e}")
    
            # print("\n--- Outliers Handling Log ---")
            for log_entry in outlier_log:
                logger.info(log_entry)
    
            if inplace:
                self.df = df
                logger.info("Outliers have been handled, and changes are saved (inplace=True).")
                return None
            else:
                logger.info("Outliers have been handled but changes are not saved (inplace=False).")
                return df
    
        except Exception as e:
            logger.exception(f"An unexpected error occurred while handling outliers: {e}")
