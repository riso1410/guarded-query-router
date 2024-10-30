import argparse
import pandas as pd

def load_and_process_csv(file_path, question_col, answer_col):
    """
    Load the CSV file and select specific columns for question and answer.
    """
    try:
        df = pd.read_csv(file_path)
        return df[[question_col, answer_col]]
    except KeyError:
        raise KeyError(f"Columns '{question_col}' or '{answer_col}' not found in {file_path}.")
    except Exception as e:
        raise Exception(f"An error occurred while processing {file_path}: {e}")

def remove_outliers(df, answer_col, method="iqr", threshold=3.0):
    """
    Remove outliers from the DataFrame in the answer column.
    
    For numeric data, use z-score or IQR to detect outliers.
    For text data, calculate outliers based on the length of each entry.
    
    Parameters:
    - method: "z_score" for z-score method or "iqr" for IQR method.
    - threshold: z-score or IQR threshold for defining outliers.
    """
    # Determine if the column is numeric or text-based
    if pd.api.types.is_numeric_dtype(df[answer_col]):
        # Numeric outlier detection
        if method == "z_score":
            mean = df[answer_col].mean()
            std_dev = df[answer_col].std()
            df = df[(df[answer_col] - mean).abs() <= threshold * std_dev]
        elif method == "iqr":
            Q1 = df[answer_col].quantile(0.25)
            Q3 = df[answer_col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[answer_col] >= Q1 - threshold * IQR) & (df[answer_col] <= Q3 + threshold * IQR)]
        else:
            raise ValueError("Method should be either 'z_score' or 'iqr'.")
    
    elif pd.api.types.is_string_dtype(df[answer_col]):
        # Text-based outlier detection based on length of each entry
        df['text_length'] = df[answer_col].apply(len)
        
        if method == "z_score":
            mean = df['text_length'].mean()
            std_dev = df['text_length'].std()
            df = df[(df['text_length'] - mean).abs() <= threshold * std_dev]
        elif method == "iqr":
            Q1 = df['text_length'].quantile(0.25)
            Q3 = df['text_length'].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df['text_length'] >= Q1 - threshold * IQR) & (df['text_length'] <= Q3 + threshold * IQR)]
        else:
            raise ValueError("Method should be either 'z_score' or 'iqr'.")
        
        # Drop the temporary 'text_length' column
        df = df.drop(columns='text_length')
    
    else:
        print(f"Skipping outlier removal for unsupported data type in column '{answer_col}'.")
    
    return df

def main(open_dataset_path, specific_dataset_path, open_question_col, open_answer_col, specific_question_col, specific_answer_col):
    """
    Load and process two CSV files, one open-domain and one specific-domain, based on provided question and answer columns.
    Perform outlier removal on the answer columns of both files.
    """
    # Process open-domain CSV
    open_df = load_and_process_csv(open_dataset_path, open_question_col, open_answer_col)
    open_df = remove_outliers(open_df, open_answer_col)
    print(f"Processed Open-Domain Data:\n{open_df.head()}\n")

    # Process specific-domain CSV
    specific_df = load_and_process_csv(specific_dataset_path, specific_question_col, specific_answer_col)
    specific_df = remove_outliers(specific_df, specific_answer_col)
    print(f"Processed Specific-Domain Data:\n{specific_df.head()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two datasets with specified question and answer columns.")
    parser.add_argument("open_dataset_path", type=str, help="Path to the open-domain dataset CSV file.")
    parser.add_argument("specific_dataset_path", type=str, help="Path to the specific-domain dataset CSV file.")
    parser.add_argument("open_question_col", type=str, help="Question column name in the open-domain dataset.")
    parser.add_argument("open_answer_col", type=str, help="Answer column name in the open-domain dataset.")
    parser.add_argument("specific_question_col", type=str, help="Question column name in the specific-domain dataset.")
    parser.add_argument("specific_answer_col", type=str, help="Answer column name in the specific-domain dataset.")
    parser.add_argument("--outlier_method", type=str, default="iqr", choices=["z_score", "iqr"],
                        help="Method for outlier detection: 'z_score' or 'iqr'.")
    parser.add_argument("--outlier_threshold", type=float, default=3.0,
                        help="Threshold for outlier detection. For 'z_score', this is the number of standard deviations. For 'iqr', this is the IQR multiplier.")

    args = parser.parse_args()

    main(
        args.open_dataset_path,
        args.specific_dataset_path,
        args.open_question_col,
        args.open_answer_col,
        args.specific_question_col,
        args.specific_answer_col
    )
