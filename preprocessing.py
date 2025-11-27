import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def remove_columns(df):
    """
    Удаляет ненужные колонки из датафрейма
    
    Args:
        df: исходный датафрейм
    Returns:
        df: очищенный датафрейм
    """
    drop_cols = ["Transaction_ID", "Customer_ID", "Name", "Email", "Phone",
                 "Address", "City", "Zipcode", "Date", "Time", "Product_Brand", "State", "products"]
    return df.drop(columns=drop_cols, errors='ignore')


def fill_missing_values(df):
    """
    Заполняет пропущенные значения в датафрейме
    
    Args:
        df: датафрейм с пропусками
    Returns:
        df: датафрейм без пропусков
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace("_", "", regex=False).str.replace(",", "", regex=False)
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def get_feature_groups(df, target_col):
    """
    Разделяет признаки на числовые и категориальные
    
    Args:
        df: датафрейм
        target_col: имя целевой переменной
    Returns:
        numeric_cols: список числовых признаков
        categorical_cols: список категориальных признаков
    """
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols


def make_preprocess_pipeline(numeric_cols, categorical_cols):
    """
    Пайплайн препроцессинга для числовых и категориальных признаков
    
    Args:
        numeric_cols: список числовых колонок 
        categorical_cols: список категориальных колонок
    Returns:
        preprocessor: объект ColumnTransformer
    """
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', categorical_pipe, categorical_cols)
    ])
    
    return preprocessor


def preprocess_dataframe(df, target_col=None):
    """
    Полная предобработка датафрейма для обучения или предсказания
    
    Args:
        df: исходный датафрейм
        target_col: имя целевой переменной (None для предсказания)
    Returns:
        df_processed: обработанный датафрейм
        target: целевая переменная (если target_col задан)
    """
    
    df_processed = remove_columns(df.copy())
    df_processed = fill_missing_values(df_processed)
    
    if target_col and target_col in df_processed.columns:
        target = df_processed[target_col]
        df_processed = df_processed.drop(columns=[target_col])
        return df_processed, target
    
    return df_processed, None

def enhanced_feature_engineering(df, target_col):
    """Расширенный фич-инжиниринг"""
    
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    for col in numeric_cols[:3]:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col != target_col:
            encoding = df.groupby(col)[target_col].mean()
            df[f'{col}_target_enc'] = df[col].map(encoding)
    
    return df

def remove_correlated_features(df, threshold=0.95):
    """Удаление сильно коррелированных признаков"""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)