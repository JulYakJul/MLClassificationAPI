from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import io

from pathlib import Path
from preprocessing import remove_columns, fill_missing_values

app = FastAPI(
    title="ML Prediction API",
    description="API для предсказания категории товара покупателя ",
    version="1.0.0"
)

MODELS_DIR = Path("models")
loaded_models = {}
label_encoder = None


class PredictionRequest(BaseModel):
    """
    Схема запроса для предсказания через JSON
    """
    data: List[Dict[str, Any]]
    model_name: Optional[str] = "Decision_Tree"

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "Decision_Tree",
                "data": [
                    {
                        "Gender": "Male",
                        "Age": 30,
                        "Product_Category": "",
                        "Quantity": 2,
                        "Price_per_Unit": 50.0,
                        "Total_Amount": 100.0,
                        "Payment_Method": "Credit Card",
                        "Customer_Satisfaction": 4
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Схема ответа с предсказаниями
    """
    predictions: List[str]
    model_used: str
    num_samples: int


def clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка DataFrame от проблемных значений перед сериализацией в JSON
    
    Args:
        df: исходный DataFrame
    Returns:
        df_clean: очищенный DataFrame
    """
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        df_clean[col] = df_clean[col].fillna(0)
        df_clean[col] = df_clean[col].apply(
            lambda x: 0 if pd.isna(x) else float(x) if abs(float(x)) < 1e10 else 0.0
        )
    
    return df_clean


def safe_json_serializable(obj):
    """
    Рекурсивно делает объект безопасным для JSON сериализации
    """
    if isinstance(obj, (list, tuple)):
        return [safe_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if pd.isna(obj) or obj in (np.inf, -np.inf):
            return 0.0
        if abs(obj) > 1e10:
            return 0.0
        if abs(obj) < 1e-10 and obj != 0:
            return 0.0
        return float(obj)
    elif isinstance(obj, str):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)


def load_model(model_name: str):
    """
    Загрузка модели из файла с обработкой проблем совместимости NumPy
    """
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_path = MODELS_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_name}' не найдена. Доступные модели: {list_available_models()}"
        )
    
    try:
        model = joblib.load(model_path)
        loaded_models[model_name] = model
        return model
    except Exception as e:
        print(f" Ошибка загрузки модели {model_name}: {e}")
        print("Пробуем альтернативный способ загрузки...")
        
        try:
            import pickle
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            if hasattr(model, 'random_state') and model_name == 'MLP':
                print("Исправляем random_state для MLP...")
                try:
                    model.random_state = np.random.RandomState(42)
                except:
                    pass
            
            loaded_models[model_name] = model
            print(f"Модель {model_name} успешно загружена альтернативным способом")
            return model
            
        except Exception as pickle_error:
            print(f"Ошибка загрузки через pickle: {pickle_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка загрузки модели '{model_name}': {str(pickle_error)}"
            )


def load_label_encoder():
    """
    Загрузка label encoder для декодирования предсказаний
    """
    global label_encoder
    if label_encoder is None:
        encoder_path = MODELS_DIR / "label_encoder.joblib"
        if not encoder_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Label encoder не найден. Обучите модели с помощью train_models.py"
            )
        label_encoder = joblib.load(encoder_path)
    return label_encoder


def list_available_models() -> List[str]:
    """
    Получение списка доступных моделей
    """
    if not MODELS_DIR.exists():
        return []
    
    models = [f.stem for f in MODELS_DIR.glob("*.joblib") if f.stem != "label_encoder"]
    return models


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка входных данных для предсказания
    """
    df_processed = remove_columns(df.copy())
    
    if 'Product_Category' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Product_Category'])
    
    df_processed = fill_missing_values(df_processed)
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if df_processed[col].isna().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    for col in categorical_cols:
        df_processed[col] = df_processed[col].astype(str)
        df_processed[col] = df_processed[col].fillna('unknown')
    
    df_processed = clean_dataframe_for_json(df_processed)
    
    print(f"Числовые колонки после обработки: {numeric_cols}")
    print(f"Категориальные колонки после обработки: {categorical_cols}")
    print(f"Типы данных после обработки: {df_processed.dtypes.to_dict()}")
    
    return df_processed

@app.on_event("startup")
async def startup_event():
    """
    Инициализация при запуске API
    """
    print("Запуск ML Prediction API...")
    print(f"Директория моделей: {MODELS_DIR.absolute()}")
    
    try:
        load_label_encoder()
        print("Label encoder загружен")
    except Exception as e:
        print(f"Ошибка загрузки label encoder: {e}")
    
    models = list_available_models()
    print(f"Доступные модели: {models}")


@app.get("/")
async def root():
    """
    Корневой эндпоинт с информацией об API
    """
    return {
        "message": "ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - предсказание через JSON",
            "/predict/csv": "POST - предсказание через CSV файл",
            "/models": "GET - список доступных моделей",
            "/health": "GET - проверка состояния сервиса"
        }
    }


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса
    """
    models = list_available_models()
    return {
        "status": "healthy",
        "models_available": len(models),
        "models": models
    }


@app.get("/models")
async def get_models():
    """
    Получение списка доступных моделей и их метаданных
    """
    models = list_available_models()
    
    models_info = {}
    for model_name in models:
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                models_info[model_name] = json.load(f)
        else:
            models_info[model_name] = {"name": model_name}
    
    return {
        "total_models": len(models),
        "models": safe_json_serializable(models_info)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_json(request: PredictionRequest):
    """
    Предсказание через JSON запрос
    """
    try:
        model = load_model(request.model_name)
        encoder = load_label_encoder()
        
        df = pd.DataFrame(request.data)
        print(f"Получены данные с колонками: {df.columns.tolist()}")
        print(f"Форма данных: {df.shape}")
        print(f"Типы данных: {df.dtypes.to_dict()}")
        
        df_processed = preprocess_input(df)
        print(f"Данные после предобработки: {df_processed.columns.tolist()}")
        print(f"Форма после предобработки: {df_processed.shape}")
        print(f"Типы данных после обработки: {df_processed.dtypes.to_dict()}")
        
        if df_processed.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Нет данных для предсказания после предобработки")
        
        if "CatBoost" in request.model_name:
            print("Обработка для CatBoost модели...")
            
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            print(f"Категориальные признаки для CatBoost: {categorical_cols}")
            
            for col in df_processed.columns:
                if col in categorical_cols:
                    df_processed[col] = df_processed[col].astype(str)
                else:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = df_processed[col].fillna(0)
        
        predictions_encoded = model.predict(df_processed)
        print(f"Получены предсказания: {predictions_encoded}")
        
        predictions = encoder.inverse_transform(predictions_encoded)
        print(f"Декодированные предсказания: {predictions}")
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_used=request.model_name,
            num_samples=len(predictions)
        )
    
    except Exception as e:
        print(f"Ошибка предсказания: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")
    

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), model_name: str = "Decision_Tree"):
    """
    Предсказание через CSV файл
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        model = load_model(model_name)
        encoder = load_label_encoder()

        df_processed = preprocess_input(df)

        predictions_encoded = model.predict(df_processed)
        predictions = encoder.inverse_transform(predictions_encoded)

        result_df = df.copy()
        result_df['Predicted_Category'] = predictions

        result_df_clean = clean_dataframe_for_json(result_df)
        
        data_with_predictions = safe_json_serializable(
            result_df_clean.to_dict(orient='records')
        )

        return JSONResponse(content={
            "predictions": safe_json_serializable(predictions.tolist()),
            "model_used": model_name,
            "num_samples": len(predictions),
            "data_with_predictions": data_with_predictions
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Запуск ML Prediction API сервера")
    print("="*60)
    print("Документация будет доступна по адресу: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)