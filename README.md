# Развёртывание моделей машинного обучения с использованием API. Решение задачи классификации на основе больших статистических данных

## Описание проекта

Полный ML-пайплайн для обучения, оценки и развёртывания моделей машинного обучения для классификации категорий товаров покупателей. Работа сделана для 1 курса магистратуры УлГТУ спецаильности “Искусственный интеллект в автоматизации проектирования”, предмет "Машинное обучение". [Оригинал задания К.В. Святова](https://github.com/ulstu/ml/blob/master/ml_course_ru/assignments/ivt_masters_2025.md)

-  Обучение нескольких ML моделей (Decision Tree, KNN, Logistic Regression, MLP, CatBoost)
-  Сохранение моделей с помощью joblib
-  Визуализация кривых обучения для анализа переобучения/недообучения
-  REST API на базе FastAPI для предсказаний
-  Поддержка JSON и CSV форматов данных
-  Контейнеризация с помощью Docker
-  Автоматизированное тестирование API

# Описание датасета 
Данные о транзакциях клиентов, включая демографическую информацию, историю покупок и отзывы о продуктах. Планируется создать интеллектуальную систему для автоматической категоризации товаров на основе характеристик транзакций, чтобы улучшить управление ассортиментом и персонализировать маркетинговые кампании.

## Задача датасета
На основе информации о транзакциях и клиентах построить модель машинного обучения, которая сможет предсказывать категорию товара.

## Бизнес-проблема 
Таргетирование рекламы - неэффективная персонализация маркетинговых кампаний в розничной сети. Решение данной проблемы лежит в создании интеллектуальной системы предсказания продуктовых категорий, представляющих интерес для каждого конкретного клиента. Такая система должна обеспечивать точную сегментацию клиентской базы на основе реальных продуктовых предпочтений, выявленных через анализ исторических данных о покупках.

## Зависимости в данных
Были проанализированы и выявлены признаки влияющие и не влияющие на результат предсказания Product_Category

Влияют: Country, Age, Gender, Income, Customer_Segment, Year, Month, Total_Purchases, Amount, Total_Amount, Product_Type, Feedback, Payment_Method, Order_Status, Ratings.

Не влияют: “Transaction_ID", "Customer_ID", "Name", "Email", "Phone", "Address", "City", "Shipping_Method", "Zipcode", "Date", "Time", "State". (поля были удалены из тестовой и обучающей выборках)

# Реализованные модели

## Деревья решений (регулируемые параметры - глубина и функция оценки изменения информации)

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_decision_tree.png?raw=true" width="650"/>
</div>

Дерево решений - это алгоритм, который моделирует процесс принятия решений в виде древовидной структуры для решения задач классификации и регрессии. Оно разбивает данные на основе правил (предикатов) в узлах, проводя анализ от корня до листа, где находится конечный результат.

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/TDT_explain.png?raw=true" width="650"/>
</div>

## KNN - k-ближайших соседей (регулируемый параметр - количество ближайших соседей)

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_knn.png?raw=true" width="650"/>
</div>

Чтобы классифицировать новую точку данных (точку запроса), алгоритм вычисляет расстояние между точкой запроса и всеми другими точками в наборе данных с помощью функции расстояния. Поиск соседей: алгоритм определяет ближайшие точки данных k (соседи) к точке запроса на основе вычисляемых расстояний.

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_knn_explain.png?raw=true" width="650"/>
</div>

## Логистическая регрессия (регулируемые параметры - степень полиномиальной функции и коэффициент регуляризации)

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_logistic_regression.png?raw=true" width="650"/>
</div>

Логистическая регрессия - это алгоритм машинного обучения, используемый для задач классификации, где предсказывается вероятность принадлежности объекта к одному из двух (или более) классов. Принцип работы заключается в применении сигмоидной функции (или "S-образной" кривой) к линейной комбинации входных признаков. В отличие от линейной регрессии, которая предсказывает числа, логистическая регрессия предсказывает вероятность принадлежности к классу (от 0 до 1). Эта функция преобразует результат в значение от 0 до 1, что интерпретируется как вероятность. Для обучения модель находит оптимальные веса признаков, минимизируя функцию потерь (например, логарифмическую функцию потерь) с помощью методов оптимизации, таких как градиентный спуск.

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_logistic_regression_explain.png?raw=true" width="650"/>
</div>

## MLP - Многослойный персептрон (регулируемые параметры - метод оптимизации, количество скрытых слоев и нейронов в скрытом слое).

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_mlp.png?raw=true" width="650"/>
</div>

Многослойный персептрон имеет входной слой и выходной слой с одним или несколькими скрытыми слоями. В MLP все нейроны одного слоя связаны со всеми нейронами следующего слоя. Здесь входной уровень принимает входные сигналы, а желаемая задача выполняется выходным слоем. А скрытые слои отвечают за все расчеты.

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_mlp_explain.png?raw=true" width="650"/>
</div>

## Градиентный бустинг

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_catboost.png?raw=true" width="650"/>
</div>

Алгоритм CatBoost основан на градиентных деревьях решений, и при обучении этой модели последовательно строится набор деревьев решений. По мере обучения каждое последующее дерево строится с меньшими потерями по сравнению с предыдущим деревом.

<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/train_catboost_explain.png?raw=true" width="650"/>
</div>

## Пример построения графика - результат предсказания модели "Дерево решений"
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/DTplot.png?raw=true" width="650"/>
</div>

## Swagger
Реализованные ендпоинты:
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/swagger.png?raw=true" width="650"/>
</div>

Запрос в формате JSON для предсказания Product_Category:
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/json.png?raw=true" width="650"/>
</div>

Успешный ответ с предсказанием:
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/json_response.png?raw=true" width="650"/>
</div>

Запрос в формате CSV файла для предсказания Product_Category:
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/csv.png?raw=true" width="650"/>
</div>

Успешный ответ с предсказаниями:
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/csv_response.png?raw=true" width="650"/>
</div>

## Docker
<div align="center">
  <img src="https://github.com/JulYakJul/MLClassificationAPI/blob/main/pictures/docker.png?raw=true" width="650"/>
</div>

# Проект
## Структура проекта

```
MLLabs/
├── preprocessing.py          # Модуль препроцессинга данных
├── train_models.py           # Скрипт обучения моделей с кривыми обучения
├── api.py                    # FastAPI сервис для предсказаний
├── buyer_definition.py       # Оригинальный скрипт обучения
├── test_api.py               # Python скрипт тестирования API
├── test_api.ps1              # PowerShell скрипт тестирования API
├── requirements.txt          # Зависимости Python
├── Dockerfile                # Конфигурация Docker
├── docker-compose.yml        # Docker Compose конфигурация
├── .dockerignore             # Исключения для Docker
├── models/                   # Сохранённые модели (создаётся автоматически)
│   ├── *.joblib             # Файлы моделей
│   ├── *_metadata.json      # Метаданные моделей
│   ├── label_encoder.joblib # Кодировщик меток классов
│   └── training_summary.json # Сводка результатов обучения
├── plots/                    # Графики кривых обучения (создаётся автоматически)
│   └── *_learning_curve.png # Кривые обучения для каждой модели
└── new_retail_data.csv       # Данные для обучения
```

## Быстрый старт

### Вариант 1: Локальная установка

#### 1. Установка зависимостей

```powershell
# Создание виртуального окружения
python -m venv venv
.\venv\Scripts\Activate.ps1

# Установка зависимостей
pip install -r requirements.txt
```

#### 2. Обучение моделей

```powershell
python train_models.py
```

**Результаты:**
```
models/
├── Decision_Tree.joblib
├── Decision_Tree_metadata.json
├── KNN.joblib
├── KNN_metadata.json
├── Logistic_Regression.joblib
├── Logistic_Regression_metadata.json
├── MLP.joblib
├── MLP_metadata.json
├── label_encoder.joblib
└── training_summary.json

plots/
├── Decision_Tree_learning_curve.png
├── KNN_learning_curve.png
├── Logistic_Regression_learning_curve.png
└── MLP_learning_curve.png
```

#### 3. Запуск API сервера

```powershell
python api.py
```

API будет доступен по адресу: `http://localhost:8000`

Интерактивная документация (Swagger UI): `http://localhost:8000/docs`

#### 4. Тестирование API

**Python скрипт:**
```powershell
python test_api.py
```

**PowerShell скрипт:**
```powershell
.\test_api.ps1
```

### Вариант 2: Docker развёртывание

#### 1. Обучение моделей (локально)

Сначала обучите модели локально, чтобы создать директорию `models/`:

```powershell
python train_models.py
```

#### 2. Сборка Docker образа

```powershell
docker build -t ml-prediction-api .
```

#### 3. Запуск контейнера

**Простой запуск:**
```powershell
docker run -p 8000:8000 -v ${PWD}/models:/app/models ml-prediction-api
```

**Использование Docker Compose:**
```powershell
docker-compose up -d
```

#### 4. Остановка контейнера

```powershell
docker-compose down
```

#### 5. Просмотр логов

```powershell
docker-compose logs -f ml-api
```

## API Endpoints

### 1. Проверка состояния
```bash
GET /health
```

**Ответ:**
```json
{
  "status": "healthy",
  "models_available": 4,
  "models": ["Decision_Tree", "KNN", "Logistic_Regression", "MLP"]
}
```

### 2. Список моделей
```bash
GET /models
```

**Ответ:**
```json
{
  "total_models": 4,
  "models": {
    "Decision_Tree": {
      "model_name": "Decision_Tree",
      "test_accuracy": 0.8542,
      "f1_macro": 0.8421,
      "best_params": {...}
    },
    ...
  }
}
```

### 3. Предсказание через JSON
```bash
POST /predict
Content-Type: application/json

{
  "model_name": "Decision_Tree",
  "data": [
    {
      "Gender": "Male",
      "Age": 30,
      "Quantity": 2,
      "Price_per_Unit": 50.0,
      "Total_Amount": 100.0,
      "Payment_Method": "Credit Card",
      "Customer_Satisfaction": 4
    }
  ]
}
```

**Ответ:**
```json
{
  "predictions": ["Electronics"],
  "model_used": "Decision_Tree",
  "num_samples": 1
}
```

### 4. Предсказание через CSV файл
```bash
POST /predict/csv
Content-Type: multipart/form-data
```

**Ответ:**
```json
{
  "predictions": ["Electronics", "Clothing"],
  "model_used": "Decision_Tree",
  "num_samples": 2,
  "data_with_predictions": [...]
}
```

### 5. Пакетное предсказание (CSV)
```bash
POST /predict/batch
Content-Type: multipart/form-data
```

Возвращает CSV файл с добавленной колонкой `Predicted_Category`.

## Оптимальные параметры моделей

Скрипт `train_models.py` автоматически подбирает оптимальные параметры для каждой модели:

### Decision Tree
- `max_depth`: глубина дерева [3, 5, 7, 10]
- `criterion`: критерий разделения ['gini', 'entropy']
- `min_samples_split`: минимальное количество образцов для разделения [2, 5, 10]

### KNN
- `n_components` (SVD): количество компонент [50, 100]
- `n_neighbors`: количество соседей [5, 10, 15]
- `metric`: метрика расстояния ['euclidean', 'manhattan']

### Logistic Regression
- `poly__degree`: степень полиномиальных признаков [1, 2]
- `C`: коэффициент регуляризации [0.1, 1.0, 10.0]

### MLP (Multi-Layer Perceptron)
- `hidden_layer_sizes`: структура слоёв [(64,), (128,), (128, 64)]
- `alpha`: коэффициент L2 регуляризации [0.001, 0.01]
- `learning_rate`: тип изменения скорости обучения ['adaptive']

## Дополнительные ресурсы

- **FastAPI документация**: https://fastapi.tiangolo.com/
- **scikit-learn документация**: https://scikit-learn.org/
- **Docker документация**: https://docs.docker.com/
