import requests
import json
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """
    Тест health check эндпоинта
    """
    print("\n" + "="*60)
    print("Тестирование /health")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(" API работает!")
        print(f"Статус: {data['status']}")
        print(f"Доступно моделей: {data['models_available']}")
        print(f"Модели: {data['models']}")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)


def test_get_models():
    """
    Тест получения списка моделей
    """
    print("\n" + "="*60)
    print("Тестирование /models")
    print("="*60)
    
    response = requests.get(f"{API_URL}/models")
    
    if response.status_code == 200:
        data = response.json()
        print(f" Всего моделей: {data['total_models']}")
        
        for model_name, metadata in data['models'].items():
            print(f"\nМодель: {model_name}")
            if 'test_accuracy' in metadata:
                print(f"   Точность: {metadata['test_accuracy']:.4f}")
                print(f"   F1 Score: {metadata['f1_macro']:.4f}")
    else:
        print(f"Ошибка: {response.status_code}")


def test_predict_json():
    """
    Тест предсказания через JSON
    """
    print("\n" + "="*60)
    print("Тестирование /predict (JSON)")
    print("="*60)
    
    test_data = {
        "model_name": "Decision_Tree",
        "data": [
            {
                "Country": "UK",
                "Age": 30,
                "Gender": "Male",
                "Income": "Medium",
                "Customer_Segment": "Regular",
                "Year": 2023,
                "Month": "January",
                "Total_Purchases": 5,
                "Amount": 150.0,
                "Total_Amount": 300.0,
                "Product_Category": "",
                "Product_Type": "Shoes",
                "Feedback": "Good",
                "Shipping_Method": "Express",
                "Payment_Method": "Credit Card",
                "Order_Status": "Shipped",
                "Ratings": 4,
                # "products": "Running shoes",
                "Quantity": 2,
                "Price_per_Unit": 50.0,
                "Customer_Satisfaction": 4
            },
            {
                "Country": "Germany",
                "Age": 25,
                "Gender": "Female",
                "Income": "Low",
                "Customer_Segment": "Premium",
                "Year": 2023,
                "Month": "February",
                "Total_Purchases": 3,
                "Amount": 100.0,
                "Total_Amount": 150.0,
                "Product_Category": "",
                "Product_Type": "Tablet",
                "Feedback": "Excellent",
                "Shipping_Method": "Standard",
                "Payment_Method": "Cash",
                "Order_Status": "Delivered",
                "Ratings": 5,
                # "products": "iPad",
                "Quantity": 1,
                "Price_per_Unit": 30.0,
                "Customer_Satisfaction": 5
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f" Предсказание успешно!")
        print(f"Модель: {result['model_used']}")
        print(f"Количество образцов: {result['num_samples']}")
        print(f"Предсказания: {result['predictions']}")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)


def test_predict_csv():
    """
    Тест предсказания через CSV файл
    """
    print("\n" + "="*60)
    print("Тестирование /predict/csv")
    print("="*60)
    
    csv_file = Path("test.csv")
    if not csv_file.exists():
        print("  Файл test.csv не найден, создаем тестовый файл...")
        
        test_df = pd.DataFrame([{
            "ID": "0x160a",
            "Customer_ID": "CUS_0xd40",
            "Month": "September",
            "Name": "Test User",
            "Age": 30,
            "SSN": "123-45-6789",
            "Occupation": "Engineer",
            "Annual_Income": 50000.0,
            "Monthly_Inhand_Salary": 3000.0,
            "Num_Bank_Accounts": 2,
            "Num_Credit_Card": 3,
            "Interest_Rate": 5,
            "Num_of_Loan": 1,
            "Type_of_Loan": "Credit-Builder Loan",
            "Delay_from_due_date": 2,
            "Num_of_Delayed_Payment": 5,
            "Changed_Credit_Limit": 10.5,
            "Num_Credit_Inquiries": 3,
            "Credit_Mix": "Good",
            "Outstanding_Debt": 1200.50,
            "Credit_Utilization_Ratio": 30.5,
            "Credit_History_Age": "10 Years and 5 Months",
            "Payment_of_Min_Amount": "Yes",
            "Total_EMI_per_month": 50.0,
            "Amount_invested_monthly": 200.0,
            "Payment_Behaviour": "High_spent_Medium_value_payments",
            "Monthly_Balance": 500.0
        }])
        
        test_df.to_csv(csv_file, index=False)
        print(" Тестовый файл создан")
    
    with open(csv_file, 'rb') as f:
        files = {'file': ('test.csv', f, 'text/csv')}
        params = {'model_name': 'Decision_Tree'}
        
        response = requests.post(
            f"{API_URL}/predict/csv",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f" Предсказание успешно!")
        print(f"Модель: {result['model_used']}")
        print(f"Количество образцов: {result['num_samples']}")
        
        output_file = Path("predictions.txt")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f"Модель: {result['model_used']}\n")
            f_out.write(f"Количество образцов: {result['num_samples']}\n\n")
            f_out.write("Предсказания:\n")
            for idx, row in enumerate(result.get('data_with_predictions', []), 1):
                f_out.write(f"Образец {idx}: {row.get('Predicted_Category', 'N/A')}\n")
        
        print(f" Предсказания сохранены в файл: {output_file}")
        
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)

def test_all_models():
    """
    Тест предсказаний для всех доступных моделей
    """
    print("\n" + "="*60)
    print("Тестирование всех моделей")
    print("="*60)
    
    # Получаем список моделей
    response = requests.get(f"{API_URL}/models")
    if response.status_code != 200:
        print("Не удалось получить список моделей")
        return
    
    models = list(response.json()['models'].keys())
    
    test_data_template = {
        "data": [
            {
                "Country": "UK",
                "Age": 30,
                "Gender": "Male", 
                "Income": "Medium",
                "Customer_Segment": "Regular",
                "Year": 2023,
                "Month": "January",
                "Total_Purchases": 5,
                "Amount": 150.0,
                "Total_Amount": 300.0,
                "Product_Category": "",
                "Product_Type": "Shoes",
                "Feedback": "Good",
                "Shipping_Method": "Express",
                "Payment_Method": "Credit Card",
                "Order_Status": "Shipped",
                "Ratings": 4,
                # "products": "Running shoes",
                "Quantity": 2,
                "Price_per_Unit": 50.0,
                "Customer_Satisfaction": 4
            }
        ]
    }
    
    for model_name in models:
        print(f"\nТестирование модели: {model_name}")
        
        test_data = test_data_template.copy()
        test_data['model_name'] = model_name
        
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    Предсказание: {result['predictions'][0]}")
        else:
            print(f"   Ошибка: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Детали ошибки: {error_detail}")
            except:
                print(f"   Текст ошибки: {response.text}")
                
def main():
    """
    Запуск всех тестов
    """
    print("="*60)
    print("ТЕСТИРОВАНИЕ ML PREDICTION API")
    print("="*60)
    print(f"API URL: {API_URL}")
    
    try:
        test_health()
        
        test_get_models()
        
        test_predict_json()
        
        test_predict_csv()
        
        test_all_models()
        
        print("\n" + "="*60)
        print(" ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\nОшибка подключения к API!")
        print("Убедитесь, что API сервер запущен: python api.py")
    except Exception as e:
        print(f"\nОшибка: {e}")


if __name__ == "__main__":
    main()
