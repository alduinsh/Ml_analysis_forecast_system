import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

def get_user_input():
    user_input = st.text_input("Введите 'all' для анализа всех продуктов или перечислите ID продуктов через запятую: ")
    if user_input.lower() == 'all':
        return None
    else:
        try:
            product_ids = list(map(int, user_input.split(',')))
            return product_ids
        except ValueError:
            st.error("Ошибка ввода. Пожалуйста, введите корректные ID.")
            return get_user_input()

def load_data(uploaded_file1, uploaded_file2, uploaded_file3, uploaded_file4):
    sales_data = pd.read_excel(uploaded_file1)
    raw_material_data = pd.read_excel(uploaded_file2)
    sales_stock = pd.read_excel(uploaded_file3)
    raw_material_stock = pd.read_excel(uploaded_file4)
    return sales_data, raw_material_data, sales_stock, raw_material_stock

def load_model():
    filename = 'finalized_model_xgb.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def preprocess_data(sales_data):
    le = LabelEncoder()
    sales_data['promo action'] = le.fit_transform(sales_data['promo action'])
    sales_data['конкуренция'] = le.fit_transform(sales_data['конкуренция'])
    features = sales_data[['price', 'cost price', 'рейтинг товаров', 'количество просмотров', 'конкуренция', 'promo action', 'seasonal_discount']]
    target = sales_data['y']
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    return features_imputed, target, imputer, features

def predict(loaded_model, imputer, features, sales_data, user_choice):
    top_products = sales_data['unique_id'].unique() if user_choice is None else user_choice
    next_week_features = sales_data[sales_data['unique_id'].isin(top_products)].groupby('unique_id').last()[list(features.columns)]
    next_week_features = imputer.transform(next_week_features)
    next_week_sales_predictions = loaded_model.predict(next_week_features)
    predicted_sales = pd.DataFrame({
        'unique_id': top_products,
        'predicted_sales': np.round(next_week_sales_predictions)
    })
    return predicted_sales

def calculate_raw_material_needs(predicted_sales, sales_stock, raw_material_data, raw_material_stock):
    predicted_sales = predicted_sales.merge(sales_stock, on='unique_id', how='left')
    predicted_sales['stock_shortage'] = predicted_sales['predicted_sales'] - predicted_sales['stock']
    predicted_sales['production_needed'] = np.where(predicted_sales['stock_shortage'] <= 0, 0, predicted_sales['stock_shortage'])
    raw_material_needs = predicted_sales[predicted_sales['production_needed'] > 0].merge(raw_material_data, on='unique_id', how='left')
    raw_material_needs = raw_material_needs.merge(raw_material_stock, on='raw_material_ID', how='left', suffixes=('_need', '_stock'))
    raw_material_needs['volume_need_adjusted'] = np.where(raw_material_needs['unit_need'] == 'мг', raw_material_needs['production_needed'] * raw_material_needs['volume_need'] / 1000, raw_material_needs['production_needed'] * raw_material_needs['volume_need'])
    raw_material_needs['volume_stock_adjusted'] = np.where(raw_material_needs['unit_stock'] == 'мг', raw_material_needs['volume_stock'] / 1000, raw_material_needs['volume_stock'])
    raw_material_needs['raw_material_needed'] = (raw_material_needs['volume_need_adjusted'] - raw_material_needs['volume_stock_adjusted']).clip(lower=0)
    raw_material_needs['unit_final'] = np.where(raw_material_needs['unit_need'] == 'мг', 'г', raw_material_needs['unit_need'])
    results = raw_material_needs[['product name_x','unique_id', 'raw_material_ID', 'list_need', 'raw_material_needed', 'unit_final']]
    return results

def main():
    st.title("Прогнозирование продаж и потребности в сырье")

    uploaded_file1 = st.file_uploader("Загрузите файл продаж", type=["xlsx"])
    uploaded_file2 = st.file_uploader("Загрузите файл сырья", type=["xlsx"])
    uploaded_file3 = st.file_uploader("Загрузите файл запасов продаж", type=["xlsx"])
    uploaded_file4 = st.file_uploader("Загрузите файл запасов сырья", type=["xlsx"])

    if uploaded_file1 is not None and uploaded_file2 is not None and uploaded_file3 is not None and uploaded_file4 is not None:
        sales_data, raw_material_data, sales_stock, raw_material_stock = load_data(uploaded_file1, uploaded_file2, uploaded_file3, uploaded_file4)
        loaded_model = load_model()
        features_imputed, target, imputer, features = preprocess_data(sales_data)
        user_choice = get_user_input()
        predicted_sales = predict(loaded_model, imputer, features, sales_data, user_choice)
        results = calculate_raw_material_needs(predicted_sales, sales_stock, raw_material_data, raw_material_stock)
        st.write("Прогноз продаж:")
        st.write(predicted_sales)
        st.write("Потребность в сырье:")
        st.write(results)
    elif uploaded_file1 is None or uploaded_file2 is None or uploaded_file3 is None or uploaded_file4 is None:
        st.warning("Пожалуйста, загрузите все 4 файла данных.")
    else:
        st.error("Ошибка загрузки файлов. Пожалуйста, попробуйте снова.")

if __name__ == "__main__":
    main()
