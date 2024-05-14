import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



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

# def load_model():
#     #filename = 'lstm_model.h5'
#     loaded_model = load_model('lstm_model.h5')
#     return loaded_model

def loading_model():
    filename = 'lstm_model.h5'
    loaded_model = load_model(filename)
    return loaded_model


def preprocess_data(sales_data):
    le = LabelEncoder()
    sales_data['promo action'] = le.fit_transform(sales_data['promo action'])
    sales_data['конкуренция'] = le.fit_transform(sales_data['конкуренция'])
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    sales_data['year'] = sales_data['date'].dt.year
    sales_data['month'] = sales_data['date'].dt.month
    sales_data['day'] = sales_data['date'].dt.day
    sales_data.drop('date', axis=1, inplace=True)

    imputer = SimpleImputer(strategy='mean')
    features = ['price', 'cost price', 'рейтинг товаров', 'количество просмотров', 'конкуренция', 'promo action', 'seasonal_discount', 'year', 'month', 'day']
    X = imputer.fit_transform(sales_data[features])
    y = sales_data['y'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y, scaler_X, scaler_y, features, imputer

def prepare_sequences(X, y, sequence_length):
    n_features = X.shape[1]
    X_sequences = []
    y_sequences = []
    for i in range(sequence_length, len(X)):
        X_sequences.append(X[i-sequence_length:i, :])
        y_sequences.append(y[i])
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    return X_sequences, y_sequences

def predict(loaded_model, scaler_X, scaler_y, imputer, features, sales_data, user_choice):
    top_products = sales_data['unique_id'].unique() if user_choice is None else user_choice
    next_week_features = sales_data[sales_data['unique_id'].isin(top_products)].groupby('unique_id').last()[features]
    next_week_features = imputer.transform(next_week_features)
    next_week_features = scaler_X.transform(next_week_features)
    next_week_features = np.repeat(next_week_features[:, np.newaxis, :], 10, axis=1)
    next_week_sales_predictions = loaded_model.predict(next_week_features)
    predicted_sales = scaler_y.inverse_transform(next_week_sales_predictions.reshape(-1, 1))
    predicted_sales_df = pd.DataFrame({
        'unique_id': top_products,
        'predicted_sales': np.round(predicted_sales.flatten())
    })
    return predicted_sales_df

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
        loaded_model = loading_model()
        X, y, scaler_X, scaler_y, features, imputer = preprocess_data(sales_data)
        sequence_length = 10
        X_sequences, y_sequences = prepare_sequences(X, y, sequence_length)
        user_choice = get_user_input()
        predicted_sales = predict(loaded_model, scaler_X, scaler_y, imputer, features, sales_data, user_choice)
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