import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\Cliente\Desktop\Python\house-price-prediction\data\raw\train.csv')
    return df

@st.cache_resource
def train_model(df):
    df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)
    limite_nulos = len(df) * 0.3
    df = df.loc[:, df.isnull().sum() < limite_nulos]
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('SalePrice', axis=1)
    y = df_encoded['SalePrice']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns, df

friendly_names = {
    'GrLivArea': 'Área construída (pés²)',
    'OverallQual': 'Qualidade geral da casa (1-10)',
    'YearBuilt': 'Ano de construção',
    'TotalBsmtSF': 'Área do porão (pés²)',
    'Neighborhood': 'Bairro',
    'HouseStyle': 'Estilo da casa',
    'Exterior1st': 'Material exterior',
}

df = load_data()
model, feature_names, df_original = train_model(df)

st.title('Previsão do Preço de Casas com Random Forest')

num_vars = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF']
cat_vars = ['Neighborhood', 'HouseStyle', 'Exterior1st']

input_data = {}
input_data['GrLivArea'] = st.slider(friendly_names['GrLivArea'], 200, 6000, 1500)
input_data['OverallQual'] = st.slider(friendly_names['OverallQual'], 1, 10, 5)
input_data['YearBuilt'] = st.number_input(friendly_names['YearBuilt'], 1870, 2021, 2000)
input_data['TotalBsmtSF'] = st.number_input(friendly_names['TotalBsmtSF'], 0, 5000, 800)

input_data['Neighborhood'] = st.selectbox(friendly_names['Neighborhood'], sorted(df_original['Neighborhood'].unique()))
input_data['HouseStyle'] = st.selectbox(friendly_names['HouseStyle'], sorted(df_original['HouseStyle'].unique()))
input_data['Exterior1st'] = st.selectbox(friendly_names['Exterior1st'], sorted(df_original['Exterior1st'].unique()))

input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

for var in num_vars:
    if var in feature_names:
        input_df.at[0, var] = input_data[var]

for var in cat_vars:
    dummy_col = f"{var}_{input_data[var]}"
    if dummy_col in feature_names:
        input_df.at[0, dummy_col] = 1

prediction = model.predict(input_df)[0]

st.markdown(f"## Preço previsto: R$ {prediction:,.2f}")