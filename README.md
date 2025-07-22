# Previsão do Preço de Casas com Random Forest

Este projeto utiliza técnicas de Machine Learning para prever o preço de casas com base em dados reais. Utilizamos o algoritmo Random Forest para treinamento do modelo e construímos uma interface interativa com Streamlit para facilitar a previsão.

## Dataset

O dataset utilizado é o [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), que contém informações detalhadas sobre casas e seus preços.

## Sobre o modelo

Pré-processamento dos dados: remoção de colunas com muitos valores nulos, preenchimento de valores faltantes, codificação one-hot de variáveis categóricas.

Algoritmo: Random Forest Regressor com 100 árvores.

Métrica de avaliação: Root Mean Squared Error (RMSE).

## Possíveis melhorias

Implementar explicação de modelo com SHAP ou LIME.

Interface mais amigável com mais variáveis de entrada.

Deploy em plataforma cloud para acesso online.