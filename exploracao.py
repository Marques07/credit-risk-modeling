# ==============================================================
# Exploração inicial - Home Credit Default Risk
# Objetivo: entender estrutura, target e qualidade dos dados
# ==============================================================

import warnings
warnings.filterwarnings('ignore')  # silencia warnings de overflow do cast

import pandas as pd
import numpy as np


def reduce_mem_usage(df):
    """Reduz uso de memória ajustando tipos numéricos pro menor possível."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                # int: tenta int8 → int16 → int32, para no primeiro que couber
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # float: tenta float16 → float32
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df



app = pd.read_csv('application_train.csv')
app = reduce_mem_usage(app)

# --------------------------------------------------------------
#  Dicionário oficial do próprio Kaggle
# --------------------------------------------------------------

''' 
desc = pd.read_csv('HomeCredit_columns_description.csv', encoding='latin-1')
desc = desc[['Table', 'Row', 'Description']]
desc.columns = ['Tabela', 'Coluna', 'Descrição']

# Salva uma versão em Excel pra consulta offline
desc.to_excel('dicionario_home_credit.xlsx', index=False)
print("\n✅ Dicionário salvo em 'dicionario_home_credit.xlsx'")
'''
#============================================================================

print("=" * 60)
print(f"Shape: {app.shape}")
print("=" * 60)


print("\nDistribuição do TARGET:")
print(app['TARGET'].value_counts(normalize=True))


print("\nTop 15 colunas com mais nulos:")
missing = app.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].head(20))


#Analisar principalmente object
print("\nTipos de dados:")
print(app.dtypes.value_counts())


# --------------------------------------------------------------
#  Amostra visual de algumas colunas-chave
# --------------------------------------------------------------
print("\nPrimeiras 10 linhas (colunas selecionadas):")
print(app[['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_2']].head(10))


# --------------------------------------------------------------
# Verificar possíveis outliers em DAYS_EMPLOYED
# --------------------------------------------------------------
print(f"\nValor máximo de DAYS_EMPLOYED: {app['DAYS_EMPLOYED'].max()}")


# --------------------------------------------------------------
# Nulos nos scores externos de crédito
# --------------------------------------------------------------
print("\nNulos em EXT_SOURCE_1, 2, 3:")
print(app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].isnull().mean())


