import numpy as np
import pandas as pd

df = pd.read_csv('/workspace/2025/LogFormer/parse_result/BGL.log_structured.csv')
#判断ParameterList列不为空的比例
print(df['ParameterList'].isnull().sum() / len(df))

