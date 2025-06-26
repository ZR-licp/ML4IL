import pandas as pd
import numpy as np

gaff_df = pd.read_csv('./data/gaff2_MD_Des.csv')
mordred_df = pd.read_csv('./data/01Des_IL.csv')

mordred_exclude_id_smiles = mordred_df.drop(columns=['ID', 'smiles'])
final_combined_df = pd.concat([gaff_df, mordred_exclude_id_smiles], axis=1)
print(f"Initial combined data has {final_combined_df.shape[0]} rows and {final_combined_df.shape[1]} columns.")

# 定义错误值类型
error_values = (np.nan, np.inf, -np.inf)

# 1）删除每列空缺值和错误值总数超过10个的列
final_combined_df = final_combined_df.loc[:, final_combined_df.isin(error_values).sum() <= 10]

print("Initial combined data after removing columns with >10 missing/error values:")
print(f"{final_combined_df.shape[0]} rows and {final_combined_df.shape[1]} columns.")

# 2）对剩下的空缺值和错误值用所在列的平均值填充
final_combined_df = final_combined_df.apply(lambda x: np.where(np.isin(x, error_values), np.nan, x))
final_combined_df.fillna(final_combined_df.mean(), inplace=True)

protected_columns = ['ID', 'smiles', 'diffusion']
protected_df = final_combined_df[protected_columns]

# 过滤掉字符串列以及保护列
numeric_df = final_combined_df.drop(columns=protected_columns).select_dtypes(include=[np.number])

# 3）删除标准差为0的列
non_zero_std = numeric_df.std() != 0
numeric_df = numeric_df.loc[:, non_zero_std]

# 4）保留数值列的参数精度到4位小数
numeric_df = numeric_df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

# 将保护列与处理后的数值列合并
final_combined_df = pd.concat([protected_df, numeric_df], axis=1)

final_combined_df.to_csv('./data/descriptors/03all_clean_Des.csv', index=False)
print(f"Cleaned data has {final_combined_df.shape[0]} rows and {final_combined_df.shape[1]} columns.")
