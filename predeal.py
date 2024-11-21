import pandas as pd
#数据预处理
# 读取CSV文件
file_path = 'data\\2016_imf_results.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 检查数据是否有缺失值
missing_values = df.isnull().sum()

# 检查数据是否有NaN值
nan_values = df.isna().sum()

# 输出结果
print("缺失值检查：")
print(missing_values)
print("\nNaN值检查：")
print(nan_values)

# 查找缺失值和NaN值的位置
missing_values_location = df[df.isnull().any(axis=1)]

# 输出结果
print("缺失值和NaN值的位置：")
print(missing_values_location)