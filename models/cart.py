import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier




data_path = 
data = pd.read_csv(data_path)

gender = data.iloc['性别']
smoke = data.iloc['吸烟史']
position = data.iloc['肿瘤位置（左肺上叶1 左肺下叶2 右肺上叶3 右肺中叶4 右肺下叶5 其他6 ）']
pathology = data.iloc['病理类型(鳞癌1 腺癌2 其他或者不确定3)']
recruitment = data.iloc['临床实验类型(是1 否2)']





# 分类数据预处理

# 输入到CART树中


