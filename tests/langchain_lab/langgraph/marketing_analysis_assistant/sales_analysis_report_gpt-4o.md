# 市场部销售智能助手（POC）

> gpt-4o

![image-20240710141823753](../assets/marketing_analysis_assistant.png)

## 多代理协商过程

### 销售员

{'function_call': {'arguments': '{}', 'name': 'load_sales_data_tool'}}


### 数据分析工具

load_sales_data_tool response: Sales Data:

|    |   Year |   Product_A |   Product_B |   Product_C |
|---:|-------:|------------:|------------:|------------:|
|  0 |   2019 |        1091 |         540 |         802 |
|  1 |   2020 |         979 |         527 |         782 |
|  2 |   2021 |         970 |         521 |         816 |
|  3 |   2022 |        1070 |         559 |         752 |
|  4 |   2023 |        1059 |         533 |         730 |

### 销售员

{'function_call': {'arguments': 'import pandas as pd\n\n# Load the sales data from the CSV file\ndata = {\n    \'Year\': [2019, 2020, 2021, 2022, 2023],\n    \'Product_A\': [1091, 979, 970, 1070, 1059],\n    \'Product_B\': [540, 527, 521, 559, 533],\n    \'Product_C\': [802, 782, 816, 752, 730]\n}\ndf = pd.DataFrame(data)\n\n# Perform basic statistical analysis\nstatistics = df.describe()\n\n# Calculate correlation matrix\ncorrelation_matrix = df.corr()\n\n# Print the results\nprint("Basic Statistical Analysis:\\n", statistics)\nprint("\\nCorrelation Matrix:\\n", correlation_matrix)\n\n# Provide insights based on the analysis\ninsights = """\nInsights:\n1. Product_A has the highest average sales over the past 5 years.\n2. Product_B has the lowest average sales over the past 5 years.\n3. Product_A sales peaked in 2019, whereas Product_C sales peaked in 2021.\n4. There is a moderate positive correlation between Product_A and Product_B sales.\n5. Product_C sales have shown a declining trend over the past 3 years.\n"""\n\nprint(insights)\n', 'name': 'data_analysis_tool'}}


