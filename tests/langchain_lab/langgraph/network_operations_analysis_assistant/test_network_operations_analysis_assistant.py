import os
import unittest
import warnings
from unittest import TestCase

import pandas as pd
from dotenv import load_dotenv

from langchain_lab.langgraph.network_operations_analysis_assistant.network_operations_analysis_assistant import \
    NetworkOperationsAnalysisAssistant

load_dotenv("./.env")


class TestNetworkOperationsAnalysisAssistant(TestCase):

    def test_main(self):
        assistant = NetworkOperationsAnalysisAssistant(openai_api_base=os.environ["OPENAI_API_BASE"],
                                                       openai_api_key=os.environ["OPENAI_API_KEY"],
                                                       model_name=os.environ["MODEL_NAME"],
                                                       recursion_limit=50)
        assistant.run()

    def test_load_data(self):
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        df = pd.read_excel("./wireless_network_statistics_data.xlsx")
        print(df.head())
        analysis_result = df.describe()
        print(analysis_result)

        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        correlation = numerical_df.corr()
        print(correlation)



if __name__ == "__main__":
    unittest.main()
