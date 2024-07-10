import os
import unittest
from unittest import TestCase

from dotenv import load_dotenv

from langchain_lab.langgraph.marketing_analysis_assistant import MarketingAnalysisAssistant

load_dotenv("../../../.env")


class TestMarketingAnalysisAssistant(TestCase):

    def test_main(self):
        assistant = MarketingAnalysisAssistant(openai_api_base=os.environ["OPENAI_API_BASE"],
                                               openai_api_key=os.environ["OPENAI_API_KEY"],
                                               model_name=os.environ["MODEL_NAME"],
                                               recursion_limit=50)
        assistant.run()


if __name__ == "__main__":
    unittest.main()
