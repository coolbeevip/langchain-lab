import os
import unittest
from unittest import TestCase

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_lab.core.conference import Conference

load_dotenv("../../../../.env")
llm = ChatOpenAI(model_name=os.environ["MODEL_NAME"],
                 openai_api_base=os.environ["OPENAI_API_BASE"],
                 openai_api_key=os.environ["OPENAI_API_KEY"],
                 temperature=0.7,
                 request_timeout=600, streaming=True)


class TestConference(TestCase):

    def test_conference_without_tool(self):
        conference = Conference(llm=llm, lang="zh")
        conference.add_agent(agent_name="数据分析专家",
                             system_message="数据分析专家，负责分析和解读数据，提供对数据的深入见解和洞察。",
                             next_agent_name="网络优化工程师",
                             entry_point=True)
        conference.add_agent(agent_name="网络优化工程师",
                             system_message="你是网络优化工程师，负责网络性能和资源管理，监控网络性能指标。",
                             next_agent_name="网络运营经理")
        conference.add_agent(agent_name="网络运营经理",
                             system_message="你是网络运营经理，负责整体网络运营策略的制定和执行，关注全国各省网络质量情况。",
                             next_agent_name="网络优化工程师")
        conference.build_graph()

        data_markdown = pd.read_excel("./wireless_network_statistics_data.xlsx").to_markdown()

        conversation = conference.invoke(humanMessage=HumanMessage(
            content="请根据数据报告进行分析，结合其他人员的建议进行补充，如果你是网络运营经理还需要最后对指标异常的省份进行单独评价"
                    f"##数据报告: \n\n{data_markdown}\n\n"
        ), recursion_limit=10)

        with open(f"network_operations_analysis_assistant.py_{llm.model_name}.md", "w") as file:
            for message in conversation:
                if message.content:
                    print(f"## {message.role}\n\n")
                    print(f"{message.content.strip()}\n\n")
                    file.write(f"## {message.role}\n\n")
                    file.write(f"{message.content.strip()}\n\n")


if __name__ == "__main__":
    unittest.main()
