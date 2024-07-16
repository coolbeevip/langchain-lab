import os
import unittest
from unittest import TestCase

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_lab.core.conference import Conference

load_dotenv("../../../../.env")
llm = ChatOpenAI(model_name=os.environ["MODEL_NAME"],
                 openai_api_base=os.environ["OPENAI_API_BASE"],
                 openai_api_key=os.environ["OPENAI_API_KEY"],
                 temperature=0.7,
                 request_timeout=600, streaming=True)


@tool
def load_sales_data_tool():
    """This tool load sales data"""
    try:
        import pandas as pd

        sales_data = pd.read_csv("./sales_data.csv")
        result = sales_data.to_markdown()
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"原始数据: \n\n{result}"


@tool
def data_analysis_tool():
    """This tool enhances analysis by providing detailed statistics and visualizations."""
    try:
        df = pd.read_csv("../../../../tests/langchain_lab/langgraph/marketing_analysis_assistant/sales_data.csv")
        analysis_result = df.describe()
        basic_stats_response = f"描述性统计:\n\n{analysis_result.to_markdown()}"
        correlation = df.corr()
        correlation_response = f"\n相关性矩阵:\n\n{correlation.to_markdown()}"
        response = f"{basic_stats_response}\n{correlation_response}"
    except Exception as e:
        response = f"Failed to analyze data. Error: {str(e)}"

    return response


class TestConference(TestCase):

    def test_conference(self):
        conference = Conference(llm=llm)
        conference.add_tool(load_sales_data_tool, data_analysis_tool)
        conference.add_agent(agent_id="Sales_Staff",
                             agent_nickname="Sales_Staff",
                             system_message="负责客户服务和产品、服务提案。回答客户问题，推荐适当的产品、服务，并记录商谈数据、销售预定数据到系统中。",
                             next_agent_name="Sales_Manager",
                             entry_point=True)
        conference.add_agent(agent_id="Sales_Manager",
                             agent_nickname="Sales_Manager",
                             system_message="负责团队管理和指导。设定销售目标，制定销售策略，监控绩效，并向团队成员提供反馈。",
                             next_agent_name="Sales_Staff")
        conference.build_graph()
        conversation = conference.invoke(humanMessage=HumanMessage(
                        content="利用事先准备好的agent和tool进行会话。"
                                "会话的主题是'调查我们公司商品A、B、C过去5年的数据，并制定本期的销售战略。"
                                "会话由sales_staff开始。"
                                "数据分析工具必须使用'./sales_data.csv'文件，并已表格形式输出数据。"
                                "数据分析工具将从'./sales_data.csv'文件中读取数据，进行基本统计和相关关系分析。"
                                "数据分析工具将输出文本形式的分析结果，并提供基于分析结果的见解。"
                                "接下来，将数据分析工具给出的分析结果和见解传达给sales_staff。"
                                "然后，sales_staff和sales_manager根据数据分析工具提供的分析结果和见解进行交流，并共同制定本期的销售策略。"
                                "sales_staff和sales_manager的会话总次数最多为20次。"
                                "最后，sales_manager在总结所有对话后，列出重要的要点并结束。"
                    ))

        with open(f"marketing_analysis_assistant_{llm.model_name}.md", "w") as file:
            for message in conversation:
                if message.content:
                    print(f"## {message.role}\n\n")
                    print(f"{message.content.strip()}\n\n")
                    file.write(f"## {message.role}\n\n")
                    file.write(f"{message.content.strip()}\n\n")


if __name__ == "__main__":
    unittest.main()
