import functools
import json
import operator
import warnings
from typing import Annotated, Sequence

import pandas as pd
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from typing_extensions import TypedDict

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


class NetworkOperationsAnalysisAssistant:
    repl = PythonREPL()

    def __init__(self, openai_api_base: str, openai_api_key: str, model_name: str, recursion_limit: int = 20):
        self.model_name = model_name
        self.recursion_limit = recursion_limit
        self.llm = ChatOpenAI(model_name=model_name, openai_api_base=openai_api_base, openai_api_key=openai_api_key, temperature=0.7, request_timeout=600, streaming=True)
        # 网络运营经理
        networkOpsManager = self.create_agent(
            self.llm,
            [self.python_repl_tool],
            system_message="负责整体网络运营策略的制定和执行，监控网络性能指标，确保服务质量，解决网络异常和紧急事件。",
        )

        # 无线网络工程师
        wirelessNetworkEngineer = self.create_agent(
            self.llm,
            [self.load_data_tool, self.python_repl_tool, self.data_analysis_tool],
            system_message="负责根据数据深入挖掘，提供建议并准备可视化报告。",
        )

        # # IT运营经理
        # itOpsManager = self.create_agent(
        #     self.llm,
        #     [self.python_repl_tool, self.load_data_tool, self.data_analysis_tool],
        #     system_message="负责协调IT资源，支持无线网络的日常运营，制定IT运营的策略和流程，评估和改进网络的运营效率及成本效益。",
        # )
        #
        # # 客户服务经理
        # customerServiceManager = self.create_agent(
        #     self.llm,
        #     [self.python_repl_tool, self.load_data_tool, self.data_analysis_tool],
        #     system_message="负责处理与网络性能相关的客户投诉和反馈，与技术团队协作，解决网络服务问题，提高客户满意度，确保高效的客户服务。",
        # )
        #
        # # 质量保证（QA）团队
        # qaTeam = self.create_agent(
        #     self.llm,
        #     [self.python_repl_tool, self.load_data_tool, self.data_analysis_tool],
        #     system_message="负责评估无线网络的服务质量和性能合规性，制定和执行质量保证测试，发现潜在问题，确保网络运行符合公司标准和法规要求。",
        # )
        #
        # # 高层管理人员 (如CIO/CTO)
        # executiveManagement = self.create_agent(
        #     self.llm,
        #     [self.python_repl_tool, self.load_data_tool, self.data_analysis_tool],
        #     system_message="负责制定整体网路战略，确保其符合公司业务目标，审阅并批准重大网络投资和优化项目，监控网络运营的关键绩效指标（KPI），确保高层次监管。",
        # )

        # 定义图
        workflow = StateGraph(AgentState)
        workflow.add_node("wirelessNetworkEngineer", functools.partial(self.graph_node_agent, agent=wirelessNetworkEngineer, name="wirelessNetworkEngineer"))
        workflow.add_node("networkOpsManager", functools.partial(self.graph_node_agent, agent=networkOpsManager, name="networkOpsManager"))
        # workflow.add_node("itOpsManager",
        #                   functools.partial(self.agent_node, agent=itOpsManager, name="itOpsManager"))
        # workflow.add_node("customerServiceManager",
        #                   functools.partial(self.agent_node, agent=customerServiceManager, name="customerServiceManager"))
        # workflow.add_node("qaTeam",
        #                   functools.partial(self.agent_node, agent=qaTeam, name="qaTeam"))
        # workflow.add_node("executiveManagement",
        #                   functools.partial(self.agent_node, agent=executiveManagement, name="executiveManagement"))
        workflow.add_node("data_tool", self.graph_node_data_tool)

        workflow.add_conditional_edges(
            "wirelessNetworkEngineer",
            self.graph_node_router,
            {"continue": "networkOpsManager", "data_tool": "data_tool", "end": END},
        )
        workflow.add_conditional_edges(
            "networkOpsManager",
            self.graph_node_router,
            {"continue": "wirelessNetworkEngineer", "data_tool": "data_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "data_tool",
            lambda x: x["sender"],
            {
                "networkOpsManager": "networkOpsManager",
                "wirelessNetworkEngineer": "wirelessNetworkEngineer",
                # "itOpsManager": "itOpsManager",
                # "customerServiceManager": "customerServiceManager",
                # "qaTeam": "qaTeam",
                # "executiveManagement": "executiveManagement",
            },
        )
        workflow.set_entry_point("wirelessNetworkEngineer")
        self.graph = workflow.compile()

        # from IPython.display import Image
        # import matplotlib.pyplot as plt
        # img = Image(self.graph.get_graph().print_ascii())
        # imgplot = plt.imshow(img)
        # plt.show()

        tools = [self.data_analysis_tool, self.python_repl_tool, self.load_data_tool]
        self.tool_executor = ToolExecutor(tools)

    @staticmethod
    @tool
    def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = NetworkOperationsAnalysisAssistant.repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

    @staticmethod
    @tool
    def load_data_tool():
        """This tool load data"""
        try:
            import pandas as pd

            data = pd.read_excel("./wireless_network_statistics_data.xlsx")
            result = data.to_markdown()
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Wireless Network Statistics Data: \n\n{result}"

    @staticmethod
    @tool
    def data_analysis_tool():
        """This tool enhances analysis by providing detailed statistics and visualizations."""
        try:
            data_response = []
            df = pd.read_excel("./wireless_network_statistics_data.xlsx")

            # 描述性统计
            analysis_result = df.describe()
            print(analysis_result)
            data_response.append(f"Basic statistics:\n\n{analysis_result.to_markdown()}")

            # 相关性统计
            numerical_df = df.select_dtypes(include=["number"])
            correlation = numerical_df.corr()
            print(correlation)
            data_response.append(f"\nCorrelation matrix:\n\n{correlation.to_markdown()}")

            response = "\n".join(data_response)
        except Exception as e:
            response = f"Failed to analyze data. Error: {str(e)}"

        return response

    @staticmethod
    def graph_node_agent(state: AgentState, agent: RunnableSequence, name: str):
        print(f"Executing {name} node!")
        result = agent.invoke(state)
        # 将代理的输出转换为适合添加到全局状态的格式。
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # 由于有严格的工作流程，可以追踪发件人。
            # 通过追踪发件人，可以知道接下来应该交给谁。
            "sender": name,
        }

    @staticmethod
    def graph_node_router(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        if "function_call" in last_message.additional_kwargs:
            return "data_tool"

        if "FINAL ANSWER" in last_message.content:
            return "end"

        return "continue"

    def graph_node_data_tool(self, state: AgentState):
        """This runs tools in the graph

        It takes in an agent action and calls that tool and returns the result."""
        messages = state["messages"]

        last_message = messages[-1]
        # 从function_call创建ToolInvocation
        tool_input = json.loads(last_message.additional_kwargs["function_call"]["arguments"])
        # 传递单个参数
        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))
        tool_name = last_message.additional_kwargs["function_call"]["name"]
        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )
        print(f"Executing data_tool[{tool_name}] node!")
        # 调用tool_executor，并返回响应。
        response = self.tool_executor.invoke(action)
        # 利用响应创建FunctionMessage。
        function_message = FunctionMessage(content=f"{tool_name} response: {str(response)}", name=action.tool)
        # 将现有列表添加
        return {"messages": [function_message]}

    @staticmethod
    def create_agent(llm, tools, system_message: str):
        functions = [convert_to_openai_function(t) for t in tools]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一个AI助手，与其他助手合作。"
                    "使用提供的工具来逐步回答问题。"
                    "如果您无法完全回答，没关系，另一个使用不同工具的助手将继续帮助您完成。尽力取得进展。"
                    "如果您或其他任何助手有最终答案或可交付成果，"
                    '请在响应中加上"FINAL ANSWER"，这样团队就知道要停止了。'
                    "您可以访问以下工具：{tool_names}。\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_functions(functions)

    def run(self):
        agent_names = {
            "networkOpsManager": "网络运营经理",
            "wirelessNetworkEngineer": "无线网络工程师",
            "data_tool": "数据分析工具",
        }
        with open(f"network_operations_analysis_assistant_report_{self.model_name}.md", "w") as f:
            f.write("# 网络运维智能助手（POC）\n\n")
            f.write(f"> {self.model_name}\n\n")
            f.write(f"```{self.graph.get_graph().draw_ascii()}\n```\n\n")
            # f.write("![image-20240710141823753](assets/marketing_analysis_assistant.png)\n\n")
            f.write("## 多代理协商过程\n\n")
            for s in self.graph.stream(
                {
                    "messages": [
                        HumanMessage(
                            content="利用事先准备好的 agent 和 tool 进行会话。"
                            "会话的主题是'分析总结无线网络统计报表，挖掘数据信息。"
                            "会话由 wirelessNetworkEngineer 开始。"
                            "数据分析工具首先加载数据。根据数据生成无线网统计报表的简要分析总结"
                            "数据分析工具对数据进行基本统计和相关关系分析。并提供基于分析结果的见解。"
                            "接下来，将数据分析工具给出的分析结果和见解传达给 networkOpsManager，并进行简要分析总结。"
                            "然后，wirelessNetworkEngineer 和 networkOpsManager 分析结果和见解进行交流，并共发现问题并制定有效措施。"
                            "wirelessNetworkEngineer 和 networkOpsManager的会话总次数最多为20次。"
                            "最后，networkOpsManager 在总结所有对话后，从总体概况、异常省份、资源和性能完成率、省份详细表现、文件传输即时率、性能合规率等方面给出综合性总结并结束对话。"
                        )
                    ],
                },
                # 图表中的最大步数
                {"recursion_limit": self.recursion_limit},
            ):
                for key in ["networkOpsManager", "wirelessNetworkEngineer", "data_tool"]:
                    if key in s:
                        messages = s[key]["messages"]
                        f.write(f"### {agent_names[key]}\n\n")
                        for msg in messages:
                            if msg.additional_kwargs:
                                f.write(f"{msg.additional_kwargs}\n")
                            f.write(msg.content)
                            f.write("\n\n")  # 结束
