import functools
import json
import operator
from typing import Annotated, Sequence

import pandas as pd
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


class MarketingAnalysisAssistant:
    repl = PythonREPL()

    def __init__(self, openai_api_base: str, openai_api_key: str, model_name: str, recursion_limit: int = 20):
        self.model_name = model_name
        self.recursion_limit = recursion_limit
        self.llm = ChatOpenAI(model_name=model_name, openai_api_base=openai_api_base, openai_api_key=openai_api_key, temperature=0.7, request_timeout=600, streaming=True)
        # 定义智能体
        sales_staff = self.create_agent(
            self.llm,
            [self.python_repl_tool, self.load_sales_data_tool, self.data_analysis_tool],
            system_message="负责客户服务和产品、服务提案。回答客户问题，推荐适当的产品、服务，并记录商谈数据、销售预定数据到系统中。",
        )

        sales_manager = self.create_agent(
            self.llm,
            [self.python_repl_tool, self.load_sales_data_tool, self.data_analysis_tool],
            system_message="负责团队管理和指导。设定销售目标，制定销售策略，监控绩效，并向团队成员提供反馈。",
        )

        sales_staff_node = functools.partial(self.agent_node, agent=sales_staff, name="sales_staff")
        sales_manager_node = functools.partial(self.agent_node, agent=sales_manager, name="sales_manager")

        # 定义图
        workflow = StateGraph(AgentState)

        workflow.add_node("sales_staff", sales_staff_node)
        workflow.add_node("sales_manager", sales_manager_node)
        workflow.add_node("sales_tool", self.sales_tool)

        workflow.add_conditional_edges(
            "sales_staff",
            self.router,
            {"continue": "sales_manager", "sales_tool": "sales_tool", "end": END},
        )
        workflow.add_conditional_edges(
            "sales_manager",
            self.router,
            {"continue": "sales_staff", "sales_tool": "sales_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "sales_tool",
            # 每个代理节点都会更新'sender'字段。
            # 工具调用节点不会被更新。
            # 换句话说，这条边将路由到调用工具的原始代理。
            lambda x: x["sender"],
            {
                "sales_staff": "sales_staff",
                "sales_manager": "sales_manager",
            },
        )
        workflow.set_entry_point("sales_staff")
        self.graph = workflow.compile()

        tools = [self.data_analysis_tool, self.python_repl_tool, self.load_sales_data_tool]
        self.tool_executor = ToolExecutor(tools)

    @staticmethod
    @tool
    def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = MarketingAnalysisAssistant.repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

    @staticmethod
    @tool
    def load_sales_data_tool():
        """This tool load sales data"""
        try:
            import pandas as pd

            sales_data = pd.read_csv("./sales_data.csv")
            result = sales_data.to_markdown()
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Sales Data: \n\n{result}"

    @staticmethod
    @tool
    def data_analysis_tool():
        """This tool enhances analysis by providing detailed statistics and visualizations."""
        try:
            df = pd.read_csv("../../../tests/langchain_lab/langgraph/sales_data.csv")
            analysis_result = df.describe()
            basic_stats_response = f"Basic statistics:\n\n{analysis_result.to_markdown()}"
            correlation = df.corr()
            correlation_response = f"\nCorrelation matrix:\n\n{correlation.to_markdown()}"
            response = f"{basic_stats_response}\n{correlation_response}"
        except Exception as e:
            response = f"Failed to analyze data. Error: {str(e)}"

        return response

    @staticmethod
    def agent_node(state, agent, name):
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
    def router(state):
        messages = state["messages"]
        last_message = messages[-1]

        if "function_call" in last_message.additional_kwargs:
            return "sales_tool"

        if "FINAL ANSWER" in last_message.content:
            return "end"

        return "continue"

    def sales_tool(self, state):
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
        print(f"Executing sales_tool[{tool_name}] node!")
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
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_functions(functions)

    def run(self):
        agent_names = {
            "sales_staff": "销售员",
            "sales_manager": "销售经理",
            "sales_tool": "数据分析工具",
        }
        with open(f"sales_analysis_report_{self.model_name}.md", "w") as f:
            f.write("# 市场部销售智能助手（POC）\n\n")
            f.write(f"> {self.model_name}\n\n")
            f.write("![image - 20240710141823753](assets / marketing_analysis_assistant.png)\n\n")
            f.write("## 多代理协商过程\n\n")
            for s in self.graph.stream(
                {
                    "messages": [
                        HumanMessage(
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
                        )
                    ],
                },
                # 图表中的最大步数
                {"recursion_limit": self.recursion_limit},
            ):
                for key in ["sales_staff", "sales_manager", "sales_tool"]:
                    if key in s:
                        messages = s[key]["messages"]
                        f.write(f"### {agent_names[key]}\n\n")
                        for msg in messages:
                            if msg.additional_kwargs:
                                f.write(f"{msg.additional_kwargs}\n")
                            f.write(msg.content)
                            f.write("\n\n")  # 结束
