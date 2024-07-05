import functools
import json
import operator
import os
from typing import Annotated, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from typing_extensions import TypedDict

load_dotenv("../../../.env")

# model_name = "gpt-3.5-turbo"
model_name = "gpt-4o"
# model_name="qwen1.5-72b-chat"
llm = ChatOpenAI(
    model_name=model_name, openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.7, request_timeout=600, streaming=True
)


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


repl = PythonREPL()


@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"


@tool
def data_analysis_tool():
    """数据分析工具."""
    try:
        df = pd.read_csv("./sales_data.csv")
        analysis_result = df.describe()
        basic_stats_response = f"Basic statistics:\n{analysis_result.to_string()}"
        correlation = df.corr()
        correlation_response = f"\nCorrelation matrix:\n{correlation.to_string()}"
        response = f"{basic_stats_response}\n{correlation_response}"
    except Exception as e:
        response = f"Failed to analyze data. Error: {str(e)}"

    return response


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


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


# 销售代表
sales_staff = create_agent(
    llm,
    [data_analysis_tool],
    system_message="负责客户服务和产品、服务提案。回答客户问题，推荐适当的产品、服务，并记录商谈数据、销售预定数据到系统中。",
)
sales_staff_node = functools.partial(agent_node, agent=sales_staff, name="sales_staff")

# 销售经理
sales_manager = create_agent(
    llm,
    [data_analysis_tool],
    system_message="负责团队管理和指导。设定销售目标，制定销售策略，监控绩效，并向团队成员提供反馈。",
)
sales_manager_node = functools.partial(agent_node, agent=sales_manager, name="sales_manager")

tools = [data_analysis_tool]
tool_executor = ToolExecutor(tools)


def tool_node(state):
    # print(f"Executing tool_node! state is {state}!")
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
    # 调用tool_executor，并返回响应。
    response = tool_executor.invoke(action)
    # 利用响应创建FunctionMessage。
    function_message = FunctionMessage(content=f"{tool_name} response: {str(response)}", name=action.tool)
    # 将现有列表添加
    return {"messages": [function_message]}


# ### 边缘逻辑的定义
def router(state):
    messages = state["messages"]
    last_message = messages[-1]

    if "function_call" in last_message.additional_kwargs:
        return "call_tool"

    if "FINAL ANSWER" in last_message.content:
        return "end"

    return "continue"


# ### 图的定义
workflow = StateGraph(AgentState)

workflow.add_node("sales_staff", sales_staff_node)
workflow.add_node("sales_manager", sales_manager_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "sales_staff",
    router,
    {"continue": "sales_manager", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "sales_manager",
    router,
    {"continue": "sales_staff", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "call_tool",
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
graph = workflow.compile()

# 执行
for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="该代码将使用预先准备的代理和工具。"
                        "对话主题是：“研究公司产品A、B、C过去5年的数据，并制定本季销售策略。"
                        "对话从销售人员开始。"
                        "数据分析工具将读取数据以文本形式输出分析结果,并进行基本统计和相关性分析。"
                        "然后，由数据分析工具得出的分析结果和见解将传达给销售人员。"
                        "接下来，销售人员和销售经理根据数据分析工具得出的分析结果和见解交换意见，制定本季销售策略。"
                        "销售人员和销售经理之间总共最多有20次对话机会。"
                        "最后，销售经理在考虑所有对话内容后，用项目符号总结重要观点并结束。"
            )
        ],
    },
    # 图表中的最大步数
    {"recursion_limit": 25},
):
    for key in ["sales_staff", "sales_manager"]:
        if key in s:
            messages = s[key]["messages"]
            for msg in messages:
                # 输出来自代理商的消息内容。
                print(msg.content)
                print("----\n")  # 结束
