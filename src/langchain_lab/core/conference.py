import functools
import json
import operator
from typing import TypedDict, Annotated, Sequence, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import Tool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.utilities import PythonREPL
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolInvocation, ToolExecutor

lang_prompts = {
    "en": {
        "agent_system_prompt_prefix": ("You are a helpful AI assistant, collaborating with other assistants."
                                       " Use the provided tools to progress towards answering the question."
                                       " If you are unable to fully answer, that's OK,"
                                       " another assistant with different tools "
                                       " will help where you left off. Execute what you can to make progress."
                                       " If you or any of the other assistants have the final answer or deliverable,"
                                       " prefix your response with FINAL ANSWER so the team knows to stop.")
    },
    "zh": {
        "agent_system_prompt_prefix": ("你是一个乐于助人的人工智能助手，正在与其他助手合作。"
                                       "如果你无法完全回答，没关系，另一个助手会帮助你完成你未完成的任务。尽你所能取得进展。"
                                       "如果你或任何其他助手有最终答案或可交付成果，"
                                       "在你的回答前面加上 'FINAL ANSWER'，这样团队就知道该停下来了。")
    }
}


class AgentRunnableSequence:
    name: str
    next_agent_name: str
    agent: RunnableSequence
    entry_point: bool = False

    def __init__(self, name, next_agent_name, agent, entry_point):
        self.name = name
        self.next_agent_name = next_agent_name
        self.agent = agent
        self.entry_point = entry_point


class ConversationMessage:
    role: str
    content: str

    def __init__(self, role, content):
        self.role = role
        self.content = content


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


class Conference:

    def __init__(self, llm: BaseChatModel = None, python_repl: bool = False, lang: str = "en"):
        self.tool_executor: ToolExecutor = None
        self.graph: CompiledGraph = None
        self.agents: List[AgentRunnableSequence] = []
        self.tools: List[Tool] = []
        self.python_repl = python_repl
        self.default_llm = llm
        if lang not in lang_prompts:
            raise ValueError(f"Language {lang} not supported. Only support {', '.join(lang_prompts.keys())}.")
        self.lang = lang

    def add_tool(self, *tools):
        for t in tools:
            self.tools.append(t)
        self.tool_executor = ToolExecutor(self.tools)

    def add_agent(self, agent_name: str, system_message: str, next_agent_name: str,
                  entry_point: bool = False, llm: BaseChatModel = None):
        functions = [convert_to_openai_function(t) for t in self.tools]

        system_content = lang_prompts[self.lang]["agent_system_prompt_prefix"]

        if len(self.tools) > 0:
            system_content = system_content + " You have access to the following tools: {tool_names}."

        system_content = system_content + "\n{system_message}"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_content
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in self.tools]))

        if llm:
            agent: RunnableSequence = prompt | llm.bind_functions(functions)
        elif self.default_llm:
            agent: RunnableSequence = prompt | self.default_llm.bind_functions(functions)
        else:
            raise ValueError("No language model provided.")

        self.agents.append(
            AgentRunnableSequence(name=agent_name,
                                  next_agent_name=next_agent_name,
                                  agent=agent,
                                  entry_point=entry_point))

    def build_graph(self):
        workflow = StateGraph(AgentState)

        # Add Agents
        for agent in self.agents:
            workflow.add_node(agent.name,
                              functools.partial(self.graph_node_agent, agent=agent.agent, name=agent.name))

        # Add ToolKit
        if len(self.tools) > 0:
            workflow.add_node("ToolKit", self.graph_node_tool_kit)

        # Add Edges
        for agent in self.agents:
            path_map = {"continue": agent.next_agent_name, "end": END}
            if len(self.tools) > 0:
                path_map["ToolKit"] = "ToolKit"
            workflow.add_conditional_edges(
                agent.name,
                self.graph_node_router,
                path_map
            )

        if len(self.tools) > 0:
            path_map = {agent.name: agent.next_agent_name for agent in self.agents}
            workflow.add_conditional_edges(
                source="ToolKit",
                path=lambda x: x["sender"],
                path_map=path_map
            )

        entry_point_agents = [agent for agent in self.agents if agent.entry_point]
        if len(entry_point_agents) > 1:
            raise ValueError("Only one agent can be an entry point.")
        elif len(entry_point_agents) == 0:
            raise ValueError("At least one agent must be an entry point.")
        else:
            workflow.set_entry_point(entry_point_agents[0].name)
            self.graph = workflow.compile()
            self.graph.get_graph().print_ascii()

    def graph_node_tool_kit(self, state: AgentState):
        last_message = state["messages"][-1]
        function_call_arguments = last_message.additional_kwargs["function_call"]["arguments"]
        tool_input = json.loads(function_call_arguments)

        # 传递单个参数
        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))
        tool_name = last_message.additional_kwargs["function_call"]["name"]
        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )
        print(f"Executing ToolKit [{tool_name}] node!")
        response = self.tool_executor.invoke(action)
        function_message = FunctionMessage(content=f"{str(response)}", name=action.tool)
        return {"messages": [function_message]}

    @staticmethod
    def graph_node_router(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        if "function_call" in last_message.additional_kwargs:
            return "ToolKit"

        if "FINAL ANSWER" in last_message.content:
            return "end"

        return "continue"

    @staticmethod
    def graph_node_agent(state: AgentState, agent: RunnableSequence, name: str):
        print(f"Executing {name} node!")
        result = agent.invoke(state)
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}))
        return {
            "messages": [result],
            "sender": name
        }

    @staticmethod
    @tool
    def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        repl = PythonREPL()
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    def invoke(self, humanMessage: HumanMessage, recursion_limit: int = 5):
        output_role = [agent.name for agent in self.agents]
        output_role.append("ToolKit")
        for s in self.graph.stream(
            {
                "messages": [humanMessage],
            },
            {"recursion_limit": recursion_limit},
        ):
            for key in output_role:
                if key in s:
                    if key != "ToolKit":
                        last_role = key
                    messages = s[key]["messages"]
                    for msg in messages:
                        if key == "ToolKit":
                            key = last_role
                        yield ConversationMessage(role=key, content=msg.content)
