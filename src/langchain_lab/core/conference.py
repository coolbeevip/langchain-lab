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


class AgentRunnableSequence:
    id: str
    nickname: str
    next_agent_name: str
    agent: RunnableSequence
    entry_point: bool = False

    def __init__(self, id, nickname, next_agent_name, agent, entry_point):
        self.id = id
        self.nickname = nickname
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

    def __init__(self, llm: BaseChatModel = None, python_repl: bool = False):
        self.tool_executor: ToolExecutor = None
        self.graph: CompiledGraph = None
        self.agents: List[AgentRunnableSequence] = []
        self.tools: List[Tool] = []
        self.python_repl = python_repl
        self.default_llm = llm

    def add_tool(self, *tools):
        for t in tools:
            self.tools.append(t)
        self.tool_executor = ToolExecutor(self.tools)

    def add_agent(self, agent_id, agent_nickname: str, system_message: str, next_agent_name: str,
                  entry_point: bool = False, llm: BaseChatModel = None):
        functions = [convert_to_openai_function(t) for t in self.tools]
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
        prompt = prompt.partial(tool_names=", ".join([t.name for t in self.tools]))

        if llm:
            agent: RunnableSequence = prompt | llm.bind_functions(functions)
        elif self.default_llm:
            agent: RunnableSequence = prompt | self.default_llm.bind_functions(functions)
        else:
            raise ValueError("No language model provided.")

        self.agents.append(
            AgentRunnableSequence(id=agent_id,
                                  nickname=agent_nickname,
                                  next_agent_name=next_agent_name,
                                  agent=agent,
                                  entry_point=entry_point))

    def build_graph(self):
        workflow = StateGraph(AgentState)

        # Add Agents
        for agent in self.agents:
            workflow.add_node(agent.nickname,
                              functools.partial(self.graph_node_agent, agent=agent.agent, name=agent.nickname))

        # Add ToolKit
        workflow.add_node("ToolKit", self.graph_node_tool_kit)

        # Add Edges
        for agent in self.agents:
            workflow.add_conditional_edges(
                agent.nickname,
                self.graph_node_router,
                {"continue": agent.next_agent_name, "ToolKit": "ToolKit", "end": END},
            )

        path_map = {agent.nickname: agent.next_agent_name for agent in self.agents}
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
            workflow.set_entry_point(entry_point_agents[0].nickname)
            self.graph = workflow.compile()

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
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
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

    def invoke(self, humanMessage: HumanMessage, recursion_limit: int = 20):
        output_role = [agent.nickname for agent in self.agents]
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
