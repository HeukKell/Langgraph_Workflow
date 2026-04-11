# 하위 agent 를 tool 로써 호출한다.
# 그래프 전체를 tool 로써 사용한다는 것이다.

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END

from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState

from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

class SupervisorOutput(BaseModel):
    next_agent : Literal["tool_korean_agent", "tool_spanish_agent", "tool_greek_agent", "__end__"]
    reasoning : str


class AgentsState(MessagesState):
    current_agent : str
    transfered_by : str
    reasoning : str


llm = init_chat_model(model="openai:gpt-4o-mini")

# 중요한것은 그래프를 만들고 그냥 리턴하지 않고
# 그래프를 만들고, 컴파일하고, 그래프를 실행하는 tool 을 반환한다는 것이다.

def make_agent_tool(tool_name, tool_description, system_prompt, tools):

    def node_agent(state : AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
                {system_prompt}

                Conversation History : 
                {state["messages"]}

            """
        )

        return {"messages" : [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("node_agent", node_agent)
    agent_builder.add_node("tools", ToolNode(tools = tools))

    agent_builder.add_edge(START, "node_agent")
    agent_builder.add_conditional_edges("node_agent", tools_condition)
    agent_builder.add_edge("tools", "node_agent")
    agent_builder.add_edge("node_agent", END)

    graph_asAgent = agent_builder.compile()

    # state 를 tool 에 주입하는방법.
    # tool 에 parameter 를 넣어주면 AI 가 그것을 활용하는데
    # state 를 tool 에 어떻게 주입할 수 있는지 알 수 있다.

    @tool(
        name_or_callable=tool_name,
        description = tool_description
    )
    def tool_withAgent(state : Annotated[dict, InjectedState]):
        # 그래프를 tool 로써 다룬다?
        result = graph_asAgent.invoke(state)  
        return result["messages"][-1].content

    return tool_withAgent

tools = [
    make_agent_tool(
        tool_name = "tool_korean_agent",
        tool_description="Use this when user is speaking korean",
        system_prompt="You're a korean customer support agent you speak in korean",
        tools=[]
    ),
    make_agent_tool(
        tool_name = "tool_spanish_agent",
        tool_description="Use this when user is speaking spanish",
        system_prompt="You're a spanish customer support agent you speak in spanish",
        tools=[]
    ),
    make_agent_tool(
        tool_name = "tool_greek_agent",
        tool_description="Use this when user is speaking greek",
        system_prompt="You're a greek customer support agent you speak in greek",
        tools=[]
    )
]

def node_supervisor(state:AgentState):
    # tool 이 3종류인데, tool 은 3종류중 무엇이든지 될 수 있다.
    # 그리고 대화에 적절한 툴이 호출되면, 해당 내용으로 그래프를 만들고, 컴파일, 호출되어 결과값만을 반환한다.
    llm_with_tools = llm.bind_tools(tools=tools)
    result = llm_with_tools.invoke(state["messages"])
    return {
        "messages" : [result],
    }

graph_builder= StateGraph(AgentsState)    

graph_builder.add_node("node_supervisor", node_supervisor)
graph_builder.add_node("tools", ToolNode(tools = tools))

graph_builder.add_edge(START, "node_supervisor")
graph_builder.add_conditional_edges("node_supervisor", tools_condition)
graph_builder.add_edge("tools", "node_supervisor")
graph_builder.add_edge("node_supervisor", END)

graph = graph_builder.compile()





