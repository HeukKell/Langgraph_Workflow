# Supervisor(감독자) Agent
# 중앙 집중식 명령 체계

# Supervisor Node 하위에 여러개의 노드가 있는 구조로
# 하위 노드는 유저에게 직접 답변하지 않고 Supervisor 에게 넘긴다
# 유저가 큰 요청을 보내도 Supervisor 가 특정 Agent 에게 시켜서 결과를 받고
# 필요하다면 Supervisor 가 그 다음 Agent 에게 시켜서 결과를 받고 해서 최종 마무리는 Supervisor 가 한다.


from typing import Literal

from langchain_core import agents
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langgraph.graph.message import MessagesState

from pydantic import BaseModel

class SupervisorOutput(BaseModel) :
    next_agent : Literal["node_korean_agent","node_spanish_agent", "node_greek_agent", "__end__"]
    reasoning : str

class AgentState(MessagesState):
    current_agent : str
    transfered_by: str
    reasoning : str   


llm = init_chat_model("openai:gpt-4o-mini") 

def make_agent(prompt : str , tools : []):

    def node_agent(state : AgentState) :
        llm_with_tools = llm.bind_tools(tools)

        response = llm_with_tools.invoke(f"""

            {prompt}

            Conversation History : 
            {state["messages"]}

        """)

        return {"messages" : [response]}

    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("node_agent", node_agent)
    agent_builder.add_node("tools", ToolNode(tools = tools))
    
    agent_builder.add_edge(START, "node_agent")
    agent_builder.add_conditional_edges("node_agent", tools_condition)
    agent_builder.add_edge("tools", "node_agent")
    agent_builder.add_edge("node_agent", END)

    return agent_builder.compile()

def node_supervisor(state : AgentState) :
    structured_llm = llm.with_structured_output(SupervisorOutput)

    response = structured_llm.invoke(
        f"""
            You are a supervisor that routes conversations to the appropriate language agent

            Analyse the customers request and the conversation history and decide wich agent should handle the conversation.

            The options for the next agent are:
            - node_greek_agent
            - node_spanish_agent
            - node_korean_agent

            <Conversation_History>
            {state.get("messages", [])}
            </Conversation_History>

            If an agent has replied end the conversation by returning __end__
        """

    )    

    if (response.next_agent == "__end__") :
        return Command(
            goto=END,
            update ={"reasoning" : response.reasoning}
        )

    return Command(
        goto=response.next_agent,
        update={"reasoning" : response.reasoning},
    )


graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "node_supervisor",
    node_supervisor,
    destinations=(
        "node_korean_agent",
        "node_spanish_agent",
        "node_greek_agent",
        END
    )
    
)

# 더이상 각 Agent 들이 tools = [handoff_tool] 을 가질 필요가 없다.
# 통제권을 자신들이 갖는게 아니라
# node_supervisor 가 통제하고, 각 agent 에게 라우팅 할것이므로

graph_builder.add_node(
    "node_korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[],
    ),
)
graph_builder.add_node(
    "node_greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[],
    ),
)
graph_builder.add_node(
    "node_spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[],
    ),
)


graph_builder.add_edge(START, "node_supervisor")
graph_builder.add_edge("node_korean_agent", "node_supervisor")
graph_builder.add_edge("node_spanish_agent", "node_supervisor")
graph_builder.add_edge("node_greek_agent", "node_supervisor")

graph = graph_builder.compile()