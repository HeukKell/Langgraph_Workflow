import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState

# ToolNode(tools = [tool_1, tool_2, ...])
# Agent 의 메세지 기록을 살펴보고 AI 모델이 Tool 을 호출하고 싶다면 Tool 을 호출해주는 기능을 한다.
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(model="openai:gpt-4o")

class AgentState(MessagesState):
    # MessagesState 의 멤버 상속

    # Agent 이전에 대한 정보를 담겠다.
    current_agent : str
    transfered_by : str


def make_agent(prompt,tools) :

    def node_agent(state : AgentState):
        llm_with_tools = llm.bind_tools(tools)
        res = llm_with_tools.invoke(
            f"""
                {prompt}

                ConversationHistory : 
                {state["messages"]}
            """
        )

        return {"messages" : [res]}

    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("node_agent", node_agent)
    agent_builder.add_node("tools", ToolNode(tools = tools))
    

    agent_builder.add_edge(START, "node_agent")

    # ToolNode 는 "tools" 또는 "__end__" 를 반환한다
    # 따로 매핑을 해주는 방법도 있고, 이름을 그대로 쓰는 방법도 있다

    # agent_builder.add_conditional_edges(
    #         "node_agent",
    #         tools_condition,  # Routes to "tools" or "__end__"
    #         {"tools": "tools", "__end__": "__end__"}
    #     )

    agent_builder.add_conditional_edges("node_agent", tools_condition)
    agent_builder.add_edge("tools", "node_agent")
    agent_builder.add_edge("node_agent", END)

    # return graph
    return agent_builder.compile() 

@tool
def handoff_tool(transfer_to : str, transfered_by:str):
    """
        Handoff to another agent.

        Use this tool when the customer speaks a language that you don't understand.

        Possible values for 'transfer_to':
        - 'node_korean_agent'
        - 'node_greek_agent'
        - 'node_spanish_agent'

        Args:
            transfer_to : The agent to transfer the conversation to
            transfered_by : The agent that transerred the conversation
    """

    return Command(
        update= {
            "current_agent" : transfer_to,
            "transfered_by" : transfered_by
        },
        goto= transfer_to,
        graph=Command.PARENT,   # 노드는 부모 그래프에서 대상을 찾아라
    )

graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "node_korean_agent",
    make_agent(
        prompt = "You're a Korean customer support agent. You only speak and understand Korean.",
        tools = [handoff_tool]
    ),
    destinations = ("node_greek_agent", "node_spanish_agent")
)

graph_builder.add_node(
    "node_greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools = [handoff_tool]
    ),
    destinations=("node_korean_agent", "node_spanish_agent")
)

graph_builder.add_node(
    "node_spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only Speak and understand Spanish.",
        tools=[handoff_tool]
    ),
    destinations=("node_korean_agent", "node_greek_agent")
)


graph_builder.add_edge(START, "node_korean_agent")
graph = graph_builder.compile()

for event in graph.stream(
    {
        "messages" : [
            {
                "role" : "user",
                "content" : "Hola! Necesito ayuda con mi cuenta."
            }
        ]
    },
    stream_mode = "updates"
) :
    print(event)