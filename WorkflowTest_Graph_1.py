from typing import Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

class EmailState(TypedDict):
    email : str
    category : Literal["spam", "normal", "urgent"]
    priority_score : int
    response : str

# 메일 카테고리 분류류
def node_categorize_email(state : EmailState):
    email = state["email"].lower()

    # urgent : 긴급함, asap : as soon as possible
    if "urgent" in email or "asap" in email : 
        category = "urgent"
    elif "offer" in email or "discount" in email:
        category = "spam"
    else :
        category = "normal"

    return {
        "category" : category,
    }

# 메일 우선도 부여
def node_assing_priority(state : EmailState):
    scores = {
        "urgent" : 10,
        "normal" : 5,
        "spam" : 1,
    }

    return {
        "priority_score" : scores[state["category"]] # 카테고리에 따라 점수화
    }

# 응답 초안 작성
def node_draft_response(state : EmailState) -> EmailState:
    responses = {
        "urgent" : "i will answer you as fast as i can",
        "normal" : "i'll get back to you soon",
        "spam" : "Go away!"
    }

    return {
        "response" : responses[state["category"]]
    }

graph_builder = StateGraph(EmailState)

graph_builder.add_node("node_categorize_email",node_categorize_email)
graph_builder.add_node("node_assing_priority",node_assing_priority)
graph_builder.add_node("node_draft_response",node_draft_response)


graph_builder.add_edge(START, "node_categorize_email")
graph_builder.add_edge("node_categorize_email","node_assing_priority")
graph_builder.add_edge("node_assing_priority","node_draft_response")
graph_builder.add_edge("node_draft_response", END)

graph = graph_builder.compile(checkpointer=checkpointer)