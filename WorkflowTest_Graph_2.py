# llm 버전 그래프프

from typing import Literal, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
checkpointer = MemorySaver()

from dotenv import load_dotenv
load_dotenv()

# llm = ChatOllama(model="qwen2.5:14b", temperature=0.7)
llm = init_chat_model("openai:gpt-4o")

class EmailState(TypedDict):
    email : str
    category : Literal["spam", "normal", "urgent"]
    priority_score : int
    response : str

    

class ST_EmailClassification(BaseModel):
    # Filed 를 사용하면 자료형의 형태를 AI 에게 더 잘 설명할 수 있다.
    category : Literal["spam", "normal", "urgent"] = Field(description="Category of the email")

class ST_PriorityScore(BaseModel):
    # Filed 를 사용하면 자료형의 형태를 AI 에게 더 잘 설명할 수 있다.
    priority_score : int = Field(description="Priority score from 1 to 10", ge=1, le=10)


# 메일 카테고리 분류류
def node_categorize_email(state : EmailState):
    email = state["email"].lower()

    category_llm = llm.with_structured_output(ST_EmailClassification)

    category_res = category_llm.invoke(
        f"""
            Classify this email into one of three categories:
            - urgent : time-sensitive, requires immediate attension
            - normal : regular business communication
            - sapm : promotional, marketing, or unwanted content

            Email : {state['email']}
        """
    )

    return {
        "category" : category_res.category
    }

# 메일 우선도 부여
def node_assing_priority(state : EmailState):
    
    priorityScore_llm = llm.with_structured_output(ST_PriorityScore)

    priorityScore_res = priorityScore_llm.invoke(f"""
        Assign a priority score from 1-10 for this {state['category']} email.

        Consider : 
        - Category : {state['category']}
        - Email content : {state.get("email", "")}

        Guidelines : 
        - Urgent emails : usually 8-10
        - Normal emails : usually 4-7
        - Spam emails : usually 1-3
    """)

    return {
        "priority_score" : priorityScore_res.priority_score
    }

# 응답 초안 작성
def node_draft_response(state : EmailState) -> EmailState:
    # 작성된 초안
    email_draft_res = llm.invoke(
        f"""
            Draft a brief, professional response for this {state['category']} email.

            Original email : {state['email']}
            Category : {state['category']}
            Priority : {state['priority_score']}/10

            Guidelines : 
            - Urgent : Acknowledge urgency, promise immediate attention
            - Normal : Professional acknowledgement, standard timeline
            - Spam : Breif notice that message was filtered

            Keep response under 2 sentences.
        """
    )
    return {
        "response" : email_draft_res.content
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