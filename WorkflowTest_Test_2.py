# 값을 주고 예상값을 검증하는 테스트

# 테스트 실행방법 1
# uv run pytest .\WorkflowTest_Test_1.py --vv
# vv 는 verbose(상세함) 을 뜻한다. 어떤 테스트가 실행되고 있는지, 이름ㄱ같은 정보나 그런것들
    # 구지 안해도 되긴하다

# 테스트 실행방법 2
# import pytest 하고
# 데코레이터 달기

import pytest
from WorkflowTest_Graph_2 import graph

from pydantic import BaseModel, Field, conset

# opeai 모델 사용하고 있다면
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
# llm = ChatOllama(model="qwen2.5:14b", temperature=0.7)
llm = init_chat_model("openai:gpt-4o")

# 유사도 측정 점수
class ST_SimilarityScore(BaseModel):

    # 예제와 얼마나 비슷한지 점수를 매긴다
    similarity_score : int = Field(
        description="How simmar is the response to the examples?",
        gt = 0,
        lt = 100
    )

RESPONSE_EXAMPLES = {
    "urgent": [
        "Thank you for your urgent message. We are addressing this immediately and will respond as soon as possible.",
        "We've received your urgent request and are prioritizing it. Our team is on it right away.",
        "This urgent matter has our immediate attention. We'll respond promptly.",
    ],
    "normal": [
        "Thank you for your email. We'll review it and get back to you within 24-48 hours.",
        "We've received your message and will respond soon. Thank you for reaching out.",
        "Thank you for contacting us. We'll process your request and respond shortly.",
        "Thank you for the update. I will review the information and follow up as needed.",
        "Thank you for the update on the project status. I will review and follow up by the end of the week.",
        "Thanks for sharing this update. We'll review and respond accordingly.",
    ],
    "spam": [
        "This message has been flagged as spam and filtered.",
        "This email has been identified as promotional content.",
        "This message has been marked as spam.",
    ],
}

def judge_response(response : str, category : str) :
    
    similarityScore_llm = llm.with_structured_output(ST_SimilarityScore)

    examples = RESPONSE_EXAMPLES[category]
    
    similarityScore = similarityScore_llm.invoke(
        f"""
            Score how similar this response is to the examples.

            Category: {category}

            Examples:
            {"\n".join(examples)}

            Response to evaluate:
            {response}

            Scoring criteria:
            - 90-100: Very similar in tone, content, and intent
            - 70-89: Similar with minor differences
            - 50-69: Moderately similar, captures main idea
            - 30-49: Some similarity but missing key elements
            - 0-29: Very different or inappropriate
        
        """
    )

    return similarityScore.similarity_score



# parametrize 에 넣어준 인수를, 대상이 되는 함수에 넣어줄 것이다.
# 그리고 2번째 인수로 넣어준 튜플 배열 입력 순서대로 함수를 호출한다
@pytest.mark.parametrize("email, expected_category, min_score, max_score",[
    ("this is urgent", "urgent", 8,10),
    ("i wanna talk with you", "normal", 4,7),
    ("i have a offer for you", "spam", 1,3)
])
def test_full_graph(email, expected_category, min_score, max_score) :

    result = graph.invoke({"email" : email}, config={"configurable":{"thread_id":1}})

    # assert 조건 // 조건이 참이 아니면 경고문을 준다
    
    assert result["category"] == expected_category
    assert min_score <= result["priority_score"] <= max_score


def test_individual_nodes():

    # 노드의 이름으로 노드를 참조 할 수 있다

    # node_categorize_email
    result = graph.nodes["node_categorize_email"].invoke(
        {"email" : "i have offer to you"},
        config={"configurable":{"thread_id":1}}
    )

    assert result["category"] == "spam"

    # node_assing_priority
    result = graph.nodes["node_assing_priority"].invoke(
        {"category" : "spam", "email":"buy this pot."},
        config={"configurable":{"thread_id":1}}
    )

    assert 1 <= result["priority_score"] <= 3

    # node_draft_response
    result = graph.nodes["node_draft_response"].invoke(
        {
            "category":"spam", 
            "email" : "Get rich quick!! i have a pyramid scheme for you", 
            "priority_score" : 1},
        config={"configurable":{"thread_id":1}}
    )

    similarity_score = judge_response(result["response"], "spam")
    assert similarity_score >= 70

# 만약 interrupt(중단) 했다가 다시 시작하고싶을때를 가정
# 또는 비용이 많이드는 노드가 있는경우, 그것을 이미 실행했다 가정
# 부분 테스트를 많이 쓴다.
def test_partial_nodes() :

    # 이미 node_categorize_email 이 실행되었다고 가정하고
    # state 를 업데이트
    graph.update_state(
        config={
            "configurable" : {"thread_id" : "1"}
        },
        # state 의 상태 지정
        values={
            "email" : "please check out this offer",
            "category" : "spam"
        },
        # 현재 어떤 노드 인것처럼 설정할것인지
        as_node="node_categorize_email"
    )

    # 입력은 필요없으니 None 을 준다
    # 그리고 node_categorize_email 이 실행되고 중단된것처럼 꾸미기 위해서
        # confg={"configurable":{"thread_id" : "1"}} 로 설정한다.
    # interrupt_after 는 특정노드가 끝난뒤에 중단하도록 설정할 수도 있다.
        # interrupt_before 도 있다.
    result = graph.invoke(
        None, 
        config={
            "configurable" : {"thread_id" : "1"}
        },
        interrupt_after ="node_draft_response"
    )

    assert 1 <= result["priority_score"] <= 3
    