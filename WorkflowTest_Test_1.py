# 값을 주고 예상값을 검증하는 테스트

# 테스트 실행방법 1
# uv run pytest .\WorkflowTest_Test_1.py --vv
# vv 는 verbose(상세함) 을 뜻한다. 어떤 테스트가 실행되고 있는지, 이름ㄱ같은 정보나 그런것들
    # 구지 안해도 되긴하다

# 테스트 실행방법 2
# import pytest 하고
# 데코레이터 달기

import pytest
from WorkflowTest_Graph_1 import graph

# parametrize 에 넣어준 인수를, 대상이 되는 함수에 넣어줄 것이다.
# 그리고 2번째 인수로 넣어준 튜플 배열 입력 순서대로 함수를 호출한다
@pytest.mark.parametrize("email, expected_category, expected_score",[
    ("this is urgent", "urgent", 10),
    ("i wanna talk with you", "normal", 5),
    ("i have a offer for you", "spam", 1)
])
def test_full_graph(email, expected_category, expected_score) :

    result = graph.invoke({"email" : email}, config={"configurable":{"thread_id":1}})

    # assert 조건 // 조건이 참이 아니면 경고문을 준다
    
    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score


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
        {"category" : "spam"},
        config={"configurable":{"thread_id":1}}
    )

    assert result["priority_score"] == 1

    # node_draft_response
    result = graph.nodes["node_draft_response"].invoke(
        {"category":"spam"},
        config={"configurable":{"thread_id":1}}
    )

    assert "Go away" in result["response"]

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

    assert result["priority_score"] == 1
    assert "Go away" in result["response"]
    