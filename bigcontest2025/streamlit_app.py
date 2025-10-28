import streamlit as st
import asyncio

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from PIL import Image
from pathlib import Path

# 환경변수
ASSETS = Path("assets")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

system_prompt = """너는 친절한 마케팅 상담가야. 지점의 데이터를 분석하고 상권경쟁보다는 상생하는 분위기의 마케팅 전략을 추천해줘. 

분석 순서는 다음과 같아.

1. 상권 특징 분석 (현재 지점의 상권의 모든 가맹점 데이터를 이용하여 분석)
   - 고객 유입 특성: 상권의 모든 가맹점의 평균 `거주이용고객비율`, `직장이용고객비율`, `유동인구이용고객비율`을 바탕으로 상권의 유형(예: 주거형, 오피스형, 혼합형)을 정의해줘. 
   - `고객유형_타입`은 가맹점의 특성이야 위에서 나눈 상권의 유형과 비교해줘
   - 주요 고객층: `남성/여성 연령대별 고객비중` 데이터를 분석해서 이 상권의 핵심 고객이 누구인지 알려줘.

2. 현재 성과 진단 (동일 업종 비교)
   지점의 성과를 `동일업종` 및 `동일상권` 평균과 비교해서 진단해줘.
   - 매출/이용 현황: `매출금액구간`, `매출건수구간`, `객단가구간`이 현재 어떤 수준이야?
   - 가게의 성장세 : 시간의 흐름에 따라 매출이 성장중인지, 하락중인지 판단하여 꺾은선 그래프로 나타내줘
   - 경쟁사 비교: `동일업종내매출순위비율`, `동일상권내매출순위비율`은 몇 %야? 상위권이야, 하위권이야?
   - 고객 관리 현황: `재방문고객비중`, `신규고객비중`은 어때?
   - 타깃 고객 : 현재 상권의 주요 연령/성별 층에 맞는 고객들이 오고있어 상권의 주요 고객층의 비율과 내 가맹점의 주요 고객 비율이랑 비교해줘

3. 종합 마케팅 전략 추천
   위 1번(상권 특징)과 2번(현재 성과)을 종합해서, [가맹점명] 지점이 당장 실행할 수 있는 구체적인 마케팅 아이디어를 3가지 추천해줘.

답변 규칙 : 중요한 부분은 굵게 중앙선은 하지마 그리고 전체지표를 나타낼 떄에는 표를 이용해서 표현해줘"""


greeting = "마케팅이 필요한 가맹점을 알려주세요  \n(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"

# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사")

def clear_chat_history():
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])  # 비율 조정 가능
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더
st.title("신한카드 소상공인 🔑 비밀상담소")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
st.image(load_image("image_gen3.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

# 초기 메시지 화면 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content.replace("<br>", "  \n"))

# LLM 모델 선택
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # 최신 Gemini 2.5 Flash 모델
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

# MCP 서버 파라미터(환경에 맞게 명령 수정)
server_params = StdioServerParameters(
    command="uv",
    args=["run","mcp_server.py"],
    env=None
)

# 사용자 입력 처리
async def process_user_input():
    """사용자 입력을 처리하는 async 함수"""
    async with stdio_client(server_params) as (read, write):
        # 스트림으로 ClientSession을 만들고
        async with ClientSession(read, write) as session:
            # 세션을 initialize 한다
            await session.initialize()

            # MCP 툴 로드
            tools = await load_mcp_tools(session)

            # 에이전트 생성
            agent = create_react_agent(llm, tools)

            # 에이전트에 전체 대화 히스토리 전달
            agent_response = await agent.ainvoke({"messages": st.session_state.messages})
            
            # AI 응답을 대화 히스토리에 추가
            ai_message = agent_response["messages"][-1]  # 마지막 메시지가 AI 응답

            return ai_message.content
            

# 사용자 입력 창
if query := st.chat_input("가맹점 이름을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=query))
    render_chat_message("user", query)

    with st.spinner("Thinking..."):
        try:
            # 사용자 입력 처리
            reply = asyncio.run(process_user_input())
            st.session_state.messages.append(AIMessage(content=reply))
            render_chat_message("assistant", reply)
        except* Exception as eg:
            # 오류 처리
            for i, exc in enumerate(eg.exceptions, 1):
                error_msg = f"오류가 발생했습니다 #{i}: {exc!r}"
                st.session_state.messages.append(AIMessage(content=error_msg))
                render_chat_message("assistant", error_msg)
