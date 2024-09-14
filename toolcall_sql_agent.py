from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_openai import ChatOpenAI 
from langchain.agents import AgentExecutor
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage

import langchain
#langchain.debug = True
db_path = './sqlagent/titanic.db'
#创建数据库连接
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')
llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=3000, temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


tools = toolkit.get_tools()

# response = model_with_tools.invoke([HumanMessage(content="请查一下现在有多少条告警!")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
#print(prompt.messages)
#prompt = hub.pull("hwchase17/react")
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    verbose=True
)



while  True:
    query = input("请输入查询语句：")
    if query == "clear":
        store.clear()
        continue
    if query == "exit":
        break
    response = agent_with_chat_history.invoke({"input":query},config={"configurable": {"session_id": "<foo>"}})["output"]
    print(response)