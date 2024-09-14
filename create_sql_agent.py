from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_openai import ChatOpenAI 
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import langchain
langchain.debug = True

db_path = './sqlagent/titanic.db'
#创建数据库连接
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')

#实例化LLM并获取sql tools
llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=3000, temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#tools = toolkit.get_tools()

from langchain_core.tools import tool
@tool
def answer_dr(input):
    """不需要查询数据库时直接回答用户的问题或者回复不知道时调用这个工具"""
    if isinstance(input, str):
        output = "直接把Action input作为final answer"
    else:
        output = "直接根据user的问题生成final answer"

    return output

import datetime
@tool
def get_datetime():
    """使用这个函数可以获取当前日期"""
    #python功能获取当前日期时间
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return str(date)

tools = [answer_dr,get_datetime]
#
#创建agent
from langchain import hub
from langchain_community.agent_toolkits import create_sql_agent
#prompt = hub.pull("hwchase17/react")
#prefix =""""1"""
suffix = """answer_dr.Begin!\n\n"Previous conversation history:\n""{chat_history}\n\n"Question: {input}\nThought: 
I should look at the tables in the database to see what I can query. Then I should query the schema of the most relevant tables.
\n{agent_scratchpad}"""
#suffix = """'Begin!\n\nQuestion: {input}\nThought: \n{agent_scratchpad}'"""
from langchain.prompts import PromptTemplate
template ="""You are an agent designed to interact with a SQL database.\nGiven an input question, create 
a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
\nUnless the user specifies a specific number of examples they wish to obtain, always limit your query 
to at most 10 results.\nYou can order the results by a relevant column to return the most interesting 
examples in the database.\nNever query for all the columns from a specific table, only ask for the relevant
columns given the question.\nYou have access to tools for interacting with the database.\nOnly use the below
tools. Only use the information returned by the below tools to construct your final answer.\nYou MUST double
check your query before executing it. If you get an error while executing a query, rewrite the query and try
again.\n\nDO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n\n\n{tools}\n\n
\nIf the question does not seem related to the database, just call the answer_dr and the action input should be "你好，我不能和你聊天哦，如果你要查询数据库，请输入你要查询的内容，否则我无法回答你的问题".\n
Use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think
about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to 
the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat
N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\n
answer_dr.Begin!\n\n"Previous conversation history:\n""{chat_history}\n\n"Question: {input}\nThought: \n
I should look at the tables in the database to see what I can query. Then I should query the schema of the 
most relevant tables.\n\n{agent_scratchpad}"""

prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools', 'chat_history'],
    template=template)
agent = create_sql_agent(llm, toolkit,extra_tools=tools,prompt=prompt)



from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#agent 添加记忆功能
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


while  True:
    query = input("请输入查询语句：")
    if query == "clear":
        store.clear()
        continue
    if query == "exit":
        break
    response = agent_with_chat_history.invoke({"input":query},config={"configurable": {"session_id": "123"}})["output"]
    print(100*"=")
    print(response)