from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_openai import ChatOpenAI 
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import langchain
#langchain.debug = True

db_path = './sqlagent/titanic.db'
#创建数据库连接
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')

#实例化LLM并获取sql tools
llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=1000, temperature=0,verbose=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


from langchain_core.tools import tool
#React框架必须要thought,Action,Observation,Final Answer,为不需要调用工具的清空定义一个直接回答的工具。
@tool
def answer_dr(input):
    """查看数据库后，问题与数据库无关时直接回答用户的问题"""
    output = "直接把Action input作为final answer"
    return output


import datetime
@tool
def get_datetime():
    """使用这个函数可以获取当前日期"""
    #python功能获取当前日期时间
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return str(date)


tools.append(answer_dr)
tools.append(get_datetime)
#创建agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

agent_executor = create_react_agent(llm, tools, checkpointer=memory,debug=True)


config = {"configurable": {"thread_id": "abc123"}}
from langchain_core.messages import HumanMessage

while  True:
    query = input("请输入查询语句：")
    if query == "clear":
        memory.clear()
        continue
    if query == "exit":
        break
    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]},config=config)["messages"]
    response = response[-1].content
    print(response)
