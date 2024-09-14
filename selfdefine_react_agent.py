from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
from langchain_openai import ChatOpenAI 
from langchain.agents import AgentExecutor
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import langchain
#langchain.debug = True
db_path = 'titanic.db'
#创建数据库连接
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')

#实例化LLM并获取sql tools
llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=1000, temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
@tool
def answer_dr(input):
    """不需要查询数据库时直接回答用户的问题"""
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


tools.append(answer_dr)
tools.append(get_datetime)
#创建agent
from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

# prompt = hub.pull("hwchase17/react")
# print(prompt)

from langchain.prompts import PromptTemplate


# # 定义 ChatPromptTemplate
prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools', 'chat_history'],
    template=(
        "Answer the following questions as best you can."
        "You have access to the following tools:\n\n"
        "{tools}\n\n"
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question\n\n"
        "Begin!\n\n"
        # 在这里加入历史对话
        "Previous conversation history:\n"
        "{chat_history}\n\n"
        # 用户的输入部分
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )
)



agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True#handle_parsing_errors=True,
#    verbose=True,
)


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
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    verbose=True,
)
import gradio as gr
# 定义聊天框的更新函数
def chatbot_response(history,query):
    # 获取机器人回复
    response = agent_with_chat_history.invoke({"input":query},config={"configurable": {"session_id": "<foo>"}})["output"]
    # 将用户消息和机器人的回复加入历史记录
    history.append((query, response))
    # 返回更新后的历史记录和一个空字符串来清空输入框
    return history, gr.update(value="")



if __name__ == "__main__":

    # 定义 Gradio 界面
    with gr.Blocks() as interface:
        gr.Markdown("# 参数智能查询系统")
        gr.Markdown("请输入问题，然后点击提交或按下回车。")
        
        chatbot = gr.Chatbot()  # 创建聊天框
        msg_input = gr.Textbox(placeholder="输入问题...")  # 输入框
        submit_btn = gr.Button("提交")  # 提交按钮

        # 当用户点击按钮时，触发响应，并清空输入框
        submit_btn.click(chatbot_response, [chatbot, msg_input], [chatbot, msg_input])
        
        # 当用户按下回车键时，触发相同的响应，并清空输入框
        msg_input.submit(chatbot_response, [chatbot, msg_input], [chatbot, msg_input])

    interface.launch()  # 启动 Gradio 界面
