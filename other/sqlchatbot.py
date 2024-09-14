from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START

#构造mysql的url
# database_uri = "mysql+pymysql://root:1qaz!QAZ@192.168.0.78/faultdata_bj?connect_timeout=10"
# engine_args = {
#     # 这里可以添加额外的引擎参数，例如连接池大小等
#     'pool_size': 10,
#     'echo': True  # 如果你想要打印出所有生成的 SQL 语句，可以设置为 True
# }


db = SQLDatabase.from_uri("sqlite:///alarm.db")

llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=1000, temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

def create_agent(llm, tools):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant,you should answer as best as you can.Use the following format:\n\nQuestion: the input question you must answer\n"
                "Thought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\n"
                "Action Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation"
                "can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\n"
                "Begin!\n\nQuestion: {input}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    #prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)
tools = toolkit.get_tools()


import operator
from typing import Annotated, Sequence, TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

import functools

from langchain_core.messages import AIMessage


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


sql_agent = create_agent(llm, tools)

sql_node = functools.partial(agent_node, agent=sql_agent, name="sql_agenter")

from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)

# Either agent can decide to end
from typing import Literal


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("sql_agenter", sql_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "sql_agenter",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)