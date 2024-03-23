import os
import secrets

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from tools import TOOLS

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview", streaming=True)

random_string = secrets.token_hex(12)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful assistant. Think step by step. Use the chat history.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", "{input}"),
    ]
)

agent = create_openai_functions_agent(llm, TOOLS, prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

async def run_agent_executor(input_text, history):
    output_lines = ""
    function_invocations = []

    async for chunk in agent_with_chat_history.astream({"input": input_text}, config={"configurable": {"session_id": random_string}}):
        if 'actions' in chunk:
            for action in chunk['actions']:
                # Log the function invocation
                function_name = action.tool
                function_args = action.tool_input
                invocation_log = f"Invoked: {function_name} with arguments {function_args}"
                function_invocations.append(invocation_log)

        if 'steps' in chunk:
            for step in chunk['steps']:
                # Log the result of the function invocation
                result_log = f"Result of {step.action.tool}: {step.observation}"
                function_invocations.append(result_log)

        if 'output' in chunk:
            # Add the final output to the log
            final_output = f"Final Output: {chunk['output']}"
            function_invocations.append(final_output)
        output_lines += "\n".join(function_invocations) + "\n"

        # Reset for the next chunk
        function_invocations = []

        yield output_lines