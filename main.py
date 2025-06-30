import os 
from dotenv import load_dotenv
from typing import cast , List
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool
from agents.run_context import RunContextWrapper


load_dotenv()

AI = os.getenv("GEMINI_API_KEY")

if not AI:
    raise ValueError(
        "api key not found"
    )

@cl.set_starters
async def set_starts() -> List[cl.Starter]:
    return[
        cl.Starter(
            label="Greetings",
            message="HEllo what can I do for you today?",
            icon="🤖",
        ),
        cl.Starter(
            label="weather",
            message="find the weather in islamabad",

        )
    ]


class MyContent:
    def __init__(self, user_id:str):
        self.user_id = user_id
        self.seem_messages = []


@function_tool
@cl.step(type="weather tool")
def get_weather(location:str , unit:str = "C") -> str:
    """
    fetch the weather of a given location returning a short description.

    """
    return f'the weather in {location} is 18 degrees {unit}.'

@function_tool
@cl.step(type="greeting tool")
def greet_user(context: RunContextWrapper[MyContent], greeting:str) -> str:
    user_id = context.context.user_id
    return f'Hello {user_id}, you said {greeting}'


@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=AI,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model = "gemini-2.0-flash", 
        openai_client= external_client,
    )

    config = RunConfig(
        model = model,
        model_provider=external_client,
        tracing_disabled= True,
    )


    cl.user_session.set("chat_history", [])

    cl.user_session.set("config", config)
    agent: Agent = Agent(
        name= "Assitant",
        tools = [greet_user, get_weather],
        instructions="you are help assitant . call greet_user tool to greet the user.  always greet the user when session starts.",
        model = model

    )

    cl.user_session.set("agent", agent)

    await cl.Message(
        content= "welcome to Nihal khan Ghauri chatbot. How can I help you today?"
    ).send()

    @cl.on_message
    async def main(message: str):
        ''' process inconming messsage and gererate response'''

        msg = cl.Message(
            content= "Thinking...",
        )

        await msg.send()

        agent: Agent = cast(Agent , cl.user_session.get("agent"))
        config: RunConfig = cast(RunConfig , cl.user_session.get("config "))

        history = cl.user_session.get("chat_history") or []

        history.append(
            {
                "role":"user",
                "content": message.content
            }
        )

        my_ctx = MyContent(user_id="nihal khan ghauri")

        try:
            print("\n[calling agent with context]\n", history, "\n")
            result = Runner.run_sync(
                agent,
                history,
                run_config=config,
                context=my_ctx
            )

            response_content =result.final_output

            msg.content = response_content
            await msg.update()

            history.append(
                {
                    "role": "assistant",
                    "content": response_content
                }
            )


            cl.user_session.set("chat_history", history)

            print(f'user: {message.content}')
            print(f"Assistant: {response_content}")

        except Exception as e:
            msg.content = f"Error : {str(e)}"
            await msg.update()
            print(f'Error : {str(e)}')





