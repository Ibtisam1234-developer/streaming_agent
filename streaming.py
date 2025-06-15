import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import dotenv
import os

dotenv.load_dotenv()
# Gemini API key and client setup
gemini_api_key = os.getenv("GOOGLE_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Set as default client
set_default_openai_client(client=external_client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# Create the agent
agent = Agent(
    name="Joker",
    instructions="You are a helpful assistant.",
    model=model
)

@cl.on_message
async def on_message(message: cl.Message):
    # Run the agent with streamed output
    result = Runner.run_streamed(agent, input=message.content)

    msg = cl.Message(content="")  # Initialize a Chainlit message
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta
            await msg.stream_token(delta)
    await msg.send()
