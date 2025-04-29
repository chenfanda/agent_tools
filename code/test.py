# import asyncio
# from pathlib import Path
# from autogen_ext.models.openai import OpenAIChatCompletionClient
# from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
# from autogen_agentchat.agents import AssistantAgent
# from autogen_core import CancellationToken
# model_client = OpenAIChatCompletionClient(
#     model="qwen2.5:14b-instruct-q8_0",
#     base_url="http://localhost:11434/v1",
#     api_key="placeholder",
#     model_info={
#         "vision": False,
#         "function_calling": True,
#         "json_output": True,
#         "family": "unknown",
#     },
# )
# async def main() -> None:
#     # Setup server params for local filesystem access
#     desktop = str(Path.home() / "Desktop")
#     server_params = StdioServerParams(
#         command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/oper/ch/code"]
#     )

#     # Get all available tools from the server
#     tools = await mcp_server_tools(server_params)

#     # Create an agent that can use all the tools
#     agent = AssistantAgent(
#         name="file_manager",
#         model_client=model_client,
#         tools=tools,  # type: ignore
#     )

#     # The agent can now use any of the filesystem tools
#     await agent.run(task="写一首杜甫的诗，然后写到test.txt ", cancellation_token=CancellationToken())


# if __name__ == "__main__":
#     asyncio.run(main())
import asyncio
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo , UserMessage
import os 
import subprocess
# from pyvirtualdisplay import Display

# # Start a virtual display
# display = Display(visible=0, size=(1024, 768))
# display.start()
# os.environ["DISPLAY"] = ":99"
# Start Xvfb
# os.environ["DISPLAY"] = ":1"
# os.environ['XAUTHORITY']='/run/user/1000/gdm/Xauthority'
env_vars = os.environ
# env_dict={}
# for line in ["APPDATA",\
#              "HOMEDRIVE",\
#              "HOMEPATH",\
#              "LOCALAPPDATA",\
#              "PATH",\
#              "PROCESSOR_ARCHITECTURE",\
#              "SYSTEMDRIVE",\
#              "SYSTEMROOT",\
#              "TEMP",\
#              "HOME",\
#              "USERNAME",\
#              "USERPROFILE"] :
#     if os.environ.get(line):
#         env_dict[line]=os.environ[line]
# env_dict['DISPLAY']=":1"
# env_dict['XAUTHORITY']='/run/user/1000/gdm/Xauthority'
async def main():
    # server_params = StdioServerParams(
    #     command="mcp", args=[ "run", "/oper/ch/code/mcp_service.py"]
    # )
    server_params = StdioServerParams(
        command="mcp", args=[ "run","/oper/ch/code/mcp_service.py" ],
        env=env_vars
    )
    tools = await mcp_server_tools(server_params)
    for tool in tools:
        # print(tool.name)
        print(f"Tool Name: {tool.name}, Description: {tool.description}")
if __name__ == "__main__":
    asyncio.run(main())

