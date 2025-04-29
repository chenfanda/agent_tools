from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import asyncio


class MCPClientManager:
    def __init__(self, server_params_list):
        self.server_params_list = server_params_list
        self.sessions = []
        self.session_index = {}
        self.tool_descriptions = ""
        self._exit_stack = AsyncExitStack()
        self._subprocesses = []

    async def _create_mcp_client(self, server_params):
        client_gen = stdio_client(server_params)
        read, write = await self._exit_stack.enter_async_context(client_gen)
        session = ClientSession(read, write)
        await self._exit_stack.enter_async_context(session)
        await session.initialize()

        # ✅ subprocess 通过 client_gen.proc 获取
        if hasattr(client_gen, "proc"):
            self._subprocesses.append(client_gen.proc)

        return {"session": session, "read": read, "write": write, "client": client_gen}

    async def initialize(self):
        self.sessions = await asyncio.gather(
            *(self._create_mcp_client(p) for p in self.server_params_list)
        )

        self.tool_descriptions = []
        self.session_index = {}

        for idx, tool in enumerate(self.sessions):
            response = await tool["session"].list_tools()
            descriptions = "\n".join(
                f"{line.name}:{line.description}。inputSchema:{line.inputSchema}"
                for line in response.tools
            )
            self.session_index.update({line.name: idx for line in response.tools})
            self.tool_descriptions.append(descriptions)

        self.tool_descriptions = "\n".join(self.tool_descriptions)
        return self.tool_descriptions, self.session_index

    async def run_tool(self, name, input_dict):
        if name not in self.session_index:
            raise ValueError(f"工具 '{name}' 不存在")

        idx = self.session_index[name]
        session = self.sessions[idx]["session"]
        content = await session.call_tool(name, input_dict)
        return content.content[0].text

    async def __aenter__(self):
        await self._exit_stack.__aenter__()
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # ✅ 先关闭所有资源
        await self._exit_stack.aclose()

        # ✅ 然后主动关闭所有子进程
        for proc in self._subprocesses:
            try:
                if proc.returncode is None:
                    proc.terminate()
                    await asyncio.sleep(0.3)
                if proc.returncode is None:
                    proc.kill()
            except Exception as e:
                print(f"⚠️ 子进程关闭异常: {e}")

response = {
    "name": "write_file",
    "input_dict": {
        "path": "/oper/ch/code/test.txt",
        "content": "床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。"
    }
}

#server_params_list = [
#    StdioServerParameters(command="npx", args=["--yes", "@modelcontextprotocol/server-filesystem", '/oper/ch/code']),
#    StdioServerParameters(command="uvx", args=["mcp-server-fetch"]),
#    StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-puppeteer"])
#]
server_params_list = [StdioServerParameters(command="uv",args=["run","/oper/ch/code/chromadb_mcp.py","--ollama_model","bge-m3","--collection_name","my_ollama_documents"])]
async def example_usage():
    async with MCPClientManager(server_params_list) as manager:
        print("可用工具:", manager.tool_descriptions)
        result = await manager.run_tool(response['name'], response['input_dict'])
        print("工具结果:", result)

asyncio.run(example_usage())
