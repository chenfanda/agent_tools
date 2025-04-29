from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from typing import Dict, List, Tuple, Any, Optional


class MCPClientManager:
    """
    A class to manage multiple MCP client sessions and interact with their tools.
    """

    def __init__(self, server_params_list):
        """
        初始化 MCPClientManager 实例，准备连接多个 MCP Server。

        参数:
            server_params_list (List[StdioServerParameters]): 每个 MCP 服务的连接参数。
        """
        self.server_params_list = server_params_list
        self.sessions = []
        self.session_index = {}
        self.tool_descriptions = ""

    async def initialize(self):
        """
        初始化所有 MCP server 的连接，并收集所有工具的描述信息。

        返回:
            Tuple[str, Dict[str, int]]: 
                - 所有工具的描述信息字符串。
                - 工具名到对应 session 索引的映射。
        """
        try:
            self.sessions = await asyncio.gather(
                *(MCPClientManager.mcp_client(p) for p in self.server_params_list)
            )
        except Exception as e:
            print(f"create session failed: {e}")

        for idx, session_data in enumerate(self.sessions):
            try:
                response = await session_data["session"].list_tools()
                descriptions = "\n".join(
                    f"{line.name}:{line.description}。inputSchema:{line.inputSchema}"
                    for line in response.tools
                )

                for line in response.tools:
                    self.session_index[line.name] = idx

                if not self.tool_descriptions:
                    self.tool_descriptions = descriptions
                else:
                    self.tool_descriptions += "\n" + descriptions
            except Exception as e:
                print(f"Error initializing session {idx}: {e}")

        return self.tool_descriptions, self.session_index

    @classmethod
    async def mcp_client(cls, server_params):
        """
        创建一个 MCP 客户端连接，初始化会话。

        参数:
            server_params (StdioServerParameters): MCP server 的连接参数。

        返回:
            Dict[str, Any]: 包含已打开的 MCP session、读写通道和客户端对象。
        """
        client = stdio_client(server_params)
        read, write = await client.__aenter__()

        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        return {"session": session, "read": read, "write": write, "client": client}

    @classmethod
    async def close_mcp_session(cls, session, read, write, client):
        """
        关闭 MCP 客户端会话和其底层资源。

        参数:
            session (ClientSession): MCP 会话实例。
            read, write: 通信通道（异步）。
            client: MCP 客户端对象。
        """
        try:
            await client.__aexit__(None, None, None)
            await session.__aexit__(None, None, None)
        except Exception:
            await read.aclose()
            await write.aclose()

    async def run_tool(self, name: str, arguments: Optional[dict]):
        """
        执行指定的工具，传入相应的输入参数。

        参数:
            name (str): 工具名称。
            input_dict (dict | None): 工具的输入参数。

        返回:
            str | None: 工具执行后的文本结果，或 None（如果出错）。

        异常:
            ValueError: 如果提供的工具名在当前 session 中不存在。
        """
        if name not in self.session_index:
            raise ValueError(f"Tool '{name}' does not exist")

        idx = self.session_index[name]
        try:
            session = self.sessions[idx].get("session")
            content = await session.call_tool(name, arguments)
            return content.content[0].text
        except Exception as e:
            print(f"Tool execution error: {e}")
            return None

    async def run_tools_parallel(self, tools_dict: List[Dict[str, Any]]):
        """
        并行运行多个工具。

        参数:
            tools_dict (List[dict]): 每个 dict 包含 `name` 和 `input_dict` 参数。

        返回:
            List[str | None]: 每个工具运行结果组成的列表，出错则返回 None。
        """
        try:
            response = await asyncio.gather(
                *(self.run_tool(**tool) for tool in tools_dict)
            )
            return response
        except Exception as e:
            print(f"Parallel tool execution error: {e}")
            return None

    async def close_all(self):
        """
        关闭所有 MCP session 和其对应的资源。
        """
        try:
            await asyncio.gather(
                *(MCPClientManager.close_mcp_session(**p) for p in self.sessions)
            )
        except Exception as e:
            print(f"Error closing sessions: {e}")


async def main():
    manager = None
    try:
        server_params_list = [
            StdioServerParameters(command="uv", args=["run", "/oper/ch/code/chromadb_mcp.py"]),
            StdioServerParameters(command="npx", args=["--yes", "@modelcontextprotocol/server-filesystem", '/oper/ch/code']),
            StdioServerParameters(command="uv", args=["run", "-m", "mcp_server_fetch"]),
            StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-puppeteer"])
        ]

        manager = MCPClientManager(server_params_list)
        descriptions, _ = await manager.initialize()
        print("Available tools:\n", descriptions)

        if descriptions:
            result = await manager.run_tool("get_collection_info", None)
            print("Tool result:\n", result)

            result = await manager.run_tool("write_file", {
                "path": "/oper/ch/code/test.txt",
                "content": "床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。"
            })
            print("Tool result:\n", result)

            parallel_tasks = [
                {"name": "get_collection_info", "input_dict": None},
                {"name": "write_file", "input_dict": {
                    "path": "/oper/ch/code/test_parallel.txt",
                    "content": "并行写入内容测试"
                }},
                # 如果有 read_file 工具可用，也可以添加：
                {"name": "read_file", "input_dict": {
                    "path": "/oper/ch/code/test.txt"
                }},
                  ]

            print("\nRunning tools in parallel...\n")
            results = await manager.run_tools_parallel(parallel_tasks)
            for i, r in enumerate(results):
                print(f"Parallel Tool Result {i+1}:", r)
    except Exception as e:
        print(f"Main function error: {e}")
    finally:
        if manager:
            await manager.close_all()
        print("Task completed")


if __name__ == "__main__":
    asyncio.run(main())
