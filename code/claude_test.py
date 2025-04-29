from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from contextlib import asynccontextmanager

class MCPClientManager:
    """
    A class to manage MCP client sessions and tool interactions.
    """
    
    def __init__(self, server_params_list):
        """
        Initialize the MCPClientManager with a list of server parameters.
        
        Args:
            server_params_list: List of server parameters for connecting to MCP services
        """
        self.server_params_list = server_params_list
        self.sessions = []
        self.session_index = {}
        self.tool_descriptions = ""
        
    @asynccontextmanager
    async def _create_session(self, server_params):
        """
        Create and initialize a single MCP client session as a context manager.
        
        Args:
            server_params: Parameters for connecting to the server
            
        Yields:
            Dict containing session, client, and connection details
        """
        client = None
        session = None
        try:
            client = stdio_client(server_params)
            read, write = await client.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            
            session_data = {
                "session": session,
                "read": read,
                "write": write,
                "client": client,
                "proc": getattr(client, "proc", None)
            }
            
            yield session_data
            
        finally:
            # Clean up in reverse order
            if session:
                try:
                    await session.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Session exit error: {e}")
            
            if client:
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Client exit error: {e}")
                    
                    # If client exit failed, try to force terminate the process
                    proc = getattr(client, "proc", None)
                    if proc and proc.returncode is None:
                        try:
                            proc.terminate()
                            await asyncio.wait_for(proc.wait(), timeout=2)
                        except (asyncio.TimeoutError, Exception):
                            try:
                                proc.kill()
                                await asyncio.wait_for(proc.wait(), timeout=1)
                            except Exception:
                                pass  # Last resort, ignore if we can't kill it
    
    async def initialize(self):
        """
        Initialize connections to all MCP servers and fetch tool descriptions.
        
        Returns:
            Tuple of (tool_descriptions, session_index)
        """
        self.sessions = []
        try:
            # Create sessions one at a time
            for p in self.server_params_list:
                async with self._create_session(p) as session_data:
                    # Store a reference to the session
                    self.sessions.append(session_data)
                    
                    # Fetch tool descriptions
                    response = await session_data['session'].list_tools()
                    descriptions = "\n".join(
                        f"{line.name}:{line.description}。inputSchema:{line.inputSchema}" 
                        for line in response.tools
                    )
                    
                    # Update session index
                    for line in response.tools:
                        self.session_index[line.name] = len(self.sessions) - 1
                    
                    # Add to tool descriptions
                    if not self.tool_descriptions:
                        self.tool_descriptions = descriptions
                    else:
                        self.tool_descriptions += "\n" + descriptions
            
            return self.tool_descriptions, self.session_index
            
        except Exception as e:
            print(f"Initialization error: {e}")
            # The sessions are already closed by the context manager
            self.sessions = []
            return "", {}
    
    async def run_tool(self, name, input_dict):
        """
        Run a specific tool with the given input.
        
        Args:
            name: Name of the tool to run
            input_dict: Input parameters for the tool
            
        Returns:
            The content returned by the tool
        """
        if name not in self.session_index:
            raise ValueError(f"Tool '{name}' does not exist")
        
        # Create a new session specifically for this tool execution
        idx = self.session_index[name]
        
        async with self._create_session(self.server_params_list[idx]) as temp_session:
            try:
                content = await temp_session['session'].call_tool(name, input_dict)
                return content.content[0].text
            except Exception as e:
                print(f"Tool execution error: {e}")
                return None
    
    async def close_all(self):
        """
        Close all open sessions.
        """
        # Sessions are now managed by context managers, so they should be closed already
        # This method is kept for backward compatibility
        self.sessions = []
        print("All sessions closed")


async def main():
    manager = None
    try: 
        server_params_list = [
            StdioServerParameters(command="uv", args=["run", "/oper/ch/code/chromadb_mcp.py"]),
#            StdioServerParameters(command="npx", args=["--yes", "@modelcontextprotocol/server-filesystem", '/oper/ch/code']),
        ]
        
        manager = MCPClientManager(server_params_list)
        descriptions, _ = await manager.initialize()
        print("Available tools:", descriptions)
        
        if descriptions:  # Only try to run tools if initialization was successful
            result = await manager.run_tool("get_collection_info", None) 
            print("Tool result:", result)
            
            result = await manager.run_tool("write_file", {
                "path": "/oper/ch/code/test.txt",
                "content": "床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。"
            }) 
            print("Tool result:", result)
            
    except Exception as e:
        print(f"Main function error: {e}")
    finally:
        if manager:
            await manager.close_all()
        print('Task completed')

if __name__ == "__main__":
    asyncio.run(main())
