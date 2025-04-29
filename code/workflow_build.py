import json
from typing import Dict, List, Any, Optional, Union, Literal,ForwardRef
from pydantic import BaseModel, Field
from mcp_client import MCPClientManager
from ollama import Client
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import asyncio
from ollama import AsyncClient
server_params = [StdioServerParameters(command="uv",args=["run","/oper/ch/code/chromadb_mcp.py","--ollama_model","bge-large-zh-v1.5:f16","--collection_name","tools_collection"])]

# 定义数据模型
from pydantic import BaseModel
from enum import Enum
from typing import Dict, Any, List, Union

async def Ollama_Asyncclient(query,prompt=None,model="gemma3:27b",host="http://localhost:11434"):
    client = AsyncClient(host)
    if not prompt:
        message = [{'role': 'user', 'content': query}]
    else:
         message = [{'role': 'user', 'content': prompt+query}]
    response = await client.chat(model = model ,messages = message, format='json')
    return response      

class Instruction(BaseModel):
    prompt: Optional[str] = None  
    query: str
    async def execute(self):
        try:
        
            response = await Ollama_Asyncclient(query = self.query,prompt = self.prompt)
         
            return {"result": response}
        except Exception as e: 
            return {"error": f"{e}" }
            
class Tool(BaseModel):
    name: str
    arguments: Dict[str, Any]
    action: str
    async def execute(self) -> Dict[str, Any]:
        try:
            descriptions,_= await manager.initialize()
            response = await manager.run_tool(self.name,self.arguments)
            await manager.close_all()
            if response:
                return {"tool_name": self.name,
                        "info": f"Executed {self.action} with {self.arguments}",
                        "result": response
                       }
            else:
                  return {"tool_name": self.name,
                          "info": f"Executed {self.action} with {self.arguments}",
                          "result": "tool call is success. "
                       }
        except Exception as e:
            return  {"tool_name": self.name,
                    "info": f"Executed {self.action} with {self.arguments}",
                    "error": str(e)
                   }
            
class ActionInfo(BaseModel):
    action_name: str
    action_type: Union[Tool, Instruction]

class WorkFlow(BaseModel):
    task_name: str
    workflow: List[ActionInfo]

class Evaluation(BaseModel):
    evaluation_previous_goal: bool
    next_goal: str
    last_step: bool
    
class StepInfo(BaseModel):
    step_id: int
    step_name: str
    step_description: str
    actions_name: List[str]
    actions: List[Union[Instruction, Tool, WorkFlow]]
    desire_output: str
    evaluation: Evaluation
    
class TaskExecutionResult(BaseModel):
    success: bool
\n根据提供的工具使用示例和新问题：\n保持action_type.name与示例一致；\n根据新问题，改写：\naction_name（简明描述新任务）；\naction_type.arguments（参数内容应与新问题一致，如文件路径、内容等）；\naction_type.action（用简洁语言描述实际动作）；\n工具使用示例:action_name=\'请作一首诗描写深圳湾海景,然后把诗写入文件 `/oper/ch/mcp_work/test.txt`\' action_type=Tool(name=\'write_file\', arguments={\'path\': \'/oper/ch/mcp_work/test.txt\', \'content\': \'深圳湾畔碧波连，\\n万象澄明映晴天。\\n白鹭翩跹逐浪远，\\n轻舟荡漾戏鸥闲。\\n长桥卧浪通霄汉，\\n灯火璀璨耀海山。\\n此景只应仙界有，\\n人间难得几回观。\'}, action="Create or overwrite a file with the poem about Shenzhen Bay\'s seascape")\n新问题:请作一首诗描写春天山景,然后把诗写入文件 `/oper/ch/mcp_work/tmp.txt`\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"$defs": {"Instruction": {"properties": {"prompt": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Prompt"}, "query": {"title": "Query", "type": "string"}}, "required": ["query"], "title": "Instruction", "type": "object"}, "Tool": {"properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"additionalProperties": true, "title": "Arguments", "type": "object"}, "action": {"title": "Action", "type": "string"}}, "required": ["name", "arguments", "action"], "title": "Tool", "type": "object"}}, "properties": {"action_name": {"title": "Action Name", "type": "string"}, "action_type": {"anyOf": [{"$ref": "#/$defs/Tool"}, {"$ref": "#/$defs/Instruction"}], "title": "Action Type"}}, "required": ["action_name", "action_type"]}\n```\n只输出符合规范的 JSON，不输出其他文字或解释。\n    output: Any
    error_message: Optional[str] = None


    
def ollama_chat(query:str,model:str='gemma3:27b') ->str:
    try:
        client = Client(host='http://localhost:11434')
        response = client.chat(model=model, messages=[{'role': 'user','content': query}])
    except Exception as e:
        print(f"{e}")
    return response['message']['content']

def search_client(query:str,n_results:5):

    manager=MCPClientManager(server_params)
    descriptions,_= await manager.initialize()
    result = await manager.run_tool("search_documents",{"query":query,"include":["documents", "metadatas"],"n_results":n_results })
    response=json.loads(result)
    return response['metadatas'][0]
    
def get_format_instructions(pydantic_object: BaseModel) -> str:
    """Return the format instructions for the JSON output."""
    schema = dict(pydantic_object.model_json_schema())
    # Remove extraneous fields
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]
    schema_str = json.dumps(reduced_schema, ensure_ascii=False)
    
    format_instructions = f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.
    Here is the output schema:
    ```
    {schema_str}
    ```"""
    return format_instructions

class IntelligentScheduler:
    def __init__(self, llm_client, search_client):
        self.llm_client = ollama_chat
        self.search_client = search_client
        self.step_pool = []  # 临时备用步骤库
        self.current_workflow = None
        self.max_tool_attempts = 3
    
    def llm_query(self, query: str, prompt: Optional[str] = None):
        """调用大模型进行查询"""
        full_prompt = (prompt or '') + query
        response = self.llm_client(full_prompt)
        return response
    
    def do_search(self, query: str, n_results: int = 5):
        """从知识库中搜索相关工作流或工具"""
        response = self.search_client(query, n_results)
        return response
    
    def workflow_recall(self, task_description: str) -> List[HistoricalWorkflow]:
        """工作流召回阶段：基于任务描述从知识库中召回相关的历史工作流"""
        recall_prompt = f"""
        请基于以下任务描述，从知识库中检索最相关的历史工作流:
        
        任务描述: {task_description}
        
        请计算相关性分数并按照相关性降序排列结果。
        """
        
        # 调用搜索API获取相关工作流
        historical_workflows = self.do_search(recall_prompt)
        
        # 计算每个历史工作流与当前任务的相关性
        scored_workflows = []
        for workflow in historical_workflows:
            # 调用LLM计算相关性分数
            relevance_prompt = f"""
            请计算以下历史工作流与当前任务的相关性分数(0-1):
            
            当前任务: {task_description}
            历史工作流: {workflow.task_name}
            历史工作流步骤: {json.dumps([step.dict() for step in workflow.steps], ensure_ascii=False)}
            
            只返回一个0到1之间的浮点数，表示相关性分数。
            """
            relevance_score = float(self.llm_query(relevance_prompt))
            workflow.relevance_score = relevance_score
            scored_workflows.append(workflow)
        
        # 按相关性分数排序
        scored_workflows.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_workflows
    
    def build_step_pool(self, historical_workflows: List[HistoricalWorkflow]):
        """构建临时备用步骤库"""
        self.step_pool = []
        for workflow in historical_workflows:
            # 只考虑相关性分数超过阈值的工作流
            if workflow.relevance_score >= 0.5:
                for step in workflow.steps:
                    self.step_pool.append({
                        "workflow_id": workflow.workflow_id,
                        "step": step,
                        "workflow_relevance": workflow.relevance_score
                    })
    
    def find_matching_step(self, current_step: StepInfo) -> Optional[HistoricalStep]:
        """在步骤池中查找与当前步骤匹配的历史步骤"""
        if not self.step_pool:
            return None
            
        best_match = None
        highest_score = 0
        
        for item in self.step_pool:
            historical_step = item["step"]
            
            # 构建步骤匹配度评估提示
            match_prompt = f"""
            请评估当前步骤与历史步骤的匹配度(0-1):
            
            当前步骤:
            - ID: {current_step.step_id}
            - 名称: {current_step.step_name}
            - 描述: {current_step.step_description}
            - 动作: {current_step.action_name}
            - 期望输出: {current_step.desire_output}
            
            历史步骤:
            - ID: {historical_step.step_id}
            - 名称: {historical_step.step_name}
            - 描述: {historical_step.step_description}
            - 动作: {historical_step.action_name}
            
            只返回一个0到1之间的浮点数，表示匹配度。
            """
            
            # 调用LLM计算匹配度
            match_score = float(self.llm_query(match_prompt))
            
            # 考虑工作流整体相关性和步骤匹配度的综合评分
            combined_score = match_score * 0.7 + item["workflow_relevance"] * 0.3
            
            if combined_score > highest_score and combined_score >= 0.7:  # 设置匹配阈值
                highest_score = combined_score
                historical_step.similarity_score = combined_score
                best_match = historical_step
        
        return best_match
    
    def tool_recall(self, step_description: str) -> List[Tools]:
        """工具召回：基于步骤描述从知识库中检索匹配的工具"""
        tool_prompt = f"""
        请基于以下步骤描述，从工具库中检索最相关的工具:
        
        步骤描述: {step_description}
        
        请返回工具名称、所需参数以及与步骤的匹配度评分。
        """
        
        # 调用搜索API获取相关工具
        tools = self.do_search(tool_prompt)
        
        # 计算每个工具与当前步骤的匹配度
        scored_tools = []
        for tool in tools:
            # 调用LLM计算匹配度
            match_prompt = f"""
            请计算以下工具与当前步骤的匹配度(0-1):
            
            当前步骤描述: {step_description}
            工具: {tool.name}
            
            只返回一个0到1之间的浮点数，表示匹配度。
            """
            match_score = float(self.llm_query(match_prompt))
            tool_with_score = (tool, match_score)
            scored_tools.append(tool_with_score)
        
        # 按匹配度排序
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scored_tools]
    
    def execute_tool(self, tool: Tools, step_info: StepInfo) -> TaskExecutionResult:
        """执行工具并返回结果"""
        # 这里应该是实际的工具执行逻辑
        # 简化示例：模拟工具执行
        try:
            # 调用相应的工具API
            # result = tool_executor.execute(tool.name, **tool.arguments)
            
            # 模拟执行结果
            success = True  # 假设执行成功
            output = f"执行工具 {tool.name} 的输出结果"\n根据提供的工具使用示例和新问题：\n保持action_type.name与示例一致；\n根据新问题，改写：\naction_name（简明描述新任务）；\naction_type.arguments（参数内容应与新问题一致，如文件路径、内容等）；\naction_type.action（用简洁语言描述实际动作）；\n工具使用示例:action_name=\'请作一首诗描写深圳湾海景,然后把诗写入文件 `/oper/ch/mcp_work/test.txt`\' action_type=Tool(name=\'write_file\', arguments={\'path\': \'/oper/ch/mcp_work/test.txt\', \'content\': \'深圳湾畔碧波连，\\n万象澄明映晴天。\\n白鹭翩跹逐浪远，\\n轻舟荡漾戏鸥闲。\\n长桥卧浪通霄汉，\\n灯火璀璨耀海山。\\n此景只应仙界有，\\n人间难得几回观。\'}, action="Create or overwrite a file with the poem about Shenzhen Bay\'s seascape")\n新问题:请作一首诗描写春天山景,然后把诗写入文件 `/oper/ch/mcp_work/tmp.txt`\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"$defs": {"Instruction": {"properties": {"prompt": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Prompt"}, "query": {"title": "Query", "type": "string"}}, "required": ["query"], "title": "Instruction", "type": "object"}, "Tool": {"properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"additionalProperties": true, "title": "Arguments", "type": "object"}, "action": {"title": "Action", "type": "string"}}, "required": ["name", "arguments", "action"], "title": "Tool", "type": "object"}}, "properties": {"action_name": {"title": "Action Name", "type": "string"}, "action_type": {"anyOf": [{"$ref": "#/$defs/Tool"}, {"$ref": "#/$defs/Instruction"}], "title": "Action Type"}}, "required": ["action_name", "action_type"]}\n```\n只输出符合规范的 JSON，不输出其他文字或解释。\n
            
            return TaskExecutionResult(
                success=success,
                output=output
            )
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                output=None,
                error_message=str(e)
            )
    
    def explore_step(self, step_info: StepInfo) -> Optional[Tools]:
        """探索阶段：尝试使用工具召回来执行当前步骤"""
        # 基于步骤描述进行工具召回
        relevant_tools = self.tool_recall(step_info.step_description)
        
        if not relevant_tools:
            return None
        
        # 尝试执行每个工具，直到成功或达到最大尝试次数
        for tool in relevant_tools[:min(len(relevant_tools), self.max_tool_attempts)]:
            # 使用LLM生成工具参数
            param_prompt = f"""
            请根据以下步骤信息，为工具 {tool.name} 生成合适的参数:
            
            步骤描述: {step_info.step_description}
            期望输出: {step_info.desire_output}
            
            请以JSON格式返回参数。
            """
            
            # 调用LLM生成参数
            param_result = self.llm_query(param_prompt)
            try:
                arguments = json.loads(param_result)
                tool.arguments = arguments
                
                # 执行工具
                execution_result = self.execute_tool(tool, step_info)
                
                if execution_result.success:
                    return tool
            except Exception as e:
                continue
        
        return None
    
    def execute_task(self, task_description: str, steps: List[StepInfo]) -> WorkflowBuild:
        """执行任务的主流程"""
        # 工作流召回阶段
        historical_workflows = self.workflow_recall(task_description)
        
        # 构建临时备用步骤库
        self.build_step_pool(historical_workflows)
        
        # 初始化新任务工作流
        new_workflow = WorkflowBuild(
            task_name=task_description,
            workflow=[]
        )
        
        # 执行阶段
        for step in steps:
            # 复用尝试
            matching_step = self.find_matching_step(step)
            
            if matching_step and matching_step.similarity_score >= 0.7:
                # 尝试复用历史步骤
                tool = matching_step.tool
                execution_result = self.execute_tool(tool, step)
                
                if execution_result.success:
                    # 复用成功
                    step.tool = tool
                    new_workflow.workflow.append(step)
                    continue
            
            # 探索尝试
            successful_tool = self.explore_step(step)
            
            if successful_tool:
                # 探索成功
                step.tool = successful_tool
                new_workflow.workflow.append(step)
            else:
                # 探索失败，标记步骤失败
                print(f"步骤 {step.step_id} 执行失败: 无法找到合适的工具")
                # 可以选择跳过或中止任务
        
        return new_workflow

# 使用示例
def main(task_description, steps, llm_client, search_client):
    scheduler = IntelligentScheduler(llm_client, search_client)
    result = scheduler.execute_task(task_description, steps)
    
    # 格式化输出
    format_instructions = get_format_instructions(WorkflowBuild)
    output = {
        "format_instructions": format_instructions,
        "result": result.dict()
    }
    
    return json.dumps(output, ensure_ascii=False, indent=2)