import json
import boto3
from langchain_core.tools import BaseTool
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

async def get_tool_spec(toolNames)-> List[BaseTool]:
            toolSpecs: List[BaseTool] = []
            all_tools = await getTools()
            filtered_tools: List[BaseTool] = [
                        tool for tool in all_tools if any(tool.name.startswith(name) for name in toolNames)
                    ]
            for tool in filtered_tools:
                tool_spec = {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {
                            "json": tool.args_schema
                        }
                    }
                }
                toolSpecs.append(tool_spec)
            return toolSpecs

all_tools = []

async def getTools() -> List[BaseTool]:

    if(len(all_tools)==0):
        mcp_client = MultiServerMCPClient(
                    {
                        "backendMcp": {
                            "url": "http://localhost:8000/mcp",
                            "transport": "streamable_http"
                        }
                    }
                )
        all_tools.extend(await mcp_client.get_tools())
    return all_tools


async def handle_tool_call(tool_name, tool_content):

    try:
        
        
        tools_dict = await getTools()
        
        tool = next((tool for tool in tools_dict if tool.name.lower() == tool_name.lower()), None)
        
        if not tool:
            return {"status": "error", "error": f"Tool '{tool_name}' not found"}
        
        # Execute the tool
        result = await tool.arun(tool_input=tool_content)

        return result
        
    except Exception as e:
        return {"status": "error", "error": f"Tool execution failed: {str(e)}"}


async def processToolUse(toolName, toolUseContent):
        """Process tool usage using MCP"""
        tool_name = toolName.lower()
            
        # Process the tool
        result = await handle_tool_call(tool_name, toolUseContent)
        
        return result

def getToolConfig(toolSpec: List[BaseTool]):
    return {"tools": toolSpec    }




async def getDividends():
    toolSpecs = await get_tool_spec(['get_dividends'])
    tool_config = getToolConfig(toolSpecs)
    print("-----------------")
    
    user_message = f"Give me dividend information for Apple stock"
    print(f"User Message: {user_message}")

    system = [
        {
            "text": "You are a friend. The user and you will engage in a spoken " +
  "dialog exchanging the transcripts of a natural real-time conversation."
        }
    ]

    messages = [{"role": "user", "content": [{"text": user_message}]}]

    inf_params = {"maxTokens": 300, "topP": 1, "temperature": 1}

    initial_response = client.converse(
        modelId=LITE_MODEL_ID,
        system=system,
        messages=messages,
        inferenceConfig=inf_params,
        additionalModelRequestFields={"inferenceConfig": {"topK": 1}},
        toolConfig=tool_config,
    )
    print("\n[Initial Response]")
    print(f"Stop Reason: {initial_response['stopReason']}")
    print(f"Content: {json.dumps(initial_response['output']['message'], indent=2)}")

    if initial_response["stopReason"] == "tool_use":
        tool_use = next(
            block["toolUse"]
            for block in initial_response["output"]["message"]["content"]
            if "toolUse" in block
        )

        if tool_use["name"] == "get_dividends":

            print(f"\nTool Name: {tool_use['name']}")
            print(f"Tool Input: {tool_use['input']}")

            tool_call_result = json.loads(await processToolUse(tool_use['name'], tool_use['input']))

            toolResult = {
                 "toolResult": 
                            {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"json": tool_call_result}],
                            }
                        }

            final_messages = [
                *messages,
                initial_response["output"]["message"],
                {
                    "role": "user",
                    "content": [
                        toolResult
                    ],
                },
            ]
            final_response = client.converse(
                modelId=LITE_MODEL_ID,
                messages=final_messages,
                inferenceConfig=inf_params,
                additionalModelRequestFields={"inferenceConfig": {"topK": 1}},
                toolConfig=tool_config
            )

            output = next(
                block["text"]
                for block in final_response["output"]["message"]["content"]
                if "text" in block
            )
            print(f"\nResponse: {output}")
        else:
            print("The divedends information tool was not called")

asyncio.run(getDividends())