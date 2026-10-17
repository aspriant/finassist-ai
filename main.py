
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
)

from financial_tools import calculate_debt_to_income_ratio
from rag_tool import search_finassist_knowledge
from document_tools import analyze_payslip
from risk_tools import assess_dti_risk
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

project = None
openai_client = None
credential = None

AGENT_NAME = os.environ.get(
    "FOUNDRY_AGENT_NAME", "finassist-ai-dev"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global project, openai_client, credential

    credential = AzureCliCredential()
    project = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=credential,
    )
    openai_client = project.get_openai_client()

    yield

    if project:
        project.close()
    if credential:
        credential.close()


app = FastAPI(
    title="FinAssist AI API",
    description=(
        "Financial document analysis, DTI calculation, "
        "policy retrieval, and risk classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class AssessmentRequest(BaseModel):
    monthly_debt: float = Field(
        gt=0,
        description="Total recurring monthly debt obligations.",
        examples=[40000],
    )


def execute_tool(name: str, arguments: dict) -> dict:
    if name == "analyze_payslip":
        return analyze_payslip()

    if name == "calculate_debt_to_income_ratio":
        return calculate_debt_to_income_ratio(
            monthly_income=arguments["monthly_income"],
            monthly_debt=arguments["monthly_debt"],
        )

    if name == "search_finassist_knowledge":
        return search_finassist_knowledge(
            query=arguments["query"]
        )

    if name == "assess_dti_risk":
        return assess_dti_risk(
            dti_ratio=arguments["dti_ratio"]
        )

    raise ValueError(f"Unsupported tool: {name}")


def run_assessment(monthly_debt: float) -> dict:
    question = (
        "Analyze the configured synthetic payslip using "
        "analyze_payslip. Use the extracted gross monthly income "
        f"and monthly debt obligations of {monthly_debt}. "
        "Calculate DTI using the deterministic calculation tool. "
        "Retrieve relevant FinAssist policy evidence, classify "
        "the DTI using the risk assessment tool, and provide a "
        "consolidated report covering calculation, policy sources, "
        "assumptions, and human-review requirements. "
        "Do not make a final lending approval decision."
    )

    conversation = openai_client.conversations.create()

    response = openai_client.responses.create(
        input=question,
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": AGENT_NAME,
                "type": "agent_reference",
            }
        },
    )

    tools_called = []
    tool_results = {}
    max_iterations = 10

    for _ in range(max_iterations):
        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not function_calls:
            return {
                "status": "completed",
                "agent": AGENT_NAME,
                "tools_called": tools_called,
                "results": tool_results,
                "answer": response.output_text,
            }

        outputs: ResponseInputParam = []

        for item in function_calls:
            try:
                arguments = json.loads(item.arguments)
                result = execute_tool(item.name, arguments)

                tools_called.append(item.name)
                tool_results[item.name] = result

                output = json.dumps(result, default=str)

            except Exception as exc:
                output = json.dumps({
                    "error": str(exc),
                    "tool": item.name,
                })

            outputs.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=output,
                )
            )

        response = openai_client.responses.create(
            input=outputs,
            conversation=conversation.id,
            extra_body={
                "agent_reference": {
                    "name": AGENT_NAME,
                    "type": "agent_reference",
                }
            },
        )

    raise RuntimeError(
        "Maximum tool-call iterations reached."
    )


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "application": "FinAssist AI",
    }


@app.post("/api/assessments")
def create_assessment(request: AssessmentRequest):
    try:
        return run_assessment(request.monthly_debt)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Assessment failed: {exc}",
        ) from exc
