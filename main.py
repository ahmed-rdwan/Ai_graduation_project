import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

from pymongo import MongoClient

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq

from allocation_engine import router as allocation_router

# Tools
from agent_tools import (
    create_ticket, manage_stock, update_task_status,
    get_inventory, search_employee, get_my_tasks,
    get_sprint_status, log_attendance,
    checkout_attendance, get_my_tickets,
    get_my_attendance, get_team_report, update_ticket_status
)

# -----------------------------------------------
# 1. Setup
# -----------------------------------------------
load_dotenv()
app = FastAPI(title="IT Management Agentic RAG API")

MAX_HISTORY_MESSAGES = 4   # رفعناها عشان السياق مهم مع الأدوات

client = MongoClient(os.getenv("MONGO_URI"))
db = client["project_management"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# 2. RAG Setup
# -----------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# -----------------------------------------------
# 3. LLM & Tools
# -----------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

tools = [
    create_ticket, manage_stock, update_task_status,
    get_inventory, search_employee, get_my_tasks,
    get_sprint_status, log_attendance,
    checkout_attendance, get_my_tickets,
    get_my_attendance, get_team_report, update_ticket_status
]

agent_llm = llm.bind_tools(tools)

# -----------------------------------------------
# 4. Request Model
# -----------------------------------------------
class ChatRequest(BaseModel):
    query: str
    user_role: str
    user_id: str
    chat_history: List[Dict[str, str]] = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------------------------
# 5. Main Endpoint
# -----------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # --- Step 1: RAG retrieval ---
       #search_filter = {"allowed_roles": {"$in": [request.user_role]}}
        retriever = vector_db.as_retriever(search_kwargs={"k": 3, "filter": search_filter})
        
        docs = await retriever.ainvoke(request.query)
        context = format_docs(docs)

        # --- Step 2: System prompt ---
        # التعديل: شلنا الـ Hardcoding تماماً وبقى دايناميك
        system_prompt = f"""You are an intelligent IT Management assistant.
Current user: role='{request.user_role}', id='{request.user_id}'.

TOOL RULES:
1. ALWAYS pass '{request.user_role}' as `user_role` and '{request.user_id}' as `user_id` to EVERY tool exactly as provided.
2. Tools give LIVE data — prefer them over the Context snapshot for anything real-time.
3. For personal requests ('my tasks', 'my tickets', 'my attendance', 'check in', 'check out') → ALWAYS call the relevant tool immediately.
4. For management/system requests ('team report', 'inventory', 'manage stock') → call the tool (the backend will handle authorization dynamically based on the user_role). Do NOT assume permissions.

AVAILABLE TOOLS SUMMARY:
- create_ticket        → report a bug or issue
- get_my_tickets       → see my open/assigned tickets
- update_task_status   → change a task's status
- get_my_tasks         → see my assigned tasks
- get_sprint_status    → sprint progress report
- get_inventory        → view stock 
- manage_stock         → add/remove stock items 
- log_attendance       → check in for today
- checkout_attendance  → check out for today
- get_my_attendance    → view my attendance history
- search_employee      → look up a colleague's info
- get_team_report      → full team status 
- update_ticket_status → close or change a ticket's status

Context from system (background knowledge — may be outdated):
{context}

ANSWER RULES:
1. For live/personal/action requests → use the appropriate tool.
2. For general IT/Project questions → answer from Context.
3. If not in Context and not related to the project → politely decline.
4. Keep responses concise and professional."""

        # --- Step 3: Build message list with history ---
        limited_history = request.chat_history[-MAX_HISTORY_MESSAGES:]
        history_messages = []
        for msg in limited_history:
            if msg["role"] == "user":
                history_messages.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(("ai", msg["content"]))

        messages = [("system", system_prompt)] + history_messages + [("human", request.query)]

        # --- Step 4: First LLM call (decide: tool or answer?) ---
        response = await agent_llm.ainvoke(messages)

        # --- Step 5: Execute tools if needed ---
        if response.tool_calls:
            messages.append(response)

            for tool_call in response.tool_calls:
                selected_tool = next(t for t in tools if t.name == tool_call["name"])
                result = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            # --- Step 6: Second LLM call to format the result naturally ---
            final_response = await agent_llm.ainvoke(messages)
            final_text = final_response.content

            if isinstance(final_text, list):
                final_text = " ".join([i.get("text", "") for i in final_text if "text" in i])

            return {
                "response": final_text,
                "role_used": request.user_role,
                "action_taken": True
            }

        else:
            final_text = response.content
            if isinstance(final_text, list):
                final_text = " ".join([i.get("text", "") for i in final_text if "text" in i])

            return {
                "response": final_text,
                "role_used": request.user_role,
                "action_taken": False
            }

    except Exception as e:
        import traceback
        print("\n❌ Internal Error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(allocation_router)