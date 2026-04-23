import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from pymongo import MongoClient

# LangChain & Cache Imports
from langchain_community.vectorstores import Chroma
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from diskcache import Cache

from allocation_engine import router as allocation_router

# Tools
from agent_tools import (
    create_ticket, manage_stock, update_task_status,
    get_inventory, search_employee, get_my_tasks,
    get_sprint_status, log_attendance,
    checkout_attendance, get_my_tickets,
    get_my_attendance, get_team_report, update_ticket_status
)

load_dotenv()
app = FastAPI(title="IT Management Agentic RAG API")

# تعريف الكاش (سيتم حفظ الملفات في فولدر اسمه api_cache)
cache = Cache("./api_cache")
MAX_HISTORY_MESSAGES = 4

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
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_vector_db():
    db_path = "./chroma_db"
    try:
        vdb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        doc_count = vdb._collection.count()
        print(f"✅ Chroma loaded — {doc_count} documents.")
        return vdb, doc_count
    except Exception as e:
        print(f"❌ Chroma load error: {e}")
        return None, 0

vector_db, doc_count = load_vector_db()

if doc_count == 0:
    print("⚠️ Vector DB is EMPTY — running setup now...")
    from setup_vector_db import setup_database
    setup_database()
    vector_db, doc_count = load_vector_db()

# تم إلغاء الـ silent_db_watcher بناءً على طلبك

# -----------------------------------------------
# 4. Endpoints
# -----------------------------------------------
@app.get("/health")
async def health_check():
    count = vector_db._collection.count() if vector_db else 0
    return {"status": "ok", "vector_db_docs": count}

@app.post("/api/rebuild-db")
async def rebuild_vector_db():
    global vector_db
    try:
        from setup_vector_db import setup_database
        setup_database()
        vector_db, count = load_vector_db()
        cache.clear() # نمسح الكاش لما نحدث الداتا بيز عشان الردود تتحدث
        return {"message": f"✅ Vector DB rebuilt and cache cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------
# 5. LLM & Tools Setup
# -----------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

tools = [
    create_ticket, manage_stock, update_task_status,
    get_inventory, search_employee, get_my_tasks,
    get_sprint_status, log_attendance,
    checkout_attendance, get_my_tickets,
    get_my_attendance, get_team_report, update_ticket_status
]

agent_llm = llm.bind_tools(tools)

class ChatRequest(BaseModel):
    query: str
    user_role: str
    user_id: str
    chat_history: List[Dict[str, str]] = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------------------------
# 6. Main Endpoint with Caching
# -----------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # --- التحقق من الكاش أولاً (الأسئلة العامة فقط) ---
        # الكاش مفيد جداً لمناقشة مشروع التخرج عشان سرعة الرد
        cache_key = f"chat_{request.user_role}_{request.query.strip().lower()}"
        cached_response = cache.get(cache_key)
        
        if cached_response:
            print("🚀 Serving from Cache!")
            return cached_response

        # --- لو مش موجود في الكاش، نكلم الـ AI ---
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        docs = await retriever.ainvoke(request.query)
        context = format_docs(docs)

        system_prompt = f"""You are an intelligent IT Management assistant.
Current user: role='{request.user_role}', id='{request.user_id}'.
Context: {context}"""

        limited_history = request.chat_history[-MAX_HISTORY_MESSAGES:]
        history_messages = []
        for msg in limited_history:
            role = "human" if msg["role"] == "user" else "ai"
            history_messages.append((role, msg["content"]))

        messages = [("system", system_prompt)] + history_messages + [("human", request.query)]

        response = await agent_llm.ainvoke(messages)

        final_text = ""
        action_taken = False

        if response.tool_calls:
            action_taken = True
            messages.append(response)
            for tool_call in response.tool_calls:
                selected_tool = next(t for t in tools if t.name == tool_call["name"])
                result = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            final_response = await agent_llm.ainvoke(messages)
            final_text = final_response.content
        else:
            final_text = response.content

        if isinstance(final_text, list):
            final_text = " ".join([i.get("text", "") for i in final_text if "text" in i])

        result_to_return = {
            "response": final_text, 
            "role_used": request.user_role, 
            "action_taken": action_taken
        }

        # --- حفظ في الكاش (فقط لو مفيش Tools اتنفذت) ---
        # الأدوات (زي الحضور) لازم تكون Live دايماً، فمش بنكيّشها
        if not action_taken:
            cache.set(cache_key, result_to_return, expire=3600) # كاش لمدة ساعة

        return result_to_return

    except Exception as e:
        import traceback
        print("\n❌ Internal Error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(allocation_router)
