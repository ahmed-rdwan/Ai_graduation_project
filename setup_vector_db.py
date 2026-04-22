import os
from pymongo import MongoClient
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# استدعاء الموديل المحلي من HuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings

# تحميل المتغيرات من .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# الاتصال بقاعدة البيانات اللايف
client = MongoClient(MONGO_URI)
db = client["project_management"]

# تجهيز موديل تحويل النصوص لأرقام (مجاني، محلي، وسريع جداً)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_database():
    documents = []
    print("Starting to extract data from MongoDB...")

    # قائمة صلاحيات عامة عشان الـ AI يقدر يقرأ السياق لأي مستخدم (الباك إند هو اللي بيحمي الـ Actions)
    dynamic_roles = ["admin", "manager", "user", "developer", "tester", "it"]

    # --- 1. Users & Team Memberships ---
    users = db["users"].find()
    for user in users:
        content = f"User Name: {user.get('name', 'Unknown')}, Role: {user.get('role', 'user')}, Email: {user.get('email', 'Unknown')}."
        if user.get("team_id"):
            user_team = db["teams"].find_one({"_id": user["team_id"]})
            if user_team:
                content += f" This user belongs to the Team: '{user_team.get('name', 'Unknown')}."
        
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(user["_id"]), "type": "user", "allowed_roles": dynamic_roles}
        ))

    # --- 2. Teams ---
    teams = db["teams"].find()
    for team in teams:
        content = f"Team Name: {team.get('name', '')}, Description: {team.get('description', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(team["_id"]), "type": "team", "allowed_roles": dynamic_roles}
        ))

    # --- 3. Projects ---
    projects = db["projects"].find()
    for project in projects:
        content = f"Project Name: {project.get('name', '')}, Description: {project.get('description', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(project["_id"]), "type": "project", "allowed_roles": dynamic_roles}
        ))

    # --- 4. Sprints ---
    sprints = db["sprints"].find()
    for sprint in sprints:
        content = f"Sprint Name: {sprint.get('name', '')}, Status: {sprint.get('status', '')}, Goal: {sprint.get('sprint_goal', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(sprint["_id"]), "type": "sprint", "allowed_roles": dynamic_roles}
        ))

    # --- 5. Backlogs ---
    backlogs = db["backlogs"].find()
    for backlog in backlogs:
        content = f"Backlog Name: {backlog.get('name', '')}, Status: {backlog.get('status', '')}, Goal: {backlog.get('backlog_goal', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(backlog["_id"]), "type": "backlog", "allowed_roles": dynamic_roles}
        ))

    # --- 6. Tasks ---
    tasks = db["tasks"].find()
    for task in tasks:
        content = f"Task Name: {task.get('name', '')}, Priority: {task.get('priority', '')}, Status: {task.get('status', '')}, Description: {task.get('description', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(task["_id"]), "type": "task", "allowed_roles": dynamic_roles} 
        ))

    # --- 7. Tickets ---
    tickets = db["tickets"].find()
    for ticket in tickets:
        content = f"Ticket Name: {ticket.get('name', '')}, Problem: {ticket.get('description', '')}, Priority: {ticket.get('priority', '')}, Status: {ticket.get('status', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(ticket["_id"]), "type": "ticket", "allowed_roles": dynamic_roles}
        ))

    # --- 8. Stock Items ---
    stock_items = db["stockitems"].find()
    for item in stock_items:
        content = f"Inventory Item: {item.get('name', '')}, Category: {item.get('category', '')}, Quantity available: {item.get('quantity', 0)}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(item["_id"]), "type": "stock", "allowed_roles": dynamic_roles} 
        ))

    # --- 9. Plans ---
    plans = db["plan"].find() # غالباً ده ماتغيرش، بس لو اتغير خليه plans
    for plan in plans:
        content = f"Subscription Plan: {plan.get('name', '')}, Cost: ${plan.get('value', 0)}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(plan["_id"]), "type": "plan", "allowed_roles": dynamic_roles} 
        ))

    # --- 10. Working Tasks ---
    print("Extracting working tasks and linking users...")
    working_tasks = db["working_task"].find()
    
    for wt in working_tasks:
        assigned_user = db["users"].find_one({"_id": wt["user_id"]})
        assigned_task = db["tasks"].find_one({"_id": wt["task_id"]})
        
        if assigned_user and assigned_task:
            content = (
                f"Work Assignment: Employee '{assigned_user.get('name', '')}' "
                f"(Role: {assigned_user.get('role', 'user')}) is currently assigned to work on "
                f"Task: '{assigned_task.get('name', '')}'. "
                f"Task Description: {assigned_task.get('description', '')}. "
                f"They started on {wt['start_date'].strftime('%Y-%m-%d')} "
                f"and are expected to finish by {wt['end_date'].strftime('%Y-%m-%d')}."
            )
            
            documents.append(Document(
                page_content=content,
                metadata={"source_id": str(wt["_id"]), "type": "working_task", "allowed_roles": dynamic_roles}
            ))

    print(f"Successfully extracted {len(documents)} documents. Embedding them locally now...")
    
    # --- الحفظ المباشر والسريع ---
    vector_db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
  
    print("✅ Database setup complete and embedded locally!")

if __name__ == "__main__":
    setup_database()