import os
from pymongo import MongoClient
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


from dotenv import load_dotenv


# استدعاء الموديل المحلي من HuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. الاتصال بقاعدة بيانات مونجو


# تحميل المتغيرات من .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# الاتصال بقاعدة البيانات اللايف
client = MongoClient(MONGO_URI)
db = client["project_management"]



# 2. تجهيز موديل تحويل النصوص لأرقام (مجاني، محلي، وسريع جداً)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_database():
    documents = []
    print("Starting to extract data from MongoDB...")

    # --- 1. Users & Team Memberships ---
    users = db["user"].find()
    for user in users:
        content = f"User Name: {user['name']}, Role/Type: {user['type']}, Email: {user['email']}."
        if user.get("team_id"):
            user_team = db["team"].find_one({"_id": user["team_id"]})
            if user_team:
                content += f" This user belongs to the Team: '{user_team['name']}'."
        
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(user["_id"]), "type": "user", "allowed_roles": ["admin", "developer", "tester"]}
        ))

    # --- 2. Teams ---
    teams = db["team"].find()
    for team in teams:
        content = f"Team Name: {team['name']}, Description: {team['description']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(team["_id"]), "type": "team", "allowed_roles": ["admin", "developer", "tester"]}
        ))

    # --- 3. Projects ---
    projects = db["project"].find()
    for project in projects:
        content = f"Project Name: {project['name']}, Description: {project['description']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(project["_id"]), "type": "project", "allowed_roles": ["admin", "developer", "tester"]}
        ))

    # --- 4. Sprints ---
    sprints = db["sprint"].find()
    for sprint in sprints:
        content = f"Sprint Name: {sprint['name']}, Status: {sprint['status']}, Goal: {sprint['sprint_goal']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(sprint["_id"]), "type": "sprint", "allowed_roles": ["admin", "developer", "tester"]}
        ))

    # --- 5. Backlogs ---
    backlogs = db["backlog"].find()
    for backlog in backlogs:
        content = f"Backlog Name: {backlog['name']}, Status: {backlog['status']}, Goal: {backlog['backlog_goal']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(backlog["_id"]), "type": "backlog", "allowed_roles": ["admin", "developer", "tester"]}
        ))

    # --- 6. Tasks ---
    tasks = db["task"].find()
    for task in tasks:
        content = f"Task Name: {task['name']}, Priority: {task['priority']}, Status: {task['status']}, Description: {task['description']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(task["_id"]), "type": "task", "allowed_roles": ["admin", "developer"]} 
        ))

    # --- 7. Tickets ---
    tickets = db["ticket"].find()
    for ticket in tickets:
        content = f"Ticket Name: {ticket['name']}, Problem: {ticket['description']}, Priority: {ticket['priority']}, Status: {ticket['status']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(ticket["_id"]), "type": "ticket", "allowed_roles": ["admin", "developer", "tester"]}
        ))

    # --- 8. Stock Items ---
    stock_items = db["stock_item"].find()
    for item in stock_items:
        content = f"Inventory Item: {item['name']}, Category: {item['category']}, Quantity available: {item['quantity']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(item["_id"]), "type": "stock", "allowed_roles": ["admin"]} 
        ))

    # --- 9. Plans ---
    plans = db["plan"].find()
    for plan in plans:
        content = f"Subscription Plan: {plan['name']}, Cost: ${plan['value']}"
        documents.append(Document(
            page_content=content,
            metadata={"source_id": str(plan["_id"]), "type": "plan", "allowed_roles": ["admin"]} 
        ))

    # --- 10. Working Tasks ---
    print("Extracting working tasks and linking users...")
    working_tasks = db["working_task"].find()
    
    for wt in working_tasks:
        assigned_user = db["user"].find_one({"_id": wt["user_id"]})
        assigned_task = db["task"].find_one({"_id": wt["task_id"]})
        
        if assigned_user and assigned_task:
            content = (
                f"Work Assignment: Employee '{assigned_user['name']}' "
                f"(Role: {assigned_user['type']}) is currently assigned to work on "
                f"Task: '{assigned_task['name']}'. "
                f"Task Description: {assigned_task['description']}. "
                f"They started on {wt['start_date'].strftime('%Y-%m-%d')} "
                f"and are expected to finish by {wt['end_date'].strftime('%Y-%m-%d')}."
            )
            
            documents.append(Document(
                page_content=content,
                metadata={"source_id": str(wt["_id"]), "type": "working_task", "allowed_roles": ["admin", "developer", "tester"]}
            ))

    print(f"Successfully extracted {len(documents)} documents. Embedding them locally now...")
    
    # --- الحفظ المباشر والسريع (بدون تعقيدات أو تأخير) ---
    vector_db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
  
    print("✅ Database setup complete and embedded locally!")

if __name__ == "__main__":
    setup_database()