import os
from pymongo import MongoClient
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["project_management"]

# استخدام موديل جوجل الجديد المدعوم
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def setup_database():
    documents = []
    print("Starting to extract data from MongoDB...")

    # 1. Users
    for user in db["users"].find():
        content = f"User Name: {user.get('name', 'Unknown')}, Role: {user.get('role', 'user')}, Email: {user.get('email', 'Unknown')}."
        if user.get("team_id"):
            user_team = db["teams"].find_one({"_id": user["team_id"]})
            if user_team: content += f" This user belongs to the Team: '{user_team.get('name', 'Unknown')}."
        documents.append(Document(page_content=content, metadata={"source_id": str(user["_id"]), "type": "user"}))

    # 2. Teams
    for team in db["teams"].find():
        documents.append(Document(page_content=f"Team Name: {team.get('name', '')}, Description: {team.get('description', '')}", metadata={"source_id": str(team["_id"]), "type": "team"}))

    # 3. Projects
    for project in db["projects"].find():
        documents.append(Document(page_content=f"Project Name: {project.get('name', '')}, Description: {project.get('description', '')}", metadata={"source_id": str(project["_id"]), "type": "project"}))

    # 4. Sprints
    for sprint in db["sprints"].find():
        documents.append(Document(page_content=f"Sprint Name: {sprint.get('name', '')}, Status: {sprint.get('status', '')}, Goal: {sprint.get('sprint_goal', '')}", metadata={"source_id": str(sprint["_id"]), "type": "sprint"}))

    # 5. Backlogs
    for backlog in db["backlogs"].find():
        documents.append(Document(page_content=f"Backlog Name: {backlog.get('name', '')}, Status: {backlog.get('status', '')}, Goal: {backlog.get('backlog_goal', '')}", metadata={"source_id": str(backlog["_id"]), "type": "backlog"}))

    # 6. Tasks
    for task in db["tasks"].find():
        documents.append(Document(page_content=f"Task Name: {task.get('name', '')}, Priority: {task.get('priority', '')}, Status: {task.get('status', '')}, Description: {task.get('description', '')}", metadata={"source_id": str(task["_id"]), "type": "task"}))

    # 7. Tickets
    for ticket in db["tickets"].find():
        documents.append(Document(page_content=f"Ticket Name: {ticket.get('name', '')}, Problem: {ticket.get('description', '')}, Priority: {ticket.get('priority', '')}, Status: {ticket.get('status', '')}", metadata={"source_id": str(ticket["_id"]), "type": "ticket"}))

    # 8. Stock Items
    for item in db["stockitems"].find():
        documents.append(Document(page_content=f"Inventory Item: {item.get('name', '')}, Category: {item.get('category', '')}, Quantity available: {item.get('quantity', 0)}", metadata={"source_id": str(item["_id"]), "type": "stock"}))

    # 9. Plans
    for plan in db["plan"].find():
        documents.append(Document(page_content=f"Subscription Plan: {plan.get('name', '')}, Cost: ${plan.get('value', 0)}", metadata={"source_id": str(plan["_id"]), "type": "plan"}))

    # 10. Working Tasks
    for wt in db["working_task"].find():
        assigned_user = db["users"].find_one({"_id": wt["user_id"]})
        assigned_task = db["tasks"].find_one({"_id": wt["task_id"]})
        if assigned_user and assigned_task:
            content = f"Work Assignment: Employee '{assigned_user.get('name', '')}' is currently assigned to work on Task: '{assigned_task.get('name', '')}'. Task Description: {assigned_task.get('description', '')}."
            documents.append(Document(page_content=content, metadata={"source_id": str(wt["_id"]), "type": "working_task"}))

    print(f"Successfully extracted {len(documents)} documents. Embedding using Google API...")
    vector_db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")
    print("✅ Database setup complete and embedded with Google!")

if __name__ == "__main__":
    setup_database()