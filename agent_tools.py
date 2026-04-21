from langchain_core.tools import tool
from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime
import os
from dotenv import load_dotenv


load_dotenv()

# الاتصال بالداتا بيز
client = MongoClient(os.getenv("MONGO_URI"))
db = client["project_management"]

# -----------------------------------------
# Tool 1: Create a Ticket   (مسموح للكل، بس الـ Assign للأدمن فقط)
# -----------------------------------------
@tool
def create_ticket(title: str, description: str, priority: str, user_role: str, user_id: str, assign_to_name: str = None) -> str:
    """
    Creates a new IT support bug or ticket.
    Use this when the user asks to report a bug, open a ticket, or report an issue.
    If the user asks to assign the ticket to a specific person, pass their name to 'assign_to_name'.
    """
    # 1. حماية إنشاء التيكت (الكل يقدر يرفع تيكت)    
    if user_role not in ["admin", "developer", "tester"]:
        return "❌ Error: You do not have permission to create tickets."

    # 2. الحماية الجديدة: الأدمن بس هو اللي يقدر يعمل Assign لشخص تاني

    if assign_to_name and user_role != "admin":
        return "❌ Access Denied: You can create tickets, but ONLY Administrators can assign them to specific employees."

    # تجهيز التيكت الأساسية

    new_ticket = {
        "name": title,
        "description": description,
        "priority": priority,
        "status": "open",  # خليناها سمول زي الفرونت/نود
        "created_by": ObjectId(user_id),
        "createdAt": datetime.datetime.utcnow(), # مطابقة للـ Node.js
        "updatedAt": datetime.datetime.utcnow(), # مطابقة للـ Node.js
        "__v": 0
    }

    assigned_msg = ""

    # لو الأدمن طلب يعمل Assign لشخص معين
    
    if assign_to_name:
        assignee = db.user.find_one({"name": {"$regex": assign_to_name, "$options": "i"}})
        if assignee:
            new_ticket["assign_to"] = assignee["_id"]
            assigned_msg = f" and assigned to {assignee['name']}"
        else:
            assigned_msg = f" (Warning: Ticket created, but couldn't find employee '{assign_to_name}')"

    db.ticket.insert_one(new_ticket)
    return f"✅ Ticket '{title}' | Priority: {priority} | Status: Open{assigned_msg}."


# -----------------------------------------
# Tool 2: Manage Stock (Admin only)
# -----------------------------------------
@tool
def manage_stock(item_name: str, quantity: int, action: str, user_role: str, user_id: str) -> str:
    """
    Adds or removes hardware items from the inventory stock.
    Action must be exactly 'add' or 'remove'.
    """
    if user_role != "admin":
        return "❌ Access Denied: Only Administrators can modify the IT stock."

    if action not in ["add", "remove"]:
        return "❌ Error: Action must be 'add' or 'remove'."

    item = db.stock_item.find_one({"name": {"$regex": item_name, "$options": "i"}})
    if not item:
        return f"❌ Error: Item '{item_name}' not found in the database."

    if action == "add":
        db.stock_item.update_one({"_id": item["_id"]}, {"$inc": {"quantity": quantity}})
    elif action == "remove":
        if item["quantity"] < quantity:
            return f"❌ Error: Not enough '{item['name']}' in stock. Current quantity: {item['quantity']}."
        db.stock_item.update_one({"_id": item["_id"]}, {"$inc": {"quantity": -quantity}})

    new_qty = item["quantity"] + quantity if action == "add" else item["quantity"] - quantity
    return f"✅ {action.capitalize()}ed {quantity} unit(s) of '{item['name']}'. New quantity: {new_qty}."


# -----------------------------------------
# Tool 3: Update Task Status
# -----------------------------------------
@tool
def update_task_status(task_name: str, new_status: str, user_role: str, user_id: str) -> str:
    """
    Updates the status of a working task (e.g., 'pending', 'in progress', 'completed').
    """
    if user_role not in ["admin", "developer", "tester"]:
        return "❌ Error: Unauthorized."

    task = db.task.find_one({"name": {"$regex": task_name, "$options": "i"}})
    if not task:
        return f"❌ Error: Could not find a task matching '{task_name}'."

    old_status = task.get("status", "unknown")
    db.task.update_one({"_id": task["_id"]}, {"$set": {"status": new_status}})
    return f"✅ Task '{task['name']}' updated: '{old_status}' → '{new_status}'."


# -----------------------------------------
# Tool 4: View Stock Inventory (Admin only)
# -----------------------------------------
@tool
def get_inventory(user_role: str, user_id: str) -> str:
    """
    Retrieves the current list of items available in the IT inventory/stock.
    """
    if user_role != "admin":
        return "❌ Access Denied: Only Administrators can view the IT stock."

    stock_items = list(db.stock_item.find())
    if not stock_items:
        return "The inventory is currently empty."

    report_lines = ["📦 Current Inventory Stock:"]
    for item in stock_items:
        report_lines.append(f"  - {item['name']} | Category: {item.get('category','—')} | Qty: {item['quantity']}")

    return "\n".join(report_lines)


# -----------------------------------------
# Tool 5: Search Employee Info
# -----------------------------------------
@tool
def search_employee(name_query: str, user_role: str, user_id: str) -> str:
    """
    Searches the database for an employee by name to get their details (role, email, team).
    Use this when the user asks about a specific person, employee, or colleague.
    """
    if user_role not in ["admin", "developer", "tester"]:
        return "❌ Error: Unauthorized to access employee directory."

    employees = list(db.user.find({"name": {"$regex": name_query, "$options": "i"}}))
    if not employees:
        return f"No employee found matching '{name_query}'."

    report = ["👥 Found the following employees:"]
    for emp in employees:
        emp_info = f"  - {emp.get('name')} | Role: {emp.get('type')} | Email: {emp.get('email')}"
        if emp.get("team_id"):
            team = db.team.find_one({"_id": emp["team_id"]})
            if team:
                emp_info += f" | Team: {team.get('name')}"
        report.append(emp_info)

    return "\n".join(report)


# -----------------------------------------
# Tool 6: Get My Tasks
# -----------------------------------------
@tool
def get_my_tasks(user_role: str, user_id: str) -> str:
    """
    Retrieves the working tasks assigned to the current user.
    Use when the user asks 'what are my tasks', 'what should I do today', or 'my work'.
    """
    my_working_tasks = list(db.working_task.find({"user_id": ObjectId(user_id)}))
    if not my_working_tasks:
        return "You currently have no tasks assigned to you."

    report = ["📋 Your assigned tasks:"]
    for wt in my_working_tasks:
        task = db.task.find_one({"_id": wt["task_id"]})
        if task:
            deadline = wt["end_date"].strftime("%Y-%m-%d")
            report.append(f"  - '{task['name']}' | Status: {task['status']} | Priority: {task['priority']} | Deadline: {deadline}")

    return "\n".join(report)


# -----------------------------------------
# Tool 7: Get Sprint Status
# -----------------------------------------
@tool
def get_sprint_status(sprint_name: str, user_role: str, user_id: str) -> str:
    """
    Provides a summary of all tasks within a specific sprint, including their statuses.
    Use when the user asks about 'sprint progress', 'how is the sprint going', or 'sprint tasks'.
    """
    if user_role not in ["admin", "developer"]:
        return "❌ Access Denied: Unauthorized to view sprint analytics."

    sprint = db.sprint.find_one({"name": {"$regex": sprint_name, "$options": "i"}})
    if not sprint:
        return f"Could not find a sprint named '{sprint_name}'."

    tasks = list(db.task.find({"sprint_id": sprint["_id"]}))
    if not tasks:
        return f"No tasks found for sprint '{sprint['name']}'."

    stats = {"completed": 0, "in progress": 0, "pending": 0}
    for t in tasks:
        status = t.get("status", "pending").lower()
        if status in stats:
            stats[status] += 1

    total = len(tasks)
    done_pct = int((stats["completed"] / total) * 100) if total else 0

    report = [
        f"📊 Sprint Report: {sprint['name']}",
        f"   Goal: {sprint.get('sprint_goal', '—')}",
        f"   Progress: {done_pct}% complete",
        f"   Total: {total} | ✅ Done: {stats['completed']} | 🔄 In Progress: {stats['in progress']} | ⏳ Pending: {stats['pending']}"
    ]
    return "\n".join(report)


# -----------------------------------------
# Tool 8: Log Attendance (Check-in)
# -----------------------------------------
@tool
def log_attendance(user_role: str, user_id: str, note: str = "") -> str:
    """
    Logs the user's check-in time for today's attendance.
    Use when the user says 'I'm here', 'log my attendance', or 'check in'.
    """
    now = datetime.datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    existing = db.attendance.find_one({"user_id": ObjectId(user_id), "date": today})
    if existing:
        checkin_time = existing["check_in"].strftime("%H:%M:%S")
        return f"⚠️ You already checked in today at {checkin_time}."

    db.attendance.insert_one({
        "user_id": ObjectId(user_id),
        "date": today,
        "check_in": now,
        "status": "present",
        "note": note
    })
    return f"✅ Check-in logged at {now.strftime('%H:%M:%S')} UTC."


# -----------------------------------------
# Tool 9: Checkout Attendance (Check-out)
# -----------------------------------------
@tool
def checkout_attendance(user_role: str, user_id: str) -> str:
    """
    Logs the user's check-out time for today.
    Use when the user says 'I'm leaving', 'check out', 'log my departure', or 'sign out'.
    """
    now = datetime.datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    record = db.attendance.find_one({"user_id": ObjectId(user_id), "date": today})
    if not record:
        return "⚠️ No check-in found for today. Please check in first."
    if record.get("check_out"):
        checkout_time = record["check_out"].strftime("%H:%M:%S")
        return f"⚠️ You already checked out today at {checkout_time}."

    db.attendance.update_one({"_id": record["_id"]}, {"$set": {"check_out": now}})

    # حساب ساعات الشغل
    hours_worked = round((now - record["check_in"]).seconds / 3600, 1)
    return f"✅ Check-out logged at {now.strftime('%H:%M:%S')} UTC. Hours worked today: {hours_worked}h."


# -----------------------------------------
# Tool 10: Get My Tickets
# -----------------------------------------
@tool
def get_my_tickets(user_role: str, user_id: str) -> str:
    """
    Retrieves tickets created by or assigned to the current user.
    Use when the user asks 'my tickets', 'what bugs did I report', or 'tickets assigned to me'.
    """
    if user_role not in ["admin", "developer", "tester"]:
        return "❌ Error: Unauthorized."

    uid = ObjectId(user_id)
    tickets = list(db.ticket.find({"$or": [{"created_by": uid}, {"assign_to": uid}]}))
    if not tickets:
        return "You have no tickets created by or assigned to you."

    report = ["🎫 Your tickets:"]
    for t in tickets:
        label = "Created" if str(t.get("created_by")) == user_id else "Assigned"
        report.append(f"  - [{label}] '{t['name']}' | Priority: {t['priority']} | Status: {t['status']}")

    return "\n".join(report)


# -----------------------------------------
# Tool 11: Get My Attendance History
# -----------------------------------------
@tool
def get_my_attendance(user_role: str, user_id: str, days: int = 7) -> str:
    """
    Shows attendance history for the current user for the last N days (default: 7).
    Use when the user asks 'my attendance', 'was I late this week', or 'my check-ins'.
    """
    if user_role not in ["admin", "developer", "tester"]:
        return "❌ Error: Unauthorized."

    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    records = list(db.attendance.find(
        {"user_id": ObjectId(user_id), "date": {"$gte": since}}
    ).sort("date", -1))

    if not records:
        return f"No attendance records found for the last {days} days."

    report = [f"🗓️ Your attendance (last {days} days):"]
    for r in records:
        date_str = r["date"].strftime("%Y-%m-%d")
        check_in = r["check_in"].strftime("%H:%M") if r.get("check_in") else "—"
        check_out = r["check_out"].strftime("%H:%M") if r.get("check_out") else "Not recorded"

        hours = ""
        if r.get("check_in") and r.get("check_out"):
            h = round((r["check_out"] - r["check_in"]).seconds / 3600, 1)
            hours = f" | {h}h worked"

        report.append(f"  - {date_str} | {r['status'].capitalize()} | In: {check_in} | Out: {check_out}{hours}")

    return "\n".join(report)


# -----------------------------------------
# Tool 12: Get Team Report (Admin only)
# -----------------------------------------
@tool
def get_team_report(user_role: str, user_id: str) -> str:
    """
    Full report of all team members: their tasks and today's attendance.
    Use when admin asks 'team status', 'who is working on what', or 'team report'.
    """
    if user_role != "admin":
        return "❌ Access Denied: Only Administrators can view the full team report."

    today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    members = list(db.user.find({"type": {"$in": ["developer", "tester"]}}))
    if not members:
        return "No team members found."

    report = ["👥 Team Report:"]
    for member in members:
        # حضور النهارده
        att = db.attendance.find_one({"user_id": member["_id"], "date": today})
        if att:
            check_in = att["check_in"].strftime("%H:%M") if att.get("check_in") else "—"
            check_out = att["check_out"].strftime("%H:%M") if att.get("check_out") else "Still in"
            att_str = f"{att['status'].capitalize()} (In: {check_in}, Out: {check_out})"
        else:
            att_str = "Absent"

        # التاسكات النشطة
        working_tasks = list(db.working_task.find({"user_id": member["_id"]}))
        active = []
        for wt in working_tasks:
            task = db.task.find_one({"_id": wt["task_id"], "status": {"$ne": "completed"}})
            if task:
                active.append(f"'{task['name']}' ({task['status']})")

        tasks_str = ", ".join(active) if active else "No active tasks"
        report.append(
            f"\n  👤 {member['name']} ({member['type']})"
            f"\n     Today: {att_str}"
            f"\n     Tasks: {tasks_str}"
        )

    return "\n".join(report)


# -----------------------------------------
# Tool 13: Update Ticket Status
# -----------------------------------------
@tool
def update_ticket_status(ticket_name: str, new_status: str, user_role: str, user_id: str) -> str:
    """
    Updates the status of an existing IT support ticket (e.g., 'Closed', 'Resolved', 'In Progress').
    Use this when the user asks to close a ticket, resolve a bug, or change a ticket's status.
    """
    if user_role not in ["admin", "developer", "tester"]:
        return "❌ Error: Unauthorized to modify tickets."

    # البحث عن التيكت بالاسم (أو جزء منه)
    ticket = db.ticket.find_one({"name": {"$regex": ticket_name, "$options": "i"}})
    if not ticket:
        return f"❌ Error: Could not find a ticket matching '{ticket_name}'."

    old_status = ticket.get("status", "Open")
    db.ticket.update_one({"_id": ticket["_id"]}, {"$set": {"status": new_status}})
    
    return f"✅ Success: Ticket '{ticket['name']}' status has been updated from '{old_status}' to '{new_status}'."