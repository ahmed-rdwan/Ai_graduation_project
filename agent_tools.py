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
# Tool 1: Create a Ticket
# -----------------------------------------
@tool
def create_ticket(title: str, description: str, priority: str, user_role: str, user_id: str, assign_to_name: str = None) -> str:
    """
    Creates a new IT support bug or ticket.
    Use this when the user asks to report a bug, open a ticket, or report an issue.
    If the user asks to assign the ticket to a specific person, pass their name to 'assign_to_name'.
    """
    new_ticket = {
        "name": title,
        "description": description,
        "priority": priority,
        "status": "open",  # حروف صغيرة زي الفرونت
        "created_by": ObjectId(user_id),
        "createdAt": datetime.datetime.utcnow(), 
        "updatedAt": datetime.datetime.utcnow(), 
        "__v": 0
    }

    assigned_msg = ""
    # الـ AI هيعمل Assign بناءً على طلب اليوزر مباشرة بدون شروط رتب
    if assign_to_name:
        assignee = db.users.find_one({"name": {"$regex": assign_to_name, "$options": "i"}})
        if assignee:
            new_ticket["assign_to"] = assignee["_id"]
            assigned_msg = f" and assigned to {assignee.get('name', 'Unknown')}"
        else:
            assigned_msg = f" (Warning: Ticket created, but couldn't find employee '{assign_to_name}')"

    # استخدام db.tickets بالجمع
    db.tickets.insert_one(new_ticket)
    return f"✅ Ticket '{title}' | Priority: {priority} | Status: open{assigned_msg}."

# -----------------------------------------
# Tool 2: Manage Stock
# -----------------------------------------
@tool
def manage_stock(item_name: str, quantity: int, action: str, user_role: str, user_id: str) -> str:
    """
    Adds or removes hardware items from the inventory stock.
    Action must be exactly 'add' or 'remove'.
    """
    if action not in ["add", "remove"]:
        return "❌ Error: Action must be 'add' or 'remove'."

    # استخدام db.stockitems بالجمع
    item = db.stockitems.find_one({"name": {"$regex": item_name, "$options": "i"}})
    if not item:
        return f"❌ Error: Item '{item_name}' not found in the database."

    if action == "add":
        db.stockitems.update_one({"_id": item["_id"]}, {"$inc": {"quantity": quantity}})
    elif action == "remove":
        if item.get("quantity", 0) < quantity:
            return f"❌ Error: Not enough '{item.get('name', item_name)}' in stock. Current quantity: {item.get('quantity', 0)}."
        db.stockitems.update_one({"_id": item["_id"]}, {"$inc": {"quantity": -quantity}})

    new_qty = item.get("quantity", 0) + quantity if action == "add" else item.get("quantity", 0) - quantity
    return f"✅ {action.capitalize()}ed {quantity} unit(s) of '{item.get('name', item_name)}'. New quantity: {new_qty}."

# -----------------------------------------
# Tool 3: Update Task Status
# -----------------------------------------
@tool
def update_task_status(task_name: str, new_status: str, user_role: str, user_id: str) -> str:
    """
    Updates the status of a working task (e.g., 'pending', 'in_progress', 'completed').
    """
    # استخدام db.tasks بالجمع
    task = db.tasks.find_one({"name": {"$regex": task_name, "$options": "i"}})
    if not task:
        return f"❌ Error: Could not find a task matching '{task_name}'."

    old_status = task.get("status", "unknown")
    db.tasks.update_one(
        {"_id": task["_id"]}, 
        {"$set": {"status": new_status, "updatedAt": datetime.datetime.utcnow()}}
    )
    return f"✅ Task '{task.get('name', task_name)}' updated: '{old_status}' → '{new_status}'."

# -----------------------------------------
# Tool 4: View Stock Inventory
# -----------------------------------------
@tool
def get_inventory(user_role: str, user_id: str) -> str:
    """
    Retrieves the current list of items available in the IT inventory/stock.
    """
    stock_items = list(db.stockitems.find())
    if not stock_items:
        return "The inventory is currently empty."

    report_lines = ["📦 Current Inventory Stock:"]
    for item in stock_items:
        report_lines.append(f"  - {item.get('name', 'Unknown')} | Category: {item.get('category','—')} | Qty: {item.get('quantity', 0)}")

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
    employees = list(db.users.find({"name": {"$regex": name_query, "$options": "i"}}))
    if not employees:
        return f"No employee found matching '{name_query}'."

    report = ["👥 Found the following employees:"]
    for emp in employees:
        emp_info = f"  - {emp.get('name')} | Role: {emp.get('role', 'user')} | Email: {emp.get('email', 'No email')}"
        if emp.get("team_id"):
            team = db.teams.find_one({"_id": emp["team_id"]})
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
        task = db.tasks.find_one({"_id": wt["task_id"]})
        if task:
            deadline = wt["end_date"].strftime("%Y-%m-%d")
            report.append(f"  - '{task.get('name', 'Unknown')}' | Status: {task.get('status', 'open')} | Priority: {task.get('priority', 'normal')} | Deadline: {deadline}")

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
    sprint = db.sprints.find_one({"name": {"$regex": sprint_name, "$options": "i"}})
    if not sprint:
        return f"Could not find a sprint named '{sprint_name}'."

    tasks = list(db.tasks.find({"sprint_id": sprint["_id"]}))
    if not tasks:
        return f"No tasks found for sprint '{sprint.get('name', sprint_name)}'."

    stats = {"completed": 0, "in_progress": 0, "pending": 0, "to_do": 0, "open": 0}
    for t in tasks:
        status = t.get("status", "pending").lower().replace(" ", "_")
        if status in stats:
            stats[status] += 1
        else:
            stats[status] = 1

    total = len(tasks)
    done_pct = int((stats.get("completed", 0) / total) * 100) if total else 0

    report = [
        f"📊 Sprint Report: {sprint.get('name', sprint_name)}",
        f"   Goal: {sprint.get('sprint_goal', '—')}",
        f"   Progress: {done_pct}% complete",
        f"   Total Tasks: {total}"
    ]
    
    for k, v in stats.items():
        if v > 0:
            report.append(f"   - {k.replace('_', ' ').title()}: {v}")
            
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
        return f"⚠️ You already checked in today at {checkin_time} UTC."

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
        return f"⚠️ You already checked out today at {checkout_time} UTC."

    db.attendance.update_one({"_id": record["_id"]}, {"$set": {"check_out": now}})

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
    uid = ObjectId(user_id)
    tickets = list(db.tickets.find({"$or": [{"created_by": uid}, {"assign_to": uid}]}))
    if not tickets:
        return "You have no tickets created by or assigned to you."

    report = ["🎫 Your tickets:"]
    for t in tickets:
        label = "Created" if str(t.get("created_by")) == user_id else "Assigned"
        report.append(f"  - [{label}] '{t.get('name', 'Unknown')}' | Priority: {t.get('priority', 'normal')} | Status: {t.get('status', 'open')}")

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

        report.append(f"  - {date_str} | {r.get('status', '').capitalize()} | In: {check_in} | Out: {check_out}{hours}")

    return "\n".join(report)

# -----------------------------------------
# Tool 12: Get Team Report
# -----------------------------------------
@tool
def get_team_report(user_role: str, user_id: str) -> str:
    """
    Full report of all team members: their tasks and today's attendance.
    Use when requested to show 'team status', 'who is working on what', or 'team report'.
    """
    today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    members = list(db.users.find())
    if not members:
        return "No team members found."

    report = ["👥 Team Report:"]
    for member in members:
        # حضور النهارده
        att = db.attendance.find_one({"user_id": member["_id"], "date": today})
        if att:
            check_in = att["check_in"].strftime("%H:%M") if att.get("check_in") else "—"
            check_out = att["check_out"].strftime("%H:%M") if att.get("check_out") else "Still in"
            att_str = f"{att.get('status', '').capitalize()} (In: {check_in}, Out: {check_out})"
        else:
            att_str = "Absent"

        # التاسكات النشطة
        working_tasks = list(db.working_task.find({"user_id": member["_id"]}))
        active = []
        for wt in working_tasks:
            task = db.tasks.find_one({"_id": wt["task_id"], "status": {"$ne": "completed"}})
            if task:
                active.append(f"'{task.get('name', 'Unknown')}' ({task.get('status', 'open')})")

        tasks_str = ", ".join(active) if active else "No active tasks"
        report.append(
            f"\n  👤 {member.get('name', 'Unknown')} ({member.get('role', 'user')})"
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
    Updates the status of an existing IT support ticket (e.g., 'closed', 'resolved', 'in_progress').
    Use this when the user asks to close a ticket, resolve a bug, or change a ticket's status.
    """
    ticket = db.tickets.find_one({"name": {"$regex": ticket_name, "$options": "i"}})
    if not ticket:
        return f"❌ Error: Could not find a ticket matching '{ticket_name}'."

    old_status = ticket.get("status", "open")
    db.tickets.update_one(
        {"_id": ticket["_id"]}, 
        {"$set": {"status": new_status, "updatedAt": datetime.datetime.utcnow()}}
    )
    
    return f"✅ Success: Ticket '{ticket.get('name', ticket_name)}' status has been updated from '{old_status}' to '{new_status}'."