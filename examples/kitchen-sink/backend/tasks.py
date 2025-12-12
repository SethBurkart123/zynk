"""
Tasks Module

Demonstrates nested models and more complex data structures.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from zynk import command


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """Status of a task."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskLabel(BaseModel):
    """A label/tag for categorizing tasks."""
    id: int
    name: str
    color: str = "#808080"


class Task(BaseModel):
    """A task item."""
    id: int
    title: str
    description: str | None = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.TODO
    labels: list[TaskLabel] = Field(default_factory=list)
    created_at: str
    due_date: str | None = None
    assigned_to: int | None = None  # User ID


class TaskStats(BaseModel):
    """Statistics about tasks."""
    total: int
    todo: int
    in_progress: int
    done: int
    cancelled: int
    by_priority: dict[str, int]


# In-memory storage
_tasks_db: dict[int, Task] = {}
_labels_db: dict[int, TaskLabel] = {
    1: TaskLabel(id=1, name="Bug", color="#ff0000"),
    2: TaskLabel(id=2, name="Feature", color="#00ff00"),
    3: TaskLabel(id=3, name="Documentation", color="#0000ff"),
}
_next_task_id = 1
_next_label_id = 4

# Initialize with sample data
_sample_tasks = [
    Task(
        id=1,
        title="Set up project structure",
        description="Create the initial folder structure and configuration files",
        priority=TaskPriority.HIGH,
        status=TaskStatus.DONE,
        labels=[_labels_db[2]],
        created_at="2024-01-01T10:00:00",
    ),
    Task(
        id=2,
        title="Implement user authentication",
        description="Add login and signup functionality",
        priority=TaskPriority.HIGH,
        status=TaskStatus.IN_PROGRESS,
        labels=[_labels_db[2]],
        created_at="2024-01-02T09:00:00",
        due_date="2024-01-15",
    ),
    Task(
        id=3,
        title="Fix login button alignment",
        priority=TaskPriority.LOW,
        status=TaskStatus.TODO,
        labels=[_labels_db[1]],
        created_at="2024-01-03T14:00:00",
    ),
    Task(
        id=4,
        title="Write API documentation",
        description="Document all endpoints and models",
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.TODO,
        labels=[_labels_db[3]],
        created_at="2024-01-04T11:00:00",
    ),
]
_tasks_db = {t.id: t for t in _sample_tasks}
_next_task_id = 5


@command
async def get_task(task_id: int) -> Task:
    """Get a task by ID."""
    if task_id not in _tasks_db:
        raise ValueError(f"Task with ID {task_id} not found")
    return _tasks_db[task_id]


@command
async def list_tasks(
    status: str | None = None,
    priority: str | None = None,
    label_id: int | None = None,
) -> list[Task]:
    """
    List tasks with optional filters.

    Can filter by status, priority, or label.
    """
    tasks = list(_tasks_db.values())

    if status:
        try:
            status_enum = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_enum]
        except ValueError:
            pass

    if priority:
        try:
            priority_enum = TaskPriority(priority)
            tasks = [t for t in tasks if t.priority == priority_enum]
        except ValueError:
            pass

    if label_id:
        tasks = [
            t for t in tasks
            if any(label.id == label_id for label in t.labels)
        ]

    # Sort by priority (urgent first) then by created_at
    priority_order = {
        TaskPriority.URGENT: 0,
        TaskPriority.HIGH: 1,
        TaskPriority.MEDIUM: 2,
        TaskPriority.LOW: 3,
    }
    tasks.sort(key=lambda t: (priority_order[t.priority], t.created_at))

    return tasks


@command
async def create_task(
    title: str,
    description: str | None = None,
    priority: str = "medium",
    due_date: str | None = None,
    label_ids: list[int] | None = None,
) -> Task:
    """Create a new task."""
    global _next_task_id

    # Validate and convert priority
    try:
        priority_enum = TaskPriority(priority)
    except ValueError:
        priority_enum = TaskPriority.MEDIUM

    # Get labels
    labels = []
    if label_ids:
        labels = [_labels_db[lid] for lid in label_ids if lid in _labels_db]

    task = Task(
        id=_next_task_id,
        title=title,
        description=description,
        priority=priority_enum,
        status=TaskStatus.TODO,
        labels=labels,
        created_at=datetime.now().isoformat(),
        due_date=due_date,
    )

    _tasks_db[_next_task_id] = task
    _next_task_id += 1

    return task


@command
async def update_task_status(task_id: int, status: str) -> Task:
    """Update the status of a task."""
    print(f"Updating task status for task ID: {task_id} to status: {status}")
    if task_id not in _tasks_db:
        raise ValueError(f"Task with ID {task_id} not found")

    try:
        status_enum = TaskStatus(status)
    except ValueError:
        raise ValueError(f"Invalid status: {status}")

    task = _tasks_db[task_id]
    updated = task.model_copy(update={"status": status_enum})
    _tasks_db[task_id] = updated

    return updated


@command
async def delete_task(task_id: int) -> bool:
    """Delete a task."""
    if task_id in _tasks_db:
        del _tasks_db[task_id]
        return True
    return False


@command
async def get_task_stats() -> TaskStats:
    """Get statistics about all tasks."""
    tasks = list(_tasks_db.values())

    by_priority = {p.value: 0 for p in TaskPriority}
    for task in tasks:
        by_priority[task.priority.value] += 1

    return TaskStats(
        total=len(tasks),
        todo=sum(1 for t in tasks if t.status == TaskStatus.TODO),
        in_progress=sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
        done=sum(1 for t in tasks if t.status == TaskStatus.DONE),
        cancelled=sum(1 for t in tasks if t.status == TaskStatus.CANCELLED),
        by_priority=by_priority,
    )


@command
async def list_labels() -> list[TaskLabel]:
    """Get all available labels."""
    return list(_labels_db.values())


@command
async def create_label(name: str, color: str = "#808080") -> TaskLabel:
    """Create a new label."""
    global _next_label_id

    label = TaskLabel(id=_next_label_id, name=name, color=color)
    _labels_db[_next_label_id] = label
    _next_label_id += 1

    return label
