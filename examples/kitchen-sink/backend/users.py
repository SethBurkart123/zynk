"""
Users Module

Demonstrates basic CRUD-like commands with Pydantic models.
"""


from pydantic import BaseModel, Field

from zynk import command


class User(BaseModel):
    """A user in the system."""
    id: int
    name: str
    email: str
    is_active: bool = True


class CreateUserRequest(BaseModel):
    """Request to create a new user."""
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")


class UpdateUserRequest(BaseModel):
    """Request to update an existing user."""
    name: str | None = None
    email: str | None = None
    is_active: bool | None = None


# In-memory "database" for demo purposes
_users_db: dict[int, User] = {
    1: User(id=1, name="Alice", email="alice@example.com"),
    2: User(id=2, name="Bob", email="bob@example.com"),
    3: User(id=3, name="Charlie", email="charlie@example.com"),
}
_next_id = 4


@command
async def get_user(user_id: int) -> User:
    """
    Get a user by ID.

    Raises an error if the user doesn't exist.
    """
    if user_id not in _users_db:
        raise ValueError(f"User with ID {user_id} not found")
    return _users_db[user_id]


@command
async def list_users(active_only: bool = False) -> list[User]:
    """
    List all users.

    Optionally filter to only active users.
    """
    users = list(_users_db.values())
    if active_only:
        users = [u for u in users if u.is_active]
    return users


@command
async def create_user(name: str, email: str) -> User:
    """
    Create a new user.

    Returns the created user with their assigned ID.
    """
    global _next_id

    # Check for duplicate email
    for user in _users_db.values():
        if user.email == email:
            raise ValueError(f"User with email {email} already exists")

    user = User(id=_next_id, name=name, email=email)
    _users_db[_next_id] = user
    _next_id += 1

    return user


@command
async def update_user(
    user_id: int,
    name: str | None = None,
    email: str | None = None,
    is_active: bool | None = None,
) -> User:
    """
    Update an existing user.

    Only provided fields will be updated.
    """
    if user_id not in _users_db:
        raise ValueError(f"User with ID {user_id} not found")

    user = _users_db[user_id]

    if name is not None:
        user = user.model_copy(update={"name": name})
    if email is not None:
        user = user.model_copy(update={"email": email})
    if is_active is not None:
        user = user.model_copy(update={"is_active": is_active})

    _users_db[user_id] = user
    return user


@command
async def delete_user(user_id: int) -> bool:
    """
    Delete a user.

    Returns True if the user was deleted, False if they didn't exist.
    """
    if user_id in _users_db:
        del _users_db[user_id]
        return True
    return False


@command
async def search_users(query: str) -> list[User]:
    """
    Search for users by name or email.

    Returns users whose name or email contains the search query.
    """
    query = query.lower()
    return [
        user for user in _users_db.values()
        if query in user.name.lower() or query in user.email.lower()
    ]
