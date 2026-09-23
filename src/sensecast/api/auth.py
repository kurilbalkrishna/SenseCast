"""Authentication / authorisation hook.

This module is deliberately small so a real identity provider can replace it
without touching any endpoint: endpoints only depend on ``current_user`` and
``ensure_store_access``.

Modes (env SENSECAST_AUTH_MODE):
  off     (default, local demo) every request is the 'demo' planner
  header  identity comes from X-User / X-Role / X-Store headers set by a trusted
          gateway or reverse proxy. Never expose this mode directly to the internet.

Roles
  admin          everything
  planner        all stores, may approve / override orders
  store_manager  own store only, may approve / override orders
  analyst        read-only
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Header, HTTPException

ROLES = {"admin", "planner", "store_manager", "analyst"}
CAN_DECIDE = {"admin", "planner", "store_manager"}


@dataclass(frozen=True)
class User:
    name: str
    role: str
    store_id: str | None = None


def auth_mode() -> str:
    return os.getenv("SENSECAST_AUTH_MODE", "off").lower()


def current_user(
    x_user: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_store: str | None = Header(default=None),
) -> User:
    if auth_mode() == "off":
        return User("demo", "planner")
    if not x_user or not x_role:
        raise HTTPException(status_code=401, detail="Missing X-User / X-Role headers")
    if x_role not in ROLES:
        raise HTTPException(status_code=403, detail=f"Unknown role '{x_role}'")
    if x_role == "store_manager" and not x_store:
        raise HTTPException(status_code=403, detail="store_manager requires X-Store")
    return User(x_user[:64], x_role, x_store)


def ensure_store_access(user: User, store_id: str) -> None:
    if user.role == "store_manager" and user.store_id != store_id:
        raise HTTPException(status_code=403, detail="Store managers can only access their own store")


def ensure_can_decide(user: User) -> None:
    if user.role not in CAN_DECIDE:
        raise HTTPException(status_code=403, detail=f"Role '{user.role}' cannot approve or override orders")
