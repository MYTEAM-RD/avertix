# ==============================================================================
# AUTHENTICATION MODULE FOR AVERTIX
# ==============================================================================

import json
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import time

# --- User Management ---
USERS_FILE = "data/users.json"

def ensure_users_file():
    """Ensure users.json file exists"""
    if not os.path.exists(USERS_FILE):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)

def load_users() -> List[Dict]:
    """Load users from JSON file"""
    ensure_users_file()
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_users(users: List[Dict]):
    """Save users to JSON file"""
    ensure_users_file()
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return hash_password(plain_password) == hashed_password

def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username"""
    users = load_users()
    for user in users:
        if user['username'] == username:
            return user
    return None

def create_user(username: str, password: str) -> Dict:
    """Create new user"""
    users = load_users()
    
    # Check if user already exists
    if get_user_by_username(username):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    user_id = len(users) + 1
    new_user = {
        "id": user_id,
        "username": username,
        "password": hash_password(password),
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    
    users.append(new_user)
    save_users(users)
    
    # Return user without password
    user_data = new_user.copy()
    del user_data['password']
    return user_data

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user"""
    user = get_user_by_username(username)
    if user and verify_password(password, user['password']):
        # Update last login
        users = load_users()
        for i, u in enumerate(users):
            if u['id'] == user['id']:
                users[i]['last_login'] = datetime.now().isoformat()
                break
        save_users(users)
        
        # Return user without password
        user_data = user.copy()
        del user_data['password']
        return user_data
    return None

# --- Session Management ---
active_sessions: Dict[str, Dict] = {}

def create_session(user: Dict) -> str:
    """Create a new session for user"""
    session_token = secrets.token_urlsafe(32)
    active_sessions[session_token] = {
        "user": user,
        "created_at": time.time(),
        "expires_at": time.time() + (24 * 60 * 60)  # 24 hours
    }
    return session_token

def get_current_user(request: Request) -> Optional[Dict]:
    """Get current user from session"""
    session_token = request.cookies.get("session_token")
    if not session_token or session_token not in active_sessions:
        return None
    
    session = active_sessions[session_token]
    if time.time() > session["expires_at"]:
        # Session expired
        del active_sessions[session_token]
        return None
    
    return session["user"]

def destroy_session(session_token: str):
    """Destroy a session"""
    if session_token in active_sessions:
        del active_sessions[session_token]

def require_auth(request: Request):
    """Dependency that requires authentication"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
