#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# Corrected imports based on exploration
from mcp.server import FastMCP
from mcp.server.fastmcp.tools import ToolManager # Corrected import
from mcp.server.fastmcp.exceptions import ToolError

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Define the base directory for agent communication relative to project root
AGENT_COMM_DIR_NAME = "agent_docs/multi_agent"

# --- Helper Functions ---

def get_comm_dir(project_root: str) -> Path:
    """Gets the absolute path to the communication directory for a project."""
    log.debug(f"Resolving comm dir for project root: {project_root}")
    root_path = Path(project_root).resolve()
    comm_path = root_path / AGENT_COMM_DIR_NAME
    try:
        comm_path.mkdir(parents=True, exist_ok=True)
        log.debug(f"Ensured comm dir exists: {comm_path}")
    except OSError as e:
        log.error(f"Failed to create comm dir {comm_path}: {e}")
        raise ToolError(f"Could not create or access agent communication directory: {comm_path}") from e
    return comm_path

def read_json_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Reads and parses a JSON file, returning None on error."""
    log.debug(f"Reading JSON file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            log.debug(f"Successfully read and parsed JSON from: {filepath}")
            return data
    except FileNotFoundError:
        log.warning(f"JSON file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON from file {filepath}: {e}")
        return None
    except OSError as e:
        log.error(f"OS error reading JSON file {filepath}: {e}")
        return None

def write_json_file(filepath: Path, data: Dict[str, Any]) -> bool:
    """Writes data to a JSON file, returning True on success."""
    log.debug(f"Writing JSON file: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        log.debug(f"Successfully wrote JSON to: {filepath}")
        return True
    except (OSError, TypeError) as e:
        log.error(f"Error writing JSON file {filepath}: {e}")
        return False

def get_current_timestamp() -> str:
    """Returns the current time in ISO 8601 format (UTC)."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

# --- Pydantic Models for Tool Inputs/Outputs (FastMCP uses these) ---

class CheckRequestsOutput(BaseModel):
    pending_requests: List[str] = Field(description="List of full paths to pending request JSON files.")

class GetRequestInput(BaseModel):
    request_filepath: str = Field(description="Full path to the request JSON file.")

class UpdateStatusInput(BaseModel):
    request_filepath: str = Field(description="Full path to the request JSON file.")
    new_status: Literal["answered", "partial", "error"] = Field(description="The new status.")
    error_message: Optional[str] = Field(None, description="Optional message if status is 'error'.")

class AddAnswerInput(BaseModel):
    request_filepath: str = Field(description="Full path to the request JSON file.")
    question_id: str = Field(description="The unique ID of the question to answer.")
    answer: Dict[str, Any] = Field(description="The answer object (should follow protocol: response_text, optional filepaths).")

class CreateFileInput(BaseModel):
    project_root: str = Field(description="The root directory of the project.")
    task_id: str = Field(description="The unique ID of the task this file relates to.")
    content: str = Field(description="The content to write to the file.")
    filename_suffix: str = Field(description="Suffix for the filename (e.g., '_response.md', '_code.py').")

class CreateFileOutput(BaseModel):
    filepath: str = Field(description="The full path to the created file.")

# --- Tool Implementations (Define functions first) ---

async def check_for_new_requests(project_root: str) -> CheckRequestsOutput:
    """
    Scans the project's agent_docs/multi_agent directory for pending requests (JSON files with status 'pending').
    """
    log.info(f"[check_for_new_requests] Scanning project root: {project_root}")
    comm_dir = get_comm_dir(project_root)
    pending_requests = []
    try:
        for item in comm_dir.glob('*.json'):
            if item.is_file():
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data and data.get("status") == "pending":
                            pending_requests.append(str(item.resolve()))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                     log.warning(f"Could not read or parse {item} to check status: {e}")
                     continue
        log.info(f"[check_for_new_requests] Found {len(pending_requests)} pending requests.")
        return CheckRequestsOutput(pending_requests=pending_requests)
    except Exception as e:
        log.error(f"[check_for_new_requests] Error scanning directory {comm_dir}: {e}")
        raise ToolError(f"Failed to scan for requests: {e}") from e

async def get_request_summary(request_filepath: str) -> Dict[str, Any]:
    """
    Reads a request JSON and returns a concise summary (task_id, questions[id, text], desired_output). Use this for quick assessment.
    """
    log.info(f"[get_request_summary] Reading summary for: {request_filepath}")
    filepath = Path(request_filepath).resolve()
    if not filepath.is_file():
         raise ToolError(f"Request file not found: {request_filepath}")
    data = read_json_file(filepath)
    if not data:
        raise ToolError(f"Failed to read or parse request file: {request_filepath}")
    summary = {
        "task_id": data.get("task_id"),
        "questions": [
            {"question_id": q.get("question_id"), "text": q.get("text")}
            for q in data.get("questions", []) if isinstance(q, dict)
        ],
        "desired_output": data.get("desired_output"),
    }
    summary_filtered = {k: v for k, v in summary.items() if v is not None}
    log.info(f"[get_request_summary] Summary generated for task_id: {summary_filtered.get('task_id')}")
    return summary_filtered

async def get_request_details(request_filepath: str) -> Dict[str, Any]:
    """
    Reads and returns the full content of a request JSON file. Use when the summary is insufficient.
    """
    log.info(f"[get_request_details] Reading full details for: {request_filepath}")
    filepath = Path(request_filepath).resolve()
    if not filepath.is_file():
         raise ToolError(f"Request file not found: {request_filepath}")
    data = read_json_file(filepath)
    if not data:
        raise ToolError(f"Failed to read or parse request file: {request_filepath}")
    log.info(f"[get_request_details] Full details retrieved for task_id: {data.get('task_id')}")
    return data

async def update_request_status(input: UpdateStatusInput) -> None:
    """
    Updates the status ('answered', 'partial', 'error') and response_timestamp of a request file.
    """
    log.info(f"[update_request_status] Updating status to '{input.new_status}' for: {input.request_filepath}")
    filepath = Path(input.request_filepath).resolve()
    if not filepath.is_file():
         raise ToolError(f"Request file not found: {input.request_filepath}")
    data = read_json_file(filepath)
    if not data:
        raise ToolError(f"Failed to read or parse request file for update: {input.request_filepath}")
    data["status"] = input.new_status
    data["response_timestamp"] = get_current_timestamp()
    if input.new_status == "error" and input.error_message:
        data["error_details"] = input.error_message
        log.warning(f"[update_request_status] Status set to error for {input.request_filepath}: {input.error_message}")
    if not write_json_file(filepath, data):
        log.error(f"[update_request_status] Failed to write updated status for: {input.request_filepath}")
        raise ToolError(f"Failed to write updated file: {input.request_filepath}")
    log.info(f"[update_request_status] Successfully updated status for task_id: {data.get('task_id')}")

async def add_answer_to_request(input: AddAnswerInput) -> None:
    """
    Adds an answer object to a specific question within a request file. Updates response_timestamp.
    """
    log.info(f"[add_answer_to_request] Adding answer to question '{input.question_id}' in: {input.request_filepath}")
    filepath = Path(input.request_filepath).resolve()
    if not filepath.is_file():
         raise ToolError(f"Request file not found: {input.request_filepath}")
    data = read_json_file(filepath)
    if not data:
        raise ToolError(f"Failed to read or parse request file for adding answer: {input.request_filepath}")
    updated = False
    if "questions" in data and isinstance(data["questions"], list):
        for question in data["questions"]:
            if isinstance(question, dict) and question.get("question_id") == input.question_id:
                question["answer"] = input.answer
                updated = True
                break
    if not updated:
        log.error(f"[add_answer_to_request] Question ID '{input.question_id}' not found in {input.request_filepath}")
        raise ToolError(f"Question ID '{input.question_id}' not found.")
    data["response_timestamp"] = get_current_timestamp()
    if not write_json_file(filepath, data):
        log.error(f"[add_answer_to_request] Failed to write file after adding answer: {input.request_filepath}")
        raise ToolError(f"Failed to write updated file: {input.request_filepath}")
    log.info(f"[add_answer_to_request] Successfully added answer to question '{input.question_id}' for task_id: {data.get('task_id')}")

async def create_associated_file(input: CreateFileInput) -> CreateFileOutput:
    """
    Creates a new file (e.g., response, code example) associated with a task in the project's agent_docs/multi_agent directory.
    """
    log.info(f"[create_associated_file] Creating file for task '{input.task_id}' with suffix '{input.filename_suffix}' in project: {input.project_root}")
    comm_dir = get_comm_dir(input.project_root)
    safe_task_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in input.task_id)
    safe_suffix = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in input.filename_suffix)
    safe_suffix = safe_suffix.lstrip('./\\')
    if not safe_suffix:
        safe_suffix = "_file"
    filename = f"{safe_task_id}{safe_suffix}"
    filepath = comm_dir / filename
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(input.content)
        output_filepath = str(filepath.resolve())
        log.info(f"[create_associated_file] Successfully created file: {output_filepath}")
        return CreateFileOutput(filepath=output_filepath)
    except OSError as e:
        log.error(f"[create_associated_file] Error writing file {filepath}: {e}")
        raise ToolError(f"Failed to create associated file: {e}") from e
    except Exception as e:
        log.error(f"[create_associated_file] Unexpected error creating file {filepath}: {e}")
        raise ToolError(f"Unexpected error creating file: {e}") from e

# --- Tool Registration ---

# Create a ToolManager instance
tools = ToolManager()

# Register functions using the add_tool method
tools.add_tool(check_for_new_requests)
tools.add_tool(get_request_summary)
tools.add_tool(get_request_details)
tools.add_tool(update_request_status)
tools.add_tool(add_answer_to_request)
tools.add_tool(create_associated_file)


# --- Main Execution ---
if __name__ == "__main__":
    log.info("Starting Multi-Agent Comm Server (FastMCP)...")
    server = FastMCP(
        tools=tools, # Pass the ToolManager instance
        server_name="multi-agent-comm-fastmcp", # Match the name in settings.json
        server_version="0.1.0",
    )
    asyncio.run(server.run())