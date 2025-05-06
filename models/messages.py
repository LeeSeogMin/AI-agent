#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Message models for agent communication in the LangGraph 멀티 에이전트 시스템
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from models.base import AgentType, IdentifiedModel, MessageType, Priority, TaskStatus


class MessageContent(BaseModel):
    """Content of a message between agents"""
    action: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Message(IdentifiedModel):
    """
    Standard message format for agent communication
    Based on the message structure from architecture.md
    """
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender: str
    receiver: str = "broadcast"  # Default to broadcast if not specified
    message_type: MessageType
    priority: Priority = Priority.NORMAL
    content: MessageContent = Field(default_factory=MessageContent)
    references: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


class TaskRequest(Message):
    """Task request from supervisor to specialist agent"""
    def __init__(self, **data):
        super().__init__(
            message_type=MessageType.REQUEST,
            **data
        )


class TaskResponse(Message):
    """Task response from specialist agent to supervisor"""
    def __init__(self, **data):
        super().__init__(
            message_type=MessageType.RESPONSE,
            **data
        )


class InfoRequest(Message):
    """Information request between agents"""
    query_type: str
    query: str

    def __init__(self, **data):
        super().__init__(
            message_type=MessageType.REQUEST,
            **data
        )


class StatusUpdate(Message):
    """Status update from any agent"""
    progress: float = 0.0
    next_steps: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(
            message_type=MessageType.NOTIFICATION,
            **data
        )


class ErrorNotification(Message):
    """Error notification from any agent"""
    error_type: str
    description: str
    severity: str = "normal"

    def __init__(self, **data):
        super().__init__(
            message_type=MessageType.ERROR,
            **data
        )


class TaskCompletion(Message):
    """Task completion notification from any agent"""
    task_id: str
    output: Dict[str, Any]
    confidence: float = 1.0

    def __init__(self, **data):
        super().__init__(
            message_type=MessageType.NOTIFICATION,
            status=TaskStatus.COMPLETED,
            **data
        ) 