#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
State models for managing the shared state and context in the LangGraph 멀티 에이전트 시스템
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from models.base import AgentType, ProcessingStage, TaskStatus


class ConversationMessage(BaseModel):
    """A message in the conversation history"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str  # agent_id or "user"
    message: str


class TaskInfo(BaseModel):
    """Information about a task in the system"""
    agent: AgentType
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result_location: Optional[str] = None  # Reference to where the result is stored


class SharedKnowledge(BaseModel):
    """Shared knowledge between agents"""
    # Dictionary for arbitrary key-value pairs
    key_values: Dict[str, Any] = Field(default_factory=dict)
    
    # Lists for specific types of information
    collected_information: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the key_values dictionary"""
        return self.key_values.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the key_values dictionary"""
        self.key_values[key] = value
    
    def add_information(self, info: Dict[str, Any]) -> None:
        """Add collected information"""
        if "timestamp" not in info:
            info["timestamp"] = datetime.utcnow().isoformat()
        self.collected_information.append(info)
    
    def add_analysis(self, analysis: Dict[str, Any]) -> None:
        """Add analysis result"""
        if "timestamp" not in analysis:
            analysis["timestamp"] = datetime.utcnow().isoformat()
        self.analysis_results.append(analysis)
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add decision"""
        if "timestamp" not in decision:
            decision["timestamp"] = datetime.utcnow().isoformat()
        self.decisions.append(decision)


class AgentContext(BaseModel):
    """Context for a specific agent"""
    agent_id: AgentType
    private_state: Dict[str, Any] = Field(default_factory=dict)
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    
    def update(self, key: str, value: Any) -> None:
        """Update the private state"""
        self.private_state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the private state"""
        return self.private_state.get(key, default)


class SharedContext(BaseModel):
    """
    Shared context across all agents in the system
    Based on the shared context structure from architecture.md
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_query: str
    parsed_intent: Optional[str] = None
    current_stage: ProcessingStage = ProcessingStage.PLANNING
    active_agents: List[AgentType] = Field(default_factory=list)
    
    task_status: Dict[str, TaskInfo] = Field(default_factory=dict)
    shared_knowledge: SharedKnowledge = Field(default_factory=SharedKnowledge)
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    
    agent_contexts: Dict[AgentType, AgentContext] = Field(default_factory=dict)
    
    def add_message(self, agent: str, message: str) -> None:
        """Add a message to the conversation history"""
        self.conversation_history.append(
            ConversationMessage(agent=agent, message=message)
        )
    
    def update_task(self, task_id: str, **kwargs) -> None:
        """Update a task's status and information"""
        if task_id not in self.task_status:
            # If task doesn't exist, we need at least the agent
            if "agent" not in kwargs:
                raise ValueError("Agent must be specified for new task")
            self.task_status[task_id] = TaskInfo(agent=kwargs.pop("agent"))
        
        # Update the existing task with any provided kwargs
        for key, value in kwargs.items():
            setattr(self.task_status[task_id], key, value)
    
    def get_agent_context(self, agent_type: AgentType) -> AgentContext:
        """Get or create an agent's context"""
        if agent_type not in self.agent_contexts:
            self.agent_contexts[agent_type] = AgentContext(agent_id=agent_type)
        return self.agent_contexts[agent_type]
    
    class Config:
        arbitrary_types_allowed = True 