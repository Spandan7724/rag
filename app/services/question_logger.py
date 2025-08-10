"""
Question Logger Service - Logs questions, document URLs, and responses to JSON files
"""
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import uuid

from app.core.config import Settings

settings = Settings()


@dataclass
class QuestionLog:
    """Individual question log entry"""
    question_id: int
    question: str
    timestamp: str
    processing_time: float
    answer: str
    answer_length: int
    sources_used: int
    similarity_scores: List[float]
    error: Optional[str] = None


@dataclass
class SessionLog:
    """Complete session log with all questions"""
    session_id: str
    timestamp: str
    document_url: str
    doc_id: str
    questions: List[QuestionLog]
    metadata: Dict[str, Any]


class QuestionLogger:
    """Thread-safe question logger that saves to JSON files"""
    
    def __init__(self):
        """Initialize question logger"""
        self._lock = threading.Lock()
        self.log_dir = Path(settings.question_log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create archived directory for old logs
        self.archived_dir = self.log_dir / "archived"
        self.archived_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Question Logger initialized: {self.log_dir}")
        print(f"  - Archived directory: {self.archived_dir}")
        print(f"  - Logging enabled: {settings.enable_question_logging}")
        print(f"  - Full responses: {settings.log_full_responses}")
        print(f"  - Retention: {settings.log_retention_days} days")
    
    def start_session(
        self, 
        document_url: str, 
        doc_id: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Start a new question session
        
        Args:
            document_url: URL of the document being processed
            doc_id: Document ID from RAG system
            metadata: Additional metadata (embedding provider, etc.)
            
        Returns:
            Session ID for this question session
        """
        if not settings.enable_question_logging:
            return ""
        
        session_id = str(uuid.uuid4())[:8]  # Short session ID
        timestamp = datetime.now().isoformat()
        
        # Initialize session log
        session_log = SessionLog(
            session_id=session_id,
            timestamp=timestamp,
            document_url=document_url,
            doc_id=doc_id,
            questions=[],
            metadata={
                **metadata,
                "session_started": timestamp,
                "logging_version": "1.0"
            }
        )
        
        print(f"Started question logging session: {session_id}")
        print(f"  - Document: {document_url}")
        print(f"  - Doc ID: {doc_id}")
        
        return session_id
    
    def log_question(
        self,
        session_id: str,
        question_id: int,
        question: str,
        processing_time: float,
        answer: str,
        sources_used: int,
        similarity_scores: List[float],
        error: Optional[str] = None
    ) -> None:
        """
        Log a single question and its response
        
        Args:
            session_id: Session ID from start_session
            question_id: Sequential question number (1, 2, 3, etc.)
            question: The question asked
            processing_time: Time taken to process the question
            answer: Generated answer
            sources_used: Number of sources used
            similarity_scores: Top similarity scores from vector search
            error: Error message if question failed
        """
        if not settings.enable_question_logging or not session_id:
            return
        
        timestamp = datetime.now().isoformat()
        
        # Create question log entry
        question_log = QuestionLog(
            question_id=question_id,
            question=question,
            timestamp=timestamp,
            processing_time=processing_time,
            answer=answer if settings.log_full_responses else answer[:200] + "..." if len(answer) > 200 else answer,
            answer_length=len(answer),
            sources_used=sources_used,
            similarity_scores=similarity_scores[:3] if similarity_scores else [],  # Top 3 scores
            error=error
        )
        
        print(f"Logging question {question_id} for session {session_id}")
        print(f"  - Question: {question[:50]}{'...' if len(question) > 50 else ''}")
        print(f"  - Processing time: {processing_time:.2f}s")
        print(f"  - Answer length: {len(answer)} characters")
        print(f"  - Sources used: {sources_used}")
        
        # Store temporarily - will be saved when session ends
        if not hasattr(self, '_active_sessions'):
            self._active_sessions = {}
        
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = []
        
        self._active_sessions[session_id].append(question_log)
    
    def end_session(
        self,
        session_id: str,
        document_url: str,
        doc_id: str,
        session_metadata: Dict[str, Any]
    ) -> bool:
        """
        End a question session and save all logged questions to file
        
        Args:
            session_id: Session ID from start_session
            document_url: URL of the document
            doc_id: Document ID
            session_metadata: Session metadata (total questions, etc.)
            
        Returns:
            True if successfully saved, False otherwise
        """
        if not settings.enable_question_logging or not session_id:
            return False
        
        if not hasattr(self, '_active_sessions') or session_id not in self._active_sessions:
            print(f"Warning: No active session found for {session_id}")
            return False
        
        try:
            with self._lock:
                # Get logged questions for this session
                questions = self._active_sessions[session_id]
                
                # Create complete session log
                session_log = SessionLog(
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    document_url=document_url,
                    doc_id=doc_id,
                    questions=questions,
                    metadata={
                        **session_metadata,
                        "session_ended": datetime.now().isoformat(),
                        "total_questions_logged": len(questions)
                    }
                )
                
                # Generate filename with date and session ID
                today = datetime.now().strftime("%Y-%m-%d")
                filename = f"questions_{today}_{session_id}.json"
                filepath = self.log_dir / filename
                
                # Save to JSON file
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(asdict(session_log), f, indent=2, ensure_ascii=False)
                
                # Clean up active session
                del self._active_sessions[session_id]
                
                print(f"Question session {session_id} saved to: {filename}")
                print(f"  - Total questions logged: {len(questions)}")
                print(f"  - File size: {filepath.stat().st_size} bytes")
                
                return True
                
        except Exception as e:
            print(f"Error saving question log for session {session_id}: {e}")
            return False
    
    def cleanup_old_logs(self) -> int:
        """
        Clean up old log files based on retention policy
        
        Returns:
            Number of files archived/deleted
        """
        if not settings.enable_question_logging:
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=settings.log_retention_days)
            archived_count = 0
            
            for log_file in self.log_dir.glob("questions_*.json"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    # Move to archived directory
                    archived_path = self.archived_dir / log_file.name
                    log_file.rename(archived_path)
                    archived_count += 1
                    print(f"Archived old log file: {log_file.name}")
            
            print(f"Archived {archived_count} old log files")
            return archived_count
            
        except Exception as e:
            print(f"Error cleaning up old logs: {e}")
            return 0
    
    def get_recent_logs(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent question logs for analysis
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of session log dictionaries
        """
        if not settings.enable_question_logging:
            return []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_logs = []
            
            for log_file in self.log_dir.glob("questions_*.json"):
                if log_file.stat().st_mtime >= cutoff_date.timestamp():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                            recent_logs.append(log_data)
                    except Exception as e:
                        print(f"Error reading log file {log_file}: {e}")
            
            # Sort by timestamp
            recent_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return recent_logs
            
        except Exception as e:
            print(f"Error retrieving recent logs: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get question logging statistics"""
        try:
            total_files = len(list(self.log_dir.glob("questions_*.json")))
            archived_files = len(list(self.archived_dir.glob("questions_*.json")))
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in self.log_dir.glob("questions_*.json"))
            
            # Count recent sessions and questions
            recent_logs = self.get_recent_logs(days=7)
            recent_sessions = len(recent_logs)
            recent_questions = sum(len(log.get('questions', [])) for log in recent_logs)
            
            return {
                "enabled": settings.enable_question_logging,
                "log_directory": str(self.log_dir),
                "total_log_files": total_files,
                "archived_files": archived_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "retention_days": settings.log_retention_days,
                "recent_sessions_7days": recent_sessions,
                "recent_questions_7days": recent_questions,
                "active_sessions": len(getattr(self, '_active_sessions', {})),
                "full_response_logging": settings.log_full_responses
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "enabled": settings.enable_question_logging
            }


# Singleton instance
_question_logger = None

def get_question_logger() -> QuestionLogger:
    """Get singleton question logger instance"""
    global _question_logger
    if _question_logger is None:
        _question_logger = QuestionLogger()
    return _question_logger