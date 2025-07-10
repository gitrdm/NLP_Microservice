"""Configuration settings for the NLP microservice."""

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Server configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8161
    max_workers: int = 10
    
    # NLTK configuration
    nltk_data_path: Optional[str] = None
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Development settings
    debug: bool = False


# Global settings instance
settings = Settings()
