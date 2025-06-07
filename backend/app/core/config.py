"""
Configuration settings for NeuroGraph AI
Production-grade configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "NeuroGraph AI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "https://neurograph-ai.com"],
        env="ALLOWED_ORIGINS"
    )
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # AI Models
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    MODEL_CACHE_DIR: str = Field(default="./models", env="MODEL_CACHE_DIR")
    
    # Protein Analysis
    PROTEIN_STRUCTURE_CACHE_SIZE: int = Field(default=1000, env="PROTEIN_STRUCTURE_CACHE_SIZE")
    ALPHAFOLD_DB_PATH: Optional[str] = Field(default=None, env="ALPHAFOLD_DB_PATH")
    
    # Knowledge Graph
    ORIGINTRAIL_NODE_URL: str = Field(
        default="https://dkg-testnet.origintrail.io",
        env="ORIGINTRAIL_NODE_URL"
    )
    ORIGINTRAIL_API_KEY: Optional[str] = Field(default=None, env="ORIGINTRAIL_API_KEY")
    
    # External APIs - Added all the missing ones
    UNIPROT_API_KEY: Optional[str] = Field(default=None, env="UNIPROT_API_KEY")
    CHEMBL_API_KEY: Optional[str] = Field(default=None, env="CHEMBL_API_KEY")
    PUBMED_API_KEY: Optional[str] = Field(default=None, env="PUBMED_API_KEY")
    CHEMBL_BASE_URL: str = "https://www.ebi.ac.uk/chembl/api/data"
    UNIPROT_BASE_URL: str = "https://rest.uniprot.org"
    
    # Blockchain/Wallet (for OriginTrail DKG)
    WALLET_ADDRESS: Optional[str] = Field(default=None, env="WALLET_ADDRESS")
    PRIVATE_KEY: Optional[str] = Field(default=None, env="PRIVATE_KEY")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # File Storage
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Performance
    WORKER_PROCESSES: int = Field(default=4, env="WORKER_PROCESSES")
    MAX_REQUESTS: int = Field(default=1000, env="MAX_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("MODEL_CACHE_DIR", "UPLOAD_DIR")
    def create_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # This is the key fix - allows extra fields without validation errors
        extra = "ignore"


# Global settings instance
settings = Settings()
