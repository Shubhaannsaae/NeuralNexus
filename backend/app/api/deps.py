"""
Dependency injection for FastAPI endpoints
Production-grade dependencies for authentication, database, and services
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.core.security import security_manager, rate_limiter
from app.services.protein_service import protein_service
from app.services.kg_service import kg_service
from app.services.ai_service import ai_service
from app.services.eliza_service import eliza_service

logger = logging.getLogger(__name__)

# Security dependencies
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    return {"user_id": user_id, "payload": payload}


async def get_current_active_user(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get current active user"""
    # Add additional checks for user status if needed
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Get user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def rate_limit_dependency(max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting dependency factory"""
    
    async def rate_limit_check(
        current_user: Optional[dict] = Depends(get_optional_user)
    ):
        # Use user ID or IP address for rate limiting
        identifier = current_user.get("user_id") if current_user else "anonymous"
        
        if not rate_limiter.is_allowed(identifier):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return True
    
    return rate_limit_check


# Service dependencies
async def get_protein_service():
    """Get protein analysis service"""
    if not protein_service.esm_model:
        await protein_service.initialize_models()
    return protein_service


async def get_kg_service():
    """Get knowledge graph service"""
    return kg_service


async def get_ai_service():
    """Get AI service"""
    if not ai_service.models:
        await ai_service.initialize()
    return ai_service


async def get_eliza_service():
    """Get Eliza integration service"""
    return eliza_service


# Database dependencies
async def get_db_session() -> AsyncSession:
    """Get database session"""
    async for session in get_db():
        yield session


# Validation dependencies
def validate_uniprot_id(uniprot_id: str) -> str:
    """Validate UniProt ID format"""
    import re
    
    # UniProt ID pattern: 6-10 alphanumeric characters
    pattern = r'^[A-Z0-9]{6,10}$'
    
    if not re.match(pattern, uniprot_id.upper()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UniProt ID format"
        )
    
    return uniprot_id.upper()


def validate_smiles(smiles: str) -> str:
    """Validate SMILES string"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        return smiles
    except ImportError:
        # If RDKit not available, do basic validation
        if not smiles or len(smiles) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid SMILES string"
            )
        return smiles
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid SMILES string"
        )


def validate_protein_sequence(sequence: str) -> str:
    """Validate protein sequence"""
    import re
    
    # Valid amino acid letters
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Remove whitespace and convert to uppercase
    sequence = sequence.replace(' ', '').replace('\n', '').upper()
    
    # Check for invalid characters
    if not all(aa in valid_aa for aa in sequence):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid amino acid sequence"
        )
    
    # Check minimum length
    if len(sequence) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sequence too short (minimum 10 amino acids)"
        )
    
    # Check maximum length
    if len(sequence) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sequence too long (maximum 5000 amino acids)"
        )
    
    return sequence


# Pagination dependencies
class PaginationParams:
    """Pagination parameters"""
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100
    ):
        self.skip = max(0, skip)
        self.limit = min(1000, max(1, limit))  # Max 1000 items per page


def get_pagination_params(
    skip: int = 0,
    limit: int = 100
) -> PaginationParams:
    """Get pagination parameters"""
    return PaginationParams(skip=skip, limit=limit)


# Query parameter dependencies
class SearchParams:
    """Search parameters"""
    
    def __init__(
        self,
        q: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        filters: Optional[dict] = None
    ):
        self.query = q
        self.sort_by = sort_by
        self.sort_order = sort_order.lower()
        self.filters = filters or {}
        
        # Validate sort order
        if self.sort_order not in ["asc", "desc"]:
            self.sort_order = "desc"


def get_search_params(
    q: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> SearchParams:
    """Get search parameters"""
    return SearchParams(
        q=q,
        sort_by=sort_by,
        sort_order=sort_order
    )


# Cache dependencies
class CacheParams:
    """Cache control parameters"""
    
    def __init__(
        self,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        force_refresh: bool = False
    ):
        self.use_cache = use_cache
        self.cache_ttl = max(60, min(86400, cache_ttl))  # 1 minute to 24 hours
        self.force_refresh = force_refresh


def get_cache_params(
    use_cache: bool = True,
    cache_ttl: int = 3600,
    force_refresh: bool = False
) -> CacheParams:
    """Get cache parameters"""
    return CacheParams(
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        force_refresh=force_refresh
    )


# Error handling dependencies
async def handle_service_errors():
    """Handle service-level errors"""
    try:
        yield
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal service error"
        )


# Logging dependencies
def get_request_logger():
    """Get request-specific logger"""
    return logging.getLogger("neurograph.api")


# Feature flag dependencies
class FeatureFlags:
    """Feature flags for API endpoints"""
    
    def __init__(self):
        self.experimental_features = True
        self.ai_predictions = True
        self.knowledge_graph = True
        self.eliza_integration = True


def get_feature_flags() -> FeatureFlags:
    """Get feature flags"""
    return FeatureFlags()
