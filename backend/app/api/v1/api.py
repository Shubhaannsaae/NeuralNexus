"""
API router configuration
Main API router that includes all endpoint modules
"""

from fastapi import APIRouter
from app.api.v1.endpoints import proteins, knowledge_graph, drugs, hypotheses

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    proteins.router,
    prefix="/proteins",
    tags=["proteins"]
)

api_router.include_router(
    knowledge_graph.router,
    prefix="/knowledge-graph",
    tags=["knowledge-graph"]
)

api_router.include_router(
    drugs.router,
    prefix="/drugs",
    tags=["drugs"]
)

api_router.include_router(
    hypotheses.router,
    prefix="/hypotheses",
    tags=["hypotheses"]
)
