"""
Knowledge Graph API endpoints
Production-grade endpoints for knowledge graph operations and queries
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import logging

from app.api.deps import (
    get_db_session, get_kg_service, get_current_user,
    get_pagination_params, PaginationParams,
    rate_limit_dependency
)
from app.services.kg_service import KnowledgeGraphService
from app.models.knowledge_graph import KnowledgeNode, KnowledgeEdge, Hypothesis

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class NodeCreateRequest(BaseModel):
    """Request model for creating knowledge node"""
    node_type: str = Field(..., description="Type of node (protein, drug, disease, pathway)")
    external_id: str = Field(..., description="External database identifier")
    name: str = Field(..., description="Node name")
    description: Optional[str] = Field(None, description="Node description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    ontology_source: Optional[str] = Field(None, description="Ontology source")
    ontology_id: Optional[str] = Field(None, description="Ontology identifier")


class EdgeCreateRequest(BaseModel):
    """Request model for creating knowledge edge"""
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    evidence_type: str = Field(default="computational", description="Type of evidence")


class GraphQueryRequest(BaseModel):
    """Request model for graph queries"""
    query_type: str = Field(..., description="Type of query")
    parameters: Dict[str, Any] = Field(..., description="Query parameters")
    limit: int = Field(default=100, ge=1, le=1000, description="Result limit")


class HypothesisGenerationRequest(BaseModel):
    """Request model for hypothesis generation"""
    research_area: str = Field(..., description="Research area")
    seed_concepts: List[str] = Field(..., description="Seed concept node IDs")
    hypothesis_type: str = Field(default="mechanistic", description="Type of hypothesis")


class NodeResponse(BaseModel):
    """Response model for knowledge node"""
    id: str
    node_type: str
    external_id: str
    name: str
    description: Optional[str]
    properties: Dict[str, Any]
    confidence_score: float
    created_at: str

    class Config:
        from_attributes = True


class EdgeResponse(BaseModel):
    """Response model for knowledge edge"""
    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence_score: float
    evidence_type: str
    created_at: str

    class Config:
        from_attributes = True


class HypothesisResponse(BaseModel):
    """Response model for hypothesis"""
    id: str
    title: str
    description: str
    hypothesis_text: str
    research_area: str
    novelty_score: float
    testability_score: Optional[float]
    potential_impact: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


# Endpoints
@router.post("/nodes", response_model=NodeResponse)
async def create_knowledge_node(
    request: NodeCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Create a new knowledge graph node
    
    - **node_type**: Type of entity (protein, drug, disease, pathway)
    - **external_id**: External database identifier
    - **name**: Human-readable name
    - **properties**: Additional properties as key-value pairs
    """
    try:
        node = await kg_service.create_knowledge_node(
            node_type=request.node_type,
            external_id=request.external_id,
            name=request.name,
            properties=request.properties,
            db=db,
            ontology_source=request.ontology_source,
            ontology_id=request.ontology_id
        )
        
        logger.info(f"Knowledge node created: {node.id} by user {current_user['user_id']}")
        
        return NodeResponse(
            id=str(node.id),
            node_type=node.node_type,
            external_id=node.external_id,
            name=node.name,
            description=node.description,
            properties=node.properties,
            confidence_score=node.confidence_score,
            created_at=node.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating knowledge node: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create knowledge node"
        )


@router.post("/edges", response_model=EdgeResponse)
async def create_knowledge_edge(
    request: EdgeCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Create a new knowledge graph edge
    
    - **source_node_id**: Source node identifier
    - **target_node_id**: Target node identifier
    - **relationship_type**: Type of relationship
    - **confidence_score**: Confidence in the relationship (0.0-1.0)
    """
    try:
        edge = await kg_service.create_knowledge_edge(
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            relationship_type=request.relationship_type,
            properties=request.properties,
            db=db,
            confidence_score=request.confidence_score,
            evidence_type=request.evidence_type
        )
        
        logger.info(f"Knowledge edge created: {edge.id} by user {current_user['user_id']}")
        
        return EdgeResponse(
            id=str(edge.id),
            source_node_id=str(edge.source_node_id),
            target_node_id=str(edge.target_node_id),
            relationship_type=edge.relationship_type,
            properties=edge.properties,
            confidence_score=edge.confidence_score,
            evidence_type=edge.evidence_type,
            created_at=edge.created_at.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating knowledge edge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create knowledge edge"
        )


@router.post("/query", response_model=Dict[str, Any])
async def query_knowledge_graph(
    request: GraphQueryRequest,
    db: AsyncSession = Depends(get_db_session),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=30))
):
    """
    Query the knowledge graph
    
    Supported query types:
    - **find_connections**: Find connections for a node
    - **shortest_path**: Find shortest path between nodes
    - **subgraph**: Extract subgraph around nodes
    - **similarity**: Find similar nodes
    """
    try:
        result = await kg_service.query_knowledge_graph(
            query_type=request.query_type,
            parameters=request.parameters,
            db=db,
            limit=request.limit
        )
        
        logger.info(f"Knowledge graph query executed: {request.query_type} by user {current_user['user_id']}")
        
        return {
            "query_type": request.query_type,
            "parameters": request.parameters,
            "result": result,
            "timestamp": "2025-06-04T23:03:00Z"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query knowledge graph"
        )


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_knowledge_node(
    node_id: str,
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_dependency(max_requests=100))
):
    """
    Get knowledge node by ID
    """
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(KnowledgeNode).where(KnowledgeNode.id == node_id)
        )
        node = result.scalar_one_or_none()
        
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge node {node_id} not found"
            )
        
        return NodeResponse(
            id=str(node.id),
            node_type=node.node_type,
            external_id=node.external_id,
            name=node.name,
            description=node.description,
            properties=node.properties,
            confidence_score=node.confidence_score,
            created_at=node.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting knowledge node: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get knowledge node"
        )


@router.get("/nodes", response_model=List[NodeResponse])
async def list_knowledge_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    search: Optional[str] = Query(None, description="Search in node names"),
    db: AsyncSession = Depends(get_db_session),
    pagination: PaginationParams = Depends(get_pagination_params),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    List knowledge nodes with optional filtering
    """
    try:
        from sqlalchemy import select, or_
        
        query = select(KnowledgeNode)
        
        # Apply filters
        if node_type:
            query = query.where(KnowledgeNode.node_type == node_type)
        
        if search:
            query = query.where(
                or_(
                    KnowledgeNode.name.ilike(f"%{search}%"),
                    KnowledgeNode.description.ilike(f"%{search}%")
                )
            )
        
        # Apply pagination
        query = query.offset(pagination.skip).limit(pagination.limit)
        
        result = await db.execute(query)
        nodes = result.scalars().all()
        
        return [
            NodeResponse(
                id=str(node.id),
                node_type=node.node_type,
                external_id=node.external_id,
                name=node.name,
                description=node.description,
                properties=node.properties,
                confidence_score=node.confidence_score,
                created_at=node.created_at.isoformat()
            )
            for node in nodes
        ]
        
    except Exception as e:
        logger.error(f"Error listing knowledge nodes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list knowledge nodes"
        )


@router.post("/hypotheses/generate", response_model=HypothesisResponse)
async def generate_hypothesis(
    request: HypothesisGenerationRequest,
    db: AsyncSession = Depends(get_db_session),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=10))
):
    """
    Generate AI-powered scientific hypothesis
    
    - **research_area**: Area of research focus
    - **seed_concepts**: List of seed concept node IDs
    - **hypothesis_type**: Type of hypothesis to generate
    """
    try:
        hypothesis = await kg_service.generate_hypothesis(
            seed_nodes=request.seed_concepts,
            research_area=request.research_area,
            db=db
        )
        
        logger.info(f"Hypothesis generated: {hypothesis.id} by user {current_user['user_id']}")
        
        return HypothesisResponse(
            id=str(hypothesis.id),
            title=hypothesis.title,
            description=hypothesis.description,
            hypothesis_text=hypothesis.hypothesis_text,
            research_area=hypothesis.research_area,
            novelty_score=hypothesis.novelty_score,
            testability_score=hypothesis.testability_score,
            potential_impact=hypothesis.potential_impact,
            created_at=hypothesis.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating hypothesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate hypothesis"
        )


@router.get("/metrics", response_model=Dict[str, Any])
async def get_knowledge_graph_metrics(
    db: AsyncSession = Depends(get_db_session),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    _: bool = Depends(rate_limit_dependency(max_requests=20))
):
    """
    Get knowledge graph quality and performance metrics
    """
    try:
        metrics = await kg_service.update_graph_metrics(db)
        
        return {
            "total_nodes": metrics.total_nodes,
            "total_edges": metrics.total_edges,
            "node_types_count": metrics.node_types_count,
            "edge_types_count": metrics.edge_types_count,
            "average_node_degree": metrics.average_node_degree,
            "graph_density": metrics.graph_density,
            "clustering_coefficient": metrics.clustering_coefficient,
            "average_path_length": metrics.average_path_length,
            "confidence_distribution": {
                "high": metrics.high_confidence_edges,
                "medium": metrics.medium_confidence_edges,
                "low": metrics.low_confidence_edges
            },
            "last_update": metrics.last_update.isoformat(),
            "data_sources": metrics.data_sources
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge graph metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get knowledge graph metrics"
        )


@router.get("/export", response_model=Dict[str, Any])
async def export_knowledge_graph(
    format: str = Query("json", description="Export format (json, rdf, cypher)"),
    node_types: Optional[List[str]] = Query(None, description="Filter by node types"),
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=5))
):
    """
    Export knowledge graph data
    
    - **format**: Export format (json, rdf, cypher)
    - **node_types**: Optional filter by node types
    """
    try:
        from sqlalchemy import select
        
        # Get nodes
        nodes_query = select(KnowledgeNode)
        if node_types:
            nodes_query = nodes_query.where(KnowledgeNode.node_type.in_(node_types))
        
        nodes_result = await db.execute(nodes_query)
        nodes = nodes_result.scalars().all()
        
        # Get edges
        edges_query = select(KnowledgeEdge)
        edges_result = await db.execute(edges_query)
        edges = edges_result.scalars().all()
        
        # Format export data
        if format == "json":
            export_data = {
                "nodes": [
                    {
                        "id": str(node.id),
                        "type": node.node_type,
                        "external_id": node.external_id,
                        "name": node.name,
                        "properties": node.properties
                    }
                    for node in nodes
                ],
                "edges": [
                    {
                        "id": str(edge.id),
                        "source": str(edge.source_node_id),
                        "target": str(edge.target_node_id),
                        "type": edge.relationship_type,
                        "confidence": edge.confidence_score
                    }
                    for edge in edges
                ]
            }
        else:
            export_data = {"message": f"Export format {format} not yet implemented"}
        
        logger.info(f"Knowledge graph exported by user {current_user['user_id']}")
        
        return {
            "format": format,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "data": export_data,
            "timestamp": "2025-06-04T23:03:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error exporting knowledge graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export knowledge graph"
        )
