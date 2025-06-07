"""
Scientific hypothesis generation and management API endpoints
Production-grade endpoints for AI-powered hypothesis generation and validation
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import logging

from app.api.deps import (
    get_db_session, get_kg_service, get_ai_service, get_eliza_service,
    get_current_user, get_pagination_params, PaginationParams,
    rate_limit_dependency
)
from app.services.kg_service import KnowledgeGraphService
from app.services.ai_service import AIModelManager
from app.services.eliza_service import ElizaIntegrationService
from app.models.knowledge_graph import Hypothesis

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class HypothesisGenerationRequest(BaseModel):
    """Request model for hypothesis generation"""
    research_question: str = Field(..., description="Research question or area")
    seed_concepts: List[str] = Field(..., description="Seed concept node IDs")
    hypothesis_type: str = Field(default="mechanistic", description="Type of hypothesis")
    background_knowledge: Optional[List[str]] = Field(default=[], description="Background knowledge")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation constraints")


class HypothesisValidationRequest(BaseModel):
    """Request model for hypothesis validation"""
    hypothesis_id: str = Field(..., description="Hypothesis ID to validate")
    validation_criteria: List[str] = Field(..., description="Validation criteria")
    experimental_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Experimental data")


class ExperimentDesignRequest(BaseModel):
    """Request model for experiment design"""
    hypothesis_id: str = Field(..., description="Hypothesis ID")
    experiment_types: List[str] = Field(..., description="Types of experiments to design")
    budget_constraints: Optional[float] = Field(None, description="Budget constraints")
    time_constraints: Optional[int] = Field(None, description="Time constraints in weeks")


class HypothesisRefinementRequest(BaseModel):
    """Request model for hypothesis refinement"""
    hypothesis_id: str = Field(..., description="Hypothesis ID to refine")
    feedback: str = Field(..., description="Feedback for refinement")
    new_evidence: Optional[List[Dict[str, Any]]] = Field(default=[], description="New evidence")


class HypothesisResponse(BaseModel):
    """Response model for hypothesis"""
    id: str
    title: str
    description: str
    hypothesis_text: str
    research_area: str
    hypothesis_type: str
    novelty_score: float
    testability_score: Optional[float]
    evidence_strength: Optional[float]
    potential_impact: Optional[str]
    validation_status: str
    created_at: str

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    """Response model for experiment design"""
    experiment_type: str
    description: str
    methodology: str
    estimated_duration: str
    estimated_cost: float
    success_probability: float
    required_resources: List[str]
    expected_outcomes: List[str]


# Endpoints
@router.post("/generate", response_model=HypothesisResponse)
async def generate_hypothesis(
    request: HypothesisGenerationRequest,
    db: AsyncSession = Depends(get_db_session),
    kg_service: KnowledgeGraphService = Depends(get_kg_service),
    ai_service: AIModelManager = Depends(get_ai_service),
    eliza_service: ElizaIntegrationService = Depends(get_eliza_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=10))
):
    """
    Generate AI-powered scientific hypothesis
    
    - **research_question**: Research question or area of interest
    - **seed_concepts**: List of seed concept node IDs from knowledge graph
    - **hypothesis_type**: Type of hypothesis (mechanistic, predictive, therapeutic)
    - **background_knowledge**: Additional background knowledge
    """
    try:
        # Use Eliza hypothesis agent for generation
        eliza_result = await eliza_service.process_agent_request(
            agent_type="hypothesis-agent",
            action="generate_hypothesis",
            parameters={
                "research_question": request.research_question,
                "seed_concepts": request.seed_concepts,
                "hypothesis_type": request.hypothesis_type,
                "background_knowledge": request.background_knowledge,
                "constraints": request.constraints
            },
            db=db
        )
        
        if not eliza_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hypothesis generation failed: {eliza_result.get('error')}"
            )
        
        hypothesis_data = eliza_result["result"]
        
        # Get the generated hypothesis from database
        from sqlalchemy import select
        result = await db.execute(
            select(Hypothesis).where(Hypothesis.id == hypothesis_data["hypothesis_id"])
        )
        hypothesis = result.scalar_one()
        
        logger.info(f"Hypothesis generated: {hypothesis.id} by user {current_user['user_id']}")
        
        return HypothesisResponse(
            id=str(hypothesis.id),
            title=hypothesis.title,
            description=hypothesis.description,
            hypothesis_text=hypothesis.hypothesis_text,
            research_area=hypothesis.research_area,
            hypothesis_type=hypothesis.hypothesis_type,
            novelty_score=hypothesis.novelty_score,
            testability_score=hypothesis.testability_score,
            evidence_strength=hypothesis.evidence_strength,
            potential_impact=hypothesis.potential_impact,
            validation_status=hypothesis.validation_status,
            created_at=hypothesis.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating hypothesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate hypothesis"
        )


@router.post("/{hypothesis_id}/validate", response_model=Dict[str, Any])
async def validate_hypothesis(
    hypothesis_id: str,
    request: HypothesisValidationRequest,
    db: AsyncSession = Depends(get_db_session),
    eliza_service: ElizaIntegrationService = Depends(get_eliza_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=20))
):
    """
    Validate scientific hypothesis using AI analysis
    
    - **hypothesis_id**: ID of hypothesis to validate
    - **validation_criteria**: List of validation criteria
    - **experimental_data**: Optional experimental data for validation
    """
    try:
        # Get hypothesis
        hypothesis = await db.get(Hypothesis, hypothesis_id)
        if not hypothesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hypothesis {hypothesis_id} not found"
            )
        
        # Use Eliza hypothesis agent for validation
        eliza_result = await eliza_service.process_agent_request(
            agent_type="hypothesis-agent",
            action="validate_hypothesis",
            parameters={
                "hypothesis_id": hypothesis_id,
                "hypothesis_text": hypothesis.hypothesis_text,
                "validation_criteria": request.validation_criteria,
                "experimental_data": request.experimental_data
            },
            db=db
        )
        
        if not eliza_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hypothesis validation failed: {eliza_result.get('error')}"
            )
        
        validation_result = eliza_result["result"]
        
        # Update hypothesis validation status
        hypothesis.validation_status = validation_result.get("validation_status", "validated")
        hypothesis.validation_score = validation_result.get("validation_score", 0.0)
        
        await db.commit()
        
        logger.info(f"Hypothesis validated: {hypothesis_id} by user {current_user['user_id']}")
        
        return {
            "hypothesis_id": hypothesis_id,
            "validation_result": validation_result,
            "updated_status": hypothesis.validation_status,
            "timestamp": "2025-06-04T23:05:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error validating hypothesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate hypothesis"
        )


@router.post("/{hypothesis_id}/design-experiments", response_model=List[ExperimentResponse])
async def design_experiments(
    hypothesis_id: str,
    request: ExperimentDesignRequest,
    db: AsyncSession = Depends(get_db_session),
    eliza_service: ElizaIntegrationService = Depends(get_eliza_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=15))
):
    """
    Design experiments to test hypothesis
    
    - **hypothesis_id**: ID of hypothesis to test
    - **experiment_types**: Types of experiments to design
    - **budget_constraints**: Optional budget constraints
    - **time_constraints**: Optional time constraints
    """
    try:
        # Get hypothesis
        hypothesis = await db.get(Hypothesis, hypothesis_id)
        if not hypothesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hypothesis {hypothesis_id} not found"
            )
        
        # Use Eliza hypothesis agent for experiment design
        eliza_result = await eliza_service.process_agent_request(
            agent_type="hypothesis-agent",
            action="design_experiments",
            parameters={
                "hypothesis_id": hypothesis_id,
                "hypothesis_text": hypothesis.hypothesis_text,
                "experiment_types": request.experiment_types,
                "budget_constraints": request.budget_constraints,
                "time_constraints": request.time_constraints
            },
            db=db
        )
        
        if not eliza_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Experiment design failed: {eliza_result.get('error')}"
            )
        
        experiments = eliza_result["result"]["experiments"]
        
        # Convert to response format
        experiment_responses = []
        for exp in experiments:
            experiment_responses.append(ExperimentResponse(
                experiment_type=exp["experiment_type"],
                description=exp["description"],
                methodology=exp.get("methodology", "Standard protocol"),
                estimated_duration=exp.get("estimated_duration", "4 weeks"),
                estimated_cost=exp.get("estimated_cost", 10000),
                success_probability=exp.get("success_probability", 0.7),
                required_resources=exp.get("required_resources", ["Lab equipment", "Reagents"]),
                expected_outcomes=exp.get("expected_outcomes", ["Validation data"])
            ))
        
        logger.info(f"Experiments designed for hypothesis {hypothesis_id} by user {current_user['user_id']}")
        
        return experiment_responses
        
    except Exception as e:
        logger.error(f"Error designing experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to design experiments"
        )


@router.post("/{hypothesis_id}/refine", response_model=HypothesisResponse)
async def refine_hypothesis(
    hypothesis_id: str,
    request: HypothesisRefinementRequest,
    db: AsyncSession = Depends(get_db_session),
    ai_service: AIModelManager = Depends(get_ai_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=15))
):
    """
    Refine hypothesis based on feedback and new evidence
    
    - **hypothesis_id**: ID of hypothesis to refine
    - **feedback**: Feedback for refinement
    - **new_evidence**: New evidence to incorporate
    """
    try:
        # Get hypothesis
        hypothesis = await db.get(Hypothesis, hypothesis_id)
        if not hypothesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hypothesis {hypothesis_id} not found"
            )
        
        # Generate refined hypothesis text
        refinement_prompt = f"""
        Original hypothesis: {hypothesis.hypothesis_text}
        
        Feedback: {request.feedback}
        
        New evidence: {request.new_evidence}
        
        Please refine the hypothesis incorporating the feedback and new evidence:
        """
        
        refined_text = await ai_service.generate_scientific_text(
            prompt=refinement_prompt,
            max_length=500
        )
        
        # Update hypothesis
        hypothesis.hypothesis_text = refined_text
        hypothesis.description = f"Refined based on feedback: {request.feedback[:100]}..."
        
        # Recalculate novelty score
        hypothesis.novelty_score = min(1.0, hypothesis.novelty_score + 0.1)
        
        await db.commit()
        await db.refresh(hypothesis)
        
        logger.info(f"Hypothesis refined: {hypothesis_id} by user {current_user['user_id']}")
        
        return HypothesisResponse(
            id=str(hypothesis.id),
            title=hypothesis.title,
            description=hypothesis.description,
            hypothesis_text=hypothesis.hypothesis_text,
            research_area=hypothesis.research_area,
            hypothesis_type=hypothesis.hypothesis_type,
            novelty_score=hypothesis.novelty_score,
            testability_score=hypothesis.testability_score,
            evidence_strength=hypothesis.evidence_strength,
            potential_impact=hypothesis.potential_impact,
            validation_status=hypothesis.validation_status,
            created_at=hypothesis.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error refining hypothesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refine hypothesis"
        )


@router.get("/{hypothesis_id}", response_model=HypothesisResponse)
async def get_hypothesis(
    hypothesis_id: str,
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_dependency(max_requests=100))
):
    """
    Get hypothesis by ID
    """
    try:
        hypothesis = await db.get(Hypothesis, hypothesis_id)
        
        if not hypothesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hypothesis {hypothesis_id} not found"
            )
        
        return HypothesisResponse(
            id=str(hypothesis.id),
            title=hypothesis.title,
            description=hypothesis.description,
            hypothesis_text=hypothesis.hypothesis_text,
            research_area=hypothesis.research_area,
            hypothesis_type=hypothesis.hypothesis_type,
            novelty_score=hypothesis.novelty_score,
            testability_score=hypothesis.testability_score,
            evidence_strength=hypothesis.evidence_strength,
            potential_impact=hypothesis.potential_impact,
            validation_status=hypothesis.validation_status,
            created_at=hypothesis.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting hypothesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get hypothesis"
        )


@router.get("/", response_model=List[HypothesisResponse])
async def list_hypotheses(
    research_area: Optional[str] = Query(None, description="Filter by research area"),
    hypothesis_type: Optional[str] = Query(None, description="Filter by hypothesis type"),
    validation_status: Optional[str] = Query(None, description="Filter by validation status"),
    min_novelty_score: Optional[float] = Query(None, description="Minimum novelty score"),
    db: AsyncSession = Depends(get_db_session),
    pagination: PaginationParams = Depends(get_pagination_params),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    List hypotheses with optional filtering
    """
    try:
        from sqlalchemy import select, and_
        
        query = select(Hypothesis)
        conditions = []
        
        # Apply filters
        if research_area:
            conditions.append(Hypothesis.research_area.ilike(f"%{research_area}%"))
        
        if hypothesis_type:
            conditions.append(Hypothesis.hypothesis_type == hypothesis_type)
        
        if validation_status:
            conditions.append(Hypothesis.validation_status == validation_status)
        
        if min_novelty_score is not None:
            conditions.append(Hypothesis.novelty_score >= min_novelty_score)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Apply pagination
        query = query.offset(pagination.skip).limit(pagination.limit)
        
        result = await db.execute(query)
        hypotheses = result.scalars().all()
        
        return [
            HypothesisResponse(
                id=str(hypothesis.id),
                title=hypothesis.title,
                description=hypothesis.description,
                hypothesis_text=hypothesis.hypothesis_text,
                research_area=hypothesis.research_area,
                hypothesis_type=hypothesis.hypothesis_type,
                novelty_score=hypothesis.novelty_score,
                testability_score=hypothesis.testability_score,
                evidence_strength=hypothesis.evidence_strength,
                potential_impact=hypothesis.potential_impact,
                validation_status=hypothesis.validation_status,
                created_at=hypothesis.created_at.isoformat()
            )
            for hypothesis in hypotheses
        ]
        
    except Exception as e:
        logger.error(f"Error listing hypotheses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list hypotheses"
        )


@router.get("/statistics", response_model=Dict[str, Any])
async def get_hypothesis_statistics(
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_dependency(max_requests=20))
):
    """
    Get hypothesis generation statistics
    """
    try:
        from sqlalchemy import select, func
        
        # Total hypotheses
        total_hypotheses = await db.scalar(select(func.count(Hypothesis.id)))
        
        # By research area
        area_counts = await db.execute(
            select(Hypothesis.research_area, func.count(Hypothesis.id))
            .group_by(Hypothesis.research_area)
            .order_by(func.count(Hypothesis.id).desc())
            .limit(10)
        )
        
        # By validation status
        status_counts = await db.execute(
            select(Hypothesis.validation_status, func.count(Hypothesis.id))
            .group_by(Hypothesis.validation_status)
        )
        
        # Average novelty score
        avg_novelty = await db.scalar(select(func.avg(Hypothesis.novelty_score)))
        
        return {
            "total_hypotheses": total_hypotheses,
            "research_areas": dict(area_counts.fetchall()),
            "validation_status_distribution": dict(status_counts.fetchall()),
            "average_novelty_score": float(avg_novelty) if avg_novelty else 0.0,
            "timestamp": "2025-06-04T23:05:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting hypothesis statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get hypothesis statistics"
        )
