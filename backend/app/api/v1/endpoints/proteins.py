"""
Protein analysis API endpoints
Production-grade endpoints for protein structure prediction and analysis
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import logging

from app.api.deps import (
    get_db_session, get_protein_service, get_current_user,
    validate_uniprot_id, validate_protein_sequence,
    get_pagination_params, PaginationParams,
    rate_limit_dependency
)
from app.services.protein_service import ProteinAnalysisService
from app.models.protein import Protein, BindingSite, ProteinStructurePrediction

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class ProteinCreateRequest(BaseModel):
    """Request model for creating protein record"""
    uniprot_id: str = Field(..., description="UniProt identifier")
    force_update: bool = Field(default=False, description="Force update if exists")


class ProteinSequenceRequest(BaseModel):
    """Request model for protein sequence analysis"""
    sequence: str = Field(..., description="Protein amino acid sequence")
    analysis_type: str = Field(default="full", description="Type of analysis to perform")


class StructurePredictionRequest(BaseModel):
    """Request model for structure prediction"""
    sequence: str = Field(..., description="Protein amino acid sequence")
    method: str = Field(default="esmfold", description="Prediction method")
    protein_id: Optional[str] = Field(None, description="Optional protein ID to store results")


class DrugBindingRequest(BaseModel):
    """Request model for drug-protein binding analysis"""
    protein_pdb: str = Field(..., description="Protein PDB structure")
    drug_smiles: str = Field(..., description="Drug SMILES string")
    binding_site_id: Optional[str] = Field(None, description="Specific binding site ID")


class ProteinResponse(BaseModel):
    """Response model for protein data"""
    id: str
    uniprot_id: str
    name: str
    description: Optional[str]
    organism: str
    sequence: str
    length: int
    molecular_weight: Optional[float]
    structure_source: Optional[str]
    structure_confidence: Optional[float]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class BindingSiteResponse(BaseModel):
    """Response model for binding site data"""
    id: str
    name: Optional[str]
    site_type: str
    residues: List[str]
    volume: Optional[float]
    druggability_score: Optional[float]
    confidence_score: Optional[float]
    prediction_method: Optional[str]

    class Config:
        from_attributes = True


class StructurePredictionResponse(BaseModel):
    """Response model for structure prediction"""
    id: str
    method: str
    model_version: Optional[str]
    global_confidence: Optional[float]
    processing_time: Optional[float]
    status: str
    created_at: str

    class Config:
        from_attributes = True


# Endpoints
@router.post("/", response_model=ProteinResponse)
async def create_protein(
    request: ProteinCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Create or update protein record from UniProt
    
    - **uniprot_id**: Valid UniProt identifier
    - **force_update**: Force update if protein already exists
    """
    try:
        # Validate UniProt ID
        uniprot_id = validate_uniprot_id(request.uniprot_id)
        
        # Create or update protein record
        protein = await protein_service.create_protein_record(
            uniprot_id=uniprot_id,
            db=db,
            force_update=request.force_update
        )
        
        logger.info(f"Protein {uniprot_id} created/updated by user {current_user['user_id']}")
        
        return ProteinResponse(
            id=str(protein.id),
            uniprot_id=protein.uniprot_id,
            name=protein.name,
            description=protein.description,
            organism=protein.organism,
            sequence=protein.sequence,
            length=protein.length,
            molecular_weight=protein.molecular_weight,
            structure_source=protein.structure_source,
            structure_confidence=protein.structure_confidence,
            created_at=protein.created_at.isoformat(),
            updated_at=protein.updated_at.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating protein {request.uniprot_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create protein record"
        )


@router.get("/{uniprot_id}", response_model=ProteinResponse)
async def get_protein(
    uniprot_id: str,
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    _: bool = Depends(rate_limit_dependency(max_requests=100))
):
    """
    Get protein by UniProt ID
    
    - **uniprot_id**: Valid UniProt identifier
    """
    try:
        # Validate and get protein
        uniprot_id = validate_uniprot_id(uniprot_id)
        protein = await protein_service.get_protein_by_uniprot_id(uniprot_id, db)
        
        if not protein:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protein {uniprot_id} not found"
            )
        
        return ProteinResponse(
            id=str(protein.id),
            uniprot_id=protein.uniprot_id,
            name=protein.name,
            description=protein.description,
            organism=protein.organism,
            sequence=protein.sequence,
            length=protein.length,
            molecular_weight=protein.molecular_weight,
            structure_source=protein.structure_source,
            structure_confidence=protein.structure_confidence,
            created_at=protein.created_at.isoformat(),
            updated_at=protein.updated_at.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/analyze-sequence", response_model=Dict[str, Any])
async def analyze_protein_sequence(
    request: ProteinSequenceRequest,
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=20))
):
    """
    Analyze protein sequence properties
    
    - **sequence**: Valid amino acid sequence
    - **analysis_type**: Type of analysis (full, structure, binding, properties)
    """
    try:
        # Validate sequence
        sequence = validate_protein_sequence(request.sequence)
        
        analysis_results = {}
        
        # Perform requested analysis
        if request.analysis_type in ["full", "structure"]:
            # Structure prediction
            structure_prediction = await protein_service.predict_protein_structure(
                sequence=sequence,
                method="esmfold"
            )
            analysis_results["structure_prediction"] = structure_prediction
            
            # Binding site prediction if structure available
            if structure_prediction.get("pdb_content"):
                binding_sites = await protein_service.predict_binding_sites(
                    pdb_content=structure_prediction["pdb_content"]
                )
                analysis_results["binding_sites"] = binding_sites
        
        if request.analysis_type in ["full", "properties"]:
            # Basic sequence properties
            analysis_results["sequence_properties"] = {
                "length": len(sequence),
                "molecular_weight": len(sequence) * 110,  # Approximate
                "amino_acid_composition": _calculate_aa_composition(sequence),
                "hydrophobicity": _calculate_hydrophobicity(sequence),
                "isoelectric_point": _estimate_isoelectric_point(sequence)
            }
        
        logger.info(f"Sequence analysis completed for user {current_user['user_id']}")
        
        return {
            "sequence_length": len(sequence),
            "analysis_type": request.analysis_type,
            "results": analysis_results,
            "timestamp": "2025-06-04T23:03:00Z"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing sequence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze protein sequence"
        )


@router.post("/predict-structure", response_model=Dict[str, Any])
async def predict_protein_structure(
    request: StructurePredictionRequest,
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=10))
):
    """
    Predict protein structure using AI models
    
    - **sequence**: Valid amino acid sequence
    - **method**: Prediction method (esmfold, alphafold)
    - **protein_id**: Optional protein ID to store results
    """
    try:
        # Validate sequence
        sequence = validate_protein_sequence(request.sequence)
        
        # Predict structure
        prediction_result = await protein_service.predict_protein_structure(
            sequence=sequence,
            method=request.method
        )
        
        # Store prediction if protein_id provided
        stored_prediction = None
        if request.protein_id:
            stored_prediction = await protein_service.store_structure_prediction(
                protein_id=request.protein_id,
                prediction_result=prediction_result,
                db=db
            )
        
        logger.info(f"Structure prediction completed for user {current_user['user_id']}")
        
        return {
            "sequence_length": len(sequence),
            "method": request.method,
            "prediction": prediction_result,
            "stored_prediction_id": str(stored_prediction.id) if stored_prediction else None,
            "timestamp": "2025-06-04T23:03:00Z"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error predicting structure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to predict protein structure"
        )


@router.get("/{uniprot_id}/binding-sites", response_model=List[BindingSiteResponse])
async def get_protein_binding_sites(
    uniprot_id: str,
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Get binding sites for a protein
    
    - **uniprot_id**: Valid UniProt identifier
    """
    try:
        # Validate and get protein
        uniprot_id = validate_uniprot_id(uniprot_id)
        protein = await protein_service.get_protein_by_uniprot_id(uniprot_id, db)
        
        if not protein:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protein {uniprot_id} not found"
            )
        
        # Get binding sites
        binding_sites = protein.binding_sites
        
        return [
            BindingSiteResponse(
                id=str(site.id),
                name=site.name,
                site_type=site.site_type,
                residues=site.residues or [],
                volume=site.volume,
                druggability_score=site.druggability_score,
                confidence_score=site.confidence_score,
                prediction_method=site.prediction_method
            )
            for site in binding_sites
        ]
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/drug-binding", response_model=Dict[str, Any])
async def analyze_drug_protein_binding(
    request: DrugBindingRequest,
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=10))
):
    """
    Analyze drug-protein binding interaction
    
    - **protein_pdb**: Protein structure in PDB format
    - **drug_smiles**: Drug molecule in SMILES format
    - **binding_site_id**: Optional specific binding site
    """
    try:
        # Analyze drug-protein interaction
        binding_analysis = await protein_service.analyze_drug_protein_interaction(
            protein_pdb=request.protein_pdb,
            drug_smiles=request.drug_smiles,
            binding_site={"id": request.binding_site_id} if request.binding_site_id else None
        )
        
        logger.info(f"Drug-protein binding analysis completed for user {current_user['user_id']}")
        
        return {
            "binding_analysis": binding_analysis,
            "timestamp": "2025-06-04T23:03:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing drug-protein binding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze drug-protein binding"
        )


@router.get("/{uniprot_id}/structure-predictions", response_model=List[StructurePredictionResponse])
async def get_structure_predictions(
    uniprot_id: str,
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    pagination: PaginationParams = Depends(get_pagination_params),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Get structure predictions for a protein
    
    - **uniprot_id**: Valid UniProt identifier
    """
    try:
        # Validate and get protein
        uniprot_id = validate_uniprot_id(uniprot_id)
        protein = await protein_service.get_protein_by_uniprot_id(uniprot_id, db)
        
        if not protein:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protein {uniprot_id} not found"
            )
        
        # Get structure predictions with pagination
        from sqlalchemy import select
        from app.models.protein import ProteinStructurePrediction
        
        query = select(ProteinStructurePrediction).where(
            ProteinStructurePrediction.protein_id == protein.id
        ).offset(pagination.skip).limit(pagination.limit)
        
        result = await db.execute(query)
        predictions = result.scalars().all()
        
        return [
            StructurePredictionResponse(
                id=str(pred.id),
                method=pred.method,
                model_version=pred.model_version,
                global_confidence=pred.global_confidence,
                processing_time=pred.processing_time,
                status=pred.status,
                created_at=pred.created_at.isoformat()
            )
            for pred in predictions
        ]
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/statistics", response_model=Dict[str, Any])
async def get_protein_statistics(
    db: AsyncSession = Depends(get_db_session),
    protein_service: ProteinAnalysisService = Depends(get_protein_service),
    _: bool = Depends(rate_limit_dependency(max_requests=20))
):
    """
    Get protein database statistics
    """
    try:
        stats = await protein_service.get_protein_statistics(db)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting protein statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get protein statistics"
        )


# Helper functions
def _calculate_aa_composition(sequence: str) -> Dict[str, float]:
    """Calculate amino acid composition"""
    aa_count = {}
    for aa in sequence:
        aa_count[aa] = aa_count.get(aa, 0) + 1
    
    total = len(sequence)
    return {aa: count / total for aa, count in aa_count.items()}


def _calculate_hydrophobicity(sequence: str) -> float:
    """Calculate average hydrophobicity"""
    hydrophobicity_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    total_hydrophobicity = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence)
    return total_hydrophobicity / len(sequence)


def _estimate_isoelectric_point(sequence: str) -> float:
    """Estimate isoelectric point"""
    # Simplified calculation
    basic_aa = sequence.count('K') + sequence.count('R') + sequence.count('H')
    acidic_aa = sequence.count('D') + sequence.count('E')
    
    if basic_aa > acidic_aa:
        return 8.5 + (basic_aa - acidic_aa) * 0.5
    elif acidic_aa > basic_aa:
        return 5.5 - (acidic_aa - basic_aa) * 0.5
    else:
        return 7.0
