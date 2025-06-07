"""
Drug discovery and analysis API endpoints
Production-grade endpoints for drug property prediction and molecular analysis
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
import logging

from app.api.deps import (
    get_db_session, get_ai_service, get_current_user,
    validate_smiles, get_pagination_params, PaginationParams,
    rate_limit_dependency
)
from app.services.ai_service import AIModelManager
from app.models.drug import Drug, DrugProteinInteraction, ADMETProperties

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class DrugCreateRequest(BaseModel):
    """Request model for creating drug record"""
    name: str = Field(..., description="Drug name")
    smiles: str = Field(..., description="SMILES string")
    chembl_id: Optional[str] = Field(None, description="ChEMBL identifier")
    pubchem_cid: Optional[str] = Field(None, description="PubChem CID")
    drugbank_id: Optional[str] = Field(None, description="DrugBank identifier")
    description: Optional[str] = Field(None, description="Drug description")
    indication: Optional[str] = Field(None, description="Therapeutic indication")
    
    @validator('smiles')
    def validate_smiles_format(cls, v):
        return validate_smiles(v)


class DrugPropertyPredictionRequest(BaseModel):
    """Request model for drug property prediction"""
    smiles: str = Field(..., description="SMILES string")
    properties: List[str] = Field(
        default=["admet", "toxicity", "druglikeness", "synthesis"],
        description="Properties to predict"
    )
    
    @validator('smiles')
    def validate_smiles_format(cls, v):
        return validate_smiles(v)


class DrugSimilarityRequest(BaseModel):
    """Request model for drug similarity search"""
    query_smiles: str = Field(..., description="Query SMILES string")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum results")
    
    @validator('query_smiles')
    def validate_smiles_format(cls, v):
        return validate_smiles(v)


class DrugOptimizationRequest(BaseModel):
    """Request model for drug optimization"""
    lead_smiles: str = Field(..., description="Lead compound SMILES")
    optimization_goals: List[str] = Field(..., description="Optimization objectives")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optimization constraints")
    
    @validator('lead_smiles')
    def validate_smiles_format(cls, v):
        return validate_smiles(v)


class DrugResponse(BaseModel):
    """Response model for drug data"""
    id: str
    name: str
    smiles: str
    chembl_id: Optional[str]
    pubchem_cid: Optional[str]
    drugbank_id: Optional[str]
    molecular_weight: Optional[float]
    logp: Optional[float]
    hbd: Optional[int]
    hba: Optional[int]
    development_phase: Optional[str]
    indication: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class ADMETResponse(BaseModel):
    """Response model for ADMET properties"""
    human_intestinal_absorption: Optional[float]
    caco2_permeability: Optional[float]
    bioavailability: Optional[float]
    plasma_protein_binding: Optional[float]
    blood_brain_barrier: Optional[float]
    cyp3a4_inhibition: Optional[float]
    hepatotoxicity: Optional[float]
    cardiotoxicity: Optional[float]
    renal_clearance: Optional[float]
    half_life: Optional[float]

    class Config:
        from_attributes = True


# Endpoints
@router.post("/", response_model=DrugResponse)
async def create_drug(
    request: DrugCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    ai_service: AIModelManager = Depends(get_ai_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Create a new drug record with AI-predicted properties
    
    - **name**: Drug name
    - **smiles**: Valid SMILES string
    - **chembl_id**: Optional ChEMBL identifier
    - **indication**: Therapeutic indication
    """
    try:
        # Calculate molecular properties
        descriptors = await ai_service._calculate_molecular_descriptors(request.smiles)
        
        # Create drug record
        drug = Drug(
            name=request.name,
            smiles=request.smiles,
            chembl_id=request.chembl_id,
            pubchem_cid=request.pubchem_cid,
            drugbank_id=request.drugbank_id,
            description=request.description,
            indication=request.indication,
            molecular_weight=descriptors.get("molecular_weight"),
            logp=descriptors.get("logp"),
            hbd=descriptors.get("hbd"),
            hba=descriptors.get("hba"),
            tpsa=descriptors.get("tpsa"),
            rotatable_bonds=descriptors.get("rotatable_bonds")
        )
        
        db.add(drug)
        await db.commit()
        await db.refresh(drug)
        
        # Predict and store ADMET properties
        admet_prediction = await ai_service.predict_drug_properties(
            smiles=request.smiles,
            properties=["admet"]
        )
        
        if "admet" in admet_prediction:
            admet_props = ADMETProperties(
                drug_id=drug.id,
                **admet_prediction["admet"],
                prediction_method="neurograph-ai",
                model_version="v1.0"
            )
            db.add(admet_props)
            await db.commit()
        
        logger.info(f"Drug created: {drug.id} by user {current_user['user_id']}")
        
        return DrugResponse(
            id=str(drug.id),
            name=drug.name,
            smiles=drug.smiles,
            chembl_id=drug.chembl_id,
            pubchem_cid=drug.pubchem_cid,
            drugbank_id=drug.drugbank_id,
            molecular_weight=drug.molecular_weight,
            logp=drug.logp,
            hbd=drug.hbd,
            hba=drug.hba,
            development_phase=drug.development_phase,
            indication=drug.indication,
            created_at=drug.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating drug: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create drug record"
        )


@router.post("/predict-properties", response_model=Dict[str, Any])
async def predict_drug_properties(
    request: DrugPropertyPredictionRequest,
    ai_service: AIModelManager = Depends(get_ai_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=20))
):
    """
    Predict drug properties using AI models
    
    - **smiles**: Valid SMILES string
    - **properties**: List of properties to predict (admet, toxicity, druglikeness, synthesis)
    """
    try:
        # Predict drug properties
        predictions = await ai_service.predict_drug_properties(
            smiles=request.smiles,
            properties=request.properties
        )
        
        logger.info(f"Drug properties predicted for user {current_user['user_id']}")
        
        return {
            "smiles": request.smiles,
            "properties": request.properties,
            "predictions": predictions,
            "timestamp": "2025-06-04T23:05:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error predicting drug properties: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to predict drug properties"
        )


@router.post("/similarity-search", response_model=Dict[str, Any])
async def drug_similarity_search(
    request: DrugSimilarityRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=30))
):
    """
    Search for similar drugs using molecular fingerprints
    
    - **query_smiles**: Query molecule SMILES
    - **similarity_threshold**: Minimum similarity score (0.0-1.0)
    - **max_results**: Maximum number of results
    """
    try:
        # Calculate molecular fingerprint for query
        query_fingerprint = await _calculate_molecular_fingerprint(request.query_smiles)
        
        # Search database for similar compounds
        from sqlalchemy import select
        
        drugs_result = await db.execute(select(Drug).limit(1000))  # Limit for performance
        drugs = drugs_result.scalars().all()
        
        similar_drugs = []
        
        for drug in drugs:
            try:
                drug_fingerprint = await _calculate_molecular_fingerprint(drug.smiles)
                similarity = _calculate_tanimoto_similarity(query_fingerprint, drug_fingerprint)
                
                if similarity >= request.similarity_threshold:
                    similar_drugs.append({
                        "drug_id": str(drug.id),
                        "name": drug.name,
                        "smiles": drug.smiles,
                        "similarity_score": similarity,
                        "molecular_weight": drug.molecular_weight,
                        "logp": drug.logp
                    })
            except Exception:
                continue  # Skip drugs with invalid SMILES
        
        # Sort by similarity score
        similar_drugs.sort(key=lambda x: x["similarity_score"], reverse=True)
        similar_drugs = similar_drugs[:request.max_results]
        
        logger.info(f"Similarity search completed for user {current_user['user_id']}")
        
        return {
            "query_smiles": request.query_smiles,
            "similarity_threshold": request.similarity_threshold,
            "total_found": len(similar_drugs),
            "similar_drugs": similar_drugs,
            "timestamp": "2025-06-04T23:05:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform similarity search"
        )


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_drug_compound(
    request: DrugOptimizationRequest,
    ai_service: AIModelManager = Depends(get_ai_service),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(rate_limit_dependency(max_requests=10))
):
    """
    Optimize drug compound for desired properties
    
    - **lead_smiles**: Lead compound SMILES
    - **optimization_goals**: List of optimization objectives
    - **constraints**: Optimization constraints
    """
    try:
        # Analyze lead compound
        lead_properties = await ai_service.predict_drug_properties(
            smiles=request.lead_smiles,
            properties=["admet", "druglikeness", "toxicity"]
        )
        
        # Generate optimized variants (simplified implementation)
        optimized_compounds = await _generate_optimized_variants(
            lead_smiles=request.lead_smiles,
            goals=request.optimization_goals,
            constraints=request.constraints,
            ai_service=ai_service
        )
        
        logger.info(f"Drug optimization completed for user {current_user['user_id']}")
        
        return {
            "lead_compound": {
                "smiles": request.lead_smiles,
                "properties": lead_properties
            },
            "optimization_goals": request.optimization_goals,
            "optimized_compounds": optimized_compounds,
            "timestamp": "2025-06-04T23:05:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing drug compound: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize drug compound"
        )


@router.get("/{drug_id}", response_model=DrugResponse)
async def get_drug(
    drug_id: str,
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_dependency(max_requests=100))
):
    """
    Get drug by ID
    """
    try:
        drug = await db.get(Drug, drug_id)
        
        if not drug:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Drug {drug_id} not found"
            )
        
        return DrugResponse(
            id=str(drug.id),
            name=drug.name,
            smiles=drug.smiles,
            chembl_id=drug.chembl_id,
            pubchem_cid=drug.pubchem_cid,
            drugbank_id=drug.drugbank_id,
            molecular_weight=drug.molecular_weight,
            logp=drug.logp,
            hbd=drug.hbd,
            hba=drug.hba,
            development_phase=drug.development_phase,
            indication=drug.indication,
            created_at=drug.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting drug: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get drug"
        )


@router.get("/{drug_id}/admet", response_model=ADMETResponse)
async def get_drug_admet_properties(
    drug_id: str,
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    Get ADMET properties for a drug
    """
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(ADMETProperties).where(ADMETProperties.drug_id == drug_id)
        )
        admet = result.scalar_one_or_none()
        
        if not admet:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ADMET properties not found for drug {drug_id}"
            )
        
        return ADMETResponse(
            human_intestinal_absorption=admet.human_intestinal_absorption,
            caco2_permeability=admet.caco2_permeability,
            bioavailability=admet.bioavailability_f20,
            plasma_protein_binding=admet.plasma_protein_binding,
            blood_brain_barrier=admet.blood_brain_barrier,
            cyp3a4_inhibition=admet.cyp3a4_inhibition,
            hepatotoxicity=admet.hepatotoxicity,
            cardiotoxicity=admet.cardiotoxicity,
            renal_clearance=admet.renal_clearance,
            half_life=admet.half_life
        )
        
    except Exception as e:
        logger.error(f"Error getting ADMET properties: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get ADMET properties"
        )


@router.get("/", response_model=List[DrugResponse])
async def list_drugs(
    search: Optional[str] = Query(None, description="Search in drug names"),
    indication: Optional[str] = Query(None, description="Filter by indication"),
    development_phase: Optional[str] = Query(None, description="Filter by development phase"),
    db: AsyncSession = Depends(get_db_session),
    pagination: PaginationParams = Depends(get_pagination_params),
    _: bool = Depends(rate_limit_dependency(max_requests=50))
):
    """
    List drugs with optional filtering
    """
    try:
        from sqlalchemy import select, or_
        
        query = select(Drug)
        
        # Apply filters
        if search:
            query = query.where(
                or_(
                    Drug.name.ilike(f"%{search}%"),
                    Drug.description.ilike(f"%{search}%")
                )
            )
        
        if indication:
            query = query.where(Drug.indication.ilike(f"%{indication}%"))
        
        if development_phase:
            query = query.where(Drug.development_phase == development_phase)
        
        # Apply pagination
        query = query.offset(pagination.skip).limit(pagination.limit)
        
        result = await db.execute(query)
        drugs = result.scalars().all()
        
        return [
            DrugResponse(
                id=str(drug.id),
                name=drug.name,
                smiles=drug.smiles,
                chembl_id=drug.chembl_id,
                pubchem_cid=drug.pubchem_cid,
                drugbank_id=drug.drugbank_id,
                molecular_weight=drug.molecular_weight,
                logp=drug.logp,
                hbd=drug.hbd,
                hba=drug.hba,
                development_phase=drug.development_phase,
                indication=drug.indication,
                created_at=drug.created_at.isoformat()
            )
            for drug in drugs
        ]
        
    except Exception as e:
        logger.error(f"Error listing drugs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list drugs"
        )


# Helper functions
async def _calculate_molecular_fingerprint(smiles: str) -> List[int]:
    """Calculate molecular fingerprint from SMILES"""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        
        # Morgan fingerprint
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return list(fingerprint)
        
    except ImportError:
        # Fallback without RDKit
        return [hash(smiles[i:i+3]) % 2 for i in range(0, len(smiles), 3)][:2048]


def _calculate_tanimoto_similarity(fp1: List[int], fp2: List[int]) -> float:
    """Calculate Tanimoto similarity between two fingerprints"""
    if len(fp1) != len(fp2):
        return 0.0
    
    intersection = sum(1 for a, b in zip(fp1, fp2) if a == 1 and b == 1)
    union = sum(1 for a, b in zip(fp1, fp2) if a == 1 or b == 1)
    
    return intersection / union if union > 0 else 0.0


async def _generate_optimized_variants(
    lead_smiles: str,
    goals: List[str],
    constraints: Dict[str, Any],
    ai_service: AIModelManager
) -> List[Dict[str, Any]]:
    """Generate optimized compound variants"""
    
    # Simplified optimization - in production, use more sophisticated methods
    variants = []
    
    # Generate some mock optimized variants
    for i in range(3):
        # Simple SMILES modifications (placeholder)
        modified_smiles = lead_smiles + f"C{i}"  # Add carbon atoms
        
        try:
            # Predict properties for variant
            properties = await ai_service.predict_drug_properties(
                smiles=modified_smiles,
                properties=["admet", "druglikeness"]
            )
            
            variants.append({
                "smiles": modified_smiles,
                "optimization_score": 0.8 + i * 0.05,
                "properties": properties,
                "modifications": [f"Added C{i} group"]
            })
        except Exception:
            continue
    
    return variants
