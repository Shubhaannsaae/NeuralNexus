"""
Protein data models
SQLAlchemy models for protein structure and analysis data
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid

from app.core.database import Base


class Protein(Base):
    """Protein structure and metadata model"""
    
    __tablename__ = "proteins"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uniprot_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    organism = Column(String(255), nullable=False)
    sequence = Column(Text, nullable=False)
    length = Column(Integer, nullable=False)
    molecular_weight = Column(Float)
    
    # Structure information
    pdb_id = Column(String(10), index=True)
    structure_source = Column(String(50))  # 'experimental', 'alphafold', 'esmfold'
    structure_confidence = Column(Float)
    structure_data = Column(JSON)  # PDB structure data
    
    # Analysis metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    binding_sites = relationship("BindingSite", back_populates="protein", cascade="all, delete-orphan")
    drug_interactions = relationship("DrugProteinInteraction", back_populates="protein")
    
    def __repr__(self):
        return f"<Protein(uniprot_id='{self.uniprot_id}', name='{self.name}')>"


class BindingSite(Base):
    """Protein binding site model"""
    
    __tablename__ = "binding_sites"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    protein_id = Column(UUID(as_uuid=True), ForeignKey("proteins.id"), nullable=False)
    name = Column(String(255))
    site_type = Column(String(50))  # 'active', 'allosteric', 'binding'
    
    # Structural information
    residues = Column(ARRAY(String))  # Array of residue identifiers
    coordinates = Column(JSON)  # 3D coordinates
    volume = Column(Float)
    surface_area = Column(Float)
    
    # Druggability metrics
    druggability_score = Column(Float)
    hydrophobicity = Column(Float)
    electrostatic_potential = Column(Float)
    
    # Analysis metadata
    prediction_method = Column(String(100))
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    protein = relationship("Protein", back_populates="binding_sites")
    
    def __repr__(self):
        return f"<BindingSite(protein_id='{self.protein_id}', name='{self.name}')>"


class ProteinStructurePrediction(Base):
    """Protein structure prediction results"""
    
    __tablename__ = "protein_structure_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    protein_id = Column(UUID(as_uuid=True), ForeignKey("proteins.id"), nullable=False)
    
    # Prediction details
    method = Column(String(50), nullable=False)  # 'alphafold', 'esmfold', 'colabfold'
    model_version = Column(String(50))
    
    # Quality metrics
    confidence_scores = Column(ARRAY(Float))  # Per-residue confidence
    global_confidence = Column(Float)
    clash_score = Column(Float)
    ramachandran_favored = Column(Float)
    
    # Structure data
    pdb_content = Column(Text)
    secondary_structure = Column(String)  # DSSP notation
    
    # Processing metadata
    processing_time = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="completed")  # 'pending', 'processing', 'completed', 'failed'
    
    def __repr__(self):
        return f"<ProteinStructurePrediction(protein_id='{self.protein_id}', method='{self.method}')>"


class ProteinFamily(Base):
    """Protein family classification"""
    
    __tablename__ = "protein_families"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pfam_id = Column(String(20), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Classification
    clan = Column(String(50))
    type = Column(String(50))  # 'Domain', 'Family', 'Motif', 'Repeat'
    
    # Statistics
    length = Column(Integer)
    num_sequences = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProteinFamily(pfam_id='{self.pfam_id}', name='{self.name}')>"


class ProteinAnnotation(Base):
    """Protein functional annotations"""
    
    __tablename__ = "protein_annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    protein_id = Column(UUID(as_uuid=True), ForeignKey("proteins.id"), nullable=False)
    
    # Annotation details
    annotation_type = Column(String(50), nullable=False)  # 'GO', 'KEGG', 'InterPro'
    annotation_id = Column(String(50), nullable=False)
    term = Column(String(255))
    description = Column(Text)
    
    # Evidence
    evidence_code = Column(String(10))
    confidence_score = Column(Float)
    source = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProteinAnnotation(protein_id='{self.protein_id}', type='{self.annotation_type}')>"
