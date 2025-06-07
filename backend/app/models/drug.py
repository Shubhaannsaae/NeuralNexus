"""
Drug and chemical compound data models
SQLAlchemy models for drug discovery and molecular analysis
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid

from app.core.database import Base


class Drug(Base):
    """Drug compound model"""
    
    __tablename__ = "drugs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Chemical identifiers
    chembl_id = Column(String(20), unique=True, index=True)
    pubchem_cid = Column(String(20), index=True)
    drugbank_id = Column(String(20), index=True)
    cas_number = Column(String(20))
    
    # Basic information
    name = Column(String(255), nullable=False)
    synonyms = Column(ARRAY(String))
    description = Column(Text)
    
    # Chemical structure
    smiles = Column(Text, nullable=False)
    inchi = Column(Text)
    inchi_key = Column(String(255), index=True)
    molecular_formula = Column(String(255))
    
    # Molecular properties
    molecular_weight = Column(Float)
    exact_mass = Column(Float)
    logp = Column(Float)  # Lipophilicity
    tpsa = Column(Float)  # Topological polar surface area
    hbd = Column(Integer)  # Hydrogen bond donors
    hba = Column(Integer)  # Hydrogen bond acceptors
    rotatable_bonds = Column(Integer)
    
    # Drug-like properties (Lipinski's Rule of Five)
    lipinski_violations = Column(Integer)
    bioavailability_score = Column(Float)
    
    # Development status
    development_phase = Column(String(50))  # 'preclinical', 'phase1', 'phase2', 'phase3', 'approved'
    approval_status = Column(String(50))
    first_approval = Column(DateTime)
    
    # Therapeutic information
    indication = Column(Text)
    mechanism_of_action = Column(Text)
    therapeutic_class = Column(String(255))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    protein_interactions = relationship("DrugProteinInteraction", back_populates="drug")
    admet_properties = relationship("ADMETProperties", back_populates="drug", uselist=False)
    
    def __repr__(self):
        return f"<Drug(name='{self.name}', chembl_id='{self.chembl_id}')>"


class DrugProteinInteraction(Base):
    """Drug-protein interaction model"""
    
    __tablename__ = "drug_protein_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drug_id = Column(UUID(as_uuid=True), ForeignKey("drugs.id"), nullable=False)
    protein_id = Column(UUID(as_uuid=True), ForeignKey("proteins.id"), nullable=False)
    
    # Interaction details
    interaction_type = Column(String(50))  # 'inhibitor', 'activator', 'agonist', 'antagonist'
    binding_affinity = Column(Float)  # IC50, Ki, Kd values in nM
    affinity_type = Column(String(10))  # 'IC50', 'Ki', 'Kd', 'EC50'
    
    # Experimental conditions
    assay_type = Column(String(100))
    organism = Column(String(100))
    cell_line = Column(String(100))
    
    # Quality metrics
    confidence_score = Column(Float)
    data_validity = Column(String(20))  # 'valid', 'intermediate', 'invalid'
    
    # Binding site information
    binding_site_id = Column(UUID(as_uuid=True), ForeignKey("binding_sites.id"))
    binding_mode = Column(String(50))  # 'competitive', 'non-competitive', 'allosteric'
    
    # Source information
    source = Column(String(100))  # 'ChEMBL', 'BindingDB', 'experimental'
    publication_doi = Column(String(255))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drug = relationship("Drug", back_populates="protein_interactions")
    protein = relationship("Protein", back_populates="drug_interactions")
    
    def __repr__(self):
        return f"<DrugProteinInteraction(drug_id='{self.drug_id}', protein_id='{self.protein_id}')>"


class ADMETProperties(Base):
    """ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties"""
    
    __tablename__ = "admet_properties"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drug_id = Column(UUID(as_uuid=True), ForeignKey("drugs.id"), nullable=False)
    
    # Absorption
    caco2_permeability = Column(Float)  # Caco-2 cell permeability
    human_intestinal_absorption = Column(Float)  # % absorbed
    bioavailability_f20 = Column(Float)  # Oral bioavailability
    
    # Distribution
    plasma_protein_binding = Column(Float)  # % bound
    volume_of_distribution = Column(Float)  # L/kg
    blood_brain_barrier = Column(Float)  # BBB permeability
    
    # Metabolism
    cyp1a2_inhibition = Column(Float)
    cyp2c19_inhibition = Column(Float)
    cyp2c9_inhibition = Column(Float)
    cyp2d6_inhibition = Column(Float)
    cyp3a4_inhibition = Column(Float)
    
    # Excretion
    renal_clearance = Column(Float)  # mL/min/kg
    half_life = Column(Float)  # hours
    
    # Toxicity
    acute_toxicity_ld50 = Column(Float)  # mg/kg
    hepatotoxicity = Column(Float)  # probability
    cardiotoxicity = Column(Float)  # probability
    mutagenicity = Column(Float)  # probability
    carcinogenicity = Column(Float)  # probability
    
    # hERG channel inhibition (cardiotoxicity)
    herg_inhibition = Column(Float)
    
    # Prediction metadata
    prediction_method = Column(String(100))
    model_version = Column(String(50))
    confidence_scores = Column(JSON)  # Per-property confidence
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drug = relationship("Drug", back_populates="admet_properties")
    
    def __repr__(self):
        return f"<ADMETProperties(drug_id='{self.drug_id}')>"


class MolecularDescriptor(Base):
    """Molecular descriptors for QSAR modeling"""
    
    __tablename__ = "molecular_descriptors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drug_id = Column(UUID(as_uuid=True), ForeignKey("drugs.id"), nullable=False)
    
    # 2D descriptors
    molecular_weight = Column(Float)
    heavy_atom_count = Column(Integer)
    ring_count = Column(Integer)
    aromatic_ring_count = Column(Integer)
    
    # Lipinski descriptors
    logp = Column(Float)
    hbd = Column(Integer)
    hba = Column(Integer)
    tpsa = Column(Float)
    
    # Additional descriptors
    formal_charge = Column(Integer)
    fractional_csp3 = Column(Float)
    num_rotatable_bonds = Column(Integer)
    num_saturated_rings = Column(Integer)
    num_aliphatic_rings = Column(Integer)
    
    # Fingerprints (as JSON)
    morgan_fingerprint = Column(JSON)
    rdkit_fingerprint = Column(JSON)
    maccs_fingerprint = Column(JSON)
    
    # 3D descriptors (if available)
    asphericity = Column(Float)
    eccentricity = Column(Float)
    inertial_shape_factor = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MolecularDescriptor(drug_id='{self.drug_id}')>"


class DrugTarget(Base):
    """Drug target information"""
    
    __tablename__ = "drug_targets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Target identification
    target_name = Column(String(255), nullable=False)
    target_type = Column(String(50))  # 'protein', 'enzyme', 'receptor', 'transporter'
    uniprot_id = Column(String(20), index=True)
    
    # Classification
    target_class = Column(String(100))
    protein_family = Column(String(100))
    
    # Druggability
    druggability_score = Column(Float)
    tractability_bucket = Column(String(20))  # 'clinical_precedence', 'predicted_tractable'
    
    # Disease association
    disease_association = Column(Text)
    therapeutic_area = Column(String(100))
    
    # Validation level
    validation_level = Column(String(50))  # 'target', 'lead', 'candidate', 'clinical', 'launched'
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<DrugTarget(name='{self.target_name}', type='{self.target_type}')>"
