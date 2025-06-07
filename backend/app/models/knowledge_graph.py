"""
Knowledge graph data models
SQLAlchemy models for scientific knowledge representation and relationships
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid

from app.core.database import Base


class KnowledgeNode(Base):
    """Knowledge graph node model"""
    
    __tablename__ = "knowledge_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Node identification
    node_type = Column(String(50), nullable=False, index=True)  # 'protein', 'drug', 'disease', 'pathway'
    external_id = Column(String(255), nullable=False, index=True)  # External database ID
    name = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Ontology information
    ontology_source = Column(String(100))  # 'GO', 'ChEBI', 'MONDO', 'KEGG'
    ontology_id = Column(String(100))
    
    # Node properties
    properties = Column(JSON)  # Flexible properties storage
    synonyms = Column(ARRAY(String))
    
    # Confidence and quality
    confidence_score = Column(Float, default=1.0)
    data_quality = Column(String(20), default="high")  # 'high', 'medium', 'low'
    
    # Provenance
    source = Column(String(100))  # Data source
    created_by = Column(String(100))  # System or user
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    outgoing_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.source_node_id", back_populates="source_node")
    incoming_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.target_node_id", back_populates="target_node")
    
    def __repr__(self):
        return f"<KnowledgeNode(type='{self.node_type}', name='{self.name}')>"


class KnowledgeEdge(Base):
    """Knowledge graph edge/relationship model"""
    
    __tablename__ = "knowledge_edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Edge endpoints
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_nodes.id"), nullable=False)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_nodes.id"), nullable=False)
    
    # Relationship information
    relationship_type = Column(String(100), nullable=False, index=True)  # 'interacts_with', 'inhibits', 'activates'
    relationship_label = Column(String(255))
    description = Column(Text)
    
    # Edge properties
    properties = Column(JSON)  # Flexible properties storage
    weight = Column(Float, default=1.0)  # Edge weight for graph algorithms
    
    # Evidence and confidence
    evidence_type = Column(String(50))  # 'experimental', 'computational', 'literature'
    confidence_score = Column(Float)
    p_value = Column(Float)
    
    # Provenance
    source = Column(String(100))  # Data source
    publication_doi = Column(String(255))
    experiment_id = Column(String(100))
    
    # Temporal information
    valid_from = Column(DateTime)
    valid_to = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    source_node = relationship("KnowledgeNode", foreign_keys=[source_node_id], back_populates="outgoing_edges")
    target_node = relationship("KnowledgeNode", foreign_keys=[target_node_id], back_populates="incoming_edges")
    
    def __repr__(self):
        return f"<KnowledgeEdge(type='{self.relationship_type}', source='{self.source_node_id}', target='{self.target_node_id}')>"


class ScientificPublication(Base):
    """Scientific publication model"""
    
    __tablename__ = "scientific_publications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Publication identifiers
    pubmed_id = Column(String(20), unique=True, index=True)
    doi = Column(String(255), unique=True, index=True)
    pmc_id = Column(String(20), index=True)
    
    # Publication details
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    authors = Column(ARRAY(String))
    journal = Column(String(255))
    
    # Publication metadata
    publication_date = Column(DateTime)
    publication_year = Column(Integer, index=True)
    volume = Column(String(50))
    issue = Column(String(50))
    pages = Column(String(50))
    
    # Content analysis
    keywords = Column(ARRAY(String))
    mesh_terms = Column(ARRAY(String))
    research_areas = Column(ARRAY(String))
    
    # Quality metrics
    citation_count = Column(Integer, default=0)
    impact_factor = Column(Float)
    quality_score = Column(Float)
    
    # Processing metadata
    processed_at = Column(DateTime)
    processing_version = Column(String(20))
    
    # Full text availability
    has_full_text = Column(Boolean, default=False)
    full_text_url = Column(String(500))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ScientificPublication(pubmed_id='{self.pubmed_id}', title='{self.title[:50]}...')>"


class Hypothesis(Base):
    """AI-generated scientific hypothesis model"""
    
    __tablename__ = "hypotheses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Hypothesis content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    hypothesis_text = Column(Text, nullable=False)
    
    # Classification
    research_area = Column(String(100))
    hypothesis_type = Column(String(50))  # 'therapeutic', 'mechanistic', 'predictive'
    novelty_score = Column(Float)
    
    # Supporting evidence
    supporting_nodes = Column(ARRAY(UUID))  # References to KnowledgeNode IDs
    supporting_edges = Column(ARRAY(UUID))  # References to KnowledgeEdge IDs
    evidence_strength = Column(Float)
    
    # AI generation metadata
    generation_method = Column(String(100))  # 'gpt-4', 'knowledge-graph-reasoning'
    model_version = Column(String(50))
    generation_parameters = Column(JSON)
    
    # Validation
    validation_status = Column(String(20), default="pending")  # 'pending', 'validated', 'rejected'
    validation_score = Column(Float)
    expert_review = Column(Text)
    
    # Testability
    testability_score = Column(Float)
    suggested_experiments = Column(JSON)
    estimated_cost = Column(Float)
    estimated_timeline = Column(Integer)  # months
    
    # Impact prediction
    potential_impact = Column(String(20))  # 'low', 'medium', 'high'
    therapeutic_relevance = Column(Float)
    commercial_potential = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Hypothesis(title='{self.title[:50]}...', novelty_score={self.novelty_score})>"


class KnowledgeGraphMetrics(Base):
    """Knowledge graph quality and performance metrics"""
    
    __tablename__ = "knowledge_graph_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Graph statistics
    total_nodes = Column(Integer)
    total_edges = Column(Integer)
    node_types_count = Column(JSON)  # Count by node type
    edge_types_count = Column(JSON)  # Count by edge type
    
    # Quality metrics
    average_node_degree = Column(Float)
    graph_density = Column(Float)
    clustering_coefficient = Column(Float)
    average_path_length = Column(Float)
    
    # Coverage metrics
    protein_coverage = Column(Float)  # % of known proteins
    drug_coverage = Column(Float)  # % of known drugs
    disease_coverage = Column(Float)  # % of known diseases
    
    # Data quality
    high_confidence_edges = Column(Integer)
    medium_confidence_edges = Column(Integer)
    low_confidence_edges = Column(Integer)
    
    # Update information
    last_update = Column(DateTime)
    data_sources = Column(ARRAY(String))
    
    # Performance metrics
    query_response_time = Column(Float)  # average in milliseconds
    indexing_time = Column(Float)  # last indexing time in seconds
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<KnowledgeGraphMetrics(nodes={self.total_nodes}, edges={self.total_edges})>"


class OntologyMapping(Base):
    """Ontology term mappings for knowledge integration"""
    
    __tablename__ = "ontology_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Source ontology
    source_ontology = Column(String(50), nullable=False)
    source_id = Column(String(100), nullable=False)
    source_term = Column(String(500))
    
    # Target ontology
    target_ontology = Column(String(50), nullable=False)
    target_id = Column(String(100), nullable=False)
    target_term = Column(String(500))
    
    # Mapping information
    mapping_type = Column(String(20))  # 'exact', 'broad', 'narrow', 'related'
    confidence_score = Column(Float)
    mapping_method = Column(String(100))  # 'manual', 'lexical', 'semantic'
    
    # Validation
    validated = Column(Boolean, default=False)
    validated_by = Column(String(100))
    validation_date = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<OntologyMapping({self.source_ontology}:{self.source_id} -> {self.target_ontology}:{self.target_id})>"
