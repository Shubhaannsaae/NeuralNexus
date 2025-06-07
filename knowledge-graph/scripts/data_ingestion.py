"""
Data Ingestion Scripts for Knowledge Graph
Production-grade implementation for ingesting scientific data into OriginTrail DKG
"""

import asyncio
import logging
import json
import aiohttp
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
import yaml
import jsonschema
import sys
import os
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load backend environment variables
from dotenv import load_dotenv
backend_env_path = backend_path / ".env"
if backend_env_path.exists():
    load_dotenv(backend_env_path)

from app.core.database import get_db

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

# Import your models (adjust imports based on your project structure)
try:
    from app.core.database import get_db
    from app.models.knowledge_graph import KnowledgeNode, KnowledgeEdge
    from app.core.config import settings
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.append('../backend')

logger = logging.getLogger(__name__)

class DataIngestionService:
    """Production-grade data ingestion service for knowledge graph"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.schemas = self._load_schemas()
        self.origintrail_client = OriginTrailClient(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _load_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for validation"""
        schemas = {}
        schema_dir = Path("schemas")
        
        for schema_file in ["protein.json", "drug.json", "disease.json"]:
            schema_path = schema_dir / schema_file
            try:
                with open(schema_path, 'r') as file:
                    schema_name = schema_file.replace('.json', '')
                    schemas[schema_name] = json.load(file)
                    logger.info(f"Loaded schema: {schema_name}")
            except FileNotFoundError:
                logger.warning(f"Schema file {schema_file} not found")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing schema {schema_file}: {e}")
        
        return schemas
    
    def validate_data(self, data: Dict, data_type: str) -> bool:
        """Validate data against JSON schema"""
        if data_type not in self.schemas:
            logger.warning(f"No schema found for data type: {data_type}")
            return True  # Allow if no schema
        
        try:
            jsonschema.validate(instance=data, schema=self.schemas[data_type])
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Validation error for {data_type}: {e.message}")
            return False
        except jsonschema.SchemaError as e:
            logger.error(f"Schema error for {data_type}: {e.message}")
            return False
    
    async def ingest_protein_data(self, protein_data: List[Dict], db: AsyncSession) -> Dict[str, int]:
        """Ingest protein data with validation and deduplication"""
        stats = {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
        for protein in protein_data:
            stats["processed"] += 1
            
            try:
                # Validate data
                if not self.validate_data(protein, "protein"):
                    stats["errors"] += 1
                    continue
                
                # Check if protein already exists
                uniprot_id = protein.get("identifier")
                existing_node = await db.execute(
                    select(KnowledgeNode).where(
                        and_(
                            KnowledgeNode.node_type == "protein",
                            KnowledgeNode.external_id == uniprot_id
                        )
                    )
                )
                existing = existing_node.scalar_one_or_none()
                
                if existing:
                    # Update existing node
                    existing.name = protein.get("name", existing.name)
                    existing.description = protein.get("description", existing.description)
                    existing.properties = protein
                    existing.updated_at = datetime.now(timezone.utc)
                    stats["updated"] += 1
                else:
                    # Create new node
                    node = KnowledgeNode(
                        node_type="protein",
                        external_id=uniprot_id,
                        name=protein.get("name"),
                        description=protein.get("description"),
                        properties=protein,
                        confidence_score=protein.get("confidence_score", 1.0),
                        source="uniprot",
                        ontology_source="uniprot",
                        ontology_id=uniprot_id,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(node)
                    stats["inserted"] += 1
                
                # Store in OriginTrail DKG
                await self.origintrail_client.create_knowledge_asset(protein, "protein")
                
            except Exception as e:
                logger.error(f"Error processing protein {protein.get('identifier', 'unknown')}: {e}")
                stats["errors"] += 1
        
        db.commit()
        logger.info(f"Protein ingestion completed: {stats}")
        return stats
    
    async def ingest_drug_data(self, drug_data: List[Dict], db: AsyncSession) -> Dict[str, int]:
        """Ingest drug data with validation and deduplication"""
        stats = {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
        for drug in drug_data:
            stats["processed"] += 1
            
            try:
                # Validate data
                if not self.validate_data(drug, "drug"):
                    stats["errors"] += 1
                    continue
                
                # Check if drug already exists
                chembl_id = drug.get("identifier")
                existing_node = await db.execute(
                    select(KnowledgeNode).where(
                        and_(
                            KnowledgeNode.node_type == "drug",
                            KnowledgeNode.external_id == chembl_id
                        )
                    )
                )
                existing = existing_node.scalar_one_or_none()
                
                if existing:
                    # Update existing node
                    existing.name = drug.get("name", existing.name)
                    existing.description = drug.get("description", existing.description)
                    existing.properties = drug
                    existing.updated_at = datetime.now(timezone.utc)
                    stats["updated"] += 1
                else:
                    # Create new node
                    node = KnowledgeNode(
                        node_type="drug",
                        external_id=chembl_id,
                        name=drug.get("name"),
                        description=drug.get("description"),
                        properties=drug,
                        confidence_score=drug.get("confidence_score", 1.0),
                        source="chembl",
                        ontology_source="chembl",
                        ontology_id=chembl_id,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(node)
                    stats["inserted"] += 1
                
                # Store in OriginTrail DKG
                await self.origintrail_client.create_knowledge_asset(drug, "drug")
                
            except Exception as e:
                logger.error(f"Error processing drug {drug.get('identifier', 'unknown')}: {e}")
                stats["errors"] += 1
        
        db.commit()
        logger.info(f"Drug ingestion completed: {stats}")
        return stats
    
    async def ingest_disease_data(self, disease_data: List[Dict], db: AsyncSession) -> Dict[str, int]:
        """Ingest disease data with validation and deduplication"""
        stats = {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
        for disease in disease_data:
            stats["processed"] += 1
            
            try:
                # Validate data
                if not self.validate_data(disease, "disease"):
                    stats["errors"] += 1
                    continue
                
                # Check if disease already exists
                mondo_id = disease.get("identifier")
                existing_node = await db.execute(
                    select(KnowledgeNode).where(
                        and_(
                            KnowledgeNode.node_type == "disease",
                            KnowledgeNode.external_id == mondo_id
                        )
                    )
                )
                existing = existing_node.scalar_one_or_none()
                
                if existing:
                    # Update existing node
                    existing.name = disease.get("name", existing.name)
                    existing.description = disease.get("description", existing.description)
                    existing.properties = disease
                    existing.updated_at = datetime.now(timezone.utc)
                    stats["updated"] += 1
                else:
                    # Create new node
                    node = KnowledgeNode(
                        node_type="disease",
                        external_id=mondo_id,
                        name=disease.get("name"),
                        description=disease.get("description"),
                        properties=disease,
                        confidence_score=disease.get("confidence_score", 1.0),
                        source="mondo",
                        ontology_source="mondo",
                        ontology_id=mondo_id,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(node)
                    stats["inserted"] += 1
                
                # Store in OriginTrail DKG
                await self.origintrail_client.create_knowledge_asset(disease, "disease")
                
            except Exception as e:
                logger.error(f"Error processing disease {disease.get('identifier', 'unknown')}: {e}")
                stats["errors"] += 1
        
        await db.commit()
        logger.info(f"Disease ingestion completed: {stats}")
        return stats
    
    async def ingest_relationships(self, relationships: List[Dict], db: AsyncSession) -> Dict[str, int]:
        """Ingest relationships (edges) with validation"""
        stats = {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
        for rel in relationships:
            stats["processed"] += 1
            
            try:
                # Validate required fields
                required_fields = ["source_node_id", "target_node_id", "relationship_type"]
                if not all(field in rel for field in required_fields):
                    logger.error(f"Missing required fields in relationship: {rel}")
                    stats["errors"] += 1
                    continue
                
                # Check if relationship already exists
                existing_edge = await db.execute(
                    select(KnowledgeEdge).where(
                        and_(
                            KnowledgeEdge.source_node_id == rel["source_node_id"],
                            KnowledgeEdge.target_node_id == rel["target_node_id"],
                            KnowledgeEdge.relationship_type == rel["relationship_type"]
                        )
                    )
                )
                existing = existing_edge.scalar_one_or_none()
                
                if existing:
                    # Update existing edge
                    existing.properties = rel.get("properties", existing.properties)
                    existing.confidence_score = rel.get("confidence_score", existing.confidence_score)
                    existing.updated_at = datetime.now(timezone.utc)
                    stats["updated"] += 1
                else:
                    # Create new edge
                    edge = KnowledgeEdge(
                        source_node_id=rel["source_node_id"],
                        target_node_id=rel["target_node_id"],
                        relationship_type=rel["relationship_type"],
                        properties=rel.get("properties", {}),
                        confidence_score=rel.get("confidence_score", 1.0),
                        evidence_type=rel.get("evidence_type", "computational"),
                        source=rel.get("source", "unknown"),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(edge)
                    stats["inserted"] += 1
                
            except Exception as e:
                logger.error(f"Error processing relationship: {e}")
                stats["errors"] += 1
        
        await db.commit()
        logger.info(f"Relationship ingestion completed: {stats}")
        return stats


class OriginTrailClient:
    """Client for OriginTrail DKG operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config["node"]["endpoint"]
        self.api_key = config["auth"]["api_key"]
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def create_knowledge_asset(self, data: Dict, asset_type: str) -> Optional[str]:
        """Create knowledge asset in OriginTrail DKG"""
        try:
            # Prepare assertion
            assertion = {
                "@context": data.get("@context", {}),
                "@type": data.get("@type", asset_type.capitalize()),
                "@id": data.get("@id", f"neurograph:{asset_type}:{data.get('identifier')}"),
                **data
            }
            
            # Add metadata
            assertion["neurograph:createdBy"] = "NeuroGraph AI"
            assertion["neurograph:createdAt"] = datetime.now(timezone.utc).isoformat()
            assertion["neurograph:version"] = "1.0"
            
            # Create knowledge asset
            payload = {
                "assertion": assertion,
                "assertionId": self._generate_assertion_id(assertion),
                "blockchain": self.config["network"]["chain_id"],
                "epochsNumber": self.config["knowledge_assets"]["default_epochs"],
                "tokenAmount": self.config["knowledge_assets"]["default_token_amount"]
            }
            
            if not self.session:
                async with self:
                    return await self._make_request("/knowledge-assets", payload)
            else:
                return await self._make_request("/knowledge-assets", payload)
                
        except Exception as e:
            logger.error(f"Failed to create knowledge asset: {e}")
            return None
    
    async def _make_request(self, endpoint: str, payload: Dict) -> Optional[str]:
        """Make HTTP request to OriginTrail node"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 201:
                    result = await response.json()
                    ual = result.get("UAL")
                    logger.info(f"Knowledge asset created with UAL: {ual}")
                    return ual
                else:
                    error_text = await response.text()
                    logger.error(f"DKG request failed: {response.status} - {error_text}")
                    return None
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in DKG request: {e}")
            return None
    
    def _generate_assertion_id(self, assertion: Dict) -> str:
        """Generate unique assertion ID"""
        assertion_str = json.dumps(assertion, sort_keys=True)
        return hashlib.sha256(assertion_str.encode()).hexdigest()


class DataSourceConnector:
    """Connector for external data sources"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def fetch_uniprot_data(self, uniprot_ids: List[str]) -> List[Dict]:
        """Fetch protein data from UniProt"""
        proteins = []
        base_url = self.config["data_sources"]["uniprot"]["base_url"]
        
        async with aiohttp.ClientSession() as session:
            for uniprot_id in uniprot_ids:
                try:
                    url = f"{base_url}/uniprotkb/{uniprot_id}.json"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            protein = self._transform_uniprot_data(data)
                            proteins.append(protein)
                        else:
                            logger.warning(f"Failed to fetch UniProt data for {uniprot_id}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching UniProt data for {uniprot_id}: {e}")
        
        return proteins
    
    def _transform_uniprot_data(self, uniprot_data: Dict) -> Dict:
        """Transform UniProt data to our schema format"""
        # Extract basic information
        protein = {
            "@context": {
                "@vocab": "https://schema.org/",
                "uniprot": "https://www.uniprot.org/uniprot/",
                "go": "http://purl.obolibrary.org/obo/GO_",
                "neurograph": "https://neurograph.ai/ontology/"
            },
            "@type": "Protein",
            "@id": f"uniprot:{uniprot_data['primaryAccession']}",
            "identifier": uniprot_data["primaryAccession"],
            "name": uniprot_data["proteinDescription"]["recommendedName"]["fullName"]["value"],
            "description": uniprot_data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "organism": {
                "@type": "Organism",
                "name": uniprot_data["organism"]["scientificName"],
                "taxonId": str(uniprot_data["organism"]["taxonId"])
            },
            "sequence": {
                "@type": "AminoAcidSequence",
                "value": uniprot_data["sequence"]["value"],
                "length": uniprot_data["sequence"]["length"],
                "molecularWeight": uniprot_data["sequence"]["molWeight"]
            }
        }
        
        # Add GO annotations
        if "uniProtKBCrossReferences" in uniprot_data:
            for ref in uniprot_data["uniProtKBCrossReferences"]:
                if ref["database"] == "GO":
                    go_term = ref["id"]
                    properties = ref.get("properties", [])
                    
                    # Determine GO category
                    category = None
                    for prop in properties:
                        if prop["key"] == "GoTerm":
                            if "F:" in prop["value"]:
                                category = "function"
                            elif "C:" in prop["value"]:
                                category = "cellularComponent"
                            elif "P:" in prop["value"]:
                                                                category = "biologicalProcess"
                    
                    if category:
                        if category not in protein:
                            protein[category] = []
                        
                        protein[category].append({
                            "@type": category.replace("c", "C").replace("f", "F").replace("b", "B"),
                            "goTerm": go_term,
                            "description": next((p["value"] for p in properties if p["key"] == "GoTerm"), "")
                        })
        
        # Add alternative names
        if "proteinDescription" in uniprot_data and "alternativeNames" in uniprot_data["proteinDescription"]:
            protein["alternativeName"] = [
                alt["fullName"]["value"] 
                for alt in uniprot_data["proteinDescription"]["alternativeNames"]
                if "fullName" in alt
            ]
        
        # Add keywords
        if "keywords" in uniprot_data:
            protein["keywords"] = [kw["name"] for kw in uniprot_data["keywords"]]
        
        # Add creation/modification dates
        protein["dateCreated"] = datetime.now(timezone.utc).isoformat()
        protein["dateModified"] = datetime.now(timezone.utc).isoformat()
        protein["creator"] = {
            "@type": "Organization",
            "name": "NeuroGraph AI",
            "url": "https://neurograph.ai"
        }
        
        return protein
    
    async def fetch_chembl_data(self, chembl_ids: List[str]) -> List[Dict]:
        """Fetch drug data from ChEMBL"""
        drugs = []
        base_url = self.config["data_sources"]["chembl"]["base_url"]
        
        async with aiohttp.ClientSession() as session:
            for chembl_id in chembl_ids:
                try:
                    url = f"{base_url}/molecule/{chembl_id}.json"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            drug = self._transform_chembl_data(data)
                            drugs.append(drug)
                        else:
                            logger.warning(f"Failed to fetch ChEMBL data for {chembl_id}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching ChEMBL data for {chembl_id}: {e}")
        
        return drugs
    
    def _transform_chembl_data(self, chembl_data: Dict) -> Dict:
        """Transform ChEMBL data to our schema format"""
        drug = {
            "@context": {
                "@vocab": "https://schema.org/",
                "chembl": "https://www.ebi.ac.uk/chembl/",
                "pubchem": "https://pubchem.ncbi.nlm.nih.gov/",
                "neurograph": "https://neurograph.ai/ontology/"
            },
            "@type": "Drug",
            "@id": f"chembl:{chembl_data['molecule_chembl_id']}",
            "identifier": chembl_data["molecule_chembl_id"],
            "name": chembl_data.get("pref_name", ""),
            "description": chembl_data.get("indication_class", ""),
            "chemicalStructure": {
                "smiles": chembl_data.get("molecule_structures", {}).get("canonical_smiles", ""),
                "inchi": chembl_data.get("molecule_structures", {}).get("standard_inchi", ""),
                "inchiKey": chembl_data.get("molecule_structures", {}).get("standard_inchi_key", ""),
                "molecularFormula": chembl_data.get("molecule_properties", {}).get("molecular_formula", "")
            }
        }
        
        # Add molecular properties
        if "molecule_properties" in chembl_data:
            props = chembl_data["molecule_properties"]
            drug["molecularProperties"] = {
                "molecularWeight": props.get("molecular_weight"),
                "logP": props.get("alogp"),
                "hBondDonors": props.get("hbd"),
                "hBondAcceptors": props.get("hba"),
                "rotatableBonds": props.get("rtb"),
                "tpsa": props.get("psa"),
                "heavyAtoms": props.get("heavy_atoms"),
                "aromaticRings": props.get("aromatic_rings")
            }
        
        # Add synonyms
        if "molecule_synonyms" in chembl_data:
            drug["synonyms"] = [syn["molecule_synonym"] for syn in chembl_data["molecule_synonyms"]]
        
        # Add development phase
        drug["developmentPhase"] = chembl_data.get("max_phase", "Unknown")
        
        # Add dates
        drug["dateCreated"] = datetime.now(timezone.utc).isoformat()
        drug["dateModified"] = datetime.now(timezone.utc).isoformat()
        drug["creator"] = {
            "@type": "Organization",
            "name": "NeuroGraph AI",
            "url": "https://neurograph.ai"
        }
        
        return drug


async def main():
    """Main function for data ingestion"""
    try:
        # Initialize services
        ingestion_service = DataIngestionService()
        data_connector = DataSourceConnector(ingestion_service.config)
        
        # Get database session
        for db in get_db():
            # Example: Ingest protein data
            uniprot_ids = ["P53_HUMAN", "EGFR_HUMAN", "BRCA1_HUMAN"]
            protein_data = await data_connector.fetch_uniprot_data(uniprot_ids)
            protein_stats = await ingestion_service.ingest_protein_data(protein_data, db)
            
            # Example: Ingest drug data
            chembl_ids = ["CHEMBL25", "CHEMBL1201585", "CHEMBL267014"]
            drug_data = await data_connector.fetch_chembl_data(chembl_ids)
            drug_stats = await ingestion_service.ingest_drug_data(drug_data, db)
            
            logger.info(f"Ingestion completed - Proteins: {protein_stats}, Drugs: {drug_stats}")
            
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
