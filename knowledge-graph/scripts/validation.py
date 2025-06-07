"""
Data Validation Scripts for Knowledge Graph
Production-grade validation for scientific data integrity and quality
"""

import asyncio
import logging
import json
import jsonschema
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from pathlib import Path
import re
import requests
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from dataclasses import dataclass
import aiohttp

# Import your models
try:
    from app.core.database import get_db
    from app.models.knowledge_graph import KnowledgeNode, KnowledgeEdge
    from app.core.config import settings
except ImportError:
    import sys
    sys.path.append('../backend')

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Data class for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float
    details: Dict[str, Any]


class DataValidator:
    """Production-grade data validator for knowledge graph"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.schemas = self._load_schemas()
        self.external_validators = ExternalValidators()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load validation configuration"""
        import yaml
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_validation_config()
    
    def _default_validation_config(self) -> Dict:
        """Default validation configuration"""
        return {
            "quality_control": {
                "min_confidence_score": 0.7,
                "max_assertion_size": "5MB",
                "validation_timeout": 30,
                "validation_rules": {
                    "protein_sequence": {
                        "min_length": 10,
                        "max_length": 5000,
                        "allowed_characters": "ACDEFGHIKLMNPQRSTVWY"
                    },
                    "drug_smiles": {
                        "min_length": 5,
                        "max_length": 500,
                        "validation_method": "rdkit"
                    },
                    "confidence_scores": {
                        "min_value": 0.0,
                        "max_value": 1.0
                    }
                }
            }
        }
    
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
                    logger.info(f"Loaded validation schema: {schema_name}")
            except FileNotFoundError:
                logger.warning(f"Schema file {schema_file} not found")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing schema {schema_file}: {e}")
        
        return schemas
    
    async def validate_protein_data(self, protein_data: Dict) -> ValidationResult:
        """Comprehensive protein data validation"""
        errors = []
        warnings = []
        score = 1.0
        details = {}
        
        try:
            # Schema validation
            schema_result = self._validate_against_schema(protein_data, "protein")
            if not schema_result.is_valid:
                errors.extend(schema_result.errors)
                score -= 0.3
            
            # Sequence validation
            sequence_result = await self._validate_protein_sequence(protein_data)
            if not sequence_result.is_valid:
                errors.extend(sequence_result.errors)
                warnings.extend(sequence_result.warnings)
                score -= 0.2
            details["sequence_validation"] = sequence_result.details
            
            # UniProt ID validation
            uniprot_result = await self._validate_uniprot_id(protein_data.get("identifier", ""))
            if not uniprot_result.is_valid:
                warnings.extend(uniprot_result.errors)
                score -= 0.1
            details["uniprot_validation"] = uniprot_result.details
            
            # GO term validation
            go_result = await self._validate_go_terms(protein_data)
            if not go_result.is_valid:
                warnings.extend(go_result.errors)
                score -= 0.1
            details["go_validation"] = go_result.details
            
            # Cross-reference validation
            xref_result = await self._validate_protein_cross_references(protein_data)
            details["cross_reference_validation"] = xref_result.details
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            score = 0.0
        
        is_valid = len(errors) == 0 and score >= self.config["quality_control"]["min_confidence_score"]
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score),
            details=details
        )
    
    async def validate_drug_data(self, drug_data: Dict) -> ValidationResult:
        """Comprehensive drug data validation"""
        errors = []
        warnings = []
        score = 1.0
        details = {}
        
        try:
            # Schema validation
            schema_result = self._validate_against_schema(drug_data, "drug")
            if not schema_result.is_valid:
                errors.extend(schema_result.errors)
                score -= 0.3
            
            # Chemical structure validation
            structure_result = await self._validate_chemical_structure(drug_data)
            if not structure_result.is_valid:
                errors.extend(structure_result.errors)
                warnings.extend(structure_result.warnings)
                score -= 0.3
            details["structure_validation"] = structure_result.details
            
            # ChEMBL ID validation
            chembl_result = await self._validate_chembl_id(drug_data.get("identifier", ""))
            if not chembl_result.is_valid:
                warnings.extend(chembl_result.errors)
                score -= 0.1
            details["chembl_validation"] = chembl_result.details
            
            # Molecular properties validation
            props_result = self._validate_molecular_properties(drug_data)
            if not props_result.is_valid:
                warnings.extend(props_result.errors)
                score -= 0.1
            details["properties_validation"] = props_result.details
            
            # Drug-likeness assessment
            druglike_result = self._assess_drug_likeness(drug_data)
            details["druglikeness_assessment"] = druglike_result.details
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            score = 0.0
        
        is_valid = len(errors) == 0 and score >= self.config["quality_control"]["min_confidence_score"]
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score),
            details=details
        )
    
    async def validate_disease_data(self, disease_data: Dict) -> ValidationResult:
        """Comprehensive disease data validation"""
        errors = []
        warnings = []
        score = 1.0
        details = {}
        
        try:
            # Schema validation
            schema_result = self._validate_against_schema(disease_data, "disease")
            if not schema_result.is_valid:
                errors.extend(schema_result.errors)
                score -= 0.3
            
            # MONDO ID validation
            mondo_result = await self._validate_mondo_id(disease_data.get("identifier", ""))
            if not mondo_result.is_valid:
                warnings.extend(mondo_result.errors)
                score -= 0.1
            details["mondo_validation"] = mondo_result.details
            
            # Clinical features validation
            clinical_result = self._validate_clinical_features(disease_data)
            if not clinical_result.is_valid:
                warnings.extend(clinical_result.errors)
                score -= 0.1
            details["clinical_validation"] = clinical_result.details
            
            # ICD-10 code validation
            icd_result = self._validate_icd10_code(disease_data)
            details["icd_validation"] = icd_result.details
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            score = 0.0
        
        is_valid = len(errors) == 0 and score >= self.config["quality_control"]["min_confidence_score"]
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score),
            details=details
        )
    
    def _validate_against_schema(self, data: Dict, data_type: str) -> ValidationResult:
        """Validate data against JSON schema"""
        if data_type not in self.schemas:
            return ValidationResult(True, [], [f"No schema found for {data_type}"], 1.0, {})
        
        try:
            jsonschema.validate(instance=data, schema=self.schemas[data_type])
            return ValidationResult(True, [], [], 1.0, {"schema_valid": True})
        except jsonschema.ValidationError as e:
            return ValidationResult(False, [f"Schema validation error: {e.message}"], [], 0.0, {"schema_valid": False})
        except jsonschema.SchemaError as e:
            return ValidationResult(False, [f"Schema error: {e.message}"], [], 0.0, {"schema_error": True})
    
    async def _validate_protein_sequence(self, protein_data: Dict) -> ValidationResult:
        """Validate protein sequence"""
        errors = []
        warnings = []
        details = {}
        
        sequence_info = protein_data.get("sequence", {})
        sequence = sequence_info.get("value", "")
        
        if not sequence:
            errors.append("Missing protein sequence")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        # Length validation
        rules = self.config["quality_control"]["validation_rules"]["protein_sequence"]
        if len(sequence) < rules["min_length"]:
            errors.append(f"Sequence too short: {len(sequence)} < {rules['min_length']}")
        elif len(sequence) > rules["max_length"]:
            warnings.append(f"Sequence very long: {len(sequence)} > {rules['max_length']}")
        
        # Character validation
        allowed_chars = set(rules["allowed_characters"])
        invalid_chars = set(sequence) - allowed_chars
        if invalid_chars:
            errors.append(f"Invalid amino acid characters: {', '.join(invalid_chars)}")
        
        # Sequence composition analysis
        composition = self._analyze_sequence_composition(sequence)
        details["composition"] = composition
        
        # Check for unusual composition
        if composition.get("stop_codons", 0) > 0:
            warnings.append("Sequence contains stop codons (*)")
        
        if composition.get("unknown_residues", 0) > len(sequence) * 0.1:
            warnings.append("High percentage of unknown residues (X)")
        
        # Molecular weight validation
        expected_mw = sequence_info.get("molecularWeight")
        calculated_mw = self._calculate_molecular_weight(sequence)
        details["calculated_molecular_weight"] = calculated_mw
        
        if expected_mw and abs(expected_mw - calculated_mw) > 1000:
            warnings.append(f"Molecular weight mismatch: expected {expected_mw}, calculated {calculated_mw}")
        
        is_valid = len(errors) == 0
        score = 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
        
        return ValidationResult(is_valid, errors, warnings, max(0.0, score), details)
    
    def _analyze_sequence_composition(self, sequence: str) -> Dict[str, Any]:
        """Analyze amino acid sequence composition"""
        composition = {}
        
        # Count amino acids
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        composition["amino_acid_counts"] = aa_counts
        composition["length"] = len(sequence)
        composition["stop_codons"] = aa_counts.get("*", 0)
        composition["unknown_residues"] = aa_counts.get("X", 0)
        
        # Calculate percentages
        total = len(sequence)
        composition["hydrophobic_percentage"] = sum(
            aa_counts.get(aa, 0) for aa in "AILMFPWV"
        ) / total * 100
        
        composition["charged_percentage"] = sum(
            aa_counts.get(aa, 0) for aa in "DEKR"
        ) / total * 100
        
        composition["polar_percentage"] = sum(
            aa_counts.get(aa, 0) for aa in "STYNQC"
        ) / total * 100
        
        return composition
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight of protein sequence"""
        # Amino acid molecular weights (average isotopic masses)
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        total_weight = 0.0
        for aa in sequence:
            total_weight += aa_weights.get(aa, 0.0)
        
        # Subtract water molecules for peptide bonds
        water_weight = 18.015
        total_weight -= (len(sequence) - 1) * water_weight
        
        return total_weight
    
    async def _validate_uniprot_id(self, uniprot_id: str) -> ValidationResult:
        """Validate UniProt ID format and existence"""
        errors = []
        details = {}
        
        if not uniprot_id:
            errors.append("Missing UniProt ID")
            return ValidationResult(False, errors, [], 0.0, details)
        
        # Format validation
        uniprot_pattern = r'^[A-Z0-9]{6,10}$'
        if not re.match(uniprot_pattern, uniprot_id):
            errors.append(f"Invalid UniProt ID format: {uniprot_id}")
        
        # Check existence (if format is valid)
        if not errors:
            exists = await self.external_validators.check_uniprot_exists(uniprot_id)
            details["exists_in_uniprot"] = exists
            if not exists:
                errors.append(f"UniProt ID not found: {uniprot_id}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    async def _validate_go_terms(self, protein_data: Dict) -> ValidationResult:
        """Validate Gene Ontology terms"""
        errors = []
        warnings = []
        details = {}
        
        go_categories = ["function", "cellularComponent", "biologicalProcess"]
        valid_terms = 0
        total_terms = 0
        
        for category in go_categories:
            if category in protein_data:
                for go_annotation in protein_data[category]:
                    go_term = go_annotation.get("goTerm", "")
                    total_terms += 1
                    
                    # Format validation
                    if re.match(r'^GO:[0-9]{7}$', go_term):
                        # Check if term exists
                        exists = await self.external_validators.check_go_term_exists(go_term)
                        if exists:
                            valid_terms += 1
                        else:
                            warnings.append(f"GO term not found: {go_term}")
                    else:
                        errors.append(f"Invalid GO term format: {go_term}")
        
        details["total_go_terms"] = total_terms
        details["valid_go_terms"] = valid_terms
        
        if total_terms > 0:
            details["validation_rate"] = valid_terms / total_terms
        
        is_valid = len(errors) == 0
        score = valid_terms / max(total_terms, 1)
        
        return ValidationResult(is_valid, errors, warnings, score, details)
    
    async def _validate_protein_cross_references(self, protein_data: Dict) -> ValidationResult:
        """Validate protein cross-references"""
        details = {}
        
        # Check for PDB structures
        if "structure" in protein_data:
            pdb_id = protein_data["structure"].get("pdbId", "")
            if pdb_id:
                pdb_exists = await self.external_validators.check_pdb_exists(pdb_id)
                details["pdb_exists"] = pdb_exists
        
        # Check for Pfam domains
        if "domains" in protein_data:
            pfam_validation = {}
            for domain in protein_data["domains"]:
                pfam_id = domain.get("pfamId", "")
                if pfam_id:
                    exists = await self.external_validators.check_pfam_exists(pfam_id)
                    pfam_validation[pfam_id] = exists
            details["pfam_validation"] = pfam_validation
        
        return ValidationResult(True, [], [], 1.0, details)
    
    async def _validate_chemical_structure(self, drug_data: Dict) -> ValidationResult:
        """Validate chemical structure (SMILES, InChI)"""
        errors = []
        warnings = []
        details = {}
        
        chemical_structure = drug_data.get("chemicalStructure", {})
        smiles = chemical_structure.get("smiles", "")
        inchi = chemical_structure.get("inchi", "")
        inchi_key = chemical_structure.get("inchiKey", "")
        
        if not smiles:
            errors.append("Missing SMILES structure")
            return ValidationResult(False, errors, warnings, 0.0, details)
        
        # SMILES validation
        smiles_result = await self._validate_smiles(smiles)
        if not smiles_result.is_valid:
            errors.extend(smiles_result.errors)
        details["smiles_validation"] = smiles_result.details
        
        # InChI validation
        if inchi:
            inchi_result = self._validate_inchi(inchi)
            if not inchi_result.is_valid:
                warnings.extend(inchi_result.errors)
            details["inchi_validation"] = inchi_result.details
        
        # InChI Key validation
        if inchi_key:
            inchi_key_result = self._validate_inchi_key(inchi_key)
            if not inchi_key_result.is_valid:
                warnings.extend(inchi_key_result.errors)
            details["inchi_key_validation"] = inchi_key_result.details
        
        # Structure consistency check
        if smiles and inchi:
            consistency_result = await self._check_structure_consistency(smiles, inchi)
            details["structure_consistency"] = consistency_result.details
            if not consistency_result.is_valid:
                warnings.extend(consistency_result.errors)
        
        is_valid = len(errors) == 0
        score = 1.0 - (len(errors) * 0.4) - (len(warnings) * 0.1)
        
        return ValidationResult(is_valid, errors, warnings, max(0.0, score), details)
    
    async def _validate_smiles(self, smiles: str) -> ValidationResult:
        """Validate SMILES string using RDKit"""
        errors = []
        details = {}
        
        try:
            # Try to import RDKit
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors.append("Invalid SMILES string")
                return ValidationResult(False, errors, [], 0.0, details)
            
            # Calculate basic properties
            details["molecular_weight"] = Descriptors.MolWt(mol)
            details["num_atoms"] = mol.GetNumAtoms()
            details["num_bonds"] = mol.GetNumBonds()
            details["num_rings"] = Descriptors.RingCount(mol)
            
            # Check for unusual structures
            if mol.GetNumAtoms() > 200:
                errors.append("Molecule too large (>200 atoms)")
            
            if mol.GetNumAtoms() < 5:
                errors.append("Molecule too small (<5 atoms)")
            
        except ImportError:
            # Fallback validation without RDKit
            details["rdkit_available"] = False
            if len(smiles) < 5:
                errors.append("SMILES string too short")
            elif len(smiles) > 500:
                errors.append("SMILES string too long")
            
            # Basic character validation
            allowed_chars = set("CNOPSFClBrI()[]=#@+-0123456789cnops")
            invalid_chars = set(smiles) - allowed_chars
            if invalid_chars:
                errors.append(f"Invalid SMILES characters: {', '.join(invalid_chars)}")
        
        except Exception as e:
            errors.append(f"SMILES validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    def _validate_inchi(self, inchi: str) -> ValidationResult:
        """Validate InChI string"""
        errors = []
        details = {}
        
        if not inchi.startswith("InChI="):
            errors.append("InChI must start with 'InChI='")
        
        # Basic format validation
        if len(inchi) < 10:
            errors.append("InChI string too short")
        elif len(inchi) > 5000:
            errors.append("InChI string too long")
        
        details["length"] = len(inchi)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    def _validate_inchi_key(self, inchi_key: str) -> ValidationResult:
        """Validate InChI Key format"""
        errors = []
        details = {}
        
        # InChI Key format: XXXXXXXXXXXXXX-YYYYYYYYYY-Z
        pattern = r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
        if not re.match(pattern, inchi_key):
            errors.append("Invalid InChI Key format")
        
        details["format_valid"] = len(errors) == 0
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    async def _check_structure_consistency(self, smiles: str, inchi: str) -> ValidationResult:
        """Check consistency between SMILES and InChI"""
        errors = []
        details = {}
        
        try:
            from rdkit import Chem
            
            # Convert SMILES to molecule
            mol_from_smiles = Chem.MolFromSmiles(smiles)
            
            # Convert InChI to molecule
            mol_from_inchi = Chem.MolFromInchi(inchi)
            
            if mol_from_smiles and mol_from_inchi:
                # Compare molecular formulas
                formula_smiles = Chem.rdMolDescriptors.CalcMolFormula(mol_from_smiles)
                formula_inchi = Chem.rdMolDescriptors.CalcMolFormula(mol_from_inchi)
                
                details["formula_from_smiles"] = formula_smiles
                details["formula_from_inchi"] = formula_inchi
                
                if formula_smiles != formula_inchi:
                    errors.append("Molecular formula mismatch between SMILES and InChI")
                
                # Compare InChI Keys
                inchi_key_smiles = Chem.MolToInchiKey(mol_from_smiles)
                inchi_key_inchi = Chem.MolToInchiKey(mol_from_inchi)
                
                details["inchi_key_from_smiles"] = inchi_key_smiles
                details["inchi_key_from_inchi"] = inchi_key_inchi
                
                if inchi_key_smiles != inchi_key_inchi:
                    errors.append("InChI Key mismatch between SMILES and InChI")
            
        except ImportError:
            details["rdkit_available"] = False
        except Exception as e:
            errors.append(f"Structure consistency check failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    async def _validate_chembl_id(self, chembl_id: str) -> ValidationResult:
        """Validate ChEMBL ID format and existence"""
        errors = []
        details = {}
        
        if not chembl_id:
            errors.append("Missing ChEMBL ID")
            return ValidationResult(False, errors, [], 0.0, details)
        
        # Format validation
        chembl_pattern = r'^CHEMBL[0-9]+$'
        if not re.match(chembl_pattern, chembl_id):
            errors.append(f"Invalid ChEMBL ID format: {chembl_id}")
        
        # Check existence
        if not errors:
            exists = await self.external_validators.check_chembl_exists(chembl_id)
            details["exists_in_chembl"] = exists
            if not exists:
                errors.append(f"ChEMBL ID not found: {chembl_id}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    def _validate_molecular_properties(self, drug_data: Dict) -> ValidationResult:
        """Validate molecular properties"""
        errors = []
        warnings = []
        details = {}
        
        mol_props = drug_data.get("molecularProperties", {})
        
        # Molecular weight validation
        mw = mol_props.get("molecularWeight")
        if mw is not None:
            if mw < 50:
                errors.append("Molecular weight too low (<50 Da)")
            elif mw > 2000:
                warnings.append("Very high molecular weight (>2000 Da)")
            details["molecular_weight_valid"] = 50 <= mw <= 2000
        
        # LogP validation
        logp = mol_props.get("logP")
        if logp is not None:
            if logp < -5 or logp > 10:
                warnings.append(f"Unusual LogP value: {logp}")
            details["logp_in_range"] = -5 <= logp <= 10
        
        # Lipinski's Rule of Five validation
        lipinski_violations = 0
        if mw and mw > 500:
            lipinski_violations += 1
        if logp and logp > 5:
            lipinski_violations += 1
        
        hbd = mol_props.get("hBondDonors", 0)
        if hbd > 5:
            lipinski_violations += 1
        
        hba = mol_props.get("hBondAcceptors", 0)
        if hba > 10:
            lipinski_violations += 1
        
        details["lipinski_violations"] = lipinski_violations
        if lipinski_violations > 1:
            warnings.append(f"Multiple Lipinski violations: {lipinski_violations}")
        
        is_valid = len(errors) == 0
        score = 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
        
        return ValidationResult(is_valid, errors, warnings, max(0.0, score), details)
    
    def _assess_drug_likeness(self, drug_data: Dict) -> ValidationResult:
        """Assess drug-likeness using multiple filters"""
        details = {}
        
        mol_props = drug_data.get("molecularProperties", {})
        
        # Lipinski's Rule of Five
        lipinski_compliant = True
        mw = mol_props.get("molecularWeight", 0)
        logp = mol_props.get("logP", 0)
        hbd = mol_props.get("hBondDonors", 0)
        hba = mol_props.get("hBondAcceptors", 0)
        
        if mw > 500 or logp > 5 or hbd > 5 or hba > 10:
            lipinski_compliant = False
        
        details["lipinski_compliant"] = lipinski_compliant
        
        # Veber's rules
        tpsa = mol_props.get("tpsa", 0)
        rotatable_bonds = mol_props.get("rotatableBonds", 0)
        veber_compliant = tpsa <= 140 and rotatable_bonds <= 10
        details["veber_compliant"] = veber_compliant
        
        # Overall drug-likeness score
        score = 0.0
        if lipinski_compliant:
                        score += 0.5
        if veber_compliant:
            score += 0.3
        
        # Additional drug-likeness factors
        aromatic_rings = mol_props.get("aromaticRings", 0)
        if aromatic_rings <= 3:
            score += 0.2
        
        details["druglikeness_score"] = score
        
        return ValidationResult(True, [], [], score, details)
    
    async def _validate_mondo_id(self, mondo_id: str) -> ValidationResult:
        """Validate MONDO ID format and existence"""
        errors = []
        details = {}
        
        if not mondo_id:
            errors.append("Missing MONDO ID")
            return ValidationResult(False, errors, [], 0.0, details)
        
        # Format validation
        mondo_pattern = r'^MONDO_[0-9]{7}$'
        if not re.match(mondo_pattern, mondo_id):
            errors.append(f"Invalid MONDO ID format: {mondo_id}")
        
        # Check existence
        if not errors:
            exists = await self.external_validators.check_mondo_exists(mondo_id)
            details["exists_in_mondo"] = exists
            if not exists:
                errors.append(f"MONDO ID not found: {mondo_id}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, [], 1.0 if is_valid else 0.0, details)
    
    def _validate_clinical_features(self, disease_data: Dict) -> ValidationResult:
        """Validate clinical features and HPO terms"""
        errors = []
        warnings = []
        details = {}
        
        clinical_features = disease_data.get("clinicalFeatures", [])
        valid_hpo_terms = 0
        total_hpo_terms = 0
        
        for feature in clinical_features:
            hpo_term = feature.get("hpoTerm", "")
            if hpo_term:
                total_hpo_terms += 1
                # Validate HPO term format
                if re.match(r'^HP:[0-9]{7}$', hpo_term):
                    valid_hpo_terms += 1
                else:
                    errors.append(f"Invalid HPO term format: {hpo_term}")
            
            # Validate frequency values
            frequency = feature.get("frequency", "")
            valid_frequencies = ["Always", "Very frequent", "Frequent", "Occasional", "Rare", "Unknown"]
            if frequency and frequency not in valid_frequencies:
                warnings.append(f"Invalid frequency value: {frequency}")
        
        details["total_hpo_terms"] = total_hpo_terms
        details["valid_hpo_terms"] = valid_hpo_terms
        
        if total_hpo_terms > 0:
            details["hpo_validation_rate"] = valid_hpo_terms / total_hpo_terms
        
        is_valid = len(errors) == 0
        score = valid_hpo_terms / max(total_hpo_terms, 1) if total_hpo_terms > 0 else 1.0
        
        return ValidationResult(is_valid, errors, warnings, score, details)
    
    def _validate_icd10_code(self, disease_data: Dict) -> ValidationResult:
        """Validate ICD-10 classification code"""
        details = {}
        
        classification = disease_data.get("classification", {})
        icd10_code = classification.get("icd10Code", "")
        
        if icd10_code:
            # ICD-10 format: Letter followed by 2 digits, optionally followed by dot and 1-2 digits
            icd10_pattern = r'^[A-Z][0-9]{2}(\.[0-9]{1,2})?$'
            format_valid = bool(re.match(icd10_pattern, icd10_code))
            details["icd10_format_valid"] = format_valid
            
            if not format_valid:
                return ValidationResult(False, [f"Invalid ICD-10 format: {icd10_code}"], [], 0.0, details)
        
        return ValidationResult(True, [], [], 1.0, details)
    
    async def validate_knowledge_graph_integrity(self, db: AsyncSession) -> ValidationResult:
        """Validate overall knowledge graph integrity"""
        errors = []
        warnings = []
        details = {}
        
        try:
            # Check for orphaned nodes
            orphaned_nodes = await self._find_orphaned_nodes(db)
            details["orphaned_nodes_count"] = len(orphaned_nodes)
            if len(orphaned_nodes) > 0:
                warnings.append(f"Found {len(orphaned_nodes)} orphaned nodes")
            
            # Check for duplicate nodes
            duplicate_nodes = await self._find_duplicate_nodes(db)
            details["duplicate_nodes_count"] = len(duplicate_nodes)
            if len(duplicate_nodes) > 0:
                errors.append(f"Found {len(duplicate_nodes)} duplicate nodes")
            
            # Check edge consistency
            edge_issues = await self._validate_edge_consistency(db)
            details["edge_issues"] = edge_issues
            if edge_issues["invalid_edges"] > 0:
                errors.append(f"Found {edge_issues['invalid_edges']} invalid edges")
            
            # Check confidence score distribution
            confidence_stats = await self._analyze_confidence_scores(db)
            details["confidence_statistics"] = confidence_stats
            if confidence_stats["low_confidence_percentage"] > 30:
                warnings.append("High percentage of low-confidence data")
            
            # Check data freshness
            freshness_stats = await self._analyze_data_freshness(db)
            details["data_freshness"] = freshness_stats
            if freshness_stats["stale_data_percentage"] > 20:
                warnings.append("Significant amount of stale data")
            
        except Exception as e:
            errors.append(f"Integrity validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        score = 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
        
        return ValidationResult(is_valid, errors, warnings, max(0.0, score), details)
    
    async def _find_orphaned_nodes(self, db: AsyncSession) -> List[str]:
        """Find nodes with no edges"""
        query = text("""
            SELECT kn.id 
            FROM knowledge_nodes kn 
            LEFT JOIN knowledge_edges ke1 ON kn.id = ke1.source_node_id 
            LEFT JOIN knowledge_edges ke2 ON kn.id = ke2.target_node_id 
            WHERE ke1.id IS NULL AND ke2.id IS NULL
        """)
        
        result = await db.execute(query)
        return [str(row[0]) for row in result.fetchall()]
    
    async def _find_duplicate_nodes(self, db: AsyncSession) -> List[Dict]:
        """Find potential duplicate nodes"""
        query = text("""
            SELECT node_type, external_id, COUNT(*) as count
            FROM knowledge_nodes 
            WHERE external_id IS NOT NULL
            GROUP BY node_type, external_id 
            HAVING COUNT(*) > 1
        """)
        
        result = await db.execute(query)
        return [
            {"node_type": row[0], "external_id": row[1], "count": row[2]}
            for row in result.fetchall()
        ]
    
    async def _validate_edge_consistency(self, db: AsyncSession) -> Dict[str, int]:
        """Validate edge consistency"""
        stats = {"total_edges": 0, "invalid_edges": 0, "missing_nodes": 0}
        
        # Count total edges
        total_result = await db.execute(select(func.count(KnowledgeEdge.id)))
        stats["total_edges"] = total_result.scalar()
        
        # Find edges with missing source or target nodes
        missing_nodes_query = text("""
            SELECT COUNT(*) 
            FROM knowledge_edges ke 
            LEFT JOIN knowledge_nodes kn1 ON ke.source_node_id = kn1.id 
            LEFT JOIN knowledge_nodes kn2 ON ke.target_node_id = kn2.id 
            WHERE kn1.id IS NULL OR kn2.id IS NULL
        """)
        
        missing_result = await db.execute(missing_nodes_query)
        stats["missing_nodes"] = missing_result.scalar()
        stats["invalid_edges"] = stats["missing_nodes"]
        
        return stats
    
    async def _analyze_confidence_scores(self, db: AsyncSession) -> Dict[str, float]:
        """Analyze confidence score distribution"""
        # Node confidence scores
        node_query = text("""
            SELECT 
                AVG(confidence_score) as avg_confidence,
                MIN(confidence_score) as min_confidence,
                MAX(confidence_score) as max_confidence,
                COUNT(CASE WHEN confidence_score < 0.5 THEN 1 END) * 100.0 / COUNT(*) as low_confidence_percentage
            FROM knowledge_nodes
        """)
        
        node_result = await db.execute(node_query)
        node_stats = node_result.fetchone()
        
        # Edge confidence scores
        edge_query = text("""
            SELECT 
                AVG(confidence_score) as avg_confidence,
                MIN(confidence_score) as min_confidence,
                MAX(confidence_score) as max_confidence,
                COUNT(CASE WHEN confidence_score < 0.5 THEN 1 END) * 100.0 / COUNT(*) as low_confidence_percentage
            FROM knowledge_edges
        """)
        
        edge_result = await db.execute(edge_query)
        edge_stats = edge_result.fetchone()
        
        return {
            "nodes": {
                "avg_confidence": float(node_stats[0]) if node_stats[0] else 0.0,
                "min_confidence": float(node_stats[1]) if node_stats[1] else 0.0,
                "max_confidence": float(node_stats[2]) if node_stats[2] else 0.0,
                "low_confidence_percentage": float(node_stats[3]) if node_stats[3] else 0.0
            },
            "edges": {
                "avg_confidence": float(edge_stats[0]) if edge_stats[0] else 0.0,
                "min_confidence": float(edge_stats[1]) if edge_stats[1] else 0.0,
                "max_confidence": float(edge_stats[2]) if edge_stats[2] else 0.0,
                "low_confidence_percentage": float(edge_stats[3]) if edge_stats[3] else 0.0
            },
            "low_confidence_percentage": max(
                float(node_stats[3]) if node_stats[3] else 0.0,
                float(edge_stats[3]) if edge_stats[3] else 0.0
            )
        }
    
    async def _analyze_data_freshness(self, db: AsyncSession) -> Dict[str, Any]:
        """Analyze data freshness"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=365)  # 1 year old
        
        # Count stale nodes
        stale_nodes_query = text("""
            SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM knowledge_nodes) as stale_percentage
            FROM knowledge_nodes 
            WHERE updated_at < :cutoff_date
        """)
        
        stale_result = await db.execute(stale_nodes_query, {"cutoff_date": cutoff_date})
        stale_percentage = stale_result.scalar() or 0.0
        
        # Average age of data
        avg_age_query = text("""
            SELECT AVG(EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400) as avg_age_days
            FROM knowledge_nodes
        """)
        
        avg_age_result = await db.execute(avg_age_query)
        avg_age_days = avg_age_result.scalar() or 0.0
        
        return {
            "stale_data_percentage": float(stale_percentage),
            "average_age_days": float(avg_age_days),
            "cutoff_date": cutoff_date.isoformat()
        }


class ExternalValidators:
    """External API validators for cross-referencing data"""
    
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=30)
    
    async def check_uniprot_exists(self, uniprot_id: str) -> bool:
        """Check if UniProt ID exists"""
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.head(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"UniProt validation failed for {uniprot_id}: {e}")
            return False
    
    async def check_go_term_exists(self, go_term: str) -> bool:
        """Check if GO term exists"""
        try:
            url = f"http://api.geneontology.org/api/ontology/term/{go_term}"
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"GO term validation failed for {go_term}: {e}")
            return False
    
    async def check_pdb_exists(self, pdb_id: str) -> bool:
        """Check if PDB structure exists"""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.head(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"PDB validation failed for {pdb_id}: {e}")
            return False
    
    async def check_pfam_exists(self, pfam_id: str) -> bool:
        """Check if Pfam family exists"""
        try:
            url = f"https://pfam.xfam.org/family/{pfam_id}"
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.head(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"Pfam validation failed for {pfam_id}: {e}")
            return False
    
    async def check_chembl_exists(self, chembl_id: str) -> bool:
        """Check if ChEMBL compound exists"""
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.head(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"ChEMBL validation failed for {chembl_id}: {e}")
            return False
    
    async def check_mondo_exists(self, mondo_id: str) -> bool:
        """Check if MONDO disease term exists"""
        try:
            # Convert MONDO_0000001 to MONDO:0000001 format for API
            mondo_api_id = mondo_id.replace("_", ":")
            url = f"https://www.ebi.ac.uk/ols/api/ontologies/mondo/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{mondo_api_id}"
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.head(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"MONDO validation failed for {mondo_id}: {e}")
            return False


async def main():
    """Main function for data validation"""
    try:
        # Initialize validator
        validator = DataValidator()
        
        # Get database session
        async for db in get_db():
            # Validate knowledge graph integrity
            integrity_result = await validator.validate_knowledge_graph_integrity(db)
            
            logger.info(f"Knowledge graph integrity validation: {integrity_result.is_valid}")
            logger.info(f"Validation score: {integrity_result.score:.2f}")
            
            if integrity_result.errors:
                logger.error(f"Validation errors: {integrity_result.errors}")
            
            if integrity_result.warnings:
                logger.warning(f"Validation warnings: {integrity_result.warnings}")
            
            # Example: Validate specific protein data
            sample_protein = {
                "@context": {
                    "@vocab": "https://schema.org/",
                    "uniprot": "https://www.uniprot.org/uniprot/",
                    "go": "http://purl.obolibrary.org/obo/GO_",
                    "neurograph": "https://neurograph.ai/ontology/"
                },
                "@type": "Protein",
                "@id": "uniprot:P04637",
                "identifier": "P04637",
                "name": "Cellular tumor antigen p53",
                "sequence": {
                    "@type": "AminoAcidSequence",
                    "value": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
                    "length": 393,
                    "molecularWeight": 43653.0
                },
                "organism": {
                    "@type": "Organism",
                    "name": "Homo sapiens",
                    "taxonId": "9606"
                }
            }
            
            protein_result = await validator.validate_protein_data(sample_protein)
            logger.info(f"Protein validation: {protein_result.is_valid}, Score: {protein_result.score:.2f}")
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
