"""
Real ESMFold integration for protein structure prediction
Production-grade implementation using Facebook's ESMFold model
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from transformers import EsmForProteinFolding, EsmTokenizer
import tempfile
import os
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class RealESMFoldPredictor:
    """Real ESMFold implementation with proper model management"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_sequence_length = 1024
        self.model_name = "facebook/esmfold_v1"
        
    async def initialize(self):
        """Initialize ESMFold model with proper error handling"""
        try:
            logger.info(f"Loading ESMFold model on {self.device}")
            
            # Load model with appropriate precision
            if torch.cuda.is_available():
                self.model = EsmForProteinFolding.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=settings.MODEL_CACHE_DIR
                )
            else:
                self.model = EsmForProteinFolding.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    cache_dir=settings.MODEL_CACHE_DIR
                ).to(self.device)
            
            self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("ESMFold model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ESMFold model: {e}")
            raise
    
    async def predict_structure(
        self,
        sequence: str,
        confidence_threshold: float = 0.7
    ) -> Dict:
        """Real structure prediction with confidence filtering"""
        
        if not self.model:
            await self.initialize()
        
        if len(sequence) > self.max_sequence_length:
            raise ValueError(f"Sequence too long: {len(sequence)} > {self.max_sequence_length}")
        
        start_time = datetime.now()
        
        try:
            # Tokenize sequence
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=self.max_sequence_length
            ).to(self.device)
            
            # Predict structure
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs["input_ids"])
                else:
                    outputs = self.model(inputs["input_ids"])
            
            # Extract results
            positions = outputs.positions.cpu().numpy()  # [1, seq_len, 37, 3]
            confidence_scores = outputs.plddt.cpu().numpy()  # [1, seq_len]
            
            # Remove batch dimension
            positions = positions[0]
            confidence_scores = confidence_scores[0]
            
            # Filter low confidence regions
            high_confidence_mask = confidence_scores >= confidence_threshold
            
            # Generate PDB with proper formatting
            pdb_content = self._generate_pdb_with_confidence(
                sequence, positions, confidence_scores
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                positions, confidence_scores, sequence
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "pdb_content": pdb_content,
                "confidence_scores": confidence_scores.tolist(),
                "positions": positions.tolist(),
                "quality_metrics": quality_metrics,
                "high_confidence_residues": int(np.sum(high_confidence_mask)),
                "processing_time": processing_time,
                "model_version": self.model_name,
                "device_used": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Structure prediction failed: {e}")
            raise
    
    def _generate_pdb_with_confidence(
        self,
        sequence: str,
        positions: np.ndarray,
        confidence_scores: np.ndarray
    ) -> str:
        """Generate properly formatted PDB with confidence in B-factor column"""
        
        pdb_lines = [
            "HEADER    PROTEIN STRUCTURE PREDICTION",
            f"TITLE     ESMFOLD PREDICTION FOR {len(sequence)} RESIDUE PROTEIN",
            f"REMARK   1 PREDICTED BY ESMFOLD",
            f"REMARK   2 MEAN CONFIDENCE: {np.mean(confidence_scores):.2f}",
            "MODEL        1"
        ]
        
        # Atom names for backbone + CB
        backbone_atoms = ["N", "CA", "C", "O"]
        
        atom_id = 1
        
        for i, (aa, pos, confidence) in enumerate(zip(sequence, positions, confidence_scores)):
            res_num = i + 1
            three_letter = self._aa_to_three_letter(aa)
            
            # Add backbone atoms
            for j, atom_name in enumerate(backbone_atoms):
                if j < len(pos) and not np.any(np.isnan(pos[j])):
                    x, y, z = pos[j]
                    
                    pdb_line = (
                        f"ATOM  {atom_id:5d}  {atom_name:<3} {three_letter} A{res_num:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{confidence:6.2f}           {atom_name[0]}  "
                    )
                    pdb_lines.append(pdb_line)
                    atom_id += 1
            
            # Add CB for non-glycine residues
            if aa != 'G' and len(pos) > 4 and not np.any(np.isnan(pos[4])):
                x, y, z = pos[4]
                pdb_line = (
                    f"ATOM  {atom_id:5d}  CB  {three_letter} A{res_num:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{confidence:6.2f}           C   "
                )
                pdb_lines.append(pdb_line)
                atom_id += 1
        
        pdb_lines.extend(["ENDMDL", "END"])
        return "\n".join(pdb_lines)
    
    def _calculate_quality_metrics(
        self,
        positions: np.ndarray,
        confidence_scores: np.ndarray,
        sequence: str
    ) -> Dict:
        """Calculate comprehensive quality metrics"""
        
        metrics = {}
        
        # Confidence statistics
        metrics["mean_confidence"] = float(np.mean(confidence_scores))
        metrics["min_confidence"] = float(np.min(confidence_scores))
        metrics["max_confidence"] = float(np.max(confidence_scores))
        metrics["confidence_std"] = float(np.std(confidence_scores))
        
        # Confidence distribution
        very_high = np.sum(confidence_scores >= 90) / len(confidence_scores)
        high = np.sum((confidence_scores >= 70) & (confidence_scores < 90)) / len(confidence_scores)
        medium = np.sum((confidence_scores >= 50) & (confidence_scores < 70)) / len(confidence_scores)
        low = np.sum(confidence_scores < 50) / len(confidence_scores)
        
        metrics["confidence_distribution"] = {
            "very_high": float(very_high),
            "high": float(high),
            "medium": float(medium),
            "low": float(low)
        }
        
        # Structural quality metrics
        ca_positions = positions[:, 1]  # CA atoms
        valid_ca = ~np.any(np.isnan(ca_positions), axis=1)
        
        if np.sum(valid_ca) > 3:
            # Calculate clash score
            metrics["clash_score"] = self._calculate_clash_score(ca_positions[valid_ca])
            
            # Calculate radius of gyration
            metrics["radius_of_gyration"] = self._calculate_radius_of_gyration(ca_positions[valid_ca])
            
            # Calculate secondary structure content estimate
            metrics["secondary_structure_estimate"] = self._estimate_secondary_structure(
                ca_positions[valid_ca]
            )
        
        return metrics
    
    def _calculate_clash_score(self, ca_positions: np.ndarray) -> float:
        """Calculate atomic clash score"""
        
        clash_count = 0
        total_pairs = 0
        min_distance = 3.0  # Minimum CA-CA distance
        
        for i in range(len(ca_positions)):
            for j in range(i + 2, len(ca_positions)):  # Skip adjacent residues
                distance = np.linalg.norm(ca_positions[i] - ca_positions[j])
                if distance < min_distance:
                    clash_count += 1
                total_pairs += 1
        
        return clash_count / max(total_pairs, 1)
    
    def _calculate_radius_of_gyration(self, ca_positions: np.ndarray) -> float:
        """Calculate radius of gyration"""
        
        center_of_mass = np.mean(ca_positions, axis=0)
        distances_squared = np.sum((ca_positions - center_of_mass) ** 2, axis=1)
        rg = np.sqrt(np.mean(distances_squared))
        
        return float(rg)
    
    def _estimate_secondary_structure(self, ca_positions: np.ndarray) -> Dict:
        """Estimate secondary structure content from CA positions"""
        
        if len(ca_positions) < 4:
            return {"helix": 0.0, "sheet": 0.0, "coil": 1.0}
        
        # Calculate phi/psi-like angles from CA positions
        helix_count = 0
        sheet_count = 0
        
        for i in range(1, len(ca_positions) - 1):
            # Simple geometric criteria
            v1 = ca_positions[i] - ca_positions[i-1]
            v2 = ca_positions[i+1] - ca_positions[i]
            
            # Angle between consecutive CA vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Distance to next-next residue
            if i < len(ca_positions) - 2:
                distance_i_plus_2 = np.linalg.norm(ca_positions[i] - ca_positions[i+2])
                
                # Helix criteria: regular angles and distances
                if 2.8 < angle < 3.5 and 5.0 < distance_i_plus_2 < 6.5:
                    helix_count += 1
                # Sheet criteria: extended conformation
                elif angle > 2.5 and distance_i_plus_2 > 6.0:
                    sheet_count += 1
        
        total = len(ca_positions) - 2
        helix_fraction = helix_count / total
        sheet_fraction = sheet_count / total
        coil_fraction = 1.0 - helix_fraction - sheet_fraction
        
        return {
            "helix": float(helix_fraction),
            "sheet": float(sheet_fraction),
            "coil": float(coil_fraction)
        }
    
    def _aa_to_three_letter(self, aa: str) -> str:
        """Convert single letter amino acid to three letter code"""
        
        aa_map = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        return aa_map.get(aa.upper(), 'UNK')
    
    async def batch_predict(
        self,
        sequences: List[str],
        max_batch_size: int = 4
    ) -> List[Dict]:
        """Batch structure prediction for multiple sequences"""
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(sequences), max_batch_size):
            batch = sequences[i:i + max_batch_size]
            
            batch_results = []
            for sequence in batch:
                try:
                    result = await self.predict_structure(sequence)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch prediction failed for sequence {i}: {e}")
                    batch_results.append({"error": str(e)})
            
            results.extend(batch_results)
            
            # Clear GPU cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results


# Global instance
esmfold_predictor = RealESMFoldPredictor()
