"""
Knowledge Graph Builder
Production-grade implementation for building and maintaining the knowledge graph
"""

import asyncio
import logging
import json
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
import aiohttp
from dataclasses import dataclass

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
class GraphMetrics:
    """Data class for graph metrics"""
    num_nodes: int
    num_edges: int
    density: float
    avg_clustering: float
    num_components: int
    diameter: int
    avg_path_length: float
    centrality_stats: Dict[str, float]


class KnowledgeGraphBuilder:
    """Production-grade knowledge graph builder"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.node_cache = {}
        self.edge_cache = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        import yaml
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "graph_building": {
                "similarity_threshold": 0.7,
                "max_edges_per_node": 100,
                "enable_inference": True,
                "batch_size": 1000
            },
            "algorithms": {
                "community_detection": "louvain",
                "centrality_measures": ["degree", "betweenness", "closeness", "eigenvector"],
                "path_finding": "dijkstra"
            }
        }
    
    async def build_graph_from_database(self, db: AsyncSession) -> GraphMetrics:
        """Build knowledge graph from database"""
        logger.info("Starting knowledge graph construction from database")
        
        try:
            # Load nodes
            nodes_stats = await self._load_nodes_from_db(db)
            logger.info(f"Loaded {nodes_stats['total']} nodes: {nodes_stats}")
            
            # Load edges
            edges_stats = await self._load_edges_from_db(db)
            logger.info(f"Loaded {edges_stats['total']} edges: {edges_stats}")
            
            # Infer additional edges
            if self.config["graph_building"]["enable_inference"]:
                inferred_edges = await self._infer_edges(db)
                logger.info(f"Inferred {len(inferred_edges)} additional edges")
            
            # Calculate graph metrics
            metrics = self._calculate_graph_metrics()
            logger.info(f"Graph construction completed: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            raise
    
    async def _load_nodes_from_db(self, db: AsyncSession) -> Dict[str, int]:
        """Load nodes from database"""
        stats = defaultdict(int)
        
        # Query all nodes
        result = await db.execute(select(KnowledgeNode))
        nodes = result.scalars().all()
        
        for node in nodes:
            # Add to NetworkX graphs
            node_attrs = {
                "node_type": node.node_type,
                "name": node.name,
                "external_id": node.external_id,
                "confidence": node.confidence_score,
                "properties": node.properties or {},
                "created_at": node.created_at,
                "updated_at": node.updated_at
            }
            
            self.graph.add_node(str(node.id), **node_attrs)
            self.directed_graph.add_node(str(node.id), **node_attrs)
            
            # Cache node
            self.node_cache[str(node.id)] = node_attrs
            
            # Update stats
            stats[node.node_type] += 1
            stats["total"] += 1
        
        return dict(stats)
    
    async def _load_edges_from_db(self, db: AsyncSession) -> Dict[str, int]:
        """Load edges from database"""
        stats = defaultdict(int)
        
        # Query all edges
        result = await db.execute(select(KnowledgeEdge))
        edges = result.scalars().all()
        
        for edge in edges:
            source_id = str(edge.source_node_id)
            target_id = str(edge.target_node_id)
            
            # Skip if nodes don't exist
            if source_id not in self.node_cache or target_id not in self.node_cache:
                continue
            
            edge_attrs = {
                "relationship_type": edge.relationship_type,
                "confidence": edge.confidence_score,
                "evidence_type": edge.evidence_type,
                "properties": edge.properties or {},
                "source": edge.source,
                "created_at": edge.created_at,
                "updated_at": edge.updated_at
            }
            
            # Add to graphs
            self.graph.add_edge(source_id, target_id, **edge_attrs)
            self.directed_graph.add_edge(source_id, target_id, **edge_attrs)
            
            # Cache edge
            edge_key = f"{source_id}-{target_id}-{edge.relationship_type}"
            self.edge_cache[edge_key] = edge_attrs
            
            # Update stats
            stats[edge.relationship_type] += 1
            stats["total"] += 1
        
        return dict(stats)
    
    async def _infer_edges(self, db: AsyncSession) -> List[Dict]:
        """Infer additional edges using various algorithms"""
        inferred_edges = []
        
        # Protein-protein interactions based on shared pathways
        ppi_edges = await self._infer_protein_interactions(db)
        inferred_edges.extend(ppi_edges)
        
        # Drug-target relationships based on similarity
        drug_target_edges = await self._infer_drug_targets(db)
        inferred_edges.extend(drug_target_edges)
        
        # Disease-protein associations
        disease_protein_edges = await self._infer_disease_associations(db)
        inferred_edges.extend(disease_protein_edges)
        
        # Pathway relationships
        pathway_edges = await self._infer_pathway_relationships(db)
        inferred_edges.extend(pathway_edges)
        
        # Add inferred edges to database
        await self._store_inferred_edges(inferred_edges, db)
        
        return inferred_edges
    
    async def _infer_protein_interactions(self, db: AsyncSession) -> List[Dict]:
        """Infer protein-protein interactions"""
        interactions = []
        
        # Get all protein nodes
        protein_nodes = {
            node_id: attrs for node_id, attrs in self.node_cache.items()
            if attrs["node_type"] == "protein"
        }
        
        # Find proteins with shared GO terms or pathways
        for protein1_id, protein1_attrs in protein_nodes.items():
            for protein2_id, protein2_attrs in protein_nodes.items():
                if protein1_id >= protein2_id:  # Avoid duplicates
                    continue
                
                # Calculate similarity based on GO terms
                similarity = self._calculate_protein_similarity(
                    protein1_attrs["properties"],
                    protein2_attrs["properties"]
                )
                
                if similarity > self.config["graph_building"]["similarity_threshold"]:
                    interactions.append({
                        "source_node_id": protein1_id,
                        "target_node_id": protein2_id,
                        "relationship_type": "interacts_with",
                        "confidence_score": similarity,
                        "evidence_type": "computational_inference",
                        "source": "neurograph_inference",
                        "properties": {
                            "similarity_score": similarity,
                            "inference_method": "go_term_similarity"
                        }
                    })
        
        logger.info(f"Inferred {len(interactions)} protein-protein interactions")
        return interactions
    
    def _calculate_protein_similarity(self, protein1_props: Dict, protein2_props: Dict) -> float:
        """Calculate similarity between two proteins based on GO terms"""
        # Extract GO terms
        go_terms1 = set()
        go_terms2 = set()
        
        for category in ["function", "cellularComponent", "biologicalProcess"]:
            if category in protein1_props:
                go_terms1.update([term.get("goTerm", "") for term in protein1_props[category]])
            if category in protein2_props:
                go_terms2.update([term.get("goTerm", "") for term in protein2_props[category]])
        
        # Calculate Jaccard similarity
        if not go_terms1 or not go_terms2:
            return 0.0
        
        intersection = len(go_terms1.intersection(go_terms2))
        union = len(go_terms1.union(go_terms2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _infer_drug_targets(self, db: AsyncSession) -> List[Dict]:
        """Infer drug-target relationships"""
        drug_targets = []
        
        # Get drug and protein nodes
        drug_nodes = {
            node_id: attrs for node_id, attrs in self.node_cache.items()
            if attrs["node_type"] == "drug"
        }
        protein_nodes = {
            node_id: attrs for node_id, attrs in self.node_cache.items()
            if attrs["node_type"] == "protein"
        }
        
        # Simple inference based on drug properties and protein function
        for drug_id, drug_attrs in drug_nodes.items():
            drug_props = drug_attrs["properties"]
            
            # Get drug therapeutic class
            therapeutic_class = drug_props.get("therapeuticClass", [])
            
            for protein_id, protein_attrs in protein_nodes.items():
                protein_props = protein_attrs["properties"]
                
                # Check if protein function matches drug therapeutic area
                confidence = self._calculate_drug_target_confidence(
                    drug_props, protein_props
                )
                
                if confidence > 0.6:  # Threshold for drug-target inference
                    drug_targets.append({
                        "source_node_id": drug_id,
                        "target_node_id": protein_id,
                        "relationship_type": "targets",
                        "confidence_score": confidence,
                        "evidence_type": "computational_inference",
                        "source": "neurograph_inference",
                        "properties": {
                            "confidence": confidence,
                            "inference_method": "therapeutic_class_matching"
                        }
                    })
        
        logger.info(f"Inferred {len(drug_targets)} drug-target relationships")
        return drug_targets
    
    def _calculate_drug_target_confidence(self, drug_props: Dict, protein_props: Dict) -> float:
        """Calculate confidence for drug-target relationship"""
        confidence = 0.0
        
        # Check therapeutic class alignment
        therapeutic_classes = drug_props.get("therapeuticClass", [])
        protein_functions = []
        
        # Extract protein function descriptions
        if "function" in protein_props:
            protein_functions.extend([
                func.get("description", "").lower() 
                for func in protein_props["function"]
            ])
        
        # Simple keyword matching
        therapeutic_keywords = {
            "kinase inhibitor": ["kinase", "phosphorylation"],
            "antibiotic": ["bacterial", "antimicrobial"],
            "antineoplastic": ["cancer", "tumor", "oncogene"],
            "cardiovascular": ["heart", "cardiovascular", "blood pressure"],
            "neurological": ["neuron", "brain", "nervous system"]
        }
        
        for therapeutic_class in therapeutic_classes:
            class_lower = therapeutic_class.lower()
            if class_lower in therapeutic_keywords:
                keywords = therapeutic_keywords[class_lower]
                for keyword in keywords:
                    for func_desc in protein_functions:
                        if keyword in func_desc:
                            confidence += 0.3
        
        return min(confidence, 1.0)
    
    async def _infer_disease_associations(self, db: AsyncSession) -> List[Dict]:
        """Infer disease-protein associations"""
        associations = []
        
        # Get disease and protein nodes
        disease_nodes = {
            node_id: attrs for node_id, attrs in self.node_cache.items()
            if attrs["node_type"] == "disease"
        }
        protein_nodes = {
            node_id: attrs for node_id, attrs in self.node_cache.items()
            if attrs["node_type"] == "protein"
        }
        
        # Infer based on disease pathways and protein functions
        for disease_id, disease_attrs in disease_nodes.items():
            disease_props = disease_attrs["properties"]
            
            for protein_id, protein_attrs in protein_nodes.items():
                protein_props = protein_attrs["properties"]
                
                confidence = self._calculate_disease_protein_confidence(
                    disease_props, protein_props
                )
                
                if confidence > 0.5:
                    associations.append({
                        "source_node_id": disease_id,
                        "target_node_id": protein_id,
                        "relationship_type": "associated_with",
                        "confidence_score": confidence,
                        "evidence_type": "computational_inference",
                        "source": "neurograph_inference",
                        "properties": {
                            "confidence": confidence,
                            "inference_method": "pathway_overlap"
                        }
                    })
        
        logger.info(f"Inferred {len(associations)} disease-protein associations")
        return associations
    
    def _calculate_disease_protein_confidence(self, disease_props: Dict, protein_props: Dict) -> float:
        """Calculate confidence for disease-protein association"""
        confidence = 0.0
        
        # Extract disease pathways
        disease_pathways = set()
        if "pathophysiology" in disease_props:
            pathways = disease_props["pathophysiology"].get("pathways", [])
            disease_pathways.update([p.lower() for p in pathways])
        
        # Extract protein pathways from GO biological processes
        protein_pathways = set()
        if "biologicalProcess" in protein_props:
            for bp in protein_props["biologicalProcess"]:
                desc = bp.get("description", "").lower()
                protein_pathways.add(desc)
        
        # Calculate pathway overlap
        if disease_pathways and protein_pathways:
            # Simple keyword matching
            overlap_count = 0
            for disease_pathway in disease_pathways:
                for protein_pathway in protein_pathways:
                    # Check for keyword overlap
                    disease_words = set(disease_pathway.split())
                    protein_words = set(protein_pathway.split())
                    if disease_words.intersection(protein_words):
                        overlap_count += 1
            
            if overlap_count > 0:
                confidence = min(overlap_count * 0.2, 1.0)
        
        return confidence
    
    async def _infer_pathway_relationships(self, db: AsyncSession) -> List[Dict]:
        """Infer pathway relationships"""
        pathway_edges = []
        
        # This would implement pathway hierarchy inference
        # For now, return empty list
        logger.info("Pathway relationship inference not implemented yet")
        
        return pathway_edges
    
    async def _store_inferred_edges(self, inferred_edges: List[Dict], db: AsyncSession):
        """Store inferred edges in database"""
        for edge_data in inferred_edges:
            try:
                # Check if edge already exists
                existing = await db.execute(
                    select(KnowledgeEdge).where(
                        and_(
                            KnowledgeEdge.source_node_id == edge_data["source_node_id"],
                            KnowledgeEdge.target_node_id == edge_data["target_node_id"],
                            KnowledgeEdge.relationship_type == edge_data["relationship_type"]
                        )
                    )
                )
                
                if not existing.scalar_one_or_none():
                    edge = KnowledgeEdge(
                        source_node_id=edge_data["source_node_id"],
                        target_node_id=edge_data["target_node_id"],
                        relationship_type=edge_data["relationship_type"],
                        confidence_score=edge_data["confidence_score"],
                        evidence_type=edge_data["evidence_type"],
                        source=edge_data["source"],
                        properties=edge_data.get("properties", {}),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(edge)
                    
            except Exception as e:
                logger.error(f"Error storing inferred edge: {e}")
        
        await db.commit()
    
    def _calculate_graph_metrics(self) -> GraphMetrics:
        """Calculate comprehensive graph metrics"""
        try:
            # Basic metrics
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            density = nx.density(self.graph)
            
            # Clustering
            avg_clustering = nx.average_clustering(self.graph) if num_nodes > 0 else 0.0
            
            # Connectivity
            num_components = nx.number_connected_components(self.graph)
            
            # For diameter and average path length, use largest component
            if num_components > 0:
                largest_component = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_component)
                
                if subgraph.number_of_nodes() > 1:
                    diameter = nx.diameter(subgraph)
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    diameter = 0
                    avg_path_length = 0.0
            else:
                diameter = 0
                avg_path_length = 0.0
            
            # Centrality measures
            centrality_stats = {}
            if num_nodes > 0:
                degree_centrality = nx.degree_centrality(self.graph)
                centrality_stats["max_degree_centrality"] = max(degree_centrality.values())
                centrality_stats["avg_degree_centrality"] = np.mean(list(degree_centrality.values()))
                
                if num_nodes > 2:
                    betweenness_centrality = nx.betweenness_centrality(self.graph)
                    centrality_stats["max_betweenness_centrality"] = max(betweenness_centrality.values())
                    centrality_stats["avg_betweenness_centrality"] = np.mean(list(betweenness_centrality.values()))
            
            return GraphMetrics(
                num_nodes=num_nodes,
                num_edges=num_edges,
                density=density,
                avg_clustering=avg_clustering,
                num_components=num_components,
                diameter=diameter,
                avg_path_length=avg_path_length,
                centrality_stats=centrality_stats
            )
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return GraphMetrics(0, 0, 0.0, 0.0, 0, 0, 0.0, {})
    
    async def find_shortest_paths(self, source_id: str, target_id: str, k: int = 5) -> List[List[str]]:
        """Find k shortest paths between two nodes"""
        try:
            paths = list(nx.shortest_simple_paths(self.graph, source_id, target_id))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Error finding shortest paths: {e}")
            return []
    
    async def detect_communities(self, algorithm: str = "louvain") -> Dict[str, List[str]]:
        """Detect communities in the graph"""
        try:
            if algorithm == "louvain":
                # Use networkx community detection
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(self.graph)
            else:
                # Default to greedy modularity
                communities = nx.algorithms.community.greedy_modularity_communities(self.graph)
            
            # Convert to dictionary format
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[f"community_{i}"] = list(community)
            
            return community_dict
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {}
    
    async def get_node_neighbors(self, node_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """Get neighbors of a node up to specified depth"""
        try:
            neighbors = {}
            
            for depth in range(1, max_depth + 1):
                if depth == 1:
                    neighbors[f"depth_{depth}"] = list(self.graph.neighbors(node_id))
                else:
                    # Get neighbors at this depth
                    depth_neighbors = set()
                    for prev_neighbor in neighbors[f"depth_{depth-1}"]:
                        depth_neighbors.update(self.graph.neighbors(prev_neighbor))
                    
                    # Remove nodes from previous depths
                    for prev_depth in range(1, depth):
                        depth_neighbors -= set(neighbors[f"depth_{prev_depth}"])
                    
                    # Remove the original node
                    depth_neighbors.discard(node_id)
                    
                    neighbors[f"depth_{depth}"] = list(depth_neighbors)
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting node neighbors: {e}")
            return {}
    
    def export_graph(self, format: str = "gexf", filename: str = "knowledge_graph") -> str:
        """Export graph to various formats"""
        try:
            if format == "gexf":
                output_file = f"{filename}.gexf"
                nx.write_gexf(self.graph, output_file)
            elif format == "graphml":
                output_file = f"{filename}.graphml"
                nx.write_graphml(self.graph, output_file)
            elif format == "json":
                output_file = f"{filename}.json"
                graph_data = nx.node_link_data(self.graph)
                with open(output_file, 'w') as f:
                    json.dump(graph_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Graph exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            raise


async def main():
    """Main function for graph building"""
    try:
        # Initialize graph builder
        builder = KnowledgeGraphBuilder()
        
        # Get database session
        async for db in get_db():
            # Build graph from database
            metrics = await builder.build_graph_from_database(db)
            
            logger.info(f"Knowledge graph built successfully: {metrics}")
            
            # Export graph
            builder.export_graph("json", "neurograph_knowledge_graph")
            
            # Detect communities
            communities = await builder.detect_communities()
            logger.info(f"Detected {len(communities)} communities")
            
    except Exception as e:
        logger.error(f"Graph building failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
