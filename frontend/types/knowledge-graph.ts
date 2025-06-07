export interface GraphNode {
  id: string;
  name: string;
  type: 'protein' | 'drug' | 'disease' | 'pathway';
  properties?: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  confidence?: number;
}

export interface KnowledgeGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}
