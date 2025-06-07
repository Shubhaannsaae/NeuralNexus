'use client';

import React from 'react';
import { NetworkIcon } from 'lucide-react';

interface KnowledgeGraphVizProps {
  onNodeSelect?: (node: any) => void;
}

export function KnowledgeGraphViz({ onNodeSelect }: KnowledgeGraphVizProps) {
  return (
    <div className="graph-viewer">
      <div className="graph-placeholder">
        <NetworkIcon size={48} />
        <p>Knowledge Graph Visualization</p>
        <p className="text-secondary">Showing connections between proteins, drugs, and pathways</p>
        <p className="text-secondary">Interactive graph visualization would be displayed here</p>
      </div>
    </div>
  );
}
