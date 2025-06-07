'use client';

import React from 'react';
import { BoxIcon } from 'lucide-react';

interface ProteinViewer3DProps {
  protein?: any;
  pdbData?: string;
}

export function ProteinViewer3D({ protein, pdbData }: ProteinViewer3DProps) {
  return (
    <div className="protein-viewer">
      <div className="viewer-placeholder">
        <BoxIcon size={48} />
        <p>3D structure of {protein?.name || 'protein'}</p>
        <p className="text-secondary">Interactive visualization would be displayed here</p>
      </div>
    </div>
  );
}
