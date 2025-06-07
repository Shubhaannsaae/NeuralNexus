'use client';

import React from 'react';

interface MolecularDockingProps {
  protein: any;
  bindingSites: any[];
}

export function MolecularDocking({ protein, bindingSites }: MolecularDockingProps) {
  return (
    <div style={{ 
      height: '400px', 
      background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', 
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '48px', marginBottom: '1rem' }}>🧬</div>
        <p style={{ fontSize: '1.25rem', fontWeight: '600' }}>Molecular Docking Simulation</p>
        <p style={{ fontSize: '0.875rem', opacity: 0.75, marginTop: '0.5rem' }}>
          Advanced docking interface coming soon
        </p>
      </div>
    </div>
  );
}
