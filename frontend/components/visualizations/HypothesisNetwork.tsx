'use client';

import React from 'react';

interface HypothesisNetworkProps {
  hypotheses: any[];
  selectedHypothesis: any;
  onHypothesisSelect: (hypothesis: any) => void;
}

export function HypothesisNetwork({ 
  hypotheses, 
  selectedHypothesis, 
  onHypothesisSelect 
}: HypothesisNetworkProps) {
  return (
    <div style={{ 
      height: '300px', 
      background: 'linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)', 
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#64748b'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '48px', marginBottom: '1rem' }}>🕸️</div>
        <p style={{ fontSize: '1.125rem', fontWeight: '600' }}>Hypothesis Network</p>
        <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
          {hypotheses.length > 0 
            ? `Visualizing ${hypotheses.length} hypotheses and their relationships`
            : 'Generate hypotheses to see network visualization'
          }
        </p>
      </div>
    </div>
  );
}
