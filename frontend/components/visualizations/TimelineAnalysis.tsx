'use client';

import React from 'react';

export function TimelineAnalysis() {
  return (
    <div style={{ 
      height: '250px', 
      background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)', 
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#64748b'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '48px', marginBottom: '1rem' }}>📊</div>
        <p style={{ fontSize: '1.125rem', fontWeight: '600' }}>Timeline Analysis</p>
        <p style={{ fontSize: '0.875rem' }}>Interactive timeline visualization</p>
      </div>
    </div>
  );
}
