'use client';

import React from 'react';
import { KnowledgeGraphViz } from '@/components/visualizations/KnowledgeGraphViz';

export default function KnowledgeGraphPage() {
  const updateKnowledgeGraph = () => {
    console.log('Updating knowledge graph...');
  };

  return (
    <div className="dashboard-page active">
      <div className="page-header">
        <h1>Knowledge Graph</h1>
        <p>Interactive exploration of biomedical relationships and pathway connections</p>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.5rem' }}>
        <div className="card">
          <div className="card__body">
            <h3>Graph Controls</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <input 
                type="text" 
                className="form-control" 
                placeholder="Search nodes..."
              />
              <div>
                <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Node Types</h4>
                {['Protein', 'Drug', 'Disease', 'Pathway'].map((type) => (
                  <label key={type} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <input type="checkbox" defaultChecked />
                    <span style={{ fontSize: '0.875rem' }}>{type}</span>
                  </label>
                ))}
              </div>
              <button className="btn btn--primary" onClick={updateKnowledgeGraph}>
                Update Graph
              </button>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="card__body">
            <h3>Graph Visualization</h3>
            <KnowledgeGraphViz />
          </div>
        </div>
      </div>
    </div>
  );
}
