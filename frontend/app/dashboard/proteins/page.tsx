'use client';

import React, { useState } from 'react';
import { ProteinViewer3D } from '@/components/visualizations/ProteinViewer3D';

export default function ProteinsPage() {
  const [selectedProtein, setSelectedProtein] = useState({
    id: "P04637",
    name: "Tumor protein p53",
    organism: "Homo sapiens", 
    length: 393,
    function: "Tumor suppressor protein"
  });

  const handleProteinSearch = () => {
    // Simulate protein search
    console.log('Searching for protein...');
  };

  return (
    <div className="dashboard-page active">
      <div className="page-header">
        <h1>Protein Analysis</h1>
        <p>Explore protein structures and analyze binding sites for drug targets</p>
      </div>
      
      <div className="protein-search">
        <div className="card">
          <div className="card__body">
            <h3>Protein Search</h3>
            <div className="search-controls" style={{ display: 'flex', gap: '1rem' }}>
              <input 
                type="text" 
                className="form-control" 
                placeholder="Enter protein ID or name (e.g., P04637)"
                style={{ flex: 1 }}
              />
              <button className="btn btn--primary" onClick={handleProteinSearch}>
                Search
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="protein-results">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="card">
            <div className="card__body">
              <h3>3D Structure Viewer</h3>
              <ProteinViewer3D protein={selectedProtein} />
            </div>
          </div>
          <div className="card">
            <div className="card__body">
              <h3>Protein Information</h3>
              <div className="protein-info">
                <div className="info-item">
                  <span className="label">ID:</span>
                  <span className="value">{selectedProtein.id}</span>
                </div>
                <div className="info-item">
                  <span className="label">Name:</span>
                  <span className="value">{selectedProtein.name}</span>
                </div>
                <div className="info-item">
                  <span className="label">Organism:</span>
                  <span className="value">{selectedProtein.organism}</span>
                </div>
                <div className="info-item">
                  <span className="label">Length:</span>
                  <span className="value">{selectedProtein.length} amino acids</span>
                </div>
                <div className="info-item">
                  <span className="label">Function:</span>
                  <span className="value">{selectedProtein.function}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
