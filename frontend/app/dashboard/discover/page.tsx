'use client';

import React, { useState } from 'react';
import { DrugSearchForm } from '@/components/forms/DrugSearchForm';

export default function DrugDiscoveryPage() {
  const [searchResults, setSearchResults] = useState([
    { id: "1", name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", molecularWeight: 180.16, admetScore: 0.85 },
    { id: "2", name: "Ibuprofen", smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", molecularWeight: 206.28, admetScore: 0.72 },
    { id: "3", name: "Donepezil", smiles: "COc1cc2c(cc1OC)C(=O)C(CC2)N3CCN(CC3)Cc4ccccc4", molecularWeight: 379.49, admetScore: 0.91 }
  ]);

  const handleSearch = async (filters: any) => {
    // Simulate API call
    console.log('Searching with filters:', filters);
  };

  const analyzeCompound = (compoundId: string) => {
    const compound = searchResults.find(d => d.id === compoundId);
    if (compound) {
      alert(`Analyzing ${compound.name}...\n\nMolecular Weight: ${compound.molecularWeight} g/mol\nADMET Score: ${compound.admetScore}\n\nDetailed analysis would be displayed in a modal or new page.`);
    }
  };

  return (
    <div className="dashboard-page active">
      <div className="page-header">
        <h1>Drug Discovery</h1>
        <p>Search and analyze molecular compounds for neurological drug development</p>
      </div>
      
      <div className="search-section">
        <div className="card">
          <div className="card__body">
            <h3>Compound Search</h3>
            <DrugSearchForm onSearch={handleSearch} />
          </div>
        </div>
      </div>

      <div className="results-section">
        <div className="card">
          <div className="card__body">
            <h3>Search Results</h3>
            <div className="compounds-table">
              <div className="table-header">
                <div>Name</div>
                <div>SMILES</div>
                <div>MW (g/mol)</div>
                <div>ADMET Score</div>
                <div>Actions</div>
              </div>
              {searchResults.map(compound => {
                const statusClass = compound.admetScore >= 0.8 ? 'success' : 
                                   compound.admetScore >= 0.7 ? 'warning' : 'error';
                
                return (
                  <div key={compound.id} className="table-row">
                    <div>{compound.name}</div>
                    <div style={{ fontFamily: 'monospace', fontSize: '12px' }}>{compound.smiles}</div>
                    <div>{compound.molecularWeight}</div>
                    <div><span className={`status status--${statusClass}`}>{compound.admetScore}</span></div>
                    <div>
                      <button 
                        className="btn btn--sm btn--outline"
                        onClick={() => analyzeCompound(compound.id)}
                      >
                        Analyze
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
