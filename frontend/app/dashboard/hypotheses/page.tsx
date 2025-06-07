'use client';

import React, { useState } from 'react';
import { HypothesisForm } from '@/components/forms/HypothesisForm';

export default function HypothesesPage() {
  const [hypotheses, setHypotheses] = useState([
    {
      id: 1,
      title: "Tau-Microtubule Interaction Inhibition",
      description: "Targeting the interaction between tau protein and microtubules through small molecule inhibitors may prevent neurodegeneration in Alzheimer's disease by stabilizing axonal transport.",
      confidence: 0.87,
      relevance: 0.92,
      domain: "Alzheimer's Disease"
    },
    {
      id: 2,
      title: "NMDA Receptor Allosteric Modulation", 
      description: "Modulation of NMDA receptor activity through allosteric mechanisms could provide neuroprotection without the side effects of direct antagonism.",
      confidence: 0.74,
      relevance: 0.81,
      domain: "Neuroprotection"
    }
  ]);

  const handleHypothesisGeneration = (params: any) => {
    console.log('Generating hypothesis with params:', params);
  };

  const exploreHypothesis = (hypothesisId: number) => {
    const hypothesis = hypotheses.find(h => h.id === hypothesisId);
    if (hypothesis) {
      alert(`Exploring hypothesis: ${hypothesis.title}\n\nThis would open a detailed analysis view with supporting evidence, related papers, and experimental suggestions.`);
    }
  };

  return (
    <div className="dashboard-page active">
      <div className="page-header">
        <h1>AI Hypotheses</h1>
        <p>Generate and validate scientific hypotheses using advanced machine learning</p>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div className="card">
          <div className="card__body">
            <h3>Generate Hypothesis</h3>
            <HypothesisForm onGenerate={handleHypothesisGeneration} />
          </div>
        </div>
        
        <div className="card">
          <div className="card__body">
            <h3>Generated Hypotheses</h3>
            <div className="hypothesis-list">
              {hypotheses.map((hypothesis, index) => {
                const confidenceStatus = hypothesis.confidence >= 0.8 ? 'success' : 
                                        hypothesis.confidence >= 0.7 ? 'warning' : 'error';
                
                return (
                  <div key={hypothesis.id} className="hypothesis-item">
                    <div className="hypothesis-header">
                      <h4>Hypothesis #{index + 1}</h4>
                      <span className={`status status--${confidenceStatus}`}>
                        {hypothesis.confidence >= 0.8 ? 'High' : hypothesis.confidence >= 0.7 ? 'Medium' : 'Low'} Confidence
                      </span>
                    </div>
                    <p>{hypothesis.description}</p>
                    <div className="hypothesis-meta">
                      <span>Confidence: {hypothesis.confidence}</span>
                      <span>Relevance Score: {hypothesis.relevance}</span>
                      <button 
                        className="btn btn--sm btn--outline"
                        onClick={() => exploreHypothesis(hypothesis.id)}
                      >
                        Explore
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
