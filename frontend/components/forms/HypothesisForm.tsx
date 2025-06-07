'use client';

import React, { useState } from 'react';

interface HypothesisFormProps {
  onGenerate: (params: any) => void;
  isLoading?: boolean;
}

export function HypothesisForm({ onGenerate, isLoading }: HypothesisFormProps) {
  const [formData, setFormData] = useState({
    domain: '',
    proteins: '',
    question: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate(formData);
  };

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label className="form-label">Research Domain</label>
        <select 
          className="form-control"
          value={formData.domain}
          onChange={(e) => handleChange('domain', e.target.value)}
          disabled={isLoading}
        >
          <option value="">Select domain</option>
          <option value="alzheimers">Alzheimer's Disease</option>
          <option value="parkinsons">Parkinson's Disease</option>
          <option value="huntingtons">Huntington's Disease</option>
          <option value="als">ALS</option>
          <option value="multiple-sclerosis">Multiple Sclerosis</option>
        </select>
      </div>
      <div className="form-group">
        <label className="form-label">Protein IDs</label>
        <input 
          type="text" 
          className="form-control" 
          placeholder="Enter protein IDs (comma-separated)"
          value={formData.proteins}
          onChange={(e) => handleChange('proteins', e.target.value)}
          disabled={isLoading}
        />
      </div>
      <div className="form-group">
        <label className="form-label">Research Question</label>
        <textarea 
          className="form-control" 
          rows={4}
          placeholder="Describe your research question or area of interest..."
          value={formData.question}
          onChange={(e) => handleChange('question', e.target.value)}
          disabled={isLoading}
          style={{ resize: 'vertical', minHeight: '100px' }}
        />
      </div>
      <button type="submit" className="btn btn--primary" disabled={isLoading}>
        {isLoading ? 'Generating...' : 'Generate Hypothesis'}
      </button>
    </form>
  );
}
