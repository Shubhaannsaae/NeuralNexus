'use client';

import React, { useState } from 'react';

interface DrugSearchFormProps {
  onSearch: (filters: any) => void;
  isLoading?: boolean;
}

export function DrugSearchForm({ onSearch, isLoading }: DrugSearchFormProps) {
  const [formData, setFormData] = useState({
    name: '',
    smiles: '',
    minWeight: '',
    maxWeight: '',
    targetClass: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(formData);
  };

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Compound Name</label>
          <input 
            type="text" 
            className="form-control" 
            placeholder="Enter compound name"
            value={formData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            disabled={isLoading}
          />
        </div>
        <div className="form-group">
          <label className="form-label">SMILES Structure</label>
          <input 
            type="text" 
            className="form-control" 
            placeholder="Enter SMILES notation"
            value={formData.smiles}
            onChange={(e) => handleChange('smiles', e.target.value)}
            disabled={isLoading}
          />
        </div>
      </div>
      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Molecular Weight Range</label>
          <div className="range-inputs">
            <input 
              type="number" 
              className="form-control" 
              placeholder="Min" 
              min="0"
              value={formData.minWeight}
              onChange={(e) => handleChange('minWeight', e.target.value)}
              disabled={isLoading}
            />
            <span>to</span>
            <input 
              type="number" 
              className="form-control" 
              placeholder="Max" 
              min="0"
              value={formData.maxWeight}
              onChange={(e) => handleChange('maxWeight', e.target.value)}
              disabled={isLoading}
            />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Target Class</label>
          <select 
            className="form-control"
            value={formData.targetClass}
            onChange={(e) => handleChange('targetClass', e.target.value)}
            disabled={isLoading}
          >
            <option value="">Select target class</option>
            <option value="kinase">Kinase</option>
            <option value="gpcr">GPCR</option>
            <option value="ion-channel">Ion Channel</option>
            <option value="enzyme">Enzyme</option>
          </select>
        </div>
      </div>
      <button type="submit" className="btn btn--primary" disabled={isLoading}>
        {isLoading ? 'Searching...' : 'Search Compounds'}
      </button>
    </form>
  );
}
