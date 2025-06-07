'use client';

import { useState, useCallback } from 'react';

export function useProteinData() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const searchProteins = useCallback(async (query: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockData = [
        { id: "P04637", name: "Tumor protein p53", organism: "Homo sapiens", length: 393, function: "Tumor suppressor protein" },
        { id: "P38398", name: "BRCA1", organism: "Homo sapiens", length: 1863, function: "DNA repair protein" },
        { id: "P42574", name: "Caspase-3", organism: "Homo sapiens", length: 277, function: "Apoptosis regulator" }
      ];
      
      return mockData.filter(protein => 
        protein.name.toLowerCase().includes(query.toLowerCase()) ||
        protein.id.toLowerCase().includes(query.toLowerCase())
      );
    } catch (err: any) {
      setError(err.message);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getProteinStructure = useCallback(async (proteinId: string) => {
    try {
      await new Promise(resolve => setTimeout(resolve, 800));
      return {
        id: proteinId,
        structure: "3D structure data would be here",
        bindingSites: [
          { id: 1, name: "Site A", residues: "123-145", affinity: 8.5 },
          { id: 2, name: "Site B", residues: "200-220", affinity: 7.2 }
        ]
      };
    } catch (err: any) {
      throw new Error(err.message);
    }
  }, []);

  return {
    searchProteins,
    getProteinStructure,
    isLoading,
    error
  };
}
