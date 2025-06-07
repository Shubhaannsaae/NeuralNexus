'use client';

import { useState, useCallback } from 'react';

export function useKnowledgeGraph() {
  const [data, setData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const searchNodes = useCallback(async (query: string, filters: any) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1200));
      
      const mockData = {
        nodes: [
          { id: 'protein1', name: 'p53', type: 'protein' },
          { id: 'drug1', name: 'Aspirin', type: 'drug' },
          { id: 'disease1', name: 'Cancer', type: 'disease' }
        ],
        edges: [
          { source: 'protein1', target: 'disease1', type: 'associated_with' },
          { source: 'drug1', target: 'protein1', type: 'targets' }
        ]
      };
      
      setData(mockData);
    } catch (err: any) {
      setError(new Error(err.message));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateGraph = useCallback(async () => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      console.log('Graph updated');
    } catch (err: any) {
      setError(new Error(err.message));
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    data,
    isLoading,
    error,
    searchNodes,
    updateGraph
  };
}
