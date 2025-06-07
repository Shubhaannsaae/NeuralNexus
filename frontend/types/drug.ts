export interface Drug {
  id: string;
  name: string;
  smiles: string;
  molecularWeight: number;
  admetScore: number;
  indication?: string;
  mechanism?: string;
}

export interface DrugSearchFilters {
  name?: string;
  smiles?: string;
  minWeight?: number;
  maxWeight?: number;
  targetClass?: string;
}
