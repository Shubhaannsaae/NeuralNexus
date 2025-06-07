export interface Protein {
    id: string;
    name: string;
    organism: string;
    length: number;
    function: string;
    sequence?: string;
    structure?: ProteinStructure;
  }
  
  export interface ProteinStructure {
    id: string;
    proteinId: string;
    pdbContent: string;
    confidence: number;
    bindingSites: BindingSite[];
  }
  
  export interface BindingSite {
    id: number;
    name: string;
    residues: string;
    affinity: number;
  }
  