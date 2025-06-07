export interface Hypothesis {
  id: number;
  title: string;
  description: string;
  confidence: number;
  relevance: number;
  domain: string;
}

export interface HypothesisGenerationParams {
  domain: string;
  proteins: string;
  question: string;
}
