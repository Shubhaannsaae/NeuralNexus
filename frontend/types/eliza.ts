export interface ElizaAgent {
  id: string;
  name: string;
  type: 'bio-agent' | 'protein-agent' | 'literature-agent' | 'hypothesis-agent';
  status: 'active' | 'inactive';
}

export interface AgentResponse {
  success: boolean;
  data?: any;
  error?: string;
}
