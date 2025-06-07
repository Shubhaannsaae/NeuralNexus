/**
 * Protein Agent Plugin for Eliza Framework
 * Production-grade protein structure analysis and prediction agent
 */

import { Plugin } from "@ai16z/eliza";
import { proteinActions } from "./structure-analyzer.ts";
import { bindingActions } from "./binding-predictor.ts";

// Combine all protein-related actions
const allProteinActions = [...proteinActions, ...bindingActions];

export const proteinAgentPlugin: Plugin = {
    name: "protein-agent",
    description: "AI agent for protein structure analysis, folding prediction, and binding site identification",
    actions: allProteinActions,
    evaluators: [], // Protein-specific evaluators would go here
    providers: [], // Protein data providers would go here
};

export default proteinAgentPlugin;
