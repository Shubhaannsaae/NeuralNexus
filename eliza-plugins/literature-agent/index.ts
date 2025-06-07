/**
 * Literature Agent Plugin for Eliza Framework
 * Production-grade scientific literature mining and knowledge extraction agent
 */

import { Plugin } from "@ai16z/eliza";
import { literatureActions } from "./paper-processor.ts";
import { knowledgeActions } from "./knowledge-extractor.ts";

// Combine all literature-related actions
const allLiteratureActions = [...literatureActions, ...knowledgeActions];

export const literatureAgentPlugin: Plugin = {
    name: "literature-agent",
    description: "AI agent for scientific literature mining, paper analysis, and knowledge extraction from research publications",
    actions: allLiteratureActions,
    evaluators: [], // Literature-specific evaluators would go here
    providers: [], // Literature data providers would go here
};

export default literatureAgentPlugin;
