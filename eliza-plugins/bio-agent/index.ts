/**
 * Bio Agent Plugin for Eliza Framework
 * Production-grade biological system analysis and drug discovery agent
 */

import { Plugin } from "@ai16z/eliza";
import { bioActions } from "./actions.ts";
import { bioEvaluators } from "./evaluators.ts";
import { bioProviders } from "./providers.ts";

export const bioAgentPlugin: Plugin = {
    name: "bio-agent",
    description: "AI agent for biological research and drug discovery",
    actions: bioActions,
    evaluators: bioEvaluators,
    providers: bioProviders,
};

export default bioAgentPlugin;
