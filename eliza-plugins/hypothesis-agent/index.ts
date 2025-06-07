/**
 * Hypothesis Agent Plugin for Eliza Framework
 * Production-grade scientific hypothesis generation and validation agent
 */

import { Plugin } from "@ai16z/eliza";
import { hypothesisActions } from "./hypothesis-generator.ts";
import { validationActions } from "./validation-engine.ts";

// Combine all hypothesis-related actions
const allHypothesisActions = [...hypothesisActions, ...validationActions];

export const hypothesisAgentPlugin: Plugin = {
    name: "hypothesis-agent",
    description: "AI agent for scientific hypothesis generation, validation, and experimental design",
    actions: allHypothesisActions,
    evaluators: [], // Hypothesis-specific evaluators would go here
    providers: [], // Hypothesis data providers would go here
};

export default hypothesisAgentPlugin;
