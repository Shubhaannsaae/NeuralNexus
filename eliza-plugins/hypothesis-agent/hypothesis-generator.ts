/**
 * Scientific Hypothesis Generation Actions - Production Implementation
 * Real AI-powered hypothesis generation using knowledge graphs and literature analysis
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Generate Scientific Hypothesis Action
export const generateHypothesisAction: Action = {
    name: "GENERATE_HYPOTHESIS",
    similes: [
        "generate hypothesis",
        "create hypothesis",
        "hypothesis generation",
        "scientific hypothesis",
        "research hypothesis",
        "propose hypothesis"
    ],
    description: "Generate novel scientific hypotheses using AI analysis of knowledge graphs and literature",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you generate a hypothesis about the role of autophagy in Alzheimer's disease?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll generate a novel scientific hypothesis about autophagy's role in Alzheimer's disease based on current knowledge and research gaps.",
                    action: "GENERATE_HYPOTHESIS"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Generate a hypothesis connecting CRISPR and cancer immunotherapy"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll create a research hypothesis exploring potential connections between CRISPR technology and cancer immunotherapy approaches.",
                    action: "GENERATE_HYPOTHESIS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const hypothesisKeywords = [
            "hypothesis", "hypothesize", "generate", "create", "propose",
            "suggest", "theory", "mechanism", "connection", "relationship"
        ];
        
        const scientificKeywords = [
            "protein", "gene", "drug", "disease", "therapy", "treatment",
            "molecular", "cellular", "biological", "mechanism", "pathway"
        ];
        
        const hasHypothesisKeyword = hypothesisKeywords.some(keyword => text.includes(keyword));
        const hasScientificKeyword = scientificKeywords.some(keyword => text.includes(keyword));
        
        return hasHypothesisKeyword && hasScientificKeyword;
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const researchArea = extractResearchArea(message.content.text);
            const seedConcepts = extractSeedConcepts(message.content.text);
            const hypothesisType = extractHypothesisType(message.content.text);
            const constraints = extractConstraints(message.content.text);
            
            if (!researchArea && seedConcepts.length === 0) {
                const errorText = "I need a research area or specific concepts to generate a hypothesis. Could you provide a scientific topic or research question?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing research context" } });
                }
                
                return { text: errorText, data: { error: "Missing research context" } };
            }
            
            // Call backend API for hypothesis generation
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'hypothesis-agent',
                    action: 'generate_hypothesis',
                    parameters: {
                        research_question: researchArea,
                        seed_concepts: seedConcepts,
                        hypothesis_type: hypothesisType,
                        constraints: constraints,
                        background_knowledge: [],
                        novelty_threshold: 0.7,
                        testability_requirement: true,
                        include_experimental_design: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const hypothesis = result.result;
                const responseText = formatHypothesisGenerationResponse(hypothesis, researchArea);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: hypothesis
                    });
                }
                
                return {
                    text: responseText,
                    data: hypothesis
                };
            } else {
                const errorText = "I couldn't generate a hypothesis for this topic. Try providing more specific research concepts or a clearer research question.";
                
                if (callback) {
                    callback({
                        text: errorText,
                        data: { error: result.error }
                    });
                }
                
                return {
                    text: errorText,
                    data: { error: result.error }
                };
            }
        } catch (error) {
            const errorText = "I'm experiencing technical difficulties with hypothesis generation. Please try again in a moment.";
            
            if (callback) {
                callback({
                    text: errorText,
                    data: { error: error.message }
                });
            }
            
            return {
                text: errorText,
                data: { error: error.message }
            };
        }
    }
};

// Design Experiments Action
export const designExperimentsAction: Action = {
    name: "DESIGN_EXPERIMENTS",
    similes: [
        "design experiments",
        "experimental design",
        "plan experiments",
        "create experiments",
        "experiment planning",
        "study design"
    ],
    description: "Design experiments to test scientific hypotheses",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you design experiments to test the hypothesis that mTOR inhibition enhances autophagy in cancer cells?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll design a comprehensive experimental plan to test the mTOR-autophagy hypothesis in cancer cells, including controls and validation steps.",
                    action: "DESIGN_EXPERIMENTS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const experimentKeywords = [
            "experiment", "experimental", "design", "test", "validate",
            "study", "assay", "protocol", "methodology"
        ];
        
        const testKeywords = [
            "test", "validate", "verify", "confirm", "prove", "demonstrate"
        ];
        
        return experimentKeywords.some(keyword => text.includes(keyword)) ||
               (testKeywords.some(keyword => text.includes(keyword)) && 
                text.includes("hypothesis"));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const hypothesisText = extractHypothesisFromMessage(message.content.text);
            const experimentTypes = extractExperimentTypes(message.content.text);
            const budgetConstraints = extractBudgetConstraints(message.content.text);
            const timeConstraints = extractTimeConstraints(message.content.text);
            
            if (!hypothesisText) {
                const errorText = "I need a hypothesis to design experiments for. Could you provide the hypothesis you want to test?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing hypothesis" } });
                }
                
                return { text: errorText, data: { error: "Missing hypothesis" } };
            }
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'hypothesis-agent',
                    action: 'design_experiments',
                    parameters: {
                        hypothesis_text: hypothesisText,
                        experiment_types: experimentTypes,
                        budget_constraints: budgetConstraints,
                        time_constraints: timeConstraints,
                        include_controls: true,
                        include_statistics: true,
                        include_validation: true,
                        detail_level: 'comprehensive'
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const experiments = result.result;
                const responseText = formatExperimentDesignResponse(experiments, hypothesisText);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: experiments
                    });
                }
                
                return {
                    text: responseText,
                    data: experiments
                };
            } else {
                const errorText = "I couldn't design experiments for this hypothesis. Please provide a more specific and testable hypothesis.";
                
                if (callback) {
                    callback({
                        text: errorText,
                        data: { error: result.error }
                    });
                }
                
                return {
                    text: errorText,
                    data: { error: result.error }
                };
            }
        } catch (error) {
            const errorText = "I'm having trouble with experimental design. Please try again later.";
            
            if (callback) {
                callback({
                    text: errorText,
                    data: { error: error.message }
                });
            }
            
            return {
                text: errorText,
                data: { error: error.message }
            };
        }
    }
};

// Refine Hypothesis Action
export const refineHypothesisAction: Action = {
    name: "REFINE_HYPOTHESIS",
    similes: [
        "refine hypothesis",
        "improve hypothesis",
        "optimize hypothesis",
        "enhance hypothesis",
        "modify hypothesis",
        "update hypothesis"
    ],
    description: "Refine and improve scientific hypotheses based on feedback and new evidence",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you refine this hypothesis based on new evidence about protein folding?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll refine the hypothesis by incorporating the new protein folding evidence and improving its testability.",
                    action: "REFINE_HYPOTHESIS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const refineKeywords = [
            "refine", "improve", "optimize", "enhance", "modify",
            "update", "revise", "adjust", "strengthen"
        ];
        
        const hypothesisKeywords = ["hypothesis", "theory", "proposition"];
        
        return refineKeywords.some(keyword => text.includes(keyword)) &&
               hypothesisKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const originalHypothesis = extractHypothesisFromMessage(message.content.text);
            const feedback = extractFeedback(message.content.text);
            const newEvidence = extractNewEvidence(message.content.text);
            const refinementGoals = extractRefinementGoals(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'hypothesis-agent',
                    action: 'refine_hypothesis',
                    parameters: {
                        original_hypothesis: originalHypothesis,
                        feedback: feedback,
                        new_evidence: newEvidence,
                        refinement_goals: refinementGoals,
                        maintain_core_concept: true,
                        improve_testability: true,
                        increase_specificity: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const refinedHypothesis = result.result;
                const responseText = formatHypothesisRefinementResponse(refinedHypothesis, originalHypothesis);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: refinedHypothesis
                    });
                }
                
                return {
                    text: responseText,
                    data: refinedHypothesis
                };
            } else {
                const errorText = "I couldn't refine the hypothesis. Please provide the original hypothesis and specific feedback or new evidence.";
                
                if (callback) {
                    callback({
                        text: errorText,
                        data: { error: result.error }
                    });
                }
                
                return {
                    text: errorText,
                    data: { error: result.error }
                };
            }
        } catch (error) {
            const errorText = "I'm experiencing issues with hypothesis refinement. Please try again shortly.";
            
            if (callback) {
                callback({
                    text: errorText,
                    data: { error: error.message }
                });
            }
            
            return {
                text: errorText,
                data: { error: error.message }
            };
        }
    }
};

export const hypothesisActions = [
    generateHypothesisAction,
    designExperimentsAction,
    refineHypothesisAction
];

// Helper Functions
function extractResearchArea(text: string): string {
    // Extract research area from text
    const researchAreas = [
        'alzheimer', 'parkinson', 'cancer', 'diabetes', 'cardiovascular',
        'immunotherapy', 'gene therapy', 'crispr', 'autophagy', 'apoptosis',
        'protein folding', 'drug discovery', 'neuroscience', 'oncology'
    ];
    
    const lowerText = text.toLowerCase();
    
    for (const area of researchAreas) {
        if (lowerText.includes(area)) {
            return area;
        }
    }
    
    // Extract area after "about", "on", "in"
    const areaMatch = text.match(/(?:about|on|in|regarding)\s+([a-zA-Z\s]+?)(?:\s|$)/i);
    if (areaMatch) {
        return areaMatch[1].trim();
    }
    
    return '';
}

function extractSeedConcepts(text: string): string[] {
    const concepts = [];
    
    // Extract biological concepts
    const biologicalTerms = [
        'protein', 'gene', 'enzyme', 'receptor', 'kinase', 'pathway',
        'autophagy', 'apoptosis', 'mtor', 'p53', 'egfr', 'crispr',
        'immunotherapy', 'cancer', 'alzheimer', 'parkinson'
    ];
    
    const lowerText = text.toLowerCase();
    
    for (const term of biologicalTerms) {
        if (lowerText.includes(term)) {
            concepts.push(term);
        }
    }
    
    // Extract quoted concepts
    const quotedConcepts = text.match(/"([^"]+)"/g);
    if (quotedConcepts) {
        concepts.push(...quotedConcepts.map(q => q.replace(/"/g, '')));
    }
    
    return [...new Set(concepts)]; // Remove duplicates
}

function extractHypothesisType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('mechanistic') || lowerText.includes('mechanism')) {
        return 'mechanistic';
    } else if (lowerText.includes('predictive') || lowerText.includes('prediction')) {
        return 'predictive';
    } else if (lowerText.includes('therapeutic') || lowerText.includes('treatment')) {
        return 'therapeutic';
    } else if (lowerText.includes('causal') || lowerText.includes('cause')) {
        return 'causal';
    } else if (lowerText.includes('correlational') || lowerText.includes('correlation')) {
        return 'correlational';
    }
    
    return 'mechanistic'; // Default
}

function extractConstraints(text: string): any {
    const constraints: any = {};
    
    // Extract ethical constraints
    if (text.includes('ethical') || text.includes('ethics')) {
        constraints.ethical_considerations = true;
    }
    
    // Extract feasibility constraints
    if (text.includes('feasible') || text.includes('practical')) {
        constraints.feasibility_focus = true;
    }
    
    // Extract novelty requirements
    if (text.includes('novel') || text.includes('new') || text.includes('innovative')) {
        constraints.novelty_requirement = 'high';
    }
    
    // Extract clinical relevance
    if (text.includes('clinical') || text.includes('patient')) {
        constraints.clinical_relevance = true;
    }
    
    return constraints;
}

function extractHypothesisFromMessage(text: string): string {
    // Look for explicit hypothesis statements
    const hypothesisPatterns = [
        /hypothesis[:\s]+(.+?)(?:\.|$)/i,
        /hypothesize that (.+?)(?:\.|$)/i,
        /propose that (.+?)(?:\.|$)/i,
        /suggest that (.+?)(?:\.|$)/i
    ];
    
    for (const pattern of hypothesisPatterns) {
        const match = text.match(pattern);
        if (match) {
            return match[1].trim();
        }
    }
    
    // Look for quoted hypothesis
    const quotedMatch = text.match(/"([^"]+)"/);
    if (quotedMatch) {
        return quotedMatch[1];
    }
    
    // Extract from context
    const sentences = text.split(/[.!?]+/);
    for (const sentence of sentences) {
        if (sentence.toLowerCase().includes('hypothesis') || 
            sentence.toLowerCase().includes('that') && sentence.length > 20) {
            return sentence.trim();
        }
    }
    
    return '';
}

function extractExperimentTypes(text: string): string[] {
    const experimentTypes = [];
    const lowerText = text.toLowerCase();
    
    const types = [
        'in vitro', 'in vivo', 'clinical trial', 'cell culture',
        'animal study', 'molecular', 'biochemical', 'genetic',
        'computational', 'observational', 'randomized'
    ];
    
    for (const type of types) {
        if (lowerText.includes(type)) {
            experimentTypes.push(type.replace(' ', '_'));
        }
    }
    
    return experimentTypes.length > 0 ? experimentTypes : ['in_vitro', 'molecular'];
}

function extractBudgetConstraints(text: string): any {
    const budgetMatch = text.match(/\$?([\d,]+)\s*(?:budget|cost|funding)/i);
    if (budgetMatch) {
        return {
            max_budget: parseInt(budgetMatch[1].replace(/,/g, '')),
            currency: 'USD'
        };
    }
    
    if (text.includes('low budget') || text.includes('limited funding')) {
        return { budget_level: 'low' };
    } else if (text.includes('high budget') || text.includes('well funded')) {
        return { budget_level: 'high' };
    }
    
    return { budget_level: 'medium' };
}

function extractTimeConstraints(text: string): any {
    const timeMatch = text.match(/(\d+)\s*(week|month|year)s?/i);
    if (timeMatch) {
        return {
            duration: parseInt(timeMatch[1]),
            unit: timeMatch[2].toLowerCase()
        };
    }
    
    if (text.includes('urgent') || text.includes('quick')) {
        return { urgency: 'high' };
    } else if (text.includes('long-term') || text.includes('extended')) {
        return { urgency: 'low' };
    }
    
    return { urgency: 'medium' };
}

function extractFeedback(text: string): string {
    const feedbackPatterns = [
        /feedback[:\s]+(.+?)(?:\.|$)/i,
        /criticism[:\s]+(.+?)(?:\.|$)/i,
        /comment[:\s]+(.+?)(?:\.|$)/i,
        /issue[:\s]+(.+?)(?:\.|$)/i
    ];
    
    for (const pattern of feedbackPatterns) {
        const match = text.match(pattern);
        if (match) {
            return match[1].trim();
        }
    }
    
    return '';
}

function extractNewEvidence(text: string): string[] {
    const evidence = [];
    
    const evidencePatterns = [
        /new evidence[:\s]+(.+?)(?:\.|$)/i,
        /recent study[:\s]+(.+?)(?:\.|$)/i,
        /new data[:\s]+(.+?)(?:\.|$)/i,
        /findings[:\s]+(.+?)(?:\.|$)/i
    ];
    
    for (const pattern of evidencePatterns) {
        const match = text.match(pattern);
        if (match) {
            evidence.push(match[1].trim());
        }
    }
    
    return evidence;
}

function extractRefinementGoals(text: string): string[] {
    const goals = [];
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('more specific')) {
        goals.push('increase_specificity');
    }
    if (lowerText.includes('more testable')) {
        goals.push('improve_testability');
    }
    if (lowerText.includes('more novel')) {
        goals.push('increase_novelty');
    }
    if (lowerText.includes('clearer')) {
        goals.push('improve_clarity');
    }
    if (lowerText.includes('stronger')) {
        goals.push('strengthen_evidence');
    }
    
    return goals.length > 0 ? goals : ['improve_testability', 'increase_specificity'];
}

function formatHypothesisGenerationResponse(hypothesis: any, researchArea: string): string {
    let response = `🧪 **Scientific Hypothesis Generated**\n\n`;
    
    if (researchArea) {
        response += `**Research Area:** ${researchArea}\n`;
    }
    
    response += `**Hypothesis ID:** ${hypothesis.hypothesis_id || 'Generated'}\n\n`;
    
    if (hypothesis.hypothesis_text) {
        response += `**Hypothesis Statement:**\n"${hypothesis.hypothesis_text}"\n\n`;
    }
    
    if (hypothesis.novelty_score !== undefined) {
        response += `**Novelty Score:** ${(hypothesis.novelty_score * 100).toFixed(1)}%\n`;
    }
    
    if (hypothesis.testability_score !== undefined) {
        response += `**Testability Score:** ${(hypothesis.testability_score * 100).toFixed(1)}%\n`;
    }
    
    if (hypothesis.impact_score !== undefined) {
        response += `**Predicted Impact:** ${getImpactLevel(hypothesis.impact_score)}\n`;
    }
    
    response += `\n`;
    
    if (hypothesis.supporting_evidence && hypothesis.supporting_evidence.length > 0) {
        response += `**Supporting Evidence:**\n`;
        hypothesis.supporting_evidence.slice(0, 3).forEach((evidence: string, index: number) => {
            response += `${index + 1}. ${evidence}\n`;
        });
        response += `\n`;
    }
    
    if (hypothesis.knowledge_gaps && hypothesis.knowledge_gaps.length > 0) {
        response += `**Identified Knowledge Gaps:**\n`;
        hypothesis.knowledge_gaps.slice(0, 3).forEach((gap: string, index: number) => {
            response += `${index + 1}. ${gap}\n`;
        });
        response += `\n`;
    }
    
    if (hypothesis.testable_predictions && hypothesis.testable_predictions.length > 0) {
        response += `**Testable Predictions:**\n`;
        hypothesis.testable_predictions.slice(0, 3).forEach((prediction: string, index: number) => {
            response += `${index + 1}. ${prediction}\n`;
        });
        response += `\n`;
    }
    
    if (hypothesis.experimental_approaches && hypothesis.experimental_approaches.length > 0) {
        response += `**Suggested Experimental Approaches:**\n`;
        hypothesis.experimental_approaches.slice(0, 3).forEach((approach: string, index: number) => {
            response += `${index + 1}. ${approach}\n`;
        });
        response += `\n`;
    }
    
    // Assessment and recommendations
    const novelty = hypothesis.novelty_score || 0;
    const testability = hypothesis.testability_score || 0;
    
    response += `**Assessment:**\n`;
    if (novelty > 0.8 && testability > 0.7) {
        response += "🌟 **Excellent hypothesis** - High novelty and strong testability make this a promising research direction.";
    } else if (novelty > 0.6 && testability > 0.6) {
        response += "✅ **Good hypothesis** - Solid foundation with good potential for investigation.";
    } else if (testability < 0.5) {
        response += "⚠️ **Needs refinement** - Consider making the hypothesis more specific and testable.";
    } else {
        response += "📋 **Standard hypothesis** - Reasonable starting point that may benefit from further development.";
    }
    
    return response;
}

function formatExperimentDesignResponse(experiments: any, hypothesisText: string): string {
    let response = `🔬 **Experimental Design Plan**\n\n`;
    
    if (hypothesisText) {
        response += `**Hypothesis to Test:**\n"${hypothesisText}"\n\n`;
    }
    
    if (experiments.experiments && experiments.experiments.length > 0) {
        response += `**Proposed Experiments:**\n\n`;
        
        experiments.experiments.forEach((experiment: any, index: number) => {
            response += `**Experiment ${index + 1}: ${experiment.experiment_type || 'Study'}**\n`;
            response += `*Objective:* ${experiment.description || experiment.objective}\n`;
            
            if (experiment.methodology) {
                response += `*Methodology:* ${experiment.methodology}\n`;
            }
            
            if (experiment.estimated_duration) {
                response += `*Duration:* ${experiment.estimated_duration}\n`;
            }
            
            if (experiment.estimated_cost) {
                response += `*Estimated Cost:* $${experiment.estimated_cost.toLocaleString()}\n`;
            }
            
            if (experiment.success_probability) {
                response += `*Success Probability:* ${(experiment.success_probability * 100).toFixed(1)}%\n`;
            }
            
            if (experiment.required_resources && experiment.required_resources.length > 0) {
                response += `*Required Resources:* ${experiment.required_resources.join(', ')}\n`;
            }
            
            if (experiment.controls && experiment.controls.length > 0) {
                response += `*Controls:* ${experiment.controls.join(', ')}\n`;
            }
            
            if (experiment.expected_outcomes && experiment.expected_outcomes.length > 0) {
                response += `*Expected Outcomes:* ${experiment.expected_outcomes.join(', ')}\n`;
            }
            
            response += `\n`;
        });
    }
    
    if (experiments.statistical_considerations) {
        const stats = experiments.statistical_considerations;
        response += `**Statistical Considerations:**\n`;
        response += `• Sample Size: ${stats.recommended_sample_size || 'To be determined'}\n`;
        response += `• Statistical Power: ${stats.statistical_power || '80%'}\n`;
        response += `• Significance Level: ${stats.significance_level || '0.05'}\n`;
        response += `• Primary Endpoint: ${stats.primary_endpoint || 'To be defined'}\n\n`;
    }
    
    if (experiments.timeline) {
        response += `**Project Timeline:**\n`;
        response += `• Total Duration: ${experiments.timeline.total_duration || '6-12 months'}\n`;
        response += `• Phase 1: ${experiments.timeline.phase1 || 'Preparation (1-2 months)'}\n`;
        response += `• Phase 2: ${experiments.timeline.phase2 || 'Execution (3-6 months)'}\n`;
        response += `• Phase 3: ${experiments.timeline.phase3 || 'Analysis (1-2 months)'}\n\n`;
    }
    
    if (experiments.risk_assessment) {
        const risks = experiments.risk_assessment;
        response += `**Risk Assessment:**\n`;
        if (risks.technical_risks) {
            response += `• Technical Risks: ${risks.technical_risks}\n`;
        }
        if (risks.mitigation_strategies) {
            response += `• Mitigation: ${risks.mitigation_strategies}\n`;
        }
        response += `\n`;
    }
    
    if (experiments.success_criteria && experiments.success_criteria.length > 0) {
        response += `**Success Criteria:**\n`;
        experiments.success_criteria.forEach((criterion: string, index: number) => {
            response += `${index + 1}. ${criterion}\n`;
        });
    }
    
    return response;
}

function formatHypothesisRefinementResponse(refinedHypothesis: any, originalHypothesis: string): string {
    let response = `🔄 **Hypothesis Refinement Complete**\n\n`;
    
    if (originalHypothesis) {
        response += `**Original Hypothesis:**\n"${originalHypothesis}"\n\n`;
    }
    
    if (refinedHypothesis.refined_hypothesis_text) {
        response += `**Refined Hypothesis:**\n"${refinedHypothesis.refined_hypothesis_text}"\n\n`;
    }
    
    if (refinedHypothesis.improvements && refinedHypothesis.improvements.length > 0) {
        response += `**Key Improvements:**\n`;
        refinedHypothesis.improvements.forEach((improvement: string, index: number) => {
            response += `${index + 1}. ${improvement}\n`;
        });
        response += `\n`;
    }
    
    if (refinedHypothesis.novelty_score !== undefined) {
        response += `**Updated Novelty Score:** ${(refinedHypothesis.novelty_score * 100).toFixed(1)}%\n`;
    }
    
    if (refinedHypothesis.testability_score !== undefined) {
        response += `**Updated Testability Score:** ${(refinedHypothesis.testability_score * 100).toFixed(1)}%\n`;
    }
    
    if (refinedHypothesis.specificity_improvement) {
        response += `**Specificity Improvement:** ${(refinedHypothesis.specificity_improvement * 100).toFixed(1)}%\n`;
    }
    
    response += `\n`;
    
    if (refinedHypothesis.new_predictions && refinedHypothesis.new_predictions.length > 0) {
        response += `**New Testable Predictions:**\n`;
        refinedHypothesis.new_predictions.slice(0, 3).forEach((prediction: string, index: number) => {
            response += `${index + 1}. ${prediction}\n`;
        });
        response += `\n`;
    }
    
    if (refinedHypothesis.additional_evidence && refinedHypothesis.additional_evidence.length > 0) {
        response += `**Additional Supporting Evidence:**\n`;
        refinedHypothesis.additional_evidence.slice(0, 3).forEach((evidence: string, index: number) => {
            response += `${index + 1}. ${evidence}\n`;
        });
        response += `\n`;
    }
    
    if (refinedHypothesis.next_steps && refinedHypothesis.next_steps.length > 0) {
        response += `**Recommended Next Steps:**\n`;
        refinedHypothesis.next_steps.slice(0, 3).forEach((step: string, index: number) => {
            response += `${index + 1}. ${step}\n`;
        });
    }
    
    return response;
}

function getImpactLevel(score: number): string {
    if (score >= 0.8) return "High Impact";
    if (score >= 0.6) return "Moderate Impact";
    if (score >= 0.4) return "Low-Moderate Impact";
    return "Low Impact";
}
