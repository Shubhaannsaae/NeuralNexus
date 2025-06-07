/**
 * Hypothesis Validation Engine - Production Implementation
 * Real validation and testing of scientific hypotheses
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Validate Hypothesis Action
export const validateHypothesisAction: Action = {
    name: "VALIDATE_HYPOTHESIS",
    similes: [
        "validate hypothesis",
        "test hypothesis",
        "hypothesis validation",
        "check hypothesis",
        "verify hypothesis",
        "hypothesis testing"
    ],
    description: "Validate scientific hypotheses against existing literature and experimental evidence",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you validate the hypothesis that autophagy dysfunction contributes to Alzheimer's pathology?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll validate this hypothesis by analyzing existing literature, experimental evidence, and assessing its consistency with current knowledge.",
                    action: "VALIDATE_HYPOTHESIS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const validationKeywords = [
            "validate", "validation", "test", "verify", "check",
            "assess", "evaluate", "examine", "analyze"
        ];
        
        const hypothesisKeywords = ["hypothesis", "theory", "proposition", "claim"];
        
        return validationKeywords.some(keyword => text.includes(keyword)) &&
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
            const hypothesisText = extractHypothesisForValidation(message.content.text);
            const validationCriteria = extractValidationCriteria(message.content.text);
            const evidenceTypes = extractEvidenceTypes(message.content.text);
            
            if (!hypothesisText) {
                const errorText = "I need a specific hypothesis to validate. Could you provide the hypothesis statement you want me to test?";
                
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
                    action: 'validate_hypothesis',
                    parameters: {
                        hypothesis_text: hypothesisText,
                        validation_criteria: validationCriteria,
                        evidence_types: evidenceTypes,
                        literature_search_depth: 'comprehensive',
                        include_contradictory_evidence: true,
                        confidence_threshold: 0.7,
                        experimental_data: {}
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const validation = result.result;
                const responseText = formatHypothesisValidationResponse(validation, hypothesisText);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: validation
                    });
                }
                
                return {
                    text: responseText,
                    data: validation
                };
            } else {
                const errorText = "I couldn't validate this hypothesis. Please ensure the hypothesis is clearly stated and scientifically testable.";
                
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
            const errorText = "I'm experiencing technical difficulties with hypothesis validation. Please try again later.";
            
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

// Assess Hypothesis Feasibility Action
export const assessFeasibilityAction: Action = {
    name: "ASSESS_FEASIBILITY",
    similes: [
        "assess feasibility",
        "feasibility assessment",
        "check feasibility",
        "evaluate feasibility",
        "feasibility analysis",
        "practical assessment"
    ],
    description: "Assess the practical feasibility of testing scientific hypotheses",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you assess the feasibility of testing this gene therapy hypothesis in clinical trials?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll assess the practical feasibility including regulatory requirements, technical challenges, timeline, and resource needs for testing this gene therapy hypothesis.",
                    action: "ASSESS_FEASIBILITY"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const feasibilityKeywords = [
            "feasibility", "feasible", "practical", "realistic",
            "doable", "achievable", "viable", "possible"
        ];
        
        return feasibilityKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const hypothesisText = extractHypothesisForValidation(message.content.text);
            const assessmentScope = extractAssessmentScope(message.content.text);
            const constraints = extractFeasibilityConstraints(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'hypothesis-agent',
                    action: 'assess_feasibility',
                    parameters: {
                        hypothesis_text: hypothesisText,
                        assessment_scope: assessmentScope,
                        constraints: constraints,
                        include_regulatory: true,
                        include_technical: true,
                        include_financial: true,
                        include_timeline: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const feasibility = result.result;
                const responseText = formatFeasibilityAssessmentResponse(feasibility, hypothesisText);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: feasibility
                    });
                }
                
                return {
                    text: responseText,
                    data: feasibility
                };
            } else {
                const errorText = "I couldn't assess the feasibility of this hypothesis. Please provide more details about the proposed research approach.";
                
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
            const errorText = "I'm having trouble with feasibility assessment. Please try again shortly.";
            
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

// Prioritize Research Hypotheses Action
export const prioritizeHypothesesAction: Action = {
    name: "PRIORITIZE_HYPOTHESES",
    similes: [
        "prioritize hypotheses",
        "rank hypotheses",
        "hypothesis prioritization",
        "research prioritization",
        "hypothesis ranking",
        "priority assessment"
    ],
    description: "Prioritize multiple research hypotheses based on impact, feasibility, and novelty",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you prioritize these three cancer research hypotheses based on their potential impact?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze and prioritize the cancer research hypotheses based on potential impact, feasibility, novelty, and resource requirements.",
                    action: "PRIORITIZE_HYPOTHESES"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const priorityKeywords = [
            "prioritize", "priority", "rank", "ranking", "order",
            "compare", "comparison", "best", "most important"
        ];
        
        const hypothesisKeywords = ["hypotheses", "hypothesis", "research", "studies"];
        
        return priorityKeywords.some(keyword => text.includes(keyword)) &&
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
            const hypothesesList = extractMultipleHypotheses(message.content.text);
            const prioritizationCriteria = extractPrioritizationCriteria(message.content.text);
            const weightings = extractCriteriaWeightings(message.content.text);
            
            if (hypothesesList.length < 2) {
                const errorText = "I need at least two hypotheses to prioritize. Could you provide multiple research hypotheses for comparison?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Insufficient hypotheses" } });
                }
                
                return { text: errorText, data: { error: "Insufficient hypotheses" } };
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
                    action: 'prioritize_hypotheses',
                    parameters: {
                        hypotheses: hypothesesList,
                        prioritization_criteria: prioritizationCriteria,
                        criteria_weightings: weightings,
                        include_risk_assessment: true,
                        include_resource_analysis: true,
                        include_timeline_analysis: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const prioritization = result.result;
                const responseText = formatHypothesisPrioritizationResponse(prioritization, hypothesesList);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: prioritization
                    });
                }
                
                return {
                    text: responseText,
                    data: prioritization
                };
            } else {
                const errorText = "I couldn't prioritize the hypotheses. Please ensure all hypotheses are clearly stated and comparable.";
                
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
            const errorText = "I'm experiencing issues with hypothesis prioritization. Please try again later.";
            
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

export const validationActions = [
    validateHypothesisAction,
    assessFeasibilityAction,
    prioritizeHypothesesAction
];

// Helper Functions
function extractHypothesisForValidation(text: string): string {
    // Look for hypothesis statements
    const hypothesisPatterns = [
        /hypothesis[:\s]+"([^"]+)"/i,
        /hypothesis[:\s]+(.+?)(?:\.|that|$)/i,
        /validate[:\s]+"([^"]+)"/i,
        /test[:\s]+"([^"]+)"/i
    ];
    
    for (const pattern of hypothesisPatterns) {
        const match = text.match(pattern);
        if (match) {
            return match[1].trim();
        }
    }
    
    // Look for "that" clauses
    const thatMatch = text.match(/that (.+?)(?:\.|$)/i);
    if (thatMatch && thatMatch[1].length > 20) {
        return thatMatch[1].trim();
    }
    
    return '';
}

function extractValidationCriteria(text: string): string[] {
    const criteria = [];
    const lowerText = text.toLowerCase();
    
    const possibleCriteria = [
        'literature_support', 'experimental_evidence', 'logical_consistency',
        'testability', 'specificity', 'novelty', 'biological_plausibility'
    ];
    
    for (const criterion of possibleCriteria) {
        const keyword = criterion.replace('_', ' ');
        if (lowerText.includes(keyword)) {
            criteria.push(criterion);
        }
    }
    
    // Default criteria if none specified
    return criteria.length > 0 ? criteria : [
        'literature_support', 'experimental_evidence', 'logical_consistency', 'testability'
    ];
}

function extractEvidenceTypes(text: string): string[] {
    const evidenceTypes = [];
    const lowerText = text.toLowerCase();
    
    const types = [
        'experimental', 'observational', 'clinical', 'computational',
        'in_vitro', 'in_vivo', 'epidemiological', 'meta_analysis'
    ];
    
    for (const type of types) {
        if (lowerText.includes(type.replace('_', ' '))) {
            evidenceTypes.push(type);
        }
    }
    
    return evidenceTypes.length > 0 ? evidenceTypes : ['experimental', 'observational'];
}

function extractAssessmentScope(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('comprehensive') || lowerText.includes('detailed')) {
        return 'comprehensive';
    } else if (lowerText.includes('quick') || lowerText.includes('brief')) {
        return 'brief';
    } else if (lowerText.includes('technical')) {
        return 'technical';
    } else if (lowerText.includes('regulatory')) {
        return 'regulatory';
    }
    
    return 'standard';
}

function extractFeasibilityConstraints(text: string): any {
    const constraints: any = {};
    
    // Budget constraints
    const budgetMatch = text.match(/budget[:\s]*\$?([\d,]+)/i);
    if (budgetMatch) {
        constraints.max_budget = parseInt(budgetMatch[1].replace(/,/g, ''));
    }
    
    // Time constraints
    const timeMatch = text.match(/(\d+)\s*(month|year)s?/i);
    if (timeMatch) {
        constraints.max_duration = {
            value: parseInt(timeMatch[1]),
            unit: timeMatch[2]
        };
    }
    
    // Regulatory constraints
    if (text.includes('fda') || text.includes('regulatory')) {
        constraints.regulatory_approval_required = true;
    }
    
    // Ethical constraints
    if (text.includes('ethical') || text.includes('human subjects')) {
        constraints.ethical_approval_required = true;
    }
    
    return constraints;
}

function extractMultipleHypotheses(text: string): string[] {
    const hypotheses = [];
    
    // Look for numbered hypotheses
    const numberedMatches = text.match(/\d+[.)]\s*([^.]+)/g);
    if (numberedMatches) {
        hypotheses.push(...numberedMatches.map(match => 
            match.replace(/^\d+[.)]\s*/, '').trim()
        ));
    }
    
    // Look for quoted hypotheses
    const quotedMatches = text.match(/"([^"]+)"/g);
    if (quotedMatches) {
        hypotheses.push(...quotedMatches.map(match => 
            match.replace(/"/g, '').trim()
        ));
    }
    
    // Look for bullet points
    const bulletMatches = text.match(/[•\-*]\s*([^•\-*\n]+)/g);
    if (bulletMatches) {
        hypotheses.push(...bulletMatches.map(match => 
            match.replace(/^[•\-*]\s*/, '').trim()
        ));
    }
    
    return hypotheses.filter(h => h.length > 10); // Filter out short matches
}

function extractPrioritizationCriteria(text: string): string[] {
    const criteria = [];
    const lowerText = text.toLowerCase();
    
    const possibleCriteria = [
        'impact', 'feasibility', 'novelty', 'cost', 'timeline',
        'risk', 'resources', 'expertise', 'technology'
    ];
    
    for (const criterion of possibleCriteria) {
        if (lowerText.includes(criterion)) {
            criteria.push(criterion);
        }
    }
    
    return criteria.length > 0 ? criteria : ['impact', 'feasibility', 'novelty'];
}

function extractCriteriaWeightings(text: string): any {
    const weightings: any = {};
    
    // Look for explicit weightings
    const weightMatches = text.match(/(\w+)[:\s]*(\d+)%/g);
    if (weightMatches) {
        for (const match of weightMatches) {
            const [criterion, weight] = match.split(/[:\s]/);
            weightings[criterion.toLowerCase()] = parseInt(weight) / 100;
        }
    }
    
    // Default weightings
    return Object.keys(weightings).length > 0 ? weightings : {
        impact: 0.4,
        feasibility: 0.3,
        novelty: 0.3
    };
}

function formatHypothesisValidationResponse(validation: any, hypothesisText: string): string {
    let response = `✅ **Hypothesis Validation Results**\n\n`;
    
    response += `**Hypothesis:** "${hypothesisText}"\n\n`;
    
    if (validation.validation_score !== undefined) {
        response += `**Overall Validation Score:** ${(validation.validation_score * 100).toFixed(1)}%\n`;
        response += `**Validation Status:** ${getValidationStatus(validation.validation_score)}\n\n`;
    }
    
    if (validation.evidence_analysis) {
        const evidence = validation.evidence_analysis;
        response += `**Evidence Analysis:**\n`;
        response += `• Supporting Evidence: ${evidence.supporting_count || 0} studies\n`;
        response += `• Contradictory Evidence: ${evidence.contradictory_count || 0} studies\n`;
        response += `• Evidence Quality: ${evidence.quality_score ? (evidence.quality_score * 100).toFixed(1) + '%' : 'Mixed'}\n`;
        response += `• Literature Coverage: ${evidence.coverage_score ? (evidence.coverage_score * 100).toFixed(1) + '%' : 'Moderate'}\n\n`;
    }
    
    if (validation.supporting_studies && validation.supporting_studies.length > 0) {
        response += `**Key Supporting Studies:**\n`;
        validation.supporting_studies.slice(0, 3).forEach((study: any, index: number) => {
            response += `${index + 1}. ${study.title || study.description}\n`;
            if (study.journal) {
                response += `   *${study.journal}* (${study.year || 'Recent'})\n`;
            }
            if (study.evidence_strength) {
                response += `   Evidence Strength: ${(study.evidence_strength * 100).toFixed(1)}%\n`;
            }
            response += `\n`;
        });
    }
    
    if (validation.contradictory_evidence && validation.contradictory_evidence.length > 0) {
        response += `**Contradictory Evidence:**\n`;
        validation.contradictory_evidence.slice(0, 2).forEach((evidence: any, index: number) => {
            response += `${index + 1}. ${evidence.description}\n`;
            if (evidence.source) {
                response += `   Source: ${evidence.source}\n`;
            }
            response += `\n`;
        });
    }
    
    if (validation.logical_consistency) {
        const logic = validation.logical_consistency;
        response += `**Logical Consistency Analysis:**\n`;
        response += `• Internal Consistency: ${logic.internal_consistency ? '✅ Consistent' : '⚠️ Issues found'}\n`;
        response += `• Causal Logic: ${logic.causal_logic_score ? (logic.causal_logic_score * 100).toFixed(1) + '%' : 'Moderate'}\n`;
        response += `• Biological Plausibility: ${logic.biological_plausibility ? '✅ Plausible' : '⚠️ Questionable'}\n\n`;
    }
    
    if (validation.testability_assessment) {
        const testability = validation.testability_assessment;
        response += `**Testability Assessment:**\n`;
        response += `• Testability Score: ${(testability.score * 100).toFixed(1)}%\n`;
        response += `• Measurable Variables: ${testability.measurable_variables || 'Yes'}\n`;
        response += `• Experimental Feasibility: ${testability.experimental_feasibility || 'Moderate'}\n`;
        response += `• Statistical Power: ${testability.statistical_power || 'Adequate'}\n\n`;
    }
    
    if (validation.recommendations && validation.recommendations.length > 0) {
        response += `**Recommendations:**\n`;
        validation.recommendations.slice(0, 3).forEach((rec: string, index: number) => {
            response += `${index + 1}. ${rec}\n`;
        });
        response += `\n`;
    }
    
    // Overall assessment
    const score = validation.validation_score || 0;
    response += `**Assessment Summary:**\n`;
    if (score > 0.8) {
        response += "🌟 **Strong hypothesis** - Well-supported by evidence and highly testable.";
    } else if (score > 0.6) {
        response += "✅ **Solid hypothesis** - Good evidence support with minor areas for improvement.";
    } else if (score > 0.4) {
        response += "⚠️ **Moderate hypothesis** - Some evidence support but requires strengthening.";
    } else {
        response += "❌ **Weak hypothesis** - Limited evidence support, significant revision needed.";
    }
    
    return response;
}

function formatFeasibilityAssessmentResponse(feasibility: any, hypothesisText: string): string {
    let response = `🔍 **Feasibility Assessment**\n\n`;
    
    if (hypothesisText) {
        response += `**Hypothesis:** "${hypothesisText}"\n\n`;
    }
    
    if (feasibility.overall_feasibility_score !== undefined) {
        response += `**Overall Feasibility Score:** ${(feasibility.overall_feasibility_score * 100).toFixed(1)}%\n`;
        response += `**Feasibility Level:** ${getFeasibilityLevel(feasibility.overall_feasibility_score)}\n\n`;
    }
    
    if (feasibility.technical_assessment) {
        const technical = feasibility.technical_assessment;
        response += `**Technical Feasibility:**\n`;
        response += `• Technical Complexity: ${technical.complexity_level || 'Moderate'}\n`;
        response += `• Required Expertise: ${technical.required_expertise || 'Specialized'}\n`;
        response += `• Technology Availability: ${technical.technology_availability || 'Available'}\n`;
        response += `• Technical Risk: ${technical.risk_level || 'Moderate'}\n\n`;
    }
    
    if (feasibility.resource_requirements) {
        const resources = feasibility.resource_requirements;
        response += `**Resource Requirements:**\n`;
        response += `• Estimated Budget: $${resources.estimated_budget?.toLocaleString() || 'TBD'}\n`;
        response += `• Personnel Needed: ${resources.personnel_count || 'Multiple'} researchers\n`;
        response += `• Equipment: ${resources.equipment_requirements?.join(', ') || 'Standard lab equipment'}\n`;
        response += `• Facilities: ${resources.facility_requirements || 'Standard research facility'}\n\n`;
    }
    
    if (feasibility.timeline_analysis) {
        const timeline = feasibility.timeline_analysis;
        response += `**Timeline Analysis:**\n`;
        response += `• Estimated Duration: ${timeline.total_duration || '12-18 months'}\n`;
        response += `• Preparation Phase: ${timeline.preparation_phase || '2-3 months'}\n`;
        response += `• Execution Phase: ${timeline.execution_phase || '6-12 months'}\n`;
        response += `• Analysis Phase: ${timeline.analysis_phase || '2-3 months'}\n\n`;
    }
    
    if (feasibility.regulatory_considerations) {
        const regulatory = feasibility.regulatory_considerations;
        response += `**Regulatory Considerations:**\n`;
        response += `• Approval Required: ${regulatory.approval_required ? 'Yes' : 'No'}\n`;
        response += `• Approval Timeline: ${regulatory.approval_timeline || 'N/A'}\n`;
        response += `• Compliance Requirements: ${regulatory.compliance_requirements || 'Standard'}\n`;
        response += `• Ethical Review: ${regulatory.ethical_review_required ? 'Required' : 'Not required'}\n\n`;
    }
    
    if (feasibility.risk_assessment) {
        const risks = feasibility.risk_assessment;
        response += `**Risk Assessment:**\n`;
        response += `• Technical Risks: ${risks.technical_risks || 'Moderate'}\n`;
        response += `• Financial Risks: ${risks.financial_risks || 'Low'}\n`;
        response += `• Timeline Risks: ${risks.timeline_risks || 'Moderate'}\n`;
        response += `• Regulatory Risks: ${risks.regulatory_risks || 'Low'}\n\n`;
    }
    
    if (feasibility.success_probability) {
        response += `**Success Probability:** ${(feasibility.success_probability * 100).toFixed(1)}%\n\n`;
    }
    
    if (feasibility.recommendations && feasibility.recommendations.length > 0) {
        response += `**Recommendations:**\n`;
        feasibility.recommendations.slice(0, 3).forEach((rec: string, index: number) => {
            response += `${index + 1}. ${rec}\n`;
        });
        response += `\n`;
    }
    
    if (feasibility.alternative_approaches && feasibility.alternative_approaches.length > 0) {
        response += `**Alternative Approaches:**\n`;
        feasibility.alternative_approaches.slice(0, 2).forEach((approach: string, index: number) => {
            response += `${index + 1}. ${approach}\n`;
        });
    }
    
    return response;
}

function formatHypothesisPrioritizationResponse(prioritization: any, hypothesesList: string[]): string {
    let response = `🏆 **Hypothesis Prioritization Results**\n\n`;
    
    response += `**Total Hypotheses Analyzed:** ${hypothesesList.length}\n`;
    response += `**Prioritization Criteria:** ${prioritization.criteria_used?.join(', ') || 'Impact, Feasibility, Novelty'}\n\n`;
    
    if (prioritization.ranked_hypotheses && prioritization.ranked_hypotheses.length > 0) {
        response += `**Priority Ranking:**\n\n`;
        
        prioritization.ranked_hypotheses.forEach((hypothesis: any, index: number) => {
            response += `**${index + 1}. Priority ${getRankLabel(index + 1)}**\n`;
            response += `*Hypothesis:* "${hypothesis.hypothesis_text || hypothesesList[hypothesis.original_index] || 'Hypothesis ' + (index + 1)}"\n`;
            
            if (hypothesis.overall_score !== undefined) {
                response += `*Overall Score:* ${(hypothesis.overall_score * 100).toFixed(1)}%\n`;
            }
            
            if (hypothesis.scores) {
                const scores = hypothesis.scores;
                response += `*Detailed Scores:*\n`;
                if (scores.impact !== undefined) {
                    response += `  • Impact: ${(scores.impact * 100).toFixed(1)}%\n`;
                }
                if (scores.feasibility !== undefined) {
                    response += `  • Feasibility: ${(scores.feasibility * 100).toFixed(1)}%\n`;
                }
                if (scores.novelty !== undefined) {
                    response += `  • Novelty: ${(scores.novelty * 100).toFixed(1)}%\n`;
                }
                if (scores.cost_effectiveness !== undefined) {
                    response += `  • Cost-Effectiveness: ${(scores.cost_effectiveness * 100).toFixed(1)}%\n`;
                }
            }
            
            if (hypothesis.strengths && hypothesis.strengths.length > 0) {
                response += `*Key Strengths:* ${hypothesis.strengths.slice(0, 2).join(', ')}\n`;
            }
            
            if (hypothesis.weaknesses && hypothesis.weaknesses.length > 0) {
                response += `*Areas for Improvement:* ${hypothesis.weaknesses.slice(0, 2).join(', ')}\n`;
            }
            
            if (hypothesis.estimated_timeline) {
                response += `*Estimated Timeline:* ${hypothesis.estimated_timeline}\n`;
            }
            
            if (hypothesis.estimated_cost) {
                response += `*Estimated Cost:* $${hypothesis.estimated_cost.toLocaleString()}\n`;
            }
            
            response += `\n`;
        });
    }
    
    if (prioritization.comparative_analysis) {
        const analysis = prioritization.comparative_analysis;
        response += `**Comparative Analysis:**\n`;
        
        if (analysis.highest_impact) {
            response += `• Highest Impact: Hypothesis ${analysis.highest_impact.rank}\n`;
        }
        if (analysis.most_feasible) {
            response += `• Most Feasible: Hypothesis ${analysis.most_feasible.rank}\n`;
        }
        if (analysis.most_novel) {
            response += `• Most Novel: Hypothesis ${analysis.most_novel.rank}\n`;
        }
        if (analysis.best_cost_benefit) {
            response += `• Best Cost-Benefit: Hypothesis ${analysis.best_cost_benefit.rank}\n`;
        }
        response += `\n`;
    }
    
    if (prioritization.resource_allocation && prioritization.resource_allocation.length > 0) {
        response += `**Recommended Resource Allocation:**\n`;
        prioritization.resource_allocation.slice(0, 3).forEach((allocation: any, index: number) => {
            response += `${index + 1}. ${allocation.hypothesis_rank}: ${allocation.percentage}% of resources\n`;
            if (allocation.rationale) {
                response += `   *Rationale:* ${allocation.rationale}\n`;
            }
        });
        response += `\n`;
    }
    
    if (prioritization.strategic_recommendations && prioritization.strategic_recommendations.length > 0) {
        response += `**Strategic Recommendations:**\n`;
        prioritization.strategic_recommendations.slice(0, 3).forEach((rec: string, index: number) => {
            response += `${index + 1}. ${rec}\n`;
        });
        response += `\n`;
    }
    
    if (prioritization.portfolio_balance) {
        const balance = prioritization.portfolio_balance;
        response += `**Portfolio Balance:**\n`;
        response += `• High-Risk/High-Reward: ${balance.high_risk_percentage || 20}%\n`;
        response += `• Moderate Risk: ${balance.moderate_risk_percentage || 60}%\n`;
        response += `• Low-Risk/Incremental: ${balance.low_risk_percentage || 20}%\n\n`;
    }
    
    response += `**Next Steps:**\n`;
    response += `1. Begin with the highest-priority hypothesis\n`;
    response += `2. Develop detailed experimental protocols\n`;
    response += `3. Secure necessary resources and approvals\n`;
    response += `4. Consider parallel development of top 2-3 hypotheses if resources allow`;
    
    return response;
}

function getValidationStatus(score: number): string {
    if (score >= 0.8) return "Strongly Validated";
    if (score >= 0.6) return "Well Validated";
    if (score >= 0.4) return "Moderately Validated";
    if (score >= 0.2) return "Weakly Validated";
    return "Poorly Validated";
}

function getFeasibilityLevel(score: number): string {
    if (score >= 0.8) return "Highly Feasible";
    if (score >= 0.6) return "Feasible";
    if (score >= 0.4) return "Moderately Feasible";
    if (score >= 0.2) return "Challenging";
    return "Not Feasible";
}

function getRankLabel(rank: number): string {
    switch (rank) {
        case 1: return "High";
        case 2: return "High";
        case 3: return "Medium";
        default: return "Low";
    }
}
