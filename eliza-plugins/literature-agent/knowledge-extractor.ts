/**
 * Knowledge Extraction Actions - Production Implementation
 * Real knowledge extraction and synthesis from scientific literature
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Extract Knowledge from Papers Action
export const extractKnowledgeAction: Action = {
    name: "EXTRACT_KNOWLEDGE",
    similes: [
        "extract knowledge",
        "knowledge extraction",
        "extract information",
        "mine knowledge",
        "extract facts",
        "knowledge mining"
    ],
    description: "Extract structured knowledge and facts from scientific literature",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you extract key knowledge from recent papers on mRNA vaccines?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll extract and structure key knowledge from recent mRNA vaccine research papers, including mechanisms, efficacy data, and safety profiles.",
                    action: "EXTRACT_KNOWLEDGE"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Extract protein-drug interactions from cancer research papers"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll extract structured information about protein-drug interactions from cancer research literature.",
                    action: "EXTRACT_KNOWLEDGE"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const extractKeywords = [
            "extract", "extraction", "mine", "mining", "knowledge",
            "information", "facts", "data", "findings"
        ];
        
        const sourceKeywords = [
            "papers", "literature", "research", "studies", "publications"
        ];
        
        return extractKeywords.some(keyword => text.includes(keyword)) &&
               sourceKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const extractionTarget = extractExtractionTarget(message.content.text);
            const knowledgeType = extractKnowledgeType(message.content.text);
            const sourceConstraints = extractSourceConstraints(message.content.text);
            
            if (!extractionTarget) {
                const errorText = "I need to know what specific knowledge you want me to extract. Could you specify the research topic or domain?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing extraction target" } });
                }
                
                return { text: errorText, data: { error: "Missing extraction target" } };
            }
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'literature-agent',
                    action: 'extract_knowledge',
                    parameters: {
                        extraction_target: extractionTarget,
                        knowledge_type: knowledgeType,
                        source_constraints: sourceConstraints,
                        max_papers: 50,
                        confidence_threshold: 0.7,
                        include_citations: true,
                        include_evidence: true,
                        structured_output: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const extractedKnowledge = result.result;
                const responseText = formatKnowledgeExtractionResponse(extractedKnowledge, extractionTarget);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: extractedKnowledge
                    });
                }
                
                return {
                    text: responseText,
                    data: extractedKnowledge
                };
            } else {
                const errorText = "I couldn't extract knowledge for this topic. Try being more specific about what information you're looking for.";
                
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
            const errorText = "I'm experiencing technical difficulties with knowledge extraction. Please try again later.";
            
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

// Synthesize Research Findings Action
export const synthesizeResearchAction: Action = {
    name: "SYNTHESIZE_RESEARCH",
    similes: [
        "synthesize research",
        "research synthesis",
        "combine findings",
        "integrate research",
        "meta-analysis",
        "systematic review"
    ],
    description: "Synthesize and integrate findings from multiple research papers",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you synthesize the current research on COVID-19 treatments?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll synthesize findings from multiple research papers on COVID-19 treatments to provide an integrated overview of current evidence.",
                    action: "SYNTHESIZE_RESEARCH"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const synthesisKeywords = [
            "synthesize", "synthesis", "combine", "integrate", "meta-analysis",
            "systematic review", "overview", "consolidate", "merge"
        ];
        
        const researchKeywords = [
            "research", "findings", "studies", "papers", "evidence", "literature"
        ];
        
        return synthesisKeywords.some(keyword => text.includes(keyword)) &&
               researchKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const researchTopic = extractResearchTopic(message.content.text);
            const synthesisType = extractSynthesisType(message.content.text);
            const inclusionCriteria = extractInclusionCriteria(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'literature-agent',
                    action: 'synthesize_research',
                    parameters: {
                        research_topic: researchTopic,
                        synthesis_type: synthesisType,
                        inclusion_criteria: inclusionCriteria,
                        max_papers: 100,
                        quality_threshold: 0.8,
                        include_meta_analysis: true,
                        include_contradictions: true,
                        evidence_grading: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const synthesis = result.result;
                const responseText = formatResearchSynthesisResponse(synthesis, researchTopic);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: synthesis
                    });
                }
                
                return {
                    text: responseText,
                    data: synthesis
                };
            } else {
                const errorText = "I couldn't synthesize research for this topic. Please provide a more specific research question.";
                
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
            const errorText = "I'm having trouble with research synthesis. Please try again shortly.";
            
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

// Generate Literature Review Action
export const generateLiteratureReviewAction: Action = {
    name: "GENERATE_LITERATURE_REVIEW",
    similes: [
        "literature review",
        "review literature",
        "comprehensive review",
        "survey literature",
        "review papers",
        "academic review"
    ],
    description: "Generate comprehensive literature reviews on scientific topics",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you generate a literature review on machine learning in drug discovery?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll generate a comprehensive literature review covering recent advances in machine learning applications for drug discovery.",
                    action: "GENERATE_LITERATURE_REVIEW"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const reviewKeywords = [
            "literature review", "review", "comprehensive review",
            "survey", "overview", "state of the art"
        ];
        
        return reviewKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const reviewTopic = extractReviewTopic(message.content.text);
            const reviewScope = extractReviewScope(message.content.text);
            const timeFrame = extractTimeFrame(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'literature-agent',
                    action: 'generate_literature_review',
                    parameters: {
                        review_topic: reviewTopic,
                        review_scope: reviewScope,
                        time_frame: timeFrame,
                        max_papers: 200,
                        include_methodology: true,
                        include_gaps: true,
                        include_future_directions: true,
                        citation_style: 'apa'
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const review = result.result;
                const responseText = formatLiteratureReviewResponse(review, reviewTopic);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: review
                    });
                }
                
                return {
                    text: responseText,
                    data: review
                };
            } else {
                const errorText = "I couldn't generate a literature review for this topic. Please provide a more focused research area.";
                
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
            const errorText = "I'm experiencing issues generating the literature review. Please try again later.";
            
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

export const knowledgeActions = [
    extractKnowledgeAction,
    synthesizeResearchAction,
    generateLiteratureReviewAction
];

// Helper Functions
function extractExtractionTarget(text: string): string {
    // Remove extraction keywords to get the target
    let target = text.toLowerCase();
    
    const extractionPrefixes = [
        'extract knowledge from', 'extract from', 'mine knowledge from',
        'extract information from', 'get knowledge from'
    ];
    
    for (const prefix of extractionPrefixes) {
        if (target.includes(prefix)) {
            target = target.split(prefix)[1]?.trim() || target;
            break;
        }
    }
    
    // Clean and extract meaningful terms
    target = target.replace(/[^\w\s-]/g, ' ').trim();
    const words = target.split(/\s+/).filter(word => word.length > 2);
    
    return words.slice(0, 5).join(' ');
}

function extractKnowledgeType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('protein') || lowerText.includes('molecular')) {
        return 'molecular_interactions';
    } else if (lowerText.includes('drug') || lowerText.includes('compound')) {
        return 'drug_information';
    } else if (lowerText.includes('disease') || lowerText.includes('clinical')) {
        return 'clinical_knowledge';
    } else if (lowerText.includes('pathway') || lowerText.includes('mechanism')) {
        return 'biological_pathways';
    } else if (lowerText.includes('gene') || lowerText.includes('genetic')) {
        return 'genetic_information';
    }
    
    return 'general_biomedical';
}

function extractSourceConstraints(text: string): any {
    const constraints: any = {};
    
    // Extract date constraints
    if (text.includes('recent')) {
        constraints.date_range = 'last_2_years';
    } else if (text.includes('last 5 years')) {
        constraints.date_range = 'last_5_years';
    }
    
    // Extract journal constraints
    const highImpactJournals = ['nature', 'science', 'cell', 'nejm', 'lancet'];
    for (const journal of highImpactJournals) {
        if (text.toLowerCase().includes(journal)) {
            constraints.journal_filter = journal;
            break;
        }
    }
    
    // Extract study type constraints
    if (text.includes('clinical trial')) {
        constraints.study_type = 'clinical_trial';
    } else if (text.includes('review')) {
        constraints.study_type = 'review';
    }
    
    return constraints;
}

function extractResearchTopic(text: string): string {
    // Extract topic after synthesis keywords
    let topic = text.toLowerCase();
    
    const synthesisPrefixes = [
        'synthesize research on', 'synthesize', 'combine findings on',
        'integrate research on', 'meta-analysis of'
    ];
    
    for (const prefix of synthesisPrefixes) {
        if (topic.includes(prefix)) {
            topic = topic.split(prefix)[1]?.trim() || topic;
            break;
        }
    }
    
    return topic.replace(/[^\w\s-]/g, ' ').trim();
}

function extractSynthesisType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('meta-analysis')) {
        return 'meta_analysis';
    } else if (lowerText.includes('systematic review')) {
        return 'systematic_review';
    } else if (lowerText.includes('narrative review')) {
        return 'narrative_review';
    }
    
    return 'narrative_synthesis';
}

function extractInclusionCriteria(text: string): any {
    const criteria: any = {};
    
    // Extract study design criteria
    if (text.includes('randomized')) {
        criteria.study_design = 'randomized_controlled_trial';
    } else if (text.includes('observational')) {
        criteria.study_design = 'observational';
    }
    
    // Extract population criteria
    if (text.includes('human')) {
        criteria.population = 'human';
    } else if (text.includes('animal')) {
        criteria.population = 'animal';
    }
    
    // Extract language criteria
    if (text.includes('english')) {
        criteria.language = 'english';
    }
    
    return criteria;
}

function extractReviewTopic(text: string): string {
    let topic = text.toLowerCase();
    
    const reviewPrefixes = [
        'literature review on', 'review of', 'comprehensive review of',
        'survey of', 'overview of'
    ];
    
    for (const prefix of reviewPrefixes) {
        if (topic.includes(prefix)) {
            topic = topic.split(prefix)[1]?.trim() || topic;
            break;
        }
    }
    
    return topic.replace(/[^\w\s-]/g, ' ').trim();
}

function extractReviewScope(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('comprehensive') || lowerText.includes('extensive')) {
        return 'comprehensive';
    } else if (lowerText.includes('focused') || lowerText.includes('specific')) {
        return 'focused';
    } else if (lowerText.includes('brief') || lowerText.includes('concise')) {
        return 'brief';
    }
    
    return 'standard';
}

function extractTimeFrame(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('last year')) {
        return 'last_1_year';
    } else if (lowerText.includes('last 3 years')) {
        return 'last_3_years';
    } else if (lowerText.includes('last 5 years')) {
        return 'last_5_years';
    } else if (lowerText.includes('last decade')) {
        return 'last_10_years';
    }
    
    return 'last_5_years';
}

function formatKnowledgeExtractionResponse(extractedKnowledge: any, target: string): string {
    let response = `🧠 **Knowledge Extraction Results**\n\n`;
    
    response += `**Target:** ${target}\n`;
    response += `**Papers Analyzed:** ${extractedKnowledge.papers_analyzed || 0}\n`;
    response += `**Knowledge Items Extracted:** ${extractedKnowledge.total_items || 0}\n\n`;
    
    if (extractedKnowledge.structured_facts && extractedKnowledge.structured_facts.length > 0) {
        response += `**Key Facts Extracted:**\n`;
        extractedKnowledge.structured_facts.slice(0, 10).forEach((fact: any, index: number) => {
            response += `${index + 1}. **${fact.subject}** ${fact.predicate} **${fact.object}**\n`;
            if (fact.confidence) {
                response += `   *Confidence:* ${(fact.confidence * 100).toFixed(1)}%\n`;
            }
            if (fact.evidence_count) {
                response += `   *Evidence:* ${fact.evidence_count} papers\n`;
            }
            response += `\n`;
        });
    }
    
    if (extractedKnowledge.entity_relationships && extractedKnowledge.entity_relationships.length > 0) {
        response += `**Entity Relationships:**\n`;
        extractedKnowledge.entity_relationships.slice(0, 5).forEach((rel: any, index: number) => {
            response += `${index + 1}. ${rel.entity1} ↔ ${rel.entity2} (${rel.relationship_type})\n`;
            if (rel.strength) {
                response += `   *Strength:* ${(rel.strength * 100).toFixed(1)}%\n`;
            }
        });
        response += `\n`;
    }
    
    if (extractedKnowledge.quantitative_data && extractedKnowledge.quantitative_data.length > 0) {
        response += `**Quantitative Findings:**\n`;
        extractedKnowledge.quantitative_data.slice(0, 5).forEach((data: any, index: number) => {
            response += `${index + 1}. ${data.measurement}: ${data.value} ${data.unit}\n`;
            if (data.context) {
                response += `   *Context:* ${data.context}\n`;
            }
        });
        response += `\n`;
    }
    
    if (extractedKnowledge.contradictions && extractedKnowledge.contradictions.length > 0) {
        response += `**Identified Contradictions:**\n`;
        extractedKnowledge.contradictions.slice(0, 3).forEach((contradiction: string, index: number) => {
            response += `${index + 1}. ${contradiction}\n`;
        });
        response += `\n`;
    }
    
    if (extractedKnowledge.knowledge_gaps && extractedKnowledge.knowledge_gaps.length > 0) {
        response += `**Knowledge Gaps Identified:**\n`;
        extractedKnowledge.knowledge_gaps.slice(0, 3).forEach((gap: string, index: number) => {
            response += `${index + 1}. ${gap}\n`;
        });
    }
    
    return response;
}

function formatResearchSynthesisResponse(synthesis: any, topic: string): string {
    let response = `🔬 **Research Synthesis**\n\n`;
    
    response += `**Topic:** ${topic}\n`;
    response += `**Papers Included:** ${synthesis.papers_included || 0}\n`;
    response += `**Synthesis Date:** ${new Date().toLocaleDateString()}\n\n`;
    
    if (synthesis.executive_summary) {
        response += `**Executive Summary:**\n${synthesis.executive_summary}\n\n`;
    }
    
    if (synthesis.key_findings && synthesis.key_findings.length > 0) {
        response += `**Key Findings:**\n`;
        synthesis.key_findings.slice(0, 5).forEach((finding: any, index: number) => {
            response += `${index + 1}. **${finding.finding}**\n`;
            response += `   *Evidence Level:* ${finding.evidence_level || 'Moderate'}\n`;
            response += `   *Supporting Studies:* ${finding.supporting_studies || 'Multiple'}\n`;
            if (finding.effect_size) {
                response += `   *Effect Size:* ${finding.effect_size}\n`;
            }
            response += `\n`;
        });
    }
    
    if (synthesis.consensus_areas && synthesis.consensus_areas.length > 0) {
        response += `**Areas of Scientific Consensus:**\n`;
        synthesis.consensus_areas.slice(0, 3).forEach((area: string, index: number) => {
            response += `${index + 1}. ${area}\n`;
        });
        response += `\n`;
    }
    
    if (synthesis.conflicting_evidence && synthesis.conflicting_evidence.length > 0) {
        response += `**Conflicting Evidence:**\n`;
        synthesis.conflicting_evidence.slice(0, 3).forEach((conflict: any, index: number) => {
            response += `${index + 1}. **${conflict.topic}**\n`;
            response += `   *Conflicting Views:* ${conflict.conflicting_views.join(' vs ')}\n`;
            response += `   *Resolution Needed:* ${conflict.resolution_needed}\n`;
            response += `\n`;
        });
    }
    
    if (synthesis.clinical_implications && synthesis.clinical_implications.length > 0) {
        response += `**Clinical Implications:**\n`;
        synthesis.clinical_implications.slice(0, 3).forEach((implication: string, index: number) => {
            response += `${index + 1}. ${implication}\n`;
        });
        response += `\n`;
    }
    
    if (synthesis.future_research_directions && synthesis.future_research_directions.length > 0) {
        response += `**Future Research Directions:**\n`;
        synthesis.future_research_directions.slice(0, 3).forEach((direction: string, index: number) => {
            response += `${index + 1}. ${direction}\n`;
        });
    }
    
    return response;
}

function formatLiteratureReviewResponse(review: any, topic: string): string {
    let response = `📖 **Literature Review**\n\n`;
    
    response += `**Topic:** ${topic}\n`;
    response += `**Review Type:** ${review.review_type || 'Narrative Review'}\n`;
    response += `**Papers Reviewed:** ${review.papers_reviewed || 0}\n`;
    response += `**Date Range:** ${review.date_range || 'Last 5 years'}\n\n`;
    
    if (review.abstract) {
        response += `**Abstract:**\n${review.abstract}\n\n`;
    }
    
    if (review.introduction) {
        response += `**Introduction:**\n${review.introduction}\n\n`;
    }
    
    if (review.main_sections && review.main_sections.length > 0) {
        response += `**Main Sections:**\n`;
        review.main_sections.forEach((section: any, index: number) => {
            response += `\n**${index + 1}. ${section.title}**\n`;
            response += `${section.content}\n`;
            
            if (section.key_papers && section.key_papers.length > 0) {
                response += `\n*Key Papers:*\n`;
                section.key_papers.slice(0, 3).forEach((paper: any) => {
                    response += `• ${paper.title} (${paper.journal}, ${paper.year})\n`;
                });
            }
        });
        response += `\n`;
    }
    
    if (review.methodology_assessment) {
        const methodology = review.methodology_assessment;
        response += `**Methodology Assessment:**\n`;
        response += `• Study Quality: ${methodology.overall_quality || 'Mixed'}\n`;
        response += `• Sample Sizes: ${methodology.sample_size_adequacy || 'Variable'}\n`;
        response += `• Methodological Rigor: ${methodology.methodological_rigor || 'Moderate'}\n\n`;
    }
    
    if (review.research_gaps && review.research_gaps.length > 0) {
        response += `**Identified Research Gaps:**\n`;
        review.research_gaps.slice(0, 5).forEach((gap: string, index: number) => {
            response += `${index + 1}. ${gap}\n`;
        });
        response += `\n`;
    }
    
    if (review.conclusions) {
        response += `**Conclusions:**\n${review.conclusions}\n\n`;
    }
    
    if (review.recommendations && review.recommendations.length > 0) {
        response += `**Recommendations:**\n`;
        review.recommendations.slice(0, 5).forEach((rec: string, index: number) => {
            response += `${index + 1}. ${rec}\n`;
        });
    }
    
    return response;
}
