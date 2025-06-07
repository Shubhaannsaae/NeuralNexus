/**
 * Bio Agent Actions - Production Implementation
 * Real biological system analysis actions following Eliza framework patterns
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Analyze Biological System Action
export const analyzeBiologicalSystemAction: Action = {
    name: "ANALYZE_BIOLOGICAL_SYSTEM",
    similes: [
        "analyze biological system",
        "study pathway",
        "examine network",
        "investigate mechanism",
        "analyze protein network",
        "study biological pathway"
    ],
    description: "Analyze biological systems, pathways, and molecular networks",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you analyze the mTOR signaling pathway and its role in cancer?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the mTOR pathway for you. Let me examine the key components and their interactions in cancer biology.",
                    action: "ANALYZE_BIOLOGICAL_SYSTEM"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What are the key proteins involved in Alzheimer's disease pathogenesis?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the protein networks involved in Alzheimer's disease pathogenesis, including amyloid and tau pathways.",
                    action: "ANALYZE_BIOLOGICAL_SYSTEM"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const keywords = [
            "analyze", "system", "pathway", "network", "mechanism",
            "protein", "gene", "biological", "molecular", "cellular"
        ];
        
        return keywords.some(keyword => text.includes(keyword)) &&
               (text.includes("biological") || text.includes("pathway") || 
                text.includes("protein") || text.includes("mechanism"));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            // Extract biological entities from message
            const entities = await extractBiologicalEntities(message.content.text, runtime);
            
            // Determine system type
            const systemType = determineSystemType(message.content.text);
            
            // Call backend API for biological system analysis
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'bio-agent',
                    action: 'analyze_biological_system',
                    parameters: {
                        entities: entities,
                        system_type: systemType,
                        analysis_depth: 3,
                        include_pathways: true,
                        include_interactions: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const analysis = result.result;
                
                // Format comprehensive response
                const responseText = formatBiologicalAnalysis(analysis, entities);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: analysis
                    });
                }
                
                return {
                    text: responseText,
                    data: analysis
                };
            } else {
                const errorText = "I encountered an issue analyzing the biological system. Could you provide more specific information about the proteins or pathways you're interested in?";
                
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
            const errorText = "I'm having trouble connecting to the analysis system. Please try again in a moment.";
            
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

// Find Drug Targets Action
export const findDrugTargetsAction: Action = {
    name: "FIND_DRUG_TARGETS",
    similes: [
        "find drug targets",
        "identify targets",
        "discover therapeutic targets",
        "target identification",
        "find therapeutic targets",
        "identify drug targets"
    ],
    description: "Identify potential drug targets for diseases using AI analysis",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What are the best drug targets for treating Parkinson's disease?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll identify promising drug targets for Parkinson's disease by analyzing the molecular pathways involved.",
                    action: "FIND_DRUG_TARGETS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const targetKeywords = ["target", "drug", "therapeutic", "treatment"];
        const diseaseKeywords = ["disease", "disorder", "cancer", "alzheimer", "parkinson", "diabetes"];
        
        return targetKeywords.some(keyword => text.includes(keyword)) &&
               diseaseKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const disease = await extractDiseaseFromMessage(message.content.text, runtime);
            const targetType = extractTargetType(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'bio-agent',
                    action: 'find_drug_targets',
                    parameters: {
                        disease: disease,
                        target_type: targetType,
                        druggability_threshold: 0.7,
                        max_targets: 10,
                        include_pathways: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const targets = result.result.targets;
                const responseText = formatDrugTargetsResponse(targets, disease);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: result.result
                    });
                }
                
                return {
                    text: responseText,
                    data: result.result
                };
            } else {
                const errorText = `I couldn't identify specific drug targets for ${disease}. Could you provide more details about the specific aspects of the disease you're interested in?`;
                
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
            const errorText = "I'm experiencing technical difficulties with target identification. Please try again shortly.";
            
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

// Predict Drug Interactions Action
export const predictDrugInteractionsAction: Action = {
    name: "PREDICT_DRUG_INTERACTIONS",
    similes: [
        "predict drug interactions",
        "analyze drug binding",
        "drug protein interaction",
        "binding prediction",
        "molecular docking"
    ],
    description: "Predict drug-protein interactions and binding affinities",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you predict how aspirin interacts with COX-2 protein?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the molecular interaction between aspirin and COX-2 protein, including binding affinity and interaction sites.",
                    action: "PREDICT_DRUG_INTERACTIONS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const interactionKeywords = ["interact", "binding", "dock", "affinity"];
        const moleculeKeywords = ["drug", "compound", "molecule", "protein"];
        
        return interactionKeywords.some(keyword => text.includes(keyword)) &&
               moleculeKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const drugInfo = extractDrugFromMessage(message.content.text);
            const proteinInfo = extractProteinFromMessage(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'bio-agent',
                    action: 'predict_interactions',
                    parameters: {
                        drug: drugInfo,
                        protein: proteinInfo,
                        interaction_type: 'binding',
                        include_docking: true,
                        include_admet: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const prediction = result.result.prediction;
                const responseText = formatInteractionPrediction(prediction, drugInfo, proteinInfo);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: result.result
                    });
                }
                
                return {
                    text: responseText,
                    data: result.result
                };
            } else {
                const errorText = "I couldn't predict the drug-protein interaction. Please provide more specific information about the drug and protein.";
                
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
            const errorText = "I'm having trouble with the interaction prediction system. Please try again later.";
            
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

export const bioActions = [
    analyzeBiologicalSystemAction,
    findDrugTargetsAction,
    predictDrugInteractionsAction
];

// Helper Functions
async function extractBiologicalEntities(text: string, runtime: IAgentRuntime): Promise<string[]> {
    // Use NLP to extract biological entities
    const bioTerms = [
        // Proteins
        'p53', 'mtor', 'egfr', 'her2', 'brca1', 'brca2', 'akt', 'pi3k',
        'ras', 'myc', 'tp53', 'pten', 'rb1', 'apc', 'vegf', 'pdgf',
        // Pathways
        'apoptosis', 'autophagy', 'glycolysis', 'oxidative phosphorylation',
        'cell cycle', 'dna repair', 'protein synthesis', 'signal transduction',
        // Diseases
        'cancer', 'alzheimer', 'parkinson', 'diabetes', 'cardiovascular',
        'neurodegeneration', 'inflammation', 'autoimmune'
    ];
    
    const words = text.toLowerCase().split(/\s+/);
    const entities = words.filter(word => 
        bioTerms.includes(word) || 
        (word.length > 3 && /^[a-z][a-z0-9]*$/i.test(word))
    );
    
    // Remove duplicates and return
    return [...new Set(entities)];
}

function determineSystemType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('pathway') || lowerText.includes('signaling')) {
        return 'pathway';
    } else if (lowerText.includes('network') || lowerText.includes('interaction')) {
        return 'network';
    } else if (lowerText.includes('protein') || lowerText.includes('enzyme')) {
        return 'protein_complex';
    } else {
        return 'biological_system';
    }
}

async function extractDiseaseFromMessage(text: string, runtime: IAgentRuntime): Promise<string> {
    const diseases = [
        'cancer', 'breast cancer', 'lung cancer', 'prostate cancer',
        'alzheimer', "alzheimer's disease", 'dementia',
        'parkinson', "parkinson's disease",
        'diabetes', 'type 1 diabetes', 'type 2 diabetes',
        'cardiovascular disease', 'heart disease',
        'stroke', 'hypertension',
        'covid-19', 'coronavirus',
        'multiple sclerosis', 'ms',
        'rheumatoid arthritis', 'arthritis'
    ];
    
    const lowerText = text.toLowerCase();
    
    for (const disease of diseases) {
        if (lowerText.includes(disease)) {
            return disease;
        }
    }
    
    // Fallback: look for disease-like terms
    const diseasePatterns = /(\w+)\s+(disease|disorder|syndrome|cancer)/gi;
    const matches = text.match(diseasePatterns);
    
    if (matches && matches.length > 0) {
        return matches[0];
    }
    
    return 'unspecified condition';
}

function extractTargetType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('protein')) return 'protein';
    if (lowerText.includes('enzyme')) return 'enzyme';
    if (lowerText.includes('receptor')) return 'receptor';
    if (lowerText.includes('kinase')) return 'kinase';
    if (lowerText.includes('channel')) return 'ion_channel';
    
    return 'protein'; // Default
}

function extractDrugFromMessage(text: string): any {
    // Extract drug names and properties
    const drugNames = [
        'aspirin', 'ibuprofen', 'acetaminophen', 'metformin',
        'insulin', 'warfarin', 'statins', 'penicillin'
    ];
    
    const lowerText = text.toLowerCase();
    
    for (const drug of drugNames) {
        if (lowerText.includes(drug)) {
            return { name: drug, type: 'known_drug' };
        }
    }
    
    // Look for SMILES or chemical formulas
    const smilesPattern = /[A-Za-z0-9@+\-\[\]()=#]/;
    if (smilesPattern.test(text)) {
        return { smiles: text.match(smilesPattern)?.[0], type: 'smiles' };
    }
    
    return { name: 'unknown compound', type: 'unknown' };
}

function extractProteinFromMessage(text: string): any {
    const proteinNames = [
        'cox-2', 'cox2', 'cyclooxygenase-2',
        'egfr', 'her2', 'p53', 'brca1', 'brca2',
        'insulin receptor', 'dopamine receptor'
    ];
    
    const lowerText = text.toLowerCase();
    
    for (const protein of proteinNames) {
        if (lowerText.includes(protein)) {
            return { name: protein, type: 'known_protein' };
        }
    }
    
    // Look for UniProt IDs
    const uniprotPattern = /[A-Z][0-9][A-Z0-9]{3}[0-9]/;
    const uniprotMatch = text.match(uniprotPattern);
    
    if (uniprotMatch) {
        return { uniprot_id: uniprotMatch[0], type: 'uniprot' };
    }
    
    return { name: 'unknown protein', type: 'unknown' };
}

function formatBiologicalAnalysis(analysis: any, entities: string[]): string {
    const components = analysis.components || [];
    const interactions = analysis.interactions || [];
    const insights = analysis.biological_insights || [];
    
    let response = `🧬 **Biological System Analysis Complete**\n\n`;
    
    if (entities.length > 0) {
        response += `**Analyzed Entities:** ${entities.join(', ')}\n\n`;
    }
    
    response += `**System Overview:**\n`;
    response += `• ${components.length} molecular components identified\n`;
    response += `• ${interactions.length} interactions mapped\n`;
    
    if (analysis.pathway_enrichment) {
        response += `• ${Object.keys(analysis.pathway_enrichment).length} enriched pathways\n`;
    }
    
    if (insights.length > 0) {
        response += `\n**Key Biological Insights:**\n`;
        insights.slice(0, 3).forEach((insight: string, index: number) => {
            response += `${index + 1}. ${insight}\n`;
        });
    }
    
    if (analysis.network_analysis) {
        const networkMetrics = analysis.network_analysis;
        response += `\n**Network Properties:**\n`;
        response += `• Network density: ${(networkMetrics.density * 100).toFixed(1)}%\n`;
        response += `• Average clustering: ${(networkMetrics.clustering_coefficient * 100).toFixed(1)}%\n`;
        
        if (networkMetrics.hub_nodes && networkMetrics.hub_nodes.length > 0) {
            response += `• Key hub nodes: ${networkMetrics.hub_nodes.slice(0, 3).join(', ')}\n`;
        }
    }
    
    return response;
}

function formatDrugTargetsResponse(targets: any[], disease: string): string {
    let response = `🎯 **Drug Target Analysis for ${disease}**\n\n`;
    
    if (targets.length === 0) {
        response += "No specific drug targets identified. Consider broadening the search criteria.";
        return response;
    }
    
    response += `**Top Drug Targets Identified:**\n\n`;
    
    targets.slice(0, 5).forEach((target: any, index: number) => {
        response += `**${index + 1}. ${target.name}**\n`;
        response += `   • Druggability Score: ${(target.druggability_score * 100).toFixed(1)}%\n`;
        
        if (target.properties && target.properties.function) {
            response += `   • Function: ${target.properties.function}\n`;
        }
        
        if (target.properties && target.properties.pathway) {
            response += `   • Pathway: ${target.properties.pathway}\n`;
        }
        
        response += `\n`;
    });
    
    if (targets.length > 5) {
        response += `*... and ${targets.length - 5} additional targets identified*\n\n`;
    }
    
    response += `**Recommendation:** Focus on targets with druggability scores >70% for optimal therapeutic potential.`;
    
    return response;
}

function formatInteractionPrediction(prediction: any, drugInfo: any, proteinInfo: any): string {
    let response = `⚗️ **Drug-Protein Interaction Prediction**\n\n`;
    
    response += `**Interaction:** ${drugInfo.name} ↔ ${proteinInfo.name}\n\n`;
    
    if (prediction.binding_affinity) {
        response += `**Binding Analysis:**\n`;
        response += `• Predicted Affinity: ${prediction.binding_affinity.toFixed(2)} (pIC50)\n`;
        response += `• Confidence: ${(prediction.confidence * 100).toFixed(1)}%\n`;
        response += `• Binding Mode: ${prediction.binding_mode || 'competitive'}\n\n`;
    }
    
    if (prediction.interaction_sites && prediction.interaction_sites.length > 0) {
        response += `**Key Interaction Sites:**\n`;
        prediction.interaction_sites.slice(0, 3).forEach((site: any, index: number) => {
            response += `• ${site.residue}: ${site.interaction_type} (${site.strength.toFixed(2)})\n`;
        });
        response += `\n`;
    }
    
    if (prediction.stability_score) {
        response += `**Complex Stability:** ${(prediction.stability_score * 100).toFixed(1)}%\n\n`;
    }
    
    response += `**Clinical Relevance:** `;
    if (prediction.binding_affinity > 7) {
        response += "High therapeutic potential - strong binding predicted.";
    } else if (prediction.binding_affinity > 5) {
        response += "Moderate therapeutic potential - optimization may be beneficial.";
    } else {
        response += "Low binding affinity - consider structural modifications.";
    }
    
    return response;
}
