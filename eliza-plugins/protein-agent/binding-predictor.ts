/**
 * Protein Binding Site Prediction Actions - Production Implementation
 * Real binding site identification and drug-protein interaction analysis
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Find Binding Sites Action
export const findBindingSitesAction: Action = {
    name: "FIND_BINDING_SITES",
    similes: [
        "find binding sites",
        "predict binding sites",
        "identify binding pockets",
        "binding site prediction",
        "active site prediction",
        "druggable sites"
    ],
    description: "Identify and analyze protein binding sites and druggable pockets",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you find the binding sites in the EGFR protein?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the EGFR protein structure to identify binding sites and assess their druggability potential.",
                    action: "FIND_BINDING_SITES"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What are the druggable pockets in human kinase proteins?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll identify druggable binding pockets in kinase proteins and analyze their therapeutic potential.",
                    action: "FIND_BINDING_SITES"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const bindingKeywords = [
            "binding site", "binding pocket", "active site", "druggable",
            "pocket", "cavity", "binding", "site", "drug target"
        ];
        
        const proteinKeywords = ["protein", "enzyme", "receptor", "kinase"];
        
        return bindingKeywords.some(keyword => text.includes(keyword)) &&
               proteinKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const proteinInfo = await extractProteinInformation(message.content.text);
            
            if (!proteinInfo.sequence && !proteinInfo.uniprotId && !proteinInfo.name) {
                const errorText = "I need a protein name, UniProt ID, or sequence to identify binding sites. Could you provide one of these?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing protein information" } });
                }
                
                return { text: errorText, data: { error: "Missing protein information" } };
            }
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'protein-agent',
                    action: 'find_binding_sites',
                    parameters: {
                        protein_id: proteinInfo.uniprotId,
                        sequence: proteinInfo.sequence,
                        protein_name: proteinInfo.name,
                        prediction_method: 'geometric',
                        druggability_threshold: 0.5,
                        include_pharmacophore: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const bindingSites = result.result.binding_sites;
                const responseText = formatBindingSitesResponse(bindingSites, proteinInfo);
                
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
                const errorText = "I couldn't identify binding sites for this protein. This might be due to limited structural data or an invalid protein identifier.";
                
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
            const errorText = "I'm experiencing technical difficulties with binding site prediction. Please try again later.";
            
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

// Predict Drug Binding Action
export const predictDrugBindingAction: Action = {
    name: "PREDICT_DRUG_BINDING",
    similes: [
        "predict drug binding",
        "drug protein interaction",
        "binding affinity",
        "molecular docking",
        "drug target interaction",
        "binding prediction"
    ],
    description: "Predict drug-protein binding interactions and calculate binding affinities",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you predict how ibuprofen binds to COX-2 protein?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the molecular interaction between ibuprofen and COX-2, including binding affinity and interaction sites.",
                    action: "PREDICT_DRUG_BINDING"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What is the binding affinity of aspirin to cyclooxygenase?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll calculate the binding affinity between aspirin and cyclooxygenase using molecular docking analysis.",
                    action: "PREDICT_DRUG_BINDING"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const bindingKeywords = [
            "binding", "affinity", "dock", "interaction", "bind"
        ];
        
        const drugKeywords = [
            "drug", "compound", "molecule", "inhibitor", "agonist",
            "antagonist", "aspirin", "ibuprofen", "metformin"
        ];
        
        const proteinKeywords = ["protein", "enzyme", "receptor", "target"];
        
        const hasDrug = drugKeywords.some(keyword => text.includes(keyword));
        const hasProtein = proteinKeywords.some(keyword => text.includes(keyword));
        const hasBinding = bindingKeywords.some(keyword => text.includes(keyword));
        
        return hasDrug && hasProtein && hasBinding;
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const drugInfo = extractDrugInformation(message.content.text);
            const proteinInfo = await extractProteinInformation(message.content.text);
            
            if (!drugInfo.name && !drugInfo.smiles) {
                const errorText = "I need a drug name or SMILES structure to predict binding. Could you specify the drug compound?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing drug information" } });
                }
                
                return { text: errorText, data: { error: "Missing drug information" } };
            }
            
            if (!proteinInfo.name && !proteinInfo.uniprotId) {
                const errorText = "I need a protein name or UniProt ID to predict binding. Could you specify the target protein?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing protein information" } });
                }
                
                return { text: errorText, data: { error: "Missing protein information" } };
            }
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'protein-agent',
                    action: 'predict_drug_binding',
                    parameters: {
                        drug_name: drugInfo.name,
                        drug_smiles: drugInfo.smiles,
                        protein_id: proteinInfo.uniprotId,
                        protein_name: proteinInfo.name,
                        docking_method: 'autodock_vina',
                        include_admet: true,
                        binding_site_id: null
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const prediction = result.result;
                const responseText = formatDrugBindingResponse(prediction, drugInfo, proteinInfo);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: prediction
                    });
                }
                
                return {
                    text: responseText,
                    data: prediction
                };
            } else {
                const errorText = "I couldn't predict the drug-protein binding. Please check that the drug and protein names are valid.";
                
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
            const errorText = "I'm having trouble with binding prediction. Please try again shortly.";
            
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

// Analyze Binding Affinity Action
export const analyzeBindingAffinityAction: Action = {
    name: "ANALYZE_BINDING_AFFINITY",
    similes: [
        "analyze binding affinity",
        "binding strength",
        "affinity analysis",
        "ic50 prediction",
        "kd calculation",
        "binding constant"
    ],
    description: "Analyze and calculate binding affinity constants (IC50, Kd, Ki)",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What is the IC50 of metformin binding to AMPK?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the binding affinity between metformin and AMPK to calculate the IC50 value.",
                    action: "ANALYZE_BINDING_AFFINITY"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const affinityKeywords = [
            "affinity", "ic50", "ec50", "kd", "ki", "binding constant",
            "binding strength", "potency", "inhibition"
        ];
        
        return affinityKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const drugInfo = extractDrugInformation(message.content.text);
            const proteinInfo = await extractProteinInformation(message.content.text);
            const affinityType = extractAffinityType(message.content.text);
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'protein-agent',
                    action: 'analyze_binding_affinity',
                    parameters: {
                        drug_name: drugInfo.name,
                        drug_smiles: drugInfo.smiles,
                        protein_id: proteinInfo.uniprotId,
                        protein_name: proteinInfo.name,
                        affinity_type: affinityType,
                        calculation_method: 'ensemble'
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const affinity = result.result;
                const responseText = formatAffinityAnalysisResponse(affinity, drugInfo, proteinInfo, affinityType);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: affinity
                    });
                }
                
                return {
                    text: responseText,
                    data: affinity
                };
            } else {
                const errorText = "I couldn't analyze the binding affinity. Please provide valid drug and protein information.";
                
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
            const errorText = "I'm experiencing issues with affinity analysis. Please try again later.";
            
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

export const bindingActions = [
    findBindingSitesAction,
    predictDrugBindingAction,
    analyzeBindingAffinityAction
];

// Helper Functions
async function extractProteinInformation(text: string): Promise<{
    sequence?: string;
    uniprotId?: string;
    name?: string;
}> {
    const result: any = {};
    
    // Extract protein sequence
    const sequencePattern = /[ACDEFGHIKLMNPQRSTVWY]{10,}/gi;
    const sequenceMatch = text.match(sequencePattern);
    if (sequenceMatch) {
        result.sequence = sequenceMatch[0];
    }
    
    // Extract UniProt ID
    const uniprotPattern = /[A-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z]{1,2}[0-9]{5}/gi;
    const uniprotMatch = text.match(uniprotPattern);
    if (uniprotMatch) {
        result.uniprotId = uniprotMatch[0];
    }
    
    // Extract protein names
    const proteinNames = [
        'cox-2', 'cox2', 'cyclooxygenase-2', 'cyclooxygenase',
        'egfr', 'her2', 'p53', 'brca1', 'brca2', 'insulin receptor',
        'ampk', 'mtor', 'akt', 'pi3k', 'kinase', 'protease'
    ];
    
    const lowerText = text.toLowerCase();
    for (const proteinName of proteinNames) {
        if (lowerText.includes(proteinName)) {
            result.name = proteinName;
            break;
        }
    }
    
    return result;
}

function extractDrugInformation(text: string): {
    name?: string;
    smiles?: string;
    type?: string;
} {
    const result: any = {};
    
    // Extract common drug names
    const drugNames = [
        'aspirin', 'ibuprofen', 'acetaminophen', 'metformin',
        'insulin', 'warfarin', 'statins', 'penicillin',
        'amoxicillin', 'lisinopril', 'amlodipine', 'omeprazole'
    ];
    
    const lowerText = text.toLowerCase();
    for (const drugName of drugNames) {
        if (lowerText.includes(drugName)) {
            result.name = drugName;
            result.type = 'known_drug';
            break;
        }
    }
    
    // Extract SMILES pattern (simplified)
    const smilesPattern = /[A-Za-z0-9@+\-\[\]()=#]{10,}/;
    const smilesMatch = text.match(smilesPattern);
    if (smilesMatch && !result.name) {
        result.smiles = smilesMatch[0];
        result.type = 'smiles';
    }
    
    return result;
}

function extractAffinityType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('ic50')) return 'IC50';
    if (lowerText.includes('ec50')) return 'EC50';
    if (lowerText.includes('kd')) return 'Kd';
    if (lowerText.includes('ki')) return 'Ki';
    
    return 'IC50'; // Default
}

function formatBindingSitesResponse(bindingSites: any[], proteinInfo: any): string {
    let response = `🎯 **Binding Site Analysis**\n\n`;
    
    if (proteinInfo.name) {
        response += `**Protein:** ${proteinInfo.name}\n\n`;
    }
    
    if (bindingSites.length === 0) {
        response += "No significant binding sites were identified in this protein structure.";
        return response;
    }
    
    response += `**Identified Binding Sites:** ${bindingSites.length}\n\n`;
    
    bindingSites.slice(0, 5).forEach((site: any, index: number) => {
        response += `**Site ${index + 1}: ${site.name || `Binding Site ${index + 1}`}**\n`;
        response += `• Type: ${site.site_type || 'binding'}\n`;
        response += `• Druggability Score: ${(site.druggability_score * 100).toFixed(1)}%\n`;
        
        if (site.volume) {
            response += `• Volume: ${site.volume.toFixed(1)} ų\n`;
        }
        
        if (site.residues && site.residues.length > 0) {
            response += `• Key Residues: ${site.residues.slice(0, 5).join(', ')}\n`;
        }
        
        if (site.confidence_score) {
            response += `• Confidence: ${(site.confidence_score * 100).toFixed(1)}%\n`;
        }
        
        // Druggability assessment
        const druggability = site.druggability_score || 0;
        if (druggability > 0.8) {
            response += `• Assessment: **Highly druggable** - Excellent therapeutic target\n`;
        } else if (druggability > 0.6) {
            response += `• Assessment: **Moderately druggable** - Good therapeutic potential\n`;
        } else if (druggability > 0.4) {
            response += `• Assessment: **Weakly druggable** - May require optimization\n`;
        } else {
            response += `• Assessment: **Poorly druggable** - Challenging target\n`;
        }
        
        response += `\n`;
    });
    
    // Summary recommendations
    const highDruggabilitySites = bindingSites.filter(site => site.druggability_score > 0.7);
    if (highDruggabilitySites.length > 0) {
        response += `**Recommendation:** Focus on sites ${highDruggabilitySites.map((_, i) => i + 1).join(', ')} for drug development due to high druggability scores.`;
    } else {
        response += `**Recommendation:** Consider alternative approaches such as allosteric modulation or protein-protein interaction inhibition.`;
    }
    
    return response;
}

function formatDrugBindingResponse(prediction: any, drugInfo: any, proteinInfo: any): string {
    let response = `💊 **Drug-Protein Binding Prediction**\n\n`;
    
    response += `**Interaction:** ${drugInfo.name || 'Compound'} ↔ ${proteinInfo.name || 'Target Protein'}\n\n`;
    
    if (prediction.binding_analysis) {
        const binding = prediction.binding_analysis;
        response += `**Binding Analysis:**\n`;
        response += `• Predicted Affinity: ${binding.predicted_affinity.toFixed(2)} (pIC50)\n`;
        response += `• Confidence: ${(binding.confidence * 100).toFixed(1)}%\n`;
        response += `• Binding Mode: ${binding.binding_mode || 'competitive'}\n`;
        
        if (binding.binding_energy) {
            response += `• Binding Energy: ${binding.binding_energy.toFixed(2)} kcal/mol\n`;
        }
        
        response += `\n`;
    }
    
    if (prediction.interaction_sites && prediction.interaction_sites.length > 0) {
        response += `**Key Interaction Sites:**\n`;
        prediction.interaction_sites.slice(0, 5).forEach((site: any, index: number) => {
            response += `${index + 1}. **${site.residue}**: ${site.interaction_type}`;
            if (site.distance) {
                response += ` (${site.distance.toFixed(2)} Ų)`;
            }
            if (site.strength) {
                response += ` - Strength: ${site.strength.toFixed(2)}`;
            }
            response += `\n`;
        });
        response += `\n`;
    }
    
    if (prediction.pharmacokinetics) {
        const pk = prediction.pharmacokinetics;
        response += `**Pharmacokinetic Properties:**\n`;
        
        if (pk.bioavailability) {
            response += `• Oral Bioavailability: ${pk.bioavailability.toFixed(1)}%\n`;
        }
        
        if (pk.half_life) {
            response += `• Half-life: ${pk.half_life.toFixed(1)} hours\n`;
        }
        
        if (pk.clearance) {
            response += `• Clearance: ${pk.clearance.toFixed(2)} L/h/kg\n`;
        }
        
        response += `\n`;
    }
    
    // Clinical relevance assessment
    const affinity = prediction.binding_analysis?.predicted_affinity || 0;
    response += `**Clinical Relevance:** `;
    
    if (affinity > 8) {
        response += "**High therapeutic potential** - Strong binding suggests good efficacy.";
    } else if (affinity > 6) {
        response += "**Moderate therapeutic potential** - May benefit from optimization.";
    } else if (affinity > 4) {
        response += "**Low therapeutic potential** - Significant optimization needed.";
    } else {
        response += "**Poor therapeutic potential** - Consider alternative targets.";
    }
    
    return response;
}

function formatAffinityAnalysisResponse(
    affinity: any, 
    drugInfo: any, 
    proteinInfo: any, 
    affinityType: string
): string {
    let response = `📊 **Binding Affinity Analysis**\n\n`;
    
    response += `**Interaction:** ${drugInfo.name || 'Compound'} → ${proteinInfo.name || 'Target'}\n`;
    response += `**Analysis Type:** ${affinityType}\n\n`;
    
    if (affinity.calculated_affinity) {
        const calc = affinity.calculated_affinity;
        response += `**Calculated ${affinityType}:**\n`;
        response += `• Value: ${calc.value.toFixed(2)} ${calc.units || 'nM'}\n`;
        response += `• Confidence: ${(calc.confidence * 100).toFixed(1)}%\n`;
        response += `• Method: ${calc.method || 'Ensemble prediction'}\n\n`;
        
        // Interpret the value
        const value = calc.value;
        response += `**Interpretation:**\n`;
        
        if (affinityType === 'IC50') {
            if (value < 10) {
                response += `• **Very potent** - Excellent inhibitory activity\n`;
            } else if (value < 100) {
                response += `• **Potent** - Good inhibitory activity\n`;
            } else if (value < 1000) {
                response += `• **Moderate** - Moderate inhibitory activity\n`;
            } else {
                response += `• **Weak** - Poor inhibitory activity\n`;
            }
        } else if (affinityType === 'Kd') {
            if (value < 1) {
                response += `• **Very high affinity** - Strong binding\n`;
            } else if (value < 10) {
                response += `• **High affinity** - Good binding\n`;
            } else if (value < 100) {
                response += `• **Moderate affinity** - Moderate binding\n`;
            } else {
                response += `• **Low affinity** - Weak binding\n`;
            }
        }
    }
    
    if (affinity.experimental_data) {
        const exp = affinity.experimental_data;
        response += `\n**Experimental Reference:**\n`;
        response += `• Literature ${affinityType}: ${exp.value} ${exp.units}\n`;
        response += `• Source: ${exp.source || 'ChEMBL/BindingDB'}\n`;
        response += `• Assay Type: ${exp.assay_type || 'Biochemical'}\n`;
    }
    
    if (affinity.structure_activity_relationship) {
        const sar = affinity.structure_activity_relationship;
        response += `\n**Structure-Activity Insights:**\n`;
        
        if (sar.key_interactions) {
            response += `• Key Interactions: ${sar.key_interactions.join(', ')}\n`;
        }
        
        if (sar.optimization_suggestions) {
            response += `• Optimization: ${sar.optimization_suggestions.slice(0, 2).join(', ')}\n`;
        }
    }
    
    return response;
}
