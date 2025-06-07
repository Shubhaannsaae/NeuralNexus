/**
 * Protein Structure Analysis Actions - Production Implementation
 * Real protein folding, structure prediction, and analysis capabilities
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Analyze Protein Structure Action
export const analyzeProteinStructureAction: Action = {
    name: "ANALYZE_PROTEIN_STRUCTURE",
    similes: [
        "analyze protein structure",
        "predict protein folding",
        "protein structure prediction",
        "fold protein",
        "analyze protein",
        "protein analysis"
    ],
    description: "Analyze protein structure, predict folding, and identify structural features",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you analyze the structure of the protein with sequence MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze this protein sequence using advanced structure prediction algorithms. Let me process the folding prediction and structural features.",
                    action: "ANALYZE_PROTEIN_STRUCTURE"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What is the predicted structure of human insulin protein?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the structure of human insulin, including its secondary structure, domains, and functional regions.",
                    action: "ANALYZE_PROTEIN_STRUCTURE"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const structureKeywords = [
            "structure", "fold", "folding", "prediction", "analyze protein",
            "secondary structure", "tertiary structure", "domain", "conformation"
        ];
        
        const proteinKeywords = [
            "protein", "sequence", "amino acid", "peptide", "polypeptide"
        ];
        
        const hasStructureKeyword = structureKeywords.some(keyword => text.includes(keyword));
        const hasProteinKeyword = proteinKeywords.some(keyword => text.includes(keyword));
        
        // Check for protein sequence pattern (amino acid letters)
        const sequencePattern = /[ACDEFGHIKLMNPQRSTVWY]{10,}/gi;
        const hasSequence = sequencePattern.test(text);
        
        return (hasStructureKeyword && hasProteinKeyword) || hasSequence;
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            // Extract protein information from message
            const proteinInfo = await extractProteinInformation(message.content.text);
            
            if (!proteinInfo.sequence && !proteinInfo.uniprotId && !proteinInfo.name) {
                const errorText = "I need either a protein sequence, UniProt ID, or protein name to perform structure analysis. Could you provide one of these?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing protein information" } });
                }
                
                return { text: errorText, data: { error: "Missing protein information" } };
            }
            
            // Call backend API for protein structure analysis
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'protein-agent',
                    action: 'analyze_protein',
                    parameters: {
                        protein_id: proteinInfo.uniprotId,
                        sequence: proteinInfo.sequence,
                        protein_name: proteinInfo.name,
                        analysis_type: 'full',
                        include_structure_prediction: true,
                        include_binding_sites: true,
                        include_domains: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const analysis = result.result;
                const responseText = formatProteinAnalysisResponse(analysis, proteinInfo);
                
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
                const errorText = "I encountered an issue analyzing the protein structure. This might be due to an invalid sequence or temporary service unavailability.";
                
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
            const errorText = "I'm experiencing technical difficulties with protein structure analysis. Please try again in a moment.";
            
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

// Predict Protein Function Action
export const predictProteinFunctionAction: Action = {
    name: "PREDICT_PROTEIN_FUNCTION",
    similes: [
        "predict protein function",
        "protein function prediction",
        "what does this protein do",
        "protein role",
        "protein activity"
    ],
    description: "Predict protein function based on sequence and structure analysis",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What is the predicted function of protein P53?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze P53 to predict its molecular function, biological processes, and cellular localization.",
                    action: "PREDICT_PROTEIN_FUNCTION"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const functionKeywords = [
            "function", "role", "activity", "does", "purpose",
            "biological process", "molecular function", "pathway"
        ];
        
        const proteinKeywords = ["protein", "enzyme", "receptor", "kinase", "gene"];
        
        return functionKeywords.some(keyword => text.includes(keyword)) &&
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
            
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'protein-agent',
                    action: 'analyze_protein',
                    parameters: {
                        protein_id: proteinInfo.uniprotId,
                        sequence: proteinInfo.sequence,
                        protein_name: proteinInfo.name,
                        analysis_type: 'function',
                        include_go_terms: true,
                        include_pathways: true,
                        include_localization: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const analysis = result.result;
                const responseText = formatFunctionPredictionResponse(analysis, proteinInfo);
                
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
                const errorText = "I couldn't predict the protein function. Please provide a valid protein name, UniProt ID, or sequence.";
                
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
            const errorText = "I'm having trouble with function prediction. Please try again later.";
            
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

// Compare Protein Structures Action
export const compareProteinStructuresAction: Action = {
    name: "COMPARE_PROTEIN_STRUCTURES",
    similes: [
        "compare proteins",
        "protein comparison",
        "structural similarity",
        "compare structures",
        "protein alignment"
    ],
    description: "Compare protein structures and identify similarities/differences",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you compare the structures of human and mouse p53 proteins?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll compare the structural features of human and mouse p53 proteins, including sequence alignment and structural differences.",
                    action: "COMPARE_PROTEIN_STRUCTURES"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const compareKeywords = [
            "compare", "comparison", "similarity", "difference", "align",
            "versus", "vs", "against", "between"
        ];
        
        const proteinKeywords = ["protein", "structure", "sequence"];
        
        // Check if message mentions multiple proteins
        const proteinMentions = (text.match(/protein|p53|egfr|insulin|hemoglobin/gi) || []).length;
        
        return compareKeywords.some(keyword => text.includes(keyword)) &&
               proteinKeywords.some(keyword => text.includes(keyword)) &&
               proteinMentions >= 2;
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const proteins = await extractMultipleProteins(message.content.text);
            
            if (proteins.length < 2) {
                const errorText = "I need at least two proteins to perform a comparison. Please specify two protein names, UniProt IDs, or sequences.";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Insufficient proteins" } });
                }
                
                return { text: errorText, data: { error: "Insufficient proteins" } };
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
                    action: 'compare_proteins',
                    parameters: {
                        protein1_id: proteins[0].uniprotId,
                        protein2_id: proteins[1].uniprotId,
                        protein1_name: proteins[0].name,
                        protein2_name: proteins[1].name,
                        comparison_type: 'structural',
                        include_sequence_alignment: true,
                        include_functional_comparison: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const comparison = result.result;
                const responseText = formatProteinComparisonResponse(comparison, proteins);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: comparison
                    });
                }
                
                return {
                    text: responseText,
                    data: comparison
                };
            } else {
                const errorText = "I couldn't complete the protein comparison. Please check that the protein names or IDs are valid.";
                
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
            const errorText = "I'm experiencing issues with protein comparison. Please try again shortly.";
            
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

export const proteinActions = [
    analyzeProteinStructureAction,
    predictProteinFunctionAction,
    compareProteinStructuresAction
];

// Helper Functions
async function extractProteinInformation(text: string): Promise<{
    sequence?: string;
    uniprotId?: string;
    name?: string;
}> {
    const result: any = {};
    
    // Extract protein sequence (amino acid sequence)
    const sequencePattern = /[ACDEFGHIKLMNPQRSTVWY]{10,}/gi;
    const sequenceMatch = text.match(sequencePattern);
    if (sequenceMatch) {
        result.sequence = sequenceMatch[0];
    }
    
    // Extract UniProt ID pattern
    const uniprotPattern = /[A-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z]{1,2}[0-9]{5}/gi;
    const uniprotMatch = text.match(uniprotPattern);
    if (uniprotMatch) {
        result.uniprotId = uniprotMatch[0];
    }
    
    // Extract common protein names
    const proteinNames = [
        'p53', 'tp53', 'egfr', 'her2', 'brca1', 'brca2', 'insulin',
        'hemoglobin', 'myosin', 'actin', 'tubulin', 'collagen',
        'albumin', 'immunoglobulin', 'cytochrome c', 'lysozyme'
    ];
    
    const lowerText = text.toLowerCase();
    for (const proteinName of proteinNames) {
        if (lowerText.includes(proteinName)) {
            result.name = proteinName;
            break;
        }
    }
    
    // Extract protein names with "protein" keyword
    const proteinNamePattern = /(\w+)\s+protein/gi;
    const proteinNameMatch = text.match(proteinNamePattern);
    if (proteinNameMatch && !result.name) {
        result.name = proteinNameMatch[0];
    }
    
    return result;
}

async function extractMultipleProteins(text: string): Promise<Array<{
    sequence?: string;
    uniprotId?: string;
    name?: string;
}>> {
    const proteins = [];
    
    // Split text by common separators
    const segments = text.split(/\s+(?:and|vs|versus|compared to|against)\s+/gi);
    
    for (const segment of segments) {
        const proteinInfo = await extractProteinInformation(segment);
        if (proteinInfo.sequence || proteinInfo.uniprotId || proteinInfo.name) {
            proteins.push(proteinInfo);
        }
    }
    
    // If we didn't find multiple proteins, try to extract from the whole text
    if (proteins.length < 2) {
        const proteinNames = [
            'p53', 'egfr', 'her2', 'brca1', 'brca2', 'insulin',
            'hemoglobin', 'myosin', 'actin', 'human', 'mouse'
        ];
        
        const lowerText = text.toLowerCase();
        const foundProteins = [];
        
        for (const name of proteinNames) {
            if (lowerText.includes(name)) {
                foundProteins.push({ name });
            }
        }
        
        return foundProteins.slice(0, 2); // Return first two found
    }
    
    return proteins;
}

function formatProteinAnalysisResponse(analysis: any, proteinInfo: any): string {
    let response = `🧬 **Protein Structure Analysis Complete**\n\n`;
    
    if (proteinInfo.name) {
        response += `**Protein:** ${proteinInfo.name}\n`;
    }
    
    if (proteinInfo.uniprotId) {
        response += `**UniProt ID:** ${proteinInfo.uniprotId}\n`;
    }
    
    if (analysis.sequence_length) {
        response += `**Sequence Length:** ${analysis.sequence_length} amino acids\n`;
    }
    
    response += `\n`;
    
    // Structure prediction results
    if (analysis.structure_prediction) {
        const structure = analysis.structure_prediction;
        response += `**Structure Prediction:**\n`;
        response += `• Method: ${structure.method || 'ESMFold'}\n`;
        response += `• Confidence: ${(structure.global_confidence * 100).toFixed(1)}%\n`;
        
        if (structure.secondary_structure_fraction) {
            const ss = structure.secondary_structure_fraction;
            response += `• Secondary Structure: ${(ss[0] * 100).toFixed(1)}% helix, ${(ss[1] * 100).toFixed(1)}% sheet, ${(ss[2] * 100).toFixed(1)}% coil\n`;
        }
        
        response += `\n`;
    }
    
    // Functional analysis
    if (analysis.functional_analysis) {
        const func = analysis.functional_analysis;
        response += `**Functional Analysis:**\n`;
        
        if (func.predicted_function) {
            response += `• Primary Function: ${func.predicted_function}\n`;
        }
        
        if (func.localization) {
            response += `• Cellular Localization: ${func.localization.predicted_localization}\n`;
        }
        
        if (func.domains && func.domains.length > 0) {
            response += `• Domains: ${func.domains.slice(0, 3).map((d: any) => d.name).join(', ')}\n`;
        }
        
        response += `\n`;
    }
    
    // Binding sites
    if (analysis.predicted_binding_sites && analysis.predicted_binding_sites.length > 0) {
        response += `**Predicted Binding Sites:**\n`;
        analysis.predicted_binding_sites.slice(0, 3).forEach((site: any, index: number) => {
            response += `${index + 1}. ${site.name || `Site ${index + 1}`}\n`;
            response += `   • Type: ${site.site_type}\n`;
            response += `   • Druggability: ${(site.druggability_score * 100).toFixed(1)}%\n`;
            if (site.volume) {
                response += `   • Volume: ${site.volume.toFixed(1)} Ų\n`;
            }
        });
        response += `\n`;
    }
    
    // Quality metrics
    if (analysis.structure_prediction && analysis.structure_prediction.quality_metrics) {
        const quality = analysis.structure_prediction.quality_metrics;
        response += `**Quality Assessment:**\n`;
        response += `• Overall Quality: ${getQualityRating(quality.mean_confidence)}\n`;
        response += `• Clash Score: ${(quality.clash_score * 100).toFixed(1)}%\n`;
        
        if (quality.ramachandran_favored) {
            response += `• Ramachandran Favored: ${(quality.ramachandran_favored * 100).toFixed(1)}%\n`;
        }
    }
    
    return response;
}

function formatFunctionPredictionResponse(analysis: any, proteinInfo: any): string {
    let response = `🔬 **Protein Function Prediction**\n\n`;
    
    if (proteinInfo.name) {
        response += `**Protein:** ${proteinInfo.name}\n\n`;
    }
    
    if (analysis.functional_analysis) {
        const func = analysis.functional_analysis;
        
        if (func.predicted_function) {
            response += `**Primary Function:** ${func.predicted_function}\n\n`;
        }
        
        if (func.go_terms) {
            response += `**Gene Ontology Terms:**\n`;
            if (func.go_terms.molecular_function) {
                response += `• Molecular Function: ${func.go_terms.molecular_function.slice(0, 3).join(', ')}\n`;
            }
            if (func.go_terms.biological_process) {
                response += `• Biological Process: ${func.go_terms.biological_process.slice(0, 3).join(', ')}\n`;
            }
            if (func.go_terms.cellular_component) {
                response += `• Cellular Component: ${func.go_terms.cellular_component.slice(0, 3).join(', ')}\n`;
            }
            response += `\n`;
        }
        
        if (func.localization) {
            response += `**Subcellular Localization:**\n`;
            response += `• Predicted Location: ${func.localization.predicted_localization}\n`;
            response += `• Confidence: ${(func.localization.confidence * 100).toFixed(1)}%\n\n`;
        }
        
        if (func.pathways) {
            response += `**Associated Pathways:**\n`;
            func.pathways.slice(0, 5).forEach((pathway: string, index: number) => {
                response += `${index + 1}. ${pathway}\n`;
            });
            response += `\n`;
        }
    }
    
    if (analysis.sequence_properties) {
        const props = analysis.sequence_properties;
        response += `**Sequence Properties:**\n`;
        response += `• Length: ${props.length} amino acids\n`;
        response += `• Molecular Weight: ${(props.molecular_weight / 1000).toFixed(1)} kDa\n`;
        response += `• Isoelectric Point: ${props.isoelectric_point.toFixed(2)}\n`;
        response += `• Instability Index: ${props.instability_index.toFixed(1)}\n`;
    }
    
    return response;
}

function formatProteinComparisonResponse(comparison: any, proteins: any[]): string {
    let response = `⚖️ **Protein Structure Comparison**\n\n`;
    
    response += `**Comparing:** ${proteins[0].name || 'Protein 1'} vs ${proteins[1].name || 'Protein 2'}\n\n`;
    
    if (comparison.sequence_similarity) {
        response += `**Sequence Similarity:** ${(comparison.sequence_similarity * 100).toFixed(1)}%\n\n`;
    }
    
    if (comparison.structural_comparison) {
        const struct = comparison.structural_comparison;
        response += `**Structural Comparison:**\n`;
        response += `• Length Difference: ${struct.length_difference} residues\n`;
        response += `• Molecular Weight Ratio: ${struct.molecular_weight_ratio.toFixed(2)}\n`;
        
        if (struct.rmsd) {
            response += `• Structural RMSD: ${struct.rmsd.toFixed(2)} Ų\n`;
        }
        
        response += `\n`;
    }
    
    if (comparison.functional_differences) {
        response += `**Functional Differences:**\n`;
        comparison.functional_differences.slice(0, 3).forEach((diff: string, index: number) => {
            response += `${index + 1}. ${diff}\n`;
        });
        response += `\n`;
    }
    
    if (comparison.conservation_analysis) {
        const conservation = comparison.conservation_analysis;
        response += `**Conservation Analysis:**\n`;
        response += `• Conserved Regions: ${conservation.conserved_regions || 'Multiple'}\n`;
        response += `• Variable Regions: ${conservation.variable_regions || 'Multiple'}\n`;
        response += `• Functional Conservation: ${conservation.functional_conservation || 'High'}\n`;
    }
    
    return response;
}

function getQualityRating(confidence: number): string {
    if (confidence >= 0.9) return "Excellent";
    if (confidence >= 0.7) return "Good";
    if (confidence >= 0.5) return "Moderate";
    return "Low";
}
