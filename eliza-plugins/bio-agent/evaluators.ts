/**
 * Bio Agent Evaluators - Production Implementation
 * Evaluators for biological analysis quality and relevance
 */

import { 
    Evaluator, 
    IAgentRuntime, 
    Memory, 
    State 
} from "@ai16z/eliza";

export const biologicalRelevanceEvaluator: Evaluator = {
    name: "BIOLOGICAL_RELEVANCE",
    similes: ["biological_relevance", "bio_relevance", "scientific_accuracy"],
    description: "Evaluates the biological relevance and scientific accuracy of responses",
    
    validate: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        const bioKeywords = [
            'protein', 'gene', 'pathway', 'enzyme', 'receptor',
            'molecule', 'drug', 'compound', 'interaction', 'binding',
            'disease', 'therapeutic', 'clinical', 'biological'
        ];
        
        return bioKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        let score = 0;
        let feedback = [];
        
        // Check for scientific terminology
        const scientificTerms = [
            'protein', 'enzyme', 'receptor', 'pathway', 'mechanism',
            'molecular', 'cellular', 'therapeutic', 'clinical', 'biomarker'
        ];
        
        const termCount = scientificTerms.filter(term => text.includes(term)).length;
        const termScore = Math.min(termCount / scientificTerms.length, 1) * 0.3;
        score += termScore;
        
        if (termScore > 0.2) {
            feedback.push("Good use of scientific terminology");
        }
        
        // Check for specific biological entities
        const specificEntities = [
            // Proteins
            'p53', 'egfr', 'her2', 'brca1', 'mtor', 'akt', 'ras',
            // Pathways
            'apoptosis', 'autophagy', 'glycolysis', 'cell cycle',
            // Diseases
            'cancer', 'alzheimer', 'parkinson', 'diabetes'
        ];
        
        const entityCount = specificEntities.filter(entity => text.includes(entity)).length;
        const entityScore = Math.min(entityCount / 3, 1) * 0.3;
        score += entityScore;
        
        if (entityScore > 0.1) {
            feedback.push("Mentions specific biological entities");
        }
        
        // Check for quantitative information
        const quantitativePatterns = [
            /\d+\.?\d*\s*(nm|μm|mm|mg|μg|ng)/gi, // Concentrations
            /\d+\.?\d*\s*%/gi, // Percentages
            /ic50|ec50|ki|kd/gi, // Binding constants
            /p\s*[<>=]\s*0\.\d+/gi // P-values
        ];
        
        const hasQuantitative = quantitativePatterns.some(pattern => pattern.test(text));
        if (hasQuantitative) {
            score += 0.2;
            feedback.push("Includes quantitative data");
        }
        
        // Check for proper scientific context
        const contextKeywords = [
            'study', 'research', 'analysis', 'experiment', 'trial',
            'evidence', 'data', 'results', 'findings', 'conclusion'
        ];
        
        const contextCount = contextKeywords.filter(keyword => text.includes(keyword)).length;
        const contextScore = Math.min(contextCount / contextKeywords.length, 1) * 0.2;
        score += contextScore;
        
        if (contextScore > 0.1) {
            feedback.push("Provides scientific context");
        }
        
        // Normalize score to 0-1 range
        score = Math.min(score, 1);
        
        return {
            score,
            feedback: feedback.join(", "),
            details: {
                scientificTerms: termScore,
                specificEntities: entityScore,
                quantitativeData: hasQuantitative ? 0.2 : 0,
                scientificContext: contextScore
            }
        };
    }
};

export const drugDiscoveryAccuracyEvaluator: Evaluator = {
    name: "DRUG_DISCOVERY_ACCURACY",
    similes: ["drug_accuracy", "pharmaceutical_accuracy", "medicinal_chemistry"],
    description: "Evaluates accuracy of drug discovery and medicinal chemistry information",
    
    validate: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        const drugKeywords = [
            'drug', 'compound', 'molecule', 'pharmaceutical', 'therapeutic',
            'target', 'binding', 'affinity', 'admet', 'toxicity',
            'clinical', 'trial', 'fda', 'approval'
        ];
        
        return drugKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        let score = 0;
        let feedback = [];
        
        // Check for drug development terminology
        const drugDevTerms = [
            'lead compound', 'hit compound', 'optimization', 'sar',
            'structure-activity relationship', 'pharmacokinetics', 'pharmacodynamics',
            'admet', 'toxicity', 'bioavailability', 'clearance'
        ];
        
        const drugTermCount = drugDevTerms.filter(term => text.includes(term)).length;
        const drugTermScore = Math.min(drugTermCount / drugDevTerms.length, 1) * 0.4;
        score += drugTermScore;
        
        if (drugTermScore > 0.2) {
            feedback.push("Uses appropriate drug development terminology");
        }
        
        // Check for specific drug targets
        const drugTargets = [
            'kinase', 'protease', 'receptor', 'channel', 'transporter',
            'gpcr', 'enzyme', 'antibody', 'protein-protein interaction'
        ];
        
        const targetCount = drugTargets.filter(target => text.includes(target)).length;
        const targetScore = Math.min(targetCount / 3, 1) * 0.3;
        score += targetScore;
        
        if (targetScore > 0.1) {
            feedback.push("Mentions relevant drug targets");
        }
        
        // Check for regulatory/clinical information
        const clinicalTerms = [
            'phase i', 'phase ii', 'phase iii', 'clinical trial',
            'fda', 'ema', 'approval', 'regulatory', 'safety',
            'efficacy', 'biomarker', 'endpoint'
        ];
        
        const clinicalCount = clinicalTerms.filter(term => text.includes(term)).length;
        const clinicalScore = Math.min(clinicalCount / clinicalTerms.length, 1) * 0.3;
        score += clinicalScore;
        
        if (clinicalScore > 0.1) {
            feedback.push("Includes clinical/regulatory context");
        }
        
        // Normalize score
        score = Math.min(score, 1);
        
        return {
            score,
            feedback: feedback.join(", "),
            details: {
                drugDevelopmentTerms: drugTermScore,
                drugTargets: targetScore,
                clinicalContext: clinicalScore
            }
        };
    }
};

export const pathwayAnalysisEvaluator: Evaluator = {
    name: "PATHWAY_ANALYSIS_QUALITY",
    similes: ["pathway_quality", "systems_biology", "network_analysis"],
    description: "Evaluates the quality of biological pathway and network analysis",
    
    validate: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        const pathwayKeywords = [
            'pathway', 'network', 'signaling', 'cascade', 'regulation',
            'upstream', 'downstream', 'feedback', 'crosstalk', 'hub'
        ];
        
        return pathwayKeywords.some(keyword => text.includes(keyword));
    },
    
    handler: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        let score = 0;
        let feedback = [];
        
        // Check for pathway terminology
        const pathwayTerms = [
            'signaling pathway', 'metabolic pathway', 'regulatory network',
            'protein interaction', 'gene expression', 'transcription',
            'translation', 'post-translational modification', 'phosphorylation'
        ];
        
        const pathwayTermCount = pathwayTerms.filter(term => text.includes(term)).length;
        const pathwayTermScore = Math.min(pathwayTermCount / pathwayTerms.length, 1) * 0.4;
        score += pathwayTermScore;
        
        if (pathwayTermScore > 0.2) {
            feedback.push("Uses appropriate pathway terminology");
        }
        
        // Check for systems biology concepts
        const systemsTerms = [
            'network topology', 'hub protein', 'bottleneck', 'modularity',
            'centrality', 'clustering', 'connectivity', 'robustness',
            'emergent property', 'systems-level'
        ];
        
        const systemsCount = systemsTerms.filter(term => text.includes(term)).length;
        const systemsScore = Math.min(systemsCount / systemsTerms.length, 1) * 0.3;
        score += systemsScore;
        
        if (systemsScore > 0.1) {
            feedback.push("Demonstrates systems biology understanding");
        }
        
        // Check for specific pathways
        const knownPathways = [
            'mtor', 'pi3k/akt', 'mapk', 'nf-kb', 'p53', 'wnt',
            'notch', 'hedgehog', 'jak/stat', 'tgf-beta',
            'apoptosis', 'autophagy', 'cell cycle', 'dna repair'
        ];
        
        const pathwayCount = knownPathways.filter(pathway => text.includes(pathway)).length;
        const specificPathwayScore = Math.min(pathwayCount / 3, 1) * 0.3;
        score += specificPathwayScore;
        
        if (specificPathwayScore > 0.1) {
            feedback.push("References specific biological pathways");
        }
        
        // Normalize score
        score = Math.min(score, 1);
        
        return {
            score,
            feedback: feedback.join(", "),
            details: {
                pathwayTerminology: pathwayTermScore,
                systemsBiology: systemsScore,
                specificPathways: specificPathwayScore
            }
        };
    }
};

export const bioEvaluators = [
    biologicalRelevanceEvaluator,
    drugDiscoveryAccuracyEvaluator,
    pathwayAnalysisEvaluator
];
