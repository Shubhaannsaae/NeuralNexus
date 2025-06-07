/**
 * Bio Agent Providers - Production Implementation
 * Data providers for biological information and context
 */

import { 
    Provider, 
    IAgentRuntime, 
    Memory, 
    State 
} from "@ai16z/eliza";

export const biologicalContextProvider: Provider = {
    name: "biological_context",
    description: "Provides biological context and background information",
    
    get: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        
        // Extract biological entities
        const entities = extractBiologicalEntities(text);
        
        // Build context based on entities
        let context = "Biological Context:\n";
        
        // Protein context
        const proteins = entities.filter(entity => isProtein(entity));
        if (proteins.length > 0) {
            context += `\nProteins mentioned: ${proteins.join(', ')}\n`;
            context += await getProteinContext(proteins);
        }
        
        // Disease context
        const diseases = entities.filter(entity => isDisease(entity));
        if (diseases.length > 0) {
            context += `\nDiseases mentioned: ${diseases.join(', ')}\n`;
            context += await getDiseaseContext(diseases);
        }
        
        // Pathway context
        const pathways = entities.filter(entity => isPathway(entity));
        if (pathways.length > 0) {
            context += `\nPathways mentioned: ${pathways.join(', ')}\n`;
            context += await getPathwayContext(pathways);
        }
        
        return context;
    }
};

export const drugInformationProvider: Provider = {
    name: "drug_information",
    description: "Provides drug and compound information",
    
    get: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        
        // Extract drug-related information
        const drugs = extractDrugEntities(text);
        
        if (drugs.length === 0) {
            return "";
        }
        
        let drugInfo = "Drug Information:\n";
        
        for (const drug of drugs.slice(0, 3)) { // Limit to 3 drugs
            drugInfo += `\n${drug.name}:\n`;
            
            const info = await getDrugInformation(drug.name);
            if (info) {
                drugInfo += `  • Class: ${info.class || 'Unknown'}\n`;
                drugInfo += `  • Mechanism: ${info.mechanism || 'Not specified'}\n`;
                drugInfo += `  • Indication: ${info.indication || 'Multiple'}\n`;
                
                if (info.targets && info.targets.length > 0) {
                    drugInfo += `  • Primary Targets: ${info.targets.slice(0, 3).join(', ')}\n`;
                }
                
                if (info.warnings && info.warnings.length > 0) {
                    drugInfo += `  • Key Warnings: ${info.warnings.slice(0, 2).join(', ')}\n`;
                }
            }
        }
        
        return drugInfo;
    }
};

export const pathwayInformationProvider: Provider = {
    name: "pathway_information",
    description: "Provides biological pathway and network information",
    
    get: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        
        // Extract pathway information
        const pathways = extractPathwayEntities(text);
        
        if (pathways.length === 0) {
            return "";
        }
        
        let pathwayInfo = "Pathway Information:\n";
        
        for (const pathway of pathways.slice(0, 2)) { // Limit to 2 pathways
            pathwayInfo += `\n${pathway}:\n`;
            
            const info = await getPathwayInformation(pathway);
            if (info) {
                pathwayInfo += `  • Function: ${info.function || 'Multiple cellular processes'}\n`;
                
                if (info.keyProteins && info.keyProteins.length > 0) {
                    pathwayInfo += `  • Key Proteins: ${info.keyProteins.slice(0, 4).join(', ')}\n`;
                }
                
                if (info.regulation) {
                    pathwayInfo += `  • Regulation: ${info.regulation}\n`;
                }
                
                if (info.diseaseAssociation && info.diseaseAssociation.length > 0) {
                    pathwayInfo += `  • Disease Links: ${info.diseaseAssociation.slice(0, 3).join(', ')}\n`;
                }
                
                if (info.therapeuticTargets && info.therapeuticTargets.length > 0) {
                    pathwayInfo += `  • Therapeutic Targets: ${info.therapeuticTargets.slice(0, 3).join(', ')}\n`;
                }
            }
        }
        
        return pathwayInfo;
    }
};

export const clinicalTrialProvider: Provider = {
    name: "clinical_trial_context",
    description: "Provides clinical trial and regulatory context",
    
    get: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
        const text = message.content.text.toLowerCase();
        
        // Check if clinical information is relevant
        const clinicalKeywords = [
            'clinical', 'trial', 'study', 'patient', 'treatment',
            'therapy', 'fda', 'approval', 'phase', 'efficacy'
        ];
        
        const hasClinicalContext = clinicalKeywords.some(keyword => text.includes(keyword));
        
        if (!hasClinicalContext) {
            return "";
        }
        
        let clinicalInfo = "Clinical Context:\n";
        
        // Extract drugs/treatments mentioned
        const treatments = extractTreatmentEntities(text);
        
        if (treatments.length > 0) {
            clinicalInfo += `\nTreatments mentioned: ${treatments.join(', ')}\n`;
            
            for (const treatment of treatments.slice(0, 2)) {
                const trialInfo = await getClinicalTrialInfo(treatment);
                if (trialInfo) {
                    clinicalInfo += `\n${treatment}:\n`;
                    clinicalInfo += `  • Development Stage: ${trialInfo.stage || 'Unknown'}\n`;
                    clinicalInfo += `  • Primary Indication: ${trialInfo.indication || 'Multiple'}\n`;
                    
                    if (trialInfo.activeTrials > 0) {
                        clinicalInfo += `  • Active Trials: ${trialInfo.activeTrials}\n`;
                    }
                    
                    if (trialInfo.approvalStatus) {
                        clinicalInfo += `  • Approval Status: ${trialInfo.approvalStatus}\n`;
                    }
                }
            }
        }
        
        // Add general clinical considerations
        if (text.includes('safety') || text.includes('adverse')) {
            clinicalInfo += `\nSafety Considerations:\n`;
            clinicalInfo += `  • Always consult healthcare providers for medical advice\n`;
            clinicalInfo += `  • Individual patient factors affect treatment decisions\n`;
            clinicalInfo += `  • Monitor for drug interactions and contraindications\n`;
        }
        
        return clinicalInfo;
    }
};

export const bioProviders = [
    biologicalContextProvider,
    drugInformationProvider,
    pathwayInformationProvider,
    clinicalTrialProvider
];

// Helper Functions
function extractBiologicalEntities(text: string): string[] {
    const bioEntities = [
        // Proteins
        'p53', 'egfr', 'her2', 'brca1', 'brca2', 'mtor', 'akt', 'pi3k',
        'ras', 'myc', 'pten', 'rb1', 'apc', 'vegf', 'pdgf', 'tgf-beta',
        // Diseases
        'cancer', 'alzheimer', 'parkinson', 'diabetes', 'hypertension',
        'stroke', 'heart disease', 'multiple sclerosis', 'arthritis',
        // Pathways
        'apoptosis', 'autophagy', 'glycolysis', 'cell cycle', 'dna repair',
        'protein synthesis', 'signal transduction', 'immune response'
    ];
    
    const lowerText = text.toLowerCase();
    return bioEntities.filter(entity => lowerText.includes(entity));
}

function isProtein(entity: string): boolean {
    const proteins = [
        'p53', 'egfr', 'her2', 'brca1', 'brca2', 'mtor', 'akt', 'pi3k',
        'ras', 'myc', 'pten', 'rb1', 'apc', 'vegf', 'pdgf', 'tgf-beta'
    ];
    return proteins.includes(entity.toLowerCase());
}

function isDisease(entity: string): boolean {
    const diseases = [
        'cancer', 'alzheimer', 'parkinson', 'diabetes', 'hypertension',
        'stroke', 'heart disease', 'multiple sclerosis', 'arthritis'
    ];
    return diseases.includes(entity.toLowerCase());
}

function isPathway(entity: string): boolean {
    const pathways = [
        'apoptosis', 'autophagy', 'glycolysis', 'cell cycle', 'dna repair',
        'protein synthesis', 'signal transduction', 'immune response'
    ];
    return pathways.includes(entity.toLowerCase());
}

function extractDrugEntities(text: string): Array<{name: string, type: string}> {
    const drugs = [
        'aspirin', 'ibuprofen', 'acetaminophen', 'metformin', 'insulin',
        'warfarin', 'statins', 'penicillin', 'amoxicillin', 'lisinopril',
        'amlodipine', 'omeprazole', 'sertraline', 'fluoxetine', 'lorazepam'
    ];
    
    const lowerText = text.toLowerCase();
    const foundDrugs = drugs.filter(drug => lowerText.includes(drug));
    
    return foundDrugs.map(drug => ({ name: drug, type: 'approved_drug' }));
}

function extractPathwayEntities(text: string): string[] {
    const pathways = [
        'mtor pathway', 'pi3k/akt pathway', 'mapk pathway', 'nf-kb pathway',
        'p53 pathway', 'wnt pathway', 'notch pathway', 'hedgehog pathway',
        'apoptosis', 'autophagy', 'cell cycle', 'dna repair',
        'glycolysis', 'oxidative phosphorylation', 'protein synthesis'
    ];
    
    const lowerText = text.toLowerCase();
    return pathways.filter(pathway => lowerText.includes(pathway));
}

function extractTreatmentEntities(text: string): string[] {
    const treatments = [
        'chemotherapy', 'immunotherapy', 'radiation therapy', 'surgery',
        'targeted therapy', 'hormone therapy', 'gene therapy', 'stem cell therapy'
    ];
    
    const lowerText = text.toLowerCase();
    return treatments.filter(treatment => lowerText.includes(treatment));
}

async function getProteinContext(proteins: string[]): Promise<string> {
    // In production, this would query a protein database
    const proteinInfo: Record<string, any> = {
        'p53': {
            function: 'Tumor suppressor protein',
            location: 'Nucleus',
            pathways: ['DNA damage response', 'Apoptosis', 'Cell cycle arrest']
        },
        'egfr': {
            function: 'Receptor tyrosine kinase',
            location: 'Cell membrane',
            pathways: ['Cell proliferation', 'Survival signaling', 'Migration']
        },
        'mtor': {
            function: 'Serine/threonine kinase',
            location: 'Cytoplasm',
            pathways: ['Protein synthesis', 'Autophagy', 'Cell growth']
        }
    };
    
    let context = "";
    for (const protein of proteins) {
        const info = proteinInfo[protein.toLowerCase()];
        if (info) {
            context += `  ${protein}: ${info.function} (${info.location})\n`;
            context += `    Pathways: ${info.pathways.join(', ')}\n`;
        }
    }
    
    return context;
}

async function getDiseaseContext(diseases: string[]): Promise<string> {
    const diseaseInfo: Record<string, any> = {
        'cancer': {
            type: 'Neoplastic disease',
            characteristics: 'Uncontrolled cell growth',
            keyPathways: ['p53', 'Cell cycle', 'Apoptosis', 'DNA repair']
        },
        'alzheimer': {
            type: 'Neurodegenerative disease',
            characteristics: 'Progressive cognitive decline',
            keyPathways: ['Amyloid processing', 'Tau pathology', 'Neuroinflammation']
        },
        'diabetes': {
            type: 'Metabolic disorder',
            characteristics: 'Impaired glucose regulation',
            keyPathways: ['Insulin signaling', 'Glucose metabolism', 'Pancreatic function']
        }
    };
    
    let context = "";
    for (const disease of diseases) {
        const info = diseaseInfo[disease.toLowerCase()];
        if (info) {
            context += `  ${disease}: ${info.type} - ${info.characteristics}\n`;
            context += `    Key pathways: ${info.keyPathways.join(', ')}\n`;
        }
    }
    
    return context;
}

async function getPathwayContext(pathways: string[]): Promise<string> {
    const pathwayInfo: Record<string, any> = {
        'apoptosis': {
            function: 'Programmed cell death',
            regulation: 'p53, Bcl-2 family proteins',
            diseases: ['Cancer', 'Neurodegeneration']
        },
        'autophagy': {
            function: 'Cellular recycling process',
            regulation: 'mTOR, AMPK, ULK1',
            diseases: ['Cancer', 'Neurodegeneration', 'Aging']
        },
        'cell cycle': {
            function: 'Cell division regulation',
            regulation: 'Cyclins, CDKs, p53',
            diseases: ['Cancer']
        }
    };
    
    let context = "";
    for (const pathway of pathways) {
        const info = pathwayInfo[pathway.toLowerCase()];
        if (info) {
            context += `  ${pathway}: ${info.function}\n`;
            context += `    Regulation: ${info.regulation}\n`;
            context += `    Disease relevance: ${info.diseases.join(', ')}\n`;
        }
    }
    
    return context;
}

async function getDrugInformation(drugName: string): Promise<any> {
    // In production, this would query drug databases
    const drugDatabase: Record<string, any> = {
        'aspirin': {
            class: 'NSAID',
            mechanism: 'COX inhibition',
            indication: 'Pain relief, anti-inflammatory, cardioprotection',
            targets: ['COX-1', 'COX-2'],
            warnings: ['GI bleeding', 'Reye syndrome in children']
        },
        'metformin': {
            class: 'Biguanide',
            mechanism: 'AMPK activation, gluconeogenesis inhibition',
            indication: 'Type 2 diabetes',
            targets: ['AMPK', 'Complex I'],
            warnings: ['Lactic acidosis', 'Kidney function monitoring']
        },
        'ibuprofen': {
            class: 'NSAID',
            mechanism: 'COX inhibition',
            indication: 'Pain relief, anti-inflammatory',
            targets: ['COX-1', 'COX-2'],
            warnings: ['GI bleeding', 'Cardiovascular risk']
        }
    };
    
    return drugDatabase[drugName.toLowerCase()];
}

async function getPathwayInformation(pathway: string): Promise<any> {
    const pathwayDatabase: Record<string, any> = {
        'mtor pathway': {
            function: 'Cell growth and metabolism regulation',
            keyProteins: ['mTOR', 'S6K1', 'eIF4E', 'AMPK'],
            regulation: 'Nutrients, growth factors, energy status',
            diseaseAssociation: ['Cancer', 'Diabetes', 'Aging'],
            therapeuticTargets: ['Rapamycin', 'Torin1', 'Metformin']
        },
        'apoptosis': {
            function: 'Programmed cell death',
            keyProteins: ['p53', 'Bcl-2', 'Bax', 'Caspases'],
            regulation: 'DNA damage, stress signals, growth factors',
            diseaseAssociation: ['Cancer', 'Neurodegeneration', 'Autoimmune'],
            therapeuticTargets: ['BCL-2 inhibitors', 'p53 activators']
        }
    };
    
    return pathwayDatabase[pathway.toLowerCase()];
}

async function getClinicalTrialInfo(treatment: string): Promise<any> {
    // In production, this would query ClinicalTrials.gov API
    const trialDatabase: Record<string, any> = {
        'immunotherapy': {
            stage: 'Various phases (I-III)',
            indication: 'Cancer, Autoimmune diseases',
            activeTrials: 1500,
            approvalStatus: 'Multiple approved agents'
        },
        'gene therapy': {
            stage: 'Mostly Phase I-II',
            indication: 'Genetic disorders, Cancer',
            activeTrials: 300,
            approvalStatus: 'Limited approvals'
        }
    };
    
    return trialDatabase[treatment.toLowerCase()];
}
