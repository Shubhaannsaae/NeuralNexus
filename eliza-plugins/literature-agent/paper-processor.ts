/**
 * Scientific Paper Processing Actions - Production Implementation
 * Real PubMed integration and paper analysis capabilities
 */

import { 
    Action, 
    IAgentRuntime, 
    Memory, 
    State, 
    HandlerCallback,
    ActionExample
} from "@ai16z/eliza";

// Search Scientific Literature Action
export const searchLiteratureAction: Action = {
    name: "SEARCH_LITERATURE",
    similes: [
        "search literature",
        "find papers",
        "search pubmed",
        "literature search",
        "find research",
        "search scientific papers"
    ],
    description: "Search scientific literature databases for research papers and publications",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you search for recent papers on CRISPR gene editing?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll search the scientific literature for recent publications on CRISPR gene editing technology.",
                    action: "SEARCH_LITERATURE"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Find research papers about Alzheimer's disease biomarkers"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll search for research papers focusing on biomarkers for Alzheimer's disease diagnosis and monitoring.",
                    action: "SEARCH_LITERATURE"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const searchKeywords = [
            "search", "find", "look for", "literature", "papers", "research",
            "publications", "studies", "articles", "pubmed"
        ];
        
        const scientificKeywords = [
            "protein", "gene", "drug", "disease", "therapy", "treatment",
            "clinical", "molecular", "cellular", "biological", "medical"
        ];
        
        const hasSearchKeyword = searchKeywords.some(keyword => text.includes(keyword));
        const hasScientificKeyword = scientificKeywords.some(keyword => text.includes(keyword));
        
        return hasSearchKeyword && (hasScientificKeyword || text.includes("paper") || text.includes("study"));
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            // Extract search query and parameters
            const searchQuery = extractSearchQuery(message.content.text);
            const searchParams = extractSearchParameters(message.content.text);
            
            if (!searchQuery) {
                const errorText = "I need a specific research topic to search for. Could you provide keywords or a research question?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing search query" } });
                }
                
                return { text: errorText, data: { error: "Missing search query" } };
            }
            
            // Call backend API for literature search
            const apiUrl = process.env.NEUROGRAPH_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/v1/eliza`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    agent_type: 'literature-agent',
                    action: 'search_literature',
                    parameters: {
                        query: searchQuery,
                        max_results: searchParams.maxResults,
                        date_range: searchParams.dateRange,
                        journal_filter: searchParams.journalFilter,
                        study_type: searchParams.studyType,
                        include_abstracts: true,
                        include_mesh_terms: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const searchResults = result.result;
                const responseText = formatLiteratureSearchResponse(searchResults, searchQuery);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: searchResults
                    });
                }
                
                return {
                    text: responseText,
                    data: searchResults
                };
            } else {
                const errorText = "I couldn't find relevant papers for your search. Try using different keywords or broadening your search terms.";
                
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
            const errorText = "I'm experiencing technical difficulties with literature search. Please try again in a moment.";
            
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

// Analyze Research Paper Action
export const analyzePaperAction: Action = {
    name: "ANALYZE_PAPER",
    similes: [
        "analyze paper",
        "summarize paper",
        "paper analysis",
        "research summary",
        "paper review",
        "study analysis"
    ],
    description: "Analyze and summarize scientific research papers",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Can you analyze this paper: PMID 12345678?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the research paper with PMID 12345678 and provide a comprehensive summary of its findings.",
                    action: "ANALYZE_PAPER"
                }
            }
        ],
        [
            {
                user: "{{user1}}",
                content: {
                    text: "Summarize the key findings from the Nature paper on protein folding"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze the Nature paper on protein folding and summarize its key findings and implications.",
                    action: "ANALYZE_PAPER"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const analysisKeywords = [
            "analyze", "summarize", "review", "summary", "analysis",
            "findings", "results", "conclusions"
        ];
        
        const paperKeywords = [
            "paper", "study", "article", "publication", "research",
            "pmid", "doi", "journal"
        ];
        
        const hasAnalysisKeyword = analysisKeywords.some(keyword => text.includes(keyword));
        const hasPaperKeyword = paperKeywords.some(keyword => text.includes(keyword));
        
        // Check for PMID or DOI patterns
        const pmidPattern = /pmid[:\s]*\d+/i;
        const doiPattern = /doi[:\s]*10\.\d+/i;
        const hasIdentifier = pmidPattern.test(text) || doiPattern.test(text);
        
        return (hasAnalysisKeyword && hasPaperKeyword) || hasIdentifier;
    },
    
    handler: async (
        runtime: IAgentRuntime, 
        message: Memory, 
        state: State,
        options: any,
        callback?: HandlerCallback
    ) => {
        try {
            const paperIdentifier = extractPaperIdentifier(message.content.text);
            const analysisType = extractAnalysisType(message.content.text);
            
            if (!paperIdentifier.pmid && !paperIdentifier.doi && !paperIdentifier.title) {
                const errorText = "I need a paper identifier (PMID, DOI) or title to analyze. Could you provide one of these?";
                
                if (callback) {
                    callback({ text: errorText, data: { error: "Missing paper identifier" } });
                }
                
                return { text: errorText, data: { error: "Missing paper identifier" } };
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
                    action: 'analyze_paper',
                    parameters: {
                        pmid: paperIdentifier.pmid,
                        doi: paperIdentifier.doi,
                        title: paperIdentifier.title,
                        analysis_type: analysisType,
                        include_methodology: true,
                        include_statistics: true,
                        include_limitations: true,
                        extract_entities: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const analysis = result.result;
                const responseText = formatPaperAnalysisResponse(analysis, paperIdentifier);
                
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
                const errorText = "I couldn't analyze this paper. It might not be available in the database or the identifier might be incorrect.";
                
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
            const errorText = "I'm having trouble analyzing the paper. Please try again later.";
            
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

// Track Research Trends Action
export const trackResearchTrendsAction: Action = {
    name: "TRACK_RESEARCH_TRENDS",
    similes: [
        "research trends",
        "trending research",
        "hot topics",
        "emerging research",
        "research patterns",
        "publication trends"
    ],
    description: "Track and analyze research trends and emerging topics in scientific literature",
    examples: [
        [
            {
                user: "{{user1}}",
                content: {
                    text: "What are the current research trends in immunotherapy?"
                }
            },
            {
                user: "{{user2}}",
                content: {
                    text: "I'll analyze recent publication patterns to identify current trends and emerging topics in immunotherapy research.",
                    action: "TRACK_RESEARCH_TRENDS"
                }
            }
        ]
    ] as ActionExample[][],
    
    validate: async (runtime: IAgentRuntime, message: Memory) => {
        const text = message.content.text.toLowerCase();
        const trendKeywords = [
            "trend", "trending", "hot", "emerging", "current", "recent",
            "latest", "new", "novel", "breakthrough", "cutting-edge"
        ];
        
        const researchKeywords = [
            "research", "study", "field", "area", "topic", "science",
            "discovery", "advancement", "development"
        ];
        
        return trendKeywords.some(keyword => text.includes(keyword)) &&
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
            const researchArea = extractResearchArea(message.content.text);
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
                    action: 'track_research_trends',
                    parameters: {
                        research_area: researchArea,
                        time_frame: timeFrame,
                        trend_analysis_depth: 'comprehensive',
                        include_citation_analysis: true,
                        include_keyword_evolution: true,
                        include_collaboration_networks: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                const trends = result.result;
                const responseText = formatResearchTrendsResponse(trends, researchArea, timeFrame);
                
                if (callback) {
                    callback({
                        text: responseText,
                        data: trends
                    });
                }
                
                return {
                    text: responseText,
                    data: trends
                };
            } else {
                const errorText = "I couldn't analyze research trends for this area. Try specifying a more specific research field.";
                
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
            const errorText = "I'm experiencing issues with trend analysis. Please try again shortly.";
            
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

export const literatureActions = [
    searchLiteratureAction,
    analyzePaperAction,
    trackResearchTrendsAction
];

// Helper Functions
function extractSearchQuery(text: string): string {
    // Remove common search prefixes
    let query = text.toLowerCase();
    
    const searchPrefixes = [
        'search for', 'find', 'look for', 'search', 'papers on',
        'research on', 'studies on', 'literature on'
    ];
    
    for (const prefix of searchPrefixes) {
        if (query.includes(prefix)) {
            query = query.split(prefix)[1]?.trim() || query;
            break;
        }
    }
    
    // Clean up the query
    query = query.replace(/[^\w\s-]/g, ' ').trim();
    
    // Extract meaningful terms
    const stopWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];
    const words = query.split(/\s+/).filter(word => 
        word.length > 2 && !stopWords.includes(word)
    );
    
    return words.join(' ');
}

function extractSearchParameters(text: string): {
    maxResults: number;
    dateRange: string;
    journalFilter?: string;
    studyType?: string;
} {
    const params: any = {
        maxResults: 20,
        dateRange: 'last_5_years'
    };
    
    // Extract number of results
    const numberMatch = text.match(/(\d+)\s*(papers?|results?|studies?)/i);
    if (numberMatch) {
        params.maxResults = Math.min(parseInt(numberMatch[1]), 100);
    }
    
    // Extract date range
    if (text.includes('recent') || text.includes('latest')) {
        params.dateRange = 'last_2_years';
    } else if (text.includes('last year')) {
        params.dateRange = 'last_1_year';
    } else if (text.includes('last decade')) {
        params.dateRange = 'last_10_years';
    }
    
    // Extract journal filter
    const journals = ['nature', 'science', 'cell', 'nejm', 'lancet', 'pnas'];
    for (const journal of journals) {
        if (text.toLowerCase().includes(journal)) {
            params.journalFilter = journal;
            break;
        }
    }
    
    // Extract study type
    const studyTypes = ['clinical trial', 'review', 'meta-analysis', 'case study'];
    for (const studyType of studyTypes) {
        if (text.toLowerCase().includes(studyType)) {
            params.studyType = studyType.replace(' ', '_');
            break;
        }
    }
    
    return params;
}

function extractPaperIdentifier(text: string): {
    pmid?: string;
    doi?: string;
    title?: string;
} {
    const identifier: any = {};
    
    // Extract PMID
    const pmidMatch = text.match(/pmid[:\s]*(\d+)/i);
    if (pmidMatch) {
        identifier.pmid = pmidMatch[1];
    }
    
    // Extract DOI
    const doiMatch = text.match(/doi[:\s]*(10\.\d+\/[^\s]+)/i);
    if (doiMatch) {
        identifier.doi = doiMatch[1];
    }
    
    // Extract title (if quoted or after "paper on")
    const titleMatch = text.match(/"([^"]+)"|paper on (.+?)(?:\s|$)/i);
    if (titleMatch) {
        identifier.title = titleMatch[1] || titleMatch[2];
    }
    
    return identifier;
}

function extractAnalysisType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('methodology') || lowerText.includes('methods')) {
        return 'methodology';
    } else if (lowerText.includes('results') || lowerText.includes('findings')) {
        return 'results';
    } else if (lowerText.includes('conclusion') || lowerText.includes('implications')) {
        return 'conclusions';
    } else if (lowerText.includes('statistical') || lowerText.includes('statistics')) {
        return 'statistics';
    }
    
    return 'comprehensive';
}

function extractResearchArea(text: string): string {
    const researchAreas = [
        'immunotherapy', 'gene therapy', 'crispr', 'cancer', 'alzheimer',
        'parkinson', 'diabetes', 'cardiovascular', 'neuroscience',
        'artificial intelligence', 'machine learning', 'drug discovery',
        'protein folding', 'genomics', 'proteomics', 'bioinformatics'
    ];
    
    const lowerText = text.toLowerCase();
    
    for (const area of researchAreas) {
        if (lowerText.includes(area)) {
            return area;
        }
    }
    
    // Extract area after "in" or "on"
    const areaMatch = text.match(/(?:in|on)\s+([a-zA-Z\s]+?)(?:\s|$)/i);
    if (areaMatch) {
        return areaMatch[1].trim();
    }
    
    return 'biomedical research';
}

function extractTimeFrame(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('last year') || lowerText.includes('past year')) {
        return 'last_1_year';
    } else if (lowerText.includes('last 2 years') || lowerText.includes('past 2 years')) {
        return 'last_2_years';
    } else if (lowerText.includes('last 5 years') || lowerText.includes('past 5 years')) {
        return 'last_5_years';
    } else if (lowerText.includes('decade')) {
        return 'last_10_years';
    } else if (lowerText.includes('recent') || lowerText.includes('current')) {
        return 'last_2_years';
    }
    
    return 'last_3_years';
}

function formatLiteratureSearchResponse(searchResults: any, query: string): string {
    let response = `📚 **Literature Search Results**\n\n`;
    
    response += `**Search Query:** "${query}"\n`;
    response += `**Results Found:** ${searchResults.total_results || 0}\n\n`;
    
    if (!searchResults.results || searchResults.results.length === 0) {
        response += "No papers found matching your search criteria. Try using different keywords or broadening your search.";
        return response;
    }
    
    response += `**Top Research Papers:**\n\n`;
    
    searchResults.results.slice(0, 5).forEach((paper: any, index: number) => {
        response += `**${index + 1}. ${paper.title}**\n`;
        
        if (paper.authors && paper.authors.length > 0) {
            const authorList = paper.authors.slice(0, 3).join(', ');
            const moreAuthors = paper.authors.length > 3 ? ` et al.` : '';
            response += `*Authors:* ${authorList}${moreAuthors}\n`;
        }
        
        if (paper.journal) {
            response += `*Journal:* ${paper.journal}`;
            if (paper.publication_date) {
                response += ` (${paper.publication_date})`;
            }
            response += `\n`;
        }
        
        if (paper.pubmed_id) {
            response += `*PMID:* ${paper.pubmed_id}\n`;
        }
        
        if (paper.abstract) {
            const shortAbstract = paper.abstract.length > 200 
                ? paper.abstract.substring(0, 200) + "..."
                : paper.abstract;
            response += `*Abstract:* ${shortAbstract}\n`;
        }
        
        if (paper.relevance_score) {
            response += `*Relevance:* ${(paper.relevance_score * 100).toFixed(1)}%\n`;
        }
        
        response += `\n`;
    });
    
    if (searchResults.results.length > 5) {
        response += `*... and ${searchResults.results.length - 5} more papers found*\n\n`;
    }
    
    // Add search suggestions
    if (searchResults.suggested_terms && searchResults.suggested_terms.length > 0) {
        response += `**Related Search Terms:** ${searchResults.suggested_terms.join(', ')}`;
    }
    
    return response;
}

function formatPaperAnalysisResponse(analysis: any, paperIdentifier: any): string {
    let response = `📄 **Research Paper Analysis**\n\n`;
    
    if (analysis.paper_info) {
        const paper = analysis.paper_info;
        response += `**Title:** ${paper.title}\n`;
        
        if (paper.authors && paper.authors.length > 0) {
            response += `**Authors:** ${paper.authors.slice(0, 5).join(', ')}`;
            if (paper.authors.length > 5) response += ` et al.`;
            response += `\n`;
        }
        
        if (paper.journal) {
            response += `**Journal:** ${paper.journal}`;
            if (paper.publication_date) {
                response += ` (${paper.publication_date})`;
            }
            response += `\n`;
        }
        
        if (paperIdentifier.pmid) {
            response += `**PMID:** ${paperIdentifier.pmid}\n`;
        }
        
        response += `\n`;
    }
    
    if (analysis.summary) {
        response += `**Summary:**\n${analysis.summary}\n\n`;
    }
    
    if (analysis.key_findings && analysis.key_findings.length > 0) {
        response += `**Key Findings:**\n`;
        analysis.key_findings.slice(0, 5).forEach((finding: string, index: number) => {
            response += `${index + 1}. ${finding}\n`;
        });
        response += `\n`;
    }
    
    if (analysis.methodology) {
        response += `**Methodology:**\n`;
        response += `• Study Type: ${analysis.methodology.study_type || 'Not specified'}\n`;
        response += `• Sample Size: ${analysis.methodology.sample_size || 'Not specified'}\n`;
        response += `• Methods: ${analysis.methodology.methods || 'See full paper'}\n\n`;
    }
    
    if (analysis.statistical_significance) {
        const stats = analysis.statistical_significance;
        response += `**Statistical Analysis:**\n`;
        response += `• Primary Endpoint: ${stats.primary_endpoint || 'Not specified'}\n`;
        response += `• P-value: ${stats.p_value || 'Not reported'}\n`;
        response += `• Confidence Interval: ${stats.confidence_interval || 'Not reported'}\n\n`;
    }
    
    if (analysis.clinical_relevance) {
        response += `**Clinical Relevance:**\n${analysis.clinical_relevance}\n\n`;
    }
    
    if (analysis.limitations && analysis.limitations.length > 0) {
        response += `**Study Limitations:**\n`;
        analysis.limitations.slice(0, 3).forEach((limitation: string, index: number) => {
            response += `${index + 1}. ${limitation}\n`;
        });
        response += `\n`;
    }
    
    if (analysis.extracted_entities) {
        const entities = analysis.extracted_entities;
        if (entities.proteins && entities.proteins.length > 0) {
            response += `**Proteins Mentioned:** ${entities.proteins.slice(0, 5).join(', ')}\n`;
        }
        if (entities.drugs && entities.drugs.length > 0) {
            response += `**Drugs/Compounds:** ${entities.drugs.slice(0, 5).join(', ')}\n`;
        }
        if (entities.diseases && entities.diseases.length > 0) {
            response += `**Diseases:** ${entities.diseases.slice(0, 5).join(', ')}\n`;
        }
    }
    
    return response;
}

function formatResearchTrendsResponse(trends: any, researchArea: string, timeFrame: string): string {
    let response = `📈 **Research Trends Analysis**\n\n`;
    
    response += `**Research Area:** ${researchArea}\n`;
    response += `**Time Frame:** ${timeFrame.replace('_', ' ')}\n`;
    response += `**Analysis Date:** ${new Date().toLocaleDateString()}\n\n`;
    
    if (trends.trending_topics && trends.trending_topics.length > 0) {
        response += `**Trending Topics:**\n`;
        trends.trending_topics.slice(0, 5).forEach((topic: any, index: number) => {
            response += `${index + 1}. **${topic.name}**\n`;
            response += `   • Growth Rate: ${(topic.growth_rate * 100).toFixed(1)}%\n`;
            response += `   • Publications: ${topic.publication_count}\n`;
            if (topic.key_terms) {
                response += `   • Key Terms: ${topic.key_terms.slice(0, 3).join(', ')}\n`;
            }
            response += `\n`;
        });
    }
    
    if (trends.emerging_keywords && trends.emerging_keywords.length > 0) {
        response += `**Emerging Keywords:**\n`;
        response += `${trends.emerging_keywords.slice(0, 10).join(', ')}\n\n`;
    }
    
    if (trends.hot_papers && trends.hot_papers.length > 0) {
        response += `**Highly Cited Recent Papers:**\n`;
        trends.hot_papers.slice(0, 3).forEach((paper: any, index: number) => {
            response += `${index + 1}. ${paper.title}\n`;
            response += `   • Citations: ${paper.citation_count}\n`;
            response += `   • Journal: ${paper.journal}\n`;
            if (paper.pmid) {
                response += `   • PMID: ${paper.pmid}\n`;
            }
            response += `\n`;
        });
    }
    
    if (trends.research_institutions && trends.research_institutions.length > 0) {
        response += `**Leading Research Institutions:**\n`;
        trends.research_institutions.slice(0, 5).forEach((institution: any, index: number) => {
            response += `${index + 1}. ${institution.name} (${institution.publication_count} papers)\n`;
        });
        response += `\n`;
    }
    
    if (trends.collaboration_networks) {
        const collab = trends.collaboration_networks;
        response += `**Collaboration Insights:**\n`;
        response += `• International Collaborations: ${(collab.international_percentage * 100).toFixed(1)}%\n`;
        response += `• Average Authors per Paper: ${collab.avg_authors_per_paper.toFixed(1)}\n`;
        response += `• Cross-disciplinary Research: ${(collab.interdisciplinary_percentage * 100).toFixed(1)}%\n\n`;
    }
    
    if (trends.future_predictions && trends.future_predictions.length > 0) {
        response += `**Predicted Future Directions:**\n`;
        trends.future_predictions.slice(0, 3).forEach((prediction: string, index: number) => {
            response += `${index + 1}. ${prediction}\n`;
        });
    }
    
    return response;
}
