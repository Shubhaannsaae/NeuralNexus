import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  try {
    const mockData = {
      nodes: [
        { id: 'protein1', name: 'p53', type: 'protein' },
        { id: 'drug1', name: 'Aspirin', type: 'drug' },
      ],
      edges: [
        { source: 'protein1', target: 'drug1', type: 'targets' }
      ]
    };
    
    return NextResponse.json(mockData);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch graph data' }, { status: 500 });
  }
}
