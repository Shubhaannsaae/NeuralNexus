import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Mock protein search
    const mockProteins = [
      { id: "P04637", name: "Tumor protein p53", organism: "Homo sapiens", length: 393 },
      { id: "P38398", name: "BRCA1", organism: "Homo sapiens", length: 1863 },
    ];
    
    return NextResponse.json({ results: mockProteins });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to search proteins' }, { status: 500 });
  }
}
