import { NextResponse } from 'next/server';

export async function POST(req: Request) {
    try {
        const { sequence } = await req.json();

        // The LSTM expects 2 features: [voltage, current].
        // We map your array of numbers into pairs. 
        // If the UI only sends voltage, we'll pair it with a default current (e.g., 0.45A)
        const formattedData = sequence.map((val: number) => [val, 0.45]);

        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sequence: formattedData }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json({ error: `Python Server Error: ${errorText}` }, { status: 500 });
        }

        const data = await response.json();

        // Send back the prediction to the Dashboard
        return NextResponse.json({
            prediction: data.prediction,
            status: data.status
        });

    } catch (error) {
        console.error("Inference route error:", error);
        return NextResponse.json({ error: "Could not connect to Python backend" }, { status: 500 });
    }
}