import { NextResponse } from 'next/server';

export async function POST(req: Request) {
    try {
        const { appliances } = await req.json();

        if (!appliances || !Array.isArray(appliances)) {
            return NextResponse.json({ error: "Invalid input: 'appliances' must be an array" }, { status: 400 });
        }

        const formattedAppliances = appliances
            .map((app: any) => {
                const applianceId =
                    typeof app?.appliance_id === 'string'
                        ? app.appliance_id.trim()
                        : typeof app?.id === 'string'
                            ? app.id.trim()
                            : '';

                if (!applianceId || !Array.isArray(app?.sequence)) {
                    return null;
                }

                const normalizedSequence = app.sequence
                    .map((point: any) => {
                        if (Array.isArray(point)) {
                            const voltage = Number(point[0]);
                            const current = Number(point[1]);
                            if (Number.isFinite(voltage) && Number.isFinite(current)) {
                                return [voltage, current];
                            }
                            return null;
                        }

                        const voltage = Number(point?.voltage);
                        const current = Number(point?.current);
                        if (Number.isFinite(voltage) && Number.isFinite(current)) {
                            return [voltage, current];
                        }

                        return null;
                    })
                    .filter((pair: any): pair is number[] => pair !== null);

                if (normalizedSequence.length === 0) {
                    return null;
                }

                return {
                    appliance_id: applianceId,
                    sequence: normalizedSequence,
                };
            })
            .filter((app: any) => app !== null);

        if (formattedAppliances.length === 0) {
            return NextResponse.json({ error: 'No valid appliance sequences found' }, { status: 400 });
        }

        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ appliances: formattedAppliances }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json({ error: `Python Server Error: ${errorText}` }, { status: 500 });
        }

        const data = await response.json();

        return NextResponse.json({ results: data.results });

    } catch (error) {
        console.error("Inference route error:", error);
        return NextResponse.json({ error: "Could not connect to AI backend" }, { status: 500 });
    }
}