"""
Interactive D3.js visualization for political compass posterior distribution

Generates an HTML file with interactive 2D marginal plots using D3.js
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformer_flow import TransformerAutoregressiveFlow, MultiModalGaussianMixture


def load_trained_model(checkpoint_path: str, device: str = 'cpu'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model_config = config['model_config']['transformer']
    model = TransformerAutoregressiveFlow(
        n_dimensions=config['model_config']['n_dimensions'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def create_target_distribution(config: dict) -> MultiModalGaussianMixture:
    """Create multi-modal Gaussian target distribution from config"""
    target_config = config['model_config']['target_distribution']

    mode_centers = np.array(target_config['mode_centers'])
    mode_weights = np.array(target_config['mode_weights'])
    n_modes = target_config['n_modes']
    mode_scale = target_config['mode_scale']

    return MultiModalGaussianMixture(
        n_modes=n_modes,
        mode_centers=mode_centers,
        mode_weights=mode_weights,
        mode_scale=mode_scale
    )


def generate_d3_visualization(
    model: TransformerAutoregressiveFlow,
    target_dist: MultiModalGaussianMixture,
    config: dict,
    output_path: str,
    n_samples: int = 5000,
    device: str = 'cpu'
):
    """Generate interactive D3.js visualization HTML"""

    # Sample from both distributions
    with torch.no_grad():
        model_samples = model.sample(n_samples, device=device).cpu().numpy()
    target_samples = target_dist.sample(n_samples, device='cpu').numpy()

    dimensions = config['dimensions']
    dimension_names = [d['name'] for d in dimensions]
    dimension_labels = [d['name'].replace('_', ' ').title() for d in dimensions]

    # Prepare data for D3
    data = {
        'dimensions': [
            {
                'name': d['name'],
                'label': d['name'].replace('_', ' ').title(),
                'description': d['description'],
                'range': d['range'],
                'labels': d['labels']
            }
            for d in dimensions
        ],
        'model_samples': model_samples.tolist(),
        'target_samples': target_samples.tolist(),
        'mode_info': [
            {
                'name': config['model_config']['target_distribution']['mode_descriptions'][i],
                'center': config['model_config']['target_distribution']['mode_centers'][i],
                'weight': config['model_config']['target_distribution']['mode_weights'][i]
            }
            for i in range(config['model_config']['target_distribution']['n_modes'])
        ]
    }

    # Generate HTML with embedded D3.js
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Political Compass Posterior - Interactive Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .control-group {{
            display: inline-block;
            margin: 0 15px;
        }}
        label {{
            margin-right: 10px;
            font-weight: bold;
        }}
        select {{
            padding: 5px 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }}
        #visualization {{
            display: flex;
            justify-content: center;
        }}
        .plot {{
            margin: 10px;
        }}
        .axis {{
            font-size: 12px;
        }}
        .axis-label {{
            font-size: 14px;
            font-weight: bold;
        }}
        .legend {{
            font-size: 13px;
        }}
        .model-point {{
            fill: rgba(220, 38, 38, 0.4);
            stroke: rgba(220, 38, 38, 0.8);
            stroke-width: 0.5;
        }}
        .target-point {{
            fill: rgba(37, 99, 235, 0.4);
            stroke: rgba(37, 99, 235, 0.8);
            stroke-width: 0.5;
        }}
        .mode-center {{
            fill: #f59e0b;
            stroke: #d97706;
            stroke-width: 2;
        }}
        .info-panel {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f9fafb;
            border-radius: 8px;
        }}
        .mode-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .mode-item {{
            padding: 12px;
            background-color: white;
            border-left: 4px solid #f59e0b;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .mode-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .mode-weight {{
            color: #666;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Political Compass Posterior Distribution</h1>
        <p class="subtitle">Transformer-based Autoregressive Flow</p>

        <div class="controls">
            <div class="control-group">
                <label for="dim-x">X Axis:</label>
                <select id="dim-x"></select>
            </div>
            <div class="control-group">
                <label for="dim-y">Y Axis:</label>
                <select id="dim-y"></select>
            </div>
            <div class="control-group">
                <label for="show-modes">Show Modes:</label>
                <input type="checkbox" id="show-modes" checked>
            </div>
        </div>

        <div id="visualization"></div>

        <div class="info-panel">
            <h3>Political Spectrum Modes</h3>
            <div class="mode-list" id="mode-list"></div>
        </div>
    </div>

    <script>
        const data = {json.dumps(data, indent=2)};

        const width = 800;
        const height = 600;
        const margin = {{ top: 40, right: 40, bottom: 60, left: 80 }};
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Populate dimension selectors
        const dimXSelect = d3.select("#dim-x");
        const dimYSelect = d3.select("#dim-y");

        data.dimensions.forEach((dim, i) => {{
            dimXSelect.append("option")
                .attr("value", i)
                .text(dim.label)
                .property("selected", i === 0);

            dimYSelect.append("option")
                .attr("value", i)
                .text(dim.label)
                .property("selected", i === 1);
        }});

        // Populate mode list
        const modeList = d3.select("#mode-list");
        data.mode_info.forEach(mode => {{
            modeList.append("div")
                .attr("class", "mode-item")
                .html(`
                    <div class="mode-name">${{mode.name}}</div>
                    <div class="mode-weight">Weight: ${{(mode.weight * 100).toFixed(1)}}%</div>
                `);
        }});

        function updatePlot() {{
            const dimX = +dimXSelect.property("value");
            const dimY = +dimYSelect.property("value");
            const showModes = d3.select("#show-modes").property("checked");

            // Clear previous plot
            d3.select("#visualization").html("");

            // Create SVG
            const svg = d3.select("#visualization")
                .append("svg")
                .attr("width", width)
                .attr("height", height);

            const g = svg.append("g")
                .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            // Scales
            const xScale = d3.scaleLinear()
                .domain([-1.5, 1.5])
                .range([0, plotWidth]);

            const yScale = d3.scaleLinear()
                .domain([-1.5, 1.5])
                .range([plotHeight, 0]);

            // Axes
            const xAxis = d3.axisBottom(xScale).ticks(10);
            const yAxis = d3.axisLeft(yScale).ticks(10);

            g.append("g")
                .attr("class", "axis")
                .attr("transform", `translate(0,${{plotHeight}})`)
                .call(xAxis);

            g.append("g")
                .attr("class", "axis")
                .call(yAxis);

            // Axis labels
            g.append("text")
                .attr("class", "axis-label")
                .attr("x", plotWidth / 2)
                .attr("y", plotHeight + 45)
                .attr("text-anchor", "middle")
                .text(data.dimensions[dimX].label);

            g.append("text")
                .attr("class", "axis-label")
                .attr("transform", "rotate(-90)")
                .attr("x", -plotHeight / 2)
                .attr("y", -55)
                .attr("text-anchor", "middle")
                .text(data.dimensions[dimY].label);

            // Plot target samples
            g.selectAll(".target-point")
                .data(data.target_samples)
                .enter()
                .append("circle")
                .attr("class", "target-point")
                .attr("cx", d => xScale(d[dimX]))
                .attr("cy", d => yScale(d[dimY]))
                .attr("r", 2);

            // Plot model samples
            g.selectAll(".model-point")
                .data(data.model_samples)
                .enter()
                .append("circle")
                .attr("class", "model-point")
                .attr("cx", d => xScale(d[dimX]))
                .attr("cy", d => yScale(d[dimY]))
                .attr("r", 2);

            // Plot mode centers
            if (showModes) {{
                g.selectAll(".mode-center")
                    .data(data.mode_info)
                    .enter()
                    .append("circle")
                    .attr("class", "mode-center")
                    .attr("cx", d => xScale(d.center[dimX]))
                    .attr("cy", d => yScale(d.center[dimY]))
                    .attr("r", 6)
                    .append("title")
                    .text(d => d.name);
            }}

            // Legend
            const legend = g.append("g")
                .attr("class", "legend")
                .attr("transform", `translate(${{plotWidth - 150}}, 10)`);

            legend.append("circle")
                .attr("cx", 0)
                .attr("cy", 0)
                .attr("r", 4)
                .attr("class", "target-point");
            legend.append("text")
                .attr("x", 10)
                .attr("y", 4)
                .text("Target Distribution");

            legend.append("circle")
                .attr("cx", 0)
                .attr("cy", 20)
                .attr("r", 4)
                .attr("class", "model-point");
            legend.append("text")
                .attr("x", 10)
                .attr("y", 24)
                .text("Model Distribution");

            if (showModes) {{
                legend.append("circle")
                    .attr("cx", 0)
                    .attr("cy", 40)
                    .attr("r", 4)
                    .attr("class", "mode-center");
                legend.append("text")
                    .attr("x", 10)
                    .attr("y", 44)
                    .text("Mode Centers");
            }}
        }}

        // Event listeners
        dimXSelect.on("change", updatePlot);
        dimYSelect.on("change", updatePlot);
        d3.select("#show-modes").on("change", updatePlot);

        // Initial plot
        updatePlot();
    </script>
</body>
</html>"""

    # Save HTML file
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Generated interactive D3 visualization: {output_path}")


def main():
    """Generate D3 visualization from trained model"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate D3 visualization')
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/interactive_viz.html',
                       help='Output HTML file')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples to visualize')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')

    args = parser.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_trained_model(args.checkpoint, device=device)

    # Create target distribution
    target_dist = create_target_distribution(config)

    # Generate visualization
    print(f"Generating visualization with {args.samples} samples...")
    generate_d3_visualization(
        model=model,
        target_dist=target_dist,
        config=config,
        output_path=args.output,
        n_samples=args.samples,
        device=device
    )

    print(f"\nVisualization complete! Open {args.output} in a web browser.")


if __name__ == '__main__':
    main()
