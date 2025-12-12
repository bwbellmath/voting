// Main visualization controller
class VotingVisualizer {
    constructor() {
        this.baseUrl = 'http://localhost:5000/api';
        this.colors = {
            'DEMOCRAT': '#3498db',
            'REPUBLICAN': '#e74c3c',
            'INDEPENDENT': '#95a5a6',
            'OTHER': '#f39c12',
            'republicans': '#e74c3c',
            'democrats': '#3498db',
            'independents': '#95a5a6'
        };
        this.init();
    }

    init() {
        // Load initial data
        this.loadSurveyTrends();
        this.loadElectionResults();
        this.loadVoteShareHeatmap();
        this.loadComparison();

        // Setup event listeners
        document.getElementById('updateBtn').addEventListener('click', () => this.updateMainViz());
    }

    async fetchData(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        return await response.json();
    }

    // Survey trends visualization
    async loadSurveyTrends() {
        const data = await this.fetchData('/surveys/party-identification');
        this.drawLineChart('#survey-chart', data);
    }

    drawLineChart(selector, data) {
        const svg = d3.select(selector);
        svg.selectAll('*').remove();

        const margin = {top: 20, right: 120, bottom: 50, left: 60};
        const width = parseInt(svg.style('width')) - margin.left - margin.right;
        const height = parseInt(svg.style('height')) - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Parse dates
        const parseDate = d3.timeParse('%Y-%m-%dT%H:%M:%S');
        const dates = data.date.map(d => parseDate(d));

        // Create scales
        const x = d3.scaleTime()
            .domain(d3.extent(dates))
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, 50])
            .range([height, 0]);

        // Add grid
        g.append('g')
            .attr('class', 'grid')
            .attr('opacity', 0.1)
            .call(d3.axisLeft(y)
                .tickSize(-width)
                .tickFormat(''));

        // Create line generator
        const line = d3.line()
            .x((d, i) => x(dates[i]))
            .y(d => y(d))
            .curve(d3.curveMonotoneX);

        // Draw lines for each party
        const parties = ['republicans', 'democrats', 'independents'];
        parties.forEach(party => {
            g.append('path')
                .datum(data[party])
                .attr('class', `line line-${party}`)
                .attr('d', line)
                .style('stroke', this.colors[party]);
        });

        // Add axes
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x));

        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(10).tickFormat(d => d + '%'));

        // Add labels
        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -margin.left + 15)
            .attr('x', -height / 2)
            .style('text-anchor', 'middle')
            .text('Party Identification (%)');

        // Add legend
        const legend = g.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${width + 10}, 0)`);

        parties.forEach((party, i) => {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 25})`);

            legendRow.append('rect')
                .attr('width', 15)
                .attr('height', 15)
                .attr('fill', this.colors[party]);

            legendRow.append('text')
                .attr('x', 20)
                .attr('y', 12)
                .text(party.charAt(0).toUpperCase() + party.slice(1));
        });
    }

    // Election results visualization
    async loadElectionResults() {
        const data = await this.fetchData('/elections/presidential/by-year');
        this.drawBarChart('#election-chart', data);
    }

    drawBarChart(selector, data) {
        const svg = d3.select(selector);
        svg.selectAll('*').remove();

        const margin = {top: 20, right: 100, bottom: 50, left: 60};
        const width = parseInt(svg.style('width')) - margin.left - margin.right;
        const height = parseInt(svg.style('height')) - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Group data by year
        const years = [...new Set(data.map(d => d.year))].sort();
        const parties = [...new Set(data.map(d => d.party))];

        // Create scales
        const x0 = d3.scaleBand()
            .domain(years)
            .range([0, width])
            .padding(0.2);

        const x1 = d3.scaleBand()
            .domain(parties)
            .range([0, x0.bandwidth()])
            .padding(0.05);

        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.votes)])
            .nice()
            .range([height, 0]);

        // Draw bars
        years.forEach(year => {
            const yearData = data.filter(d => d.year === year);

            const yearGroup = g.append('g')
                .attr('transform', `translate(${x0(year)}, 0)`);

            yearGroup.selectAll('rect')
                .data(yearData)
                .enter()
                .append('rect')
                .attr('class', 'bar')
                .attr('x', d => x1(d.party))
                .attr('y', d => y(d.votes))
                .attr('width', x1.bandwidth())
                .attr('height', d => height - y(d.votes))
                .attr('fill', d => this.colors[d.party] || this.colors.OTHER);
        });

        // Add axes
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x0).tickValues(years.filter((_, i) => i % 4 === 0)));

        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).tickFormat(d => (d / 1000000) + 'M'));

        // Add label
        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -margin.left + 15)
            .attr('x', -height / 2)
            .style('text-anchor', 'middle')
            .text('Total Votes (Millions)');

        // Add legend
        const legend = g.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${width + 10}, 0)`);

        parties.forEach((party, i) => {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 25})`);

            legendRow.append('rect')
                .attr('width', 15)
                .attr('height', 15)
                .attr('fill', this.colors[party] || this.colors.OTHER);

            legendRow.append('text')
                .attr('x', 20)
                .attr('y', 12)
                .text(party);
        });
    }

    // Vote share heatmap
    async loadVoteShareHeatmap() {
        const data = await this.fetchData('/elections/presidential/vote-shares');
        this.drawHeatmap('#heatmap-chart', data);
    }

    drawHeatmap(selector, data) {
        const svg = d3.select(selector);
        svg.selectAll('*').remove();

        const margin = {top: 20, right: 100, bottom: 100, left: 60};
        const width = parseInt(svg.style('width')) - margin.left - margin.right;
        const height = parseInt(svg.style('height')) - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Filter to show only Democrat and Republican vote shares
        const filteredData = data.filter(d =>
            (d.party === 'DEMOCRAT' || d.party === 'REPUBLICAN') && d.vote_share > 0
        );

        // Get unique years and states
        const years = [...new Set(filteredData.map(d => d.year))].sort();
        const states = [...new Set(filteredData.map(d => d.state))].sort().slice(0, 20); // Show first 20 states

        // Create scales
        const x = d3.scaleBand()
            .domain(years)
            .range([0, width])
            .padding(0.05);

        const y = d3.scaleBand()
            .domain(states)
            .range([0, height])
            .padding(0.05);

        const colorScale = d3.scaleSequential(d3.interpolateRdBu)
            .domain([0, 100]);

        // Draw cells
        filteredData.forEach(d => {
            if (states.includes(d.state)) {
                g.append('rect')
                    .attr('x', x(d.year))
                    .attr('y', y(d.state))
                    .attr('width', x.bandwidth())
                    .attr('height', y.bandwidth())
                    .attr('fill', colorScale(d.vote_share))
                    .attr('opacity', 0.8);
            }
        });

        // Add axes
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).tickValues(years.filter((_, i) => i % 4 === 0)))
            .selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end');

        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y));

        // Add title
        g.append('text')
            .attr('x', width / 2)
            .attr('y', -5)
            .style('text-anchor', 'middle')
            .text('Vote Share by State and Year (%)');
    }

    // Comparison visualization
    async loadComparison() {
        const surveys = await this.fetchData('/surveys/leaners');
        const elections = await this.fetchData('/elections/presidential/by-year');
        this.drawComparison('#comparison-chart', surveys, elections);
    }

    drawComparison(selector, surveys, elections) {
        const svg = d3.select(selector);
        svg.selectAll('*').remove();

        const margin = {top: 20, right: 120, bottom: 50, left: 60};
        const width = parseInt(svg.style('width')) - margin.left - margin.right;
        const height = parseInt(svg.style('height')) - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Parse dates
        const parseDate = d3.timeParse('%Y-%m-%dT%H:%M:%S');
        const dates = surveys.date.map(d => parseDate(d));

        // Create scales
        const x = d3.scaleTime()
            .domain(d3.extent(dates))
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([30, 55])
            .range([height, 0]);

        // Create line generator
        const line = d3.line()
            .x((d, i) => x(dates[i]))
            .y(d => y(d))
            .curve(d3.curveMonotoneX);

        // Draw survey lines
        g.append('path')
            .datum(surveys.republican_leaners)
            .attr('class', 'line')
            .attr('d', line)
            .style('stroke', this.colors.republicans)
            .style('stroke-width', 2)
            .style('stroke-dasharray', '5,5');

        g.append('path')
            .datum(surveys.democratic_leaners)
            .attr('class', 'line')
            .attr('d', line)
            .style('stroke', this.colors.democrats)
            .style('stroke-width', 2)
            .style('stroke-dasharray', '5,5');

        // Add axes
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x));

        g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).tickFormat(d => d + '%'));

        // Add legend
        const legend = g.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${width + 10}, 0)`);

        const legendData = [
            {label: 'Rep Leaners', color: this.colors.republicans},
            {label: 'Dem Leaners', color: this.colors.democrats}
        ];

        legendData.forEach((d, i) => {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 25})`);

            legendRow.append('line')
                .attr('x1', 0)
                .attr('x2', 15)
                .attr('y1', 7)
                .attr('y2', 7)
                .style('stroke', d.color)
                .style('stroke-width', 2)
                .style('stroke-dasharray', '5,5');

            legendRow.append('text')
                .attr('x', 20)
                .attr('y', 12)
                .text(d.label);
        });
    }

    // Main visualization updater
    async updateMainViz() {
        const dataType = document.getElementById('dataType').value;
        const aggregation = document.getElementById('aggregation').value;
        const vizType = document.getElementById('vizType').value;

        console.log(`Updating: ${dataType} - ${aggregation} - ${vizType}`);

        // Implement main viz updates based on controls
        // This is a placeholder for dynamic updates
    }
}

// Initialize visualizer when page loads
document.addEventListener('DOMContentLoaded', () => {
    const viz = new VotingVisualizer();
});
