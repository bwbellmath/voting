"""
Flask server for D3 visualization interface.
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path
from data_loaders import SurveyLoader, ElectionLoader, DataAggregator

app = Flask(__name__)
CORS(app)

# Initialize loaders
survey_loader = SurveyLoader()
election_loader = ElectionLoader()
aggregator = DataAggregator()


@app.route('/')
def index():
    """Serve the main visualization page"""
    return render_template('index.html')


@app.route('/api/surveys/party-identification')
def get_party_identification():
    """Get party identification survey data"""
    data = survey_loader.load_party_identification()
    return jsonify(data.to_dict())


@app.route('/api/surveys/leaners')
def get_leaners():
    """Get leaner survey data"""
    data = survey_loader.load_leaner_data()
    return jsonify(data.to_dict())


@app.route('/api/elections/presidential')
def get_presidential():
    """Get presidential election results"""
    # Get query parameters for filtering
    years_str = request.args.get('years')
    states_str = request.args.get('states')

    years = [int(y) for y in years_str.split(',') if y] if years_str else None
    states = states_str.split(',') if states_str else None

    results = election_loader.load_presidential_results(years=years, states=states)

    return jsonify(results.to_dict())


@app.route('/api/elections/presidential/by-year')
def get_presidential_by_year():
    """Get presidential results aggregated by year"""
    results = election_loader.load_presidential_results()
    agg_data = aggregator.aggregate_by_year(results)
    return jsonify(agg_data.to_dict('records'))


@app.route('/api/elections/presidential/by-state')
def get_presidential_by_state():
    """Get presidential results aggregated by state"""
    results = election_loader.load_presidential_results()
    agg_data = aggregator.aggregate_by_state(results)
    return jsonify(agg_data.to_dict('records'))


@app.route('/api/elections/presidential/vote-shares')
def get_presidential_vote_shares():
    """Get presidential vote shares by year and state"""
    results = election_loader.load_presidential_results()
    vote_shares = aggregator.calculate_vote_shares(results)
    return jsonify(vote_shares.to_dict('records'))


@app.route('/api/elections/senate')
def get_senate():
    """Get senate election results"""
    years_str = request.args.get('years')
    states_str = request.args.get('states')

    years = [int(y) for y in years_str.split(',') if y] if years_str else None
    states = states_str.split(',') if states_str else None

    results = election_loader.load_senate_results(years=years, states=states)
    return jsonify(results.to_dict())


@app.route('/api/elections/senate/by-year')
def get_senate_by_year():
    """Get senate results aggregated by year"""
    results = election_loader.load_senate_results()
    agg_data = aggregator.aggregate_by_year(results)
    return jsonify(agg_data.to_dict('records'))


@app.route('/api/elections/house')
def get_house():
    """Get house election results"""
    years_str = request.args.get('years')
    states_str = request.args.get('states')

    years = [int(y) for y in years_str.split(',') if y] if years_str else None
    states = states_str.split(',') if states_str else None

    results = election_loader.load_house_results(years=years, states=states)
    return jsonify(results.to_dict())


@app.route('/api/metadata')
def get_metadata():
    """Get metadata about available data"""
    # Load a sample to get available years and states
    pres = election_loader.load_presidential_results()

    import pandas as pd
    df = pd.DataFrame({
        'year': pres.year,
        'state': pres.state
    })

    return jsonify({
        'years': sorted(df['year'].unique().tolist()),
        'states': sorted(df['state'].unique().tolist())
    })


if __name__ == '__main__':
    print("Starting visualization server...")
    print("Open your browser to http://localhost:5000")
    app.run(debug=True, port=5000)
